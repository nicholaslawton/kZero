//! Persistence of self-play progress between sessions.
//!
//! A checkpoint stores the games already completed in the current generation file and the
//! committed moves of games still in progress. Moves are stored as policy indices and replayed
//! from the default start position on load, which also restores any history kept by the board.
//! Unfinished searches are not stored.

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use board_game::board::Board;
use board_game::pov::ScalarPov;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;
use serde::{Deserialize, Serialize};

use kz_core::mapping::BoardMapper;
use kz_core::network::ZeroEvaluation;
use kz_core::zero::values::ZeroValuesPov;

use crate::simulation::{Position, Simulation};

const VERSION: u32 = 1;
const FILE_NAME: &str = "game_checkpoint.json";
const CONSUMED_FILE_NAME: &str = "game_checkpoint.consumed.json";

/// An in-progress game waiting to be continued by a generator.
#[derive(Debug)]
pub struct ResumedGame<B: Board> {
    pub id: usize,
    pub moves: Vec<B::Move>,
}

#[derive(Debug)]
pub struct LoadedCheckpoint<B: Board> {
    pub finished: Vec<Simulation<'static, B>>,
    pub active: Vec<Vec<Position<'static, B>>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedCheckpoint {
    version: u32,
    game: String,
    gen: u32,
    finished: Vec<Vec<SavedMove>>,
    active: Vec<Vec<SavedMove>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedMove {
    mv: usize,
    full_search: bool,
    visits: u64,
    zero: SavedEvaluation,
    net: SavedEvaluation,
}

// Floats are stored as bit patterns so values round-trip exactly, including NaN.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedEvaluation {
    values: [u32; 5],
    policy: Vec<u32>,
}

pub fn checkpoint_path(output_folder: &str) -> PathBuf {
    Path::new(output_folder).join(FILE_NAME)
}

pub fn save<'p, B: Board + 'p, M: BoardMapper<B>>(
    output_folder: &str,
    game: &str,
    gen: u32,
    mapper: M,
    finished: &[Simulation<'static, B>],
    active: impl IntoIterator<Item = &'p [Position<'static, B>]>,
) -> io::Result<()> {
    let saved = SavedCheckpoint {
        version: VERSION,
        game: game.to_owned(),
        gen,
        finished: finished.iter().map(|sim| save_moves(&sim.positions, mapper)).collect(),
        active: active
            .into_iter()
            .map(|positions| save_moves(positions, mapper))
            .collect(),
    };

    let path = checkpoint_path(output_folder);
    let tmp = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&tmp)?);
    serde_json::to_writer(&mut writer, &saved)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    std::fs::rename(&tmp, &path)?;
    File::open(output_folder)?.sync_all()?;
    Ok(())
}

/// Load the checkpoint for `gen`, if any. Every move is validated by replaying it from `start`.
pub fn load<B: Board, M: BoardMapper<B>>(
    output_folder: &str,
    game: &str,
    gen: u32,
    start: &B,
    mapper: M,
) -> io::Result<Option<LoadedCheckpoint<B>>> {
    let path = checkpoint_path(output_folder);
    if !path.exists() {
        return Ok(None);
    }
    let saved: SavedCheckpoint = serde_json::from_reader(BufReader::new(File::open(&path)?))?;

    if saved.version != VERSION {
        return Err(invalid(format!("unsupported version {}", saved.version)));
    }
    if saved.game != game {
        return Err(invalid(format!(
            "checkpoint is for game '{}', not '{}'",
            saved.game, game
        )));
    }
    if saved.gen != gen {
        return Err(invalid(format!(
            "checkpoint is for generation {}, but selfplay is starting at generation {}",
            saved.gen, gen
        )));
    }

    let mut finished = vec![];
    for moves in &saved.finished {
        let (positions, final_board) = replay(start, moves, mapper)?;
        finished.push(Simulation { positions, final_board });
    }
    let mut active = vec![];
    for moves in &saved.active {
        let (positions, board) = replay(start, moves, mapper)?;
        // A checkpoint can land between a game's final move and its completion message.
        if board.is_done() {
            finished.push(Simulation {
                positions,
                final_board: board,
            });
        } else {
            active.push(positions);
        }
    }

    Ok(Some(LoadedCheckpoint { finished, active }))
}

/// Keep the loaded checkpoint for manual recovery, but make sure it is never loaded twice.
pub fn mark_consumed(output_folder: &str) -> io::Result<()> {
    std::fs::rename(
        checkpoint_path(output_folder),
        Path::new(output_folder).join(CONSUMED_FILE_NAME),
    )?;
    File::open(output_folder)?.sync_all()
}

fn save_moves<B: Board, M: BoardMapper<B>>(positions: &[Position<B>], mapper: M) -> Vec<SavedMove> {
    positions
        .iter()
        .map(|position| SavedMove {
            mv: mapper.move_to_index(&position.board, position.played_mv),
            full_search: position.is_full_search,
            visits: position.zero_visits,
            zero: save_evaluation(&position.zero_evaluation),
            net: save_evaluation(&position.net_evaluation),
        })
        .collect()
}

fn save_evaluation(eval: &ZeroEvaluation) -> SavedEvaluation {
    SavedEvaluation {
        values: eval.values.to_slice().map(f32::to_bits),
        policy: eval.policy.iter().map(|p| p.to_bits()).collect(),
    }
}

fn load_evaluation(saved: &SavedEvaluation) -> ZeroEvaluation<'static> {
    let [value, win, draw, loss, moves_left] = saved.values.map(f32::from_bits);
    ZeroEvaluation {
        values: ZeroValuesPov {
            value: ScalarPov::new(value),
            wdl: WDL { win, draw, loss },
            moves_left,
        },
        policy: Cow::Owned(saved.policy.iter().map(|&p| f32::from_bits(p)).collect()),
    }
}

fn replay<B: Board, M: BoardMapper<B>>(
    start: &B,
    moves: &[SavedMove],
    mapper: M,
) -> io::Result<(Vec<Position<'static, B>>, B)> {
    let mut board = start.clone();
    let mut positions = Vec::with_capacity(moves.len());

    for (i, saved) in moves.iter().enumerate() {
        let mut available = 0;
        if let Ok(moves) = board.available_moves() {
            moves.for_each(|_| available += 1);
        }
        if saved.zero.policy.len() != available || saved.net.policy.len() != available {
            return Err(invalid(format!("move {i} has a policy for the wrong number of moves")));
        }

        let mv = mapper
            .index_to_move(&board, saved.mv)
            .filter(|&mv| board.is_available_move(mv) == Ok(true))
            .ok_or_else(|| invalid(format!("move {i} is not available")))?;

        positions.push(Position {
            board: board.clone(),
            is_full_search: saved.full_search,
            played_mv: mv,
            zero_visits: saved.visits,
            zero_evaluation: load_evaluation(&saved.zero),
            net_evaluation: load_evaluation(&saved.net),
        });
        board
            .play(mv)
            .map_err(|e| invalid(format!("move {i} could not be played: {e:?}")))?;
    }

    Ok((positions, board))
}

fn invalid(message: String) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Invalid selfplay checkpoint: {message}"),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use board_game::games::ttt::TTTBoard;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    use kz_core::mapping::oriflambe::repetition::{OriflambeRepetitionBoard, OriflambeRepetitionMapper};
    use kz_core::mapping::ttt::TTTStdMapper;

    use super::*;
    use crate::binary_output::BinaryOutput;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("kz-game-checkpoint-{}-{}", std::process::id(), id));
            std::fs::create_dir_all(&path).unwrap();
            TestDir(path)
        }

        fn folder(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn evaluation(rng: &mut StdRng, moves: usize, nan_moves_left: bool) -> ZeroEvaluation<'static> {
        let win = rng.gen_range(0.0..0.5);
        let draw = rng.gen_range(0.0..0.5);
        ZeroEvaluation {
            values: ZeroValuesPov {
                value: ScalarPov::new(rng.gen_range(-1.0..1.0)),
                wdl: WDL {
                    win,
                    draw,
                    loss: 1.0 - win - draw,
                },
                moves_left: if nan_moves_left { f32::NAN } else { rng.gen() },
            },
            policy: Cow::Owned(vec![1.0 / moves as f32; moves]),
        }
    }

    fn play(rng: &mut StdRng, max_moves: usize) -> (Vec<Position<'static, TTTBoard>>, TTTBoard) {
        play_from(TTTBoard::default(), rng, max_moves)
    }

    fn play_from<B: Board>(start: B, rng: &mut StdRng, max_moves: usize) -> (Vec<Position<'static, B>>, B) {
        let mut board = start;
        let mut positions = vec![];
        while !board.is_done() && positions.len() < max_moves {
            let moves: Vec<_> = board.available_moves().unwrap().collect();
            let mv = *moves.choose(rng).unwrap();
            positions.push(Position {
                board: board.clone(),
                is_full_search: rng.gen(),
                played_mv: mv,
                zero_visits: rng.gen_range(1..1000),
                zero_evaluation: evaluation(rng, moves.len(), false),
                net_evaluation: evaluation(rng, moves.len(), true),
            });
            board.play(mv).unwrap();
        }
        (positions, board)
    }

    fn play_again(sim: &Simulation<'static, TTTBoard>) -> Simulation<'static, TTTBoard> {
        let positions = sim
            .positions
            .iter()
            .map(|p| Position {
                board: p.board.clone(),
                is_full_search: p.is_full_search,
                played_mv: p.played_mv,
                zero_visits: p.zero_visits,
                zero_evaluation: p.zero_evaluation.clone(),
                net_evaluation: p.net_evaluation.clone(),
            })
            .collect();
        Simulation {
            positions,
            final_board: sim.final_board.clone(),
        }
    }

    fn output_bytes(dir: &TestDir, name: &str, games: &[Simulation<TTTBoard>]) -> Vec<Vec<u8>> {
        let path = dir.0.join(name);
        let mut output = BinaryOutput::new(&path, "ttt", TTTStdMapper).unwrap();
        for game in games {
            output.append(game).unwrap();
        }
        output.finish().unwrap();
        ["bin", "off", "json"]
            .iter()
            .map(|ext| std::fs::read(path.with_extension(ext)).unwrap())
            .collect()
    }

    #[test]
    fn reloaded_games_produce_identical_training_data() {
        let dir = TestDir::new();
        let mut rng = StdRng::seed_from_u64(0);
        let finished: Vec<_> = (0..3)
            .map(|_| {
                let (positions, final_board) = play(&mut rng, usize::MAX);
                Simulation { positions, final_board }
            })
            .collect();
        let mut active: Vec<_> = [2, 5].iter().map(|&n| play(&mut rng, n).0).collect();
        // the same finished game, recorded as still in progress
        let (positions, final_board) = play(&mut rng.clone(), usize::MAX);
        active.push(play(&mut rng, usize::MAX).0);
        let mut expected_finished: Vec<_> = finished.iter().map(|sim| play_again(sim)).collect();
        expected_finished.push(Simulation { positions, final_board });

        save(
            dir.folder(),
            "ttt",
            4,
            TTTStdMapper,
            &finished,
            active.iter().map(Vec::as_slice),
        )
        .unwrap();
        let loaded = load(dir.folder(), "ttt", 4, &TTTBoard::default(), TTTStdMapper)
            .unwrap()
            .unwrap();

        assert_eq!(
            output_bytes(&dir, "expected", &expected_finished),
            output_bytes(&dir, "actual", &loaded.finished)
        );

        active.pop();
        assert_eq!(loaded.active.len(), active.len());
        for (expected, actual) in active.iter().zip(&loaded.active) {
            assert_eq!(expected.len(), actual.len());
            for (e, a) in expected.iter().zip(actual) {
                assert_eq!(e.board, a.board);
                assert_eq!(e.played_mv, a.played_mv);
                assert_eq!(e.is_full_search, a.is_full_search);
                assert_eq!(e.zero_visits, a.zero_visits);
                for (e, a) in [
                    (&e.zero_evaluation, &a.zero_evaluation),
                    (&e.net_evaluation, &a.net_evaluation),
                ] {
                    assert_eq!(
                        e.values.to_slice().map(f32::to_bits),
                        a.values.to_slice().map(f32::to_bits)
                    );
                    assert_eq!(e.policy, a.policy);
                }
            }
        }

        mark_consumed(dir.folder()).unwrap();
        assert!(load(dir.folder(), "ttt", 4, &TTTBoard::default(), TTTStdMapper)
            .unwrap()
            .is_none());
    }

    #[test]
    fn oriflambe_games_replay_from_saved_moves() {
        let dir = TestDir::new();
        let mut rng = StdRng::seed_from_u64(2);
        let start = OriflambeRepetitionBoard::default();
        let finished: Vec<_> = (0..2)
            .map(|_| {
                let (positions, final_board) = play_from(start.clone(), &mut rng, 400);
                Simulation { positions, final_board }
            })
            .collect();
        let active: Vec<_> = (0..4)
            .map(|i| play_from(start.clone(), &mut rng, 50 * i + 1).0)
            .collect();
        let mapper = OriflambeRepetitionMapper;

        save(
            dir.folder(),
            "oriflambe",
            0,
            mapper,
            &finished,
            active.iter().map(Vec::as_slice),
        )
        .unwrap();
        let loaded = load(dir.folder(), "oriflambe", 0, &start, mapper).unwrap().unwrap();

        let moves = |positions: &[Position<OriflambeRepetitionBoard>]| -> Vec<_> {
            positions.iter().map(|p| (p.board.clone(), p.played_mv)).collect()
        };
        let original = finished.iter().map(|s| &s.positions).chain(&active);
        let reloaded = loaded.finished.iter().map(|s| &s.positions).chain(&loaded.active);
        assert_eq!(original.clone().count(), reloaded.clone().count());
        for (e, a) in original.zip(reloaded) {
            assert_eq!(moves(e), moves(a));
        }
        for (e, a) in finished.iter().zip(&loaded.finished) {
            assert_eq!(e.final_board, a.final_board);
        }
    }

    #[test]
    fn mismatched_or_invalid_checkpoints_are_rejected() {
        let dir = TestDir::new();
        let mut rng = StdRng::seed_from_u64(1);
        let active = vec![play(&mut rng, 3).0];
        save(
            dir.folder(),
            "ttt",
            2,
            TTTStdMapper,
            &[],
            active.iter().map(Vec::as_slice),
        )
        .unwrap();

        let load_gen = |gen| load(dir.folder(), "ttt", gen, &TTTBoard::default(), TTTStdMapper);
        assert!(load_gen(3).is_err());
        assert!(load(dir.folder(), "sttt", 2, &TTTBoard::default(), TTTStdMapper).is_err());

        // replay the first move twice, which is never legal
        let path = checkpoint_path(dir.folder());
        let mut saved: SavedCheckpoint = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        saved.active[0][1].mv = saved.active[0][0].mv;
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        assert!(load_gen(2).is_err());
    }
}
