use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write};
use std::time::Instant;

use board_game::board::Board;
use flume::Receiver;
use itertools::Itertools;

use kz_core::mapping::BoardMapper;

use crate::binary_output::BinaryOutput;
use crate::server::game_checkpoint;
use crate::server::protocol::{Evals, GeneratorUpdate, ServerUpdate};
use crate::simulation::{Position, Simulation};

const STATS_INTERVAL_SECS: f32 = 60.0;

pub fn collector_main<B: Board, M: BoardMapper<B>, W: Write>(
    game: &str,
    mut writer: BufWriter<W>,
    muzero: bool,
    games_per_file: usize,
    first_gen: u32,
    output_folder: &str,
    mapper: M,
    update_receiver: Receiver<GeneratorUpdate<B>>,
    resumed_finished: Vec<Simulation<'static, B>>,
    mut resumed_pending: HashMap<usize, Vec<Position<'static, B>>>,
) {
    create_dir_all(&output_folder).expect("Failed to create output folder");

    let mut output = GenerationOutput::new(game, output_folder, mapper, games_per_file, first_gen);
    for simulation in resumed_finished {
        output.append(simulation, &mut writer);
    }

    let mut total_games = 0;
    let mut total_moves = 0;
    let mut counter = Counter::default();

    let mut last_print_time = Instant::now();
    let mut curr_game_lengths = HashMap::new();
    let mut active_games: HashMap<usize, Vec<Position<'static, B>>> = HashMap::new();

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,

            GeneratorUpdate::Checkpoint => {
                if muzero {
                    eprintln!("Checkpoints are only supported for AlphaZero");
                    std::process::exit(1);
                }

                let active = active_games
                    .values()
                    .chain(resumed_pending.values())
                    .filter(|positions| !positions.is_empty())
                    .map(|positions| positions.as_slice())
                    .collect_vec();
                let active_count = active.len();
                let games = output.games.len();

                match game_checkpoint::save(output_folder, game, output.gen, mapper, &output.games, active) {
                    Ok(()) => {
                        println!(
                            "Checkpoint complete: generation {} with {} finished and {} in-progress games, exiting",
                            output.gen, games, active_count
                        );
                        let message = ServerUpdate::Checkpointed {
                            index: output.gen,
                            games,
                            active: active_count,
                        };
                        send_update(&mut writer, &message);
                        // Unfinished searches are discarded, so there is nothing left to shut down cleanly.
                        std::process::exit(0);
                    }
                    Err(e) => {
                        eprintln!("Failed to save selfplay checkpoint: {e}");
                        std::process::exit(1);
                    }
                }
            }

            GeneratorUpdate::StartedSimulation { generator_id } => {
                curr_game_lengths.insert(generator_id, 0);
                active_games.insert(generator_id, vec![]);
            }

            GeneratorUpdate::ResumedSimulation { generator_id, saved_id } => {
                let positions = resumed_pending
                    .remove(&saved_id)
                    .expect("Resumed game was already taken");
                curr_game_lengths.insert(generator_id, positions.len());
                active_games.insert(generator_id, positions);
            }

            GeneratorUpdate::FinishedMove {
                generator_id,
                curr_game_length,
            } => {
                curr_game_lengths.insert(generator_id, curr_game_length);
                counter.moves += 1;
            }

            GeneratorUpdate::RecordedMove { generator_id, position } => {
                let positions = active_games
                    .get_mut(&generator_id)
                    .expect("Move recorded for a game that was not started");
                positions.push(position);
                curr_game_lengths.insert(generator_id, positions.len());
                counter.moves += 1;
            }

            GeneratorUpdate::FinishedSimulation {
                generator_id,
                simulation,
            } => {
                counter.games += 1;
                curr_game_lengths.remove(&generator_id);
                output.append(simulation, &mut writer);
            }

            GeneratorUpdate::FinishedGame {
                generator_id,
                final_board,
            } => {
                counter.games += 1;
                curr_game_lengths.remove(&generator_id);
                let positions = active_games
                    .remove(&generator_id)
                    .expect("Game finished that was not started");
                output.append(Simulation { positions, final_board }, &mut writer);
            }

            GeneratorUpdate::RootEvals(evals) => {
                counter.root_evals += evals;
            }
            GeneratorUpdate::ExpandEvals(evals) => {
                counter.expand_evals += evals;
            }
        }

        // periodically print stats
        let now = Instant::now();
        let delta = (now - last_print_time).as_secs_f32();
        if delta >= STATS_INTERVAL_SECS {
            total_games += counter.games;
            total_moves += counter.moves;

            let info = counter
                .to_string(delta, total_moves, total_games, &curr_game_lengths, muzero)
                .unwrap();
            print!("{}", info);

            counter = Counter::default();
            last_print_time = now;
        }
    }

    send_update(&mut writer, &ServerUpdate::Stopped);
}

fn send_update(writer: &mut BufWriter<impl Write>, message: &ServerUpdate) {
    writer
        .write_all(serde_json::to_string(message).unwrap().as_bytes())
        .unwrap();
    writer.write_all(&[b'\n']).unwrap();
    writer.flush().unwrap();
}

/// The generation file currently being written, with its games kept in memory for checkpoints.
struct GenerationOutput<'a, B: Board, M: BoardMapper<B>> {
    game: &'a str,
    folder: &'a str,
    mapper: M,
    games_per_file: usize,

    gen: u32,
    output: BinaryOutput<B, M>,
    games: Vec<Simulation<'static, B>>,
}

impl<'a, B: Board, M: BoardMapper<B>> GenerationOutput<'a, B, M> {
    fn new(game: &'a str, folder: &'a str, mapper: M, games_per_file: usize, gen: u32) -> Self {
        GenerationOutput {
            game,
            folder,
            mapper,
            games_per_file,
            gen,
            output: open_output(game, folder, mapper, gen),
            games: vec![],
        }
    }

    // write game to disk, possibly starting a new generation
    fn append(&mut self, simulation: Simulation<'static, B>, writer: &mut BufWriter<impl Write>) {
        self.output
            .append(&simulation)
            .expect("Error during simulation appending");
        self.games.push(simulation);

        if self.output.game_count() >= self.games_per_file {
            self.output.finish().expect("Error while finishing output file");

            let prev_i = self.gen;
            self.gen += 1;
            self.output = open_output(self.game, self.folder, self.mapper, self.gen);
            self.games.clear();

            send_update(writer, &ServerUpdate::FinishedFile { index: prev_i });
        }
    }
}

fn open_output<B: Board, M: BoardMapper<B>>(game: &str, folder: &str, mapper: M, gen: u32) -> BinaryOutput<B, M> {
    let path = format!("{}/games_{}", folder, gen);
    println!("Collector: start writing to {}", path);
    BinaryOutput::new(path, game, mapper).expect("Error while creating output files")
}

#[derive(Default, Debug)]
struct Counter {
    moves: u64,
    games: u64,

    root_evals: Evals,
    expand_evals: Evals,
}

impl Counter {
    fn to_string(
        &self,
        delta: f32,
        total_moves: u64,
        total_games: u64,
        game_lengths: &HashMap<usize, usize>,
        muzero: bool,
    ) -> Result<String, std::fmt::Error> {
        let move_throughput = self.moves as f32 / delta;
        let games_per_hour = self.games as f32 / delta * 3600.0;

        let min_game_length = game_lengths.values().copied().min().unwrap_or(0);
        let max_game_length = game_lengths.values().copied().max().unwrap_or(0);
        let mean_game_length = game_lengths.values().copied().sum::<usize>() as f32 / game_lengths.len() as f32;

        let mut result = String::new();
        let f = &mut result;

        writeln!(f, "Selfplay info:")?;
        if muzero {
            write_evals(f, "expand evals", self.expand_evals, delta);
            write_evals(f, "root   evals", self.root_evals, delta);
        } else {
            assert_eq!(self.root_evals, Evals::default());
            write_evals(f, "evals", self.expand_evals, delta);
        }
        writeln!(
            f,
            "  moves: {:.2}/s => {}, games: {:.1}/h => {}",
            move_throughput, total_moves, games_per_hour, total_games
        )?;
        writeln!(
            f,
            "  running games: {}, lengths: min {} max {} mean {:.2}",
            game_lengths.len(),
            min_game_length,
            max_game_length,
            mean_game_length
        )?;

        Ok(result)
    }
}

fn write_evals(f: &mut String, name: &str, evals: Evals, delta: f32) {
    let Evals {
        real,
        potential,
        cached,
    } = evals;

    let real_tp = real as f32 / delta;
    let cached_tp = cached as f32 / delta;
    let potential_tp = potential as f32 / delta;

    let hit_rate = (cached as f32) / (real + cached) as f32;
    let fill_rate = (real as f32) / (potential as f32);

    writeln!(
        f,
        "  {}/s: real: {:.1}, cached: {:.1}, potential: {:.1} (hit: {:.2}, fill: {:.2})",
        name, real_tp, cached_tp, potential_tp, hit_rate, fill_rate
    )
    .unwrap();
}
