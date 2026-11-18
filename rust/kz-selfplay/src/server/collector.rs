use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write as _};
use std::time::Instant;

use board_game::board::Board;
use flume::Receiver;

use kz_core::mapping::BoardMapper;

use crate::binary_output::BinaryOutput;
use crate::server::protocol::{Evals, GeneratorUpdate, ServerUpdate};

const STATS_INTERVAL_SECS: f32 = 60.0;

pub fn collector_main<B: Board, W: std::io::Write>(
    game: &str,
    mut writer: BufWriter<W>,
    muzero: bool,
    games_per_file: usize,
    first_gen: u32,
    output_folder: &str,
    mapper: impl BoardMapper<B>,
    update_receiver: Receiver<GeneratorUpdate<B>>,
) -> BufWriter<W> {
    let new_output = |gen: u32| {
        let path = format!("{}/games_{}", output_folder, gen);
        println!("Collector: start writing to {}", path);
        BinaryOutput::new(path, game, mapper).expect("Error while creating output files")
    };

    create_dir_all(&output_folder).expect("Failed to create output folder");

    let mut curr_gen = first_gen;
    let mut curr_output = new_output(curr_gen);

    let mut total_games = 0;
    let mut total_moves = 0;
    let mut counter = Counter::default();

    let mut last_print_time = Instant::now();
    let mut curr_game_lengths = HashMap::new();
    let mut drain_remaining: Option<usize> = None;
    let mut drained_generators = HashSet::new();

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,

            GeneratorUpdate::DrainStarted { generators } => {
                assert!(drain_remaining.is_none(), "Duplicate drain request");
                drain_remaining = Some(generators);
                println!("Drain started: {} games draining", generators);
            }

            GeneratorUpdate::GeneratorDrained { generator_id } => {
                let remaining = drain_remaining
                    .as_mut()
                    .expect("Generator drained before drain request");
                assert!(drained_generators.insert(generator_id), "Generator drained twice");
                *remaining = remaining.checked_sub(1).expect("Too many drained generators");
                println!("Drain progress: {} games draining", remaining);
                if *remaining == 0 {
                    curr_output
                        .checkpoint()
                        .expect("Error checkpointing partial selfplay output");
                    println!(
                        "Drain complete: {} games saved in partial generation {}",
                        curr_output.game_count(),
                        curr_gen
                    );
                    let message = ServerUpdate::Drained {
                        index: curr_gen,
                        games: curr_output.game_count(),
                    };
                    writer
                        .write_all(serde_json::to_string(&message).unwrap().as_bytes())
                        .unwrap();
                    writer.write_all(b"\n").unwrap();
                    writer.flush().unwrap();
                }
            }

            GeneratorUpdate::DrainRejected { reason } => {
                let message = ServerUpdate::DrainRejected { reason };
                writer
                    .write_all(serde_json::to_string(&message).unwrap().as_bytes())
                    .unwrap();
                writer.write_all(b"\n").unwrap();
                writer.flush().unwrap();
            }

            GeneratorUpdate::StartedSimulation { generator_id } => {
                curr_game_lengths.insert(generator_id, 0);
            }

            GeneratorUpdate::FinishedMove {
                generator_id,
                curr_game_length,
            } => {
                curr_game_lengths.insert(generator_id, curr_game_length);
                counter.moves += 1;
            }

            GeneratorUpdate::FinishedSimulation {
                generator_id,
                simulation,
            } => {
                counter.games += 1;
                curr_game_lengths.remove(&generator_id);

                // write file to disk, possibly starting a new generation
                curr_output
                    .append(&simulation)
                    .expect("Error during simulation appending");

                if curr_output.game_count() >= games_per_file {
                    curr_output.finish().expect("Error while finishing output file");

                    let prev_i = curr_gen;
                    curr_gen += 1;
                    curr_output = new_output(curr_gen);

                    let message = ServerUpdate::FinishedFile { index: prev_i };
                    writer
                        .write_all(serde_json::to_string(&message).unwrap().as_bytes())
                        .unwrap();
                    writer.write_all(&[b'\n']).unwrap();
                    writer.flush().unwrap();
                }
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
                .to_string(
                    delta,
                    total_moves,
                    total_games,
                    &curr_game_lengths,
                    muzero,
                    drain_remaining,
                )
                .unwrap();
            print!("{}", info);

            counter = Counter::default();
            last_print_time = now;
        }
    }

    writer
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
        drain_remaining: Option<usize>,
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
        write!(
            f,
            "  running games: {}, lengths: min {} max {} mean {:.2}",
            game_lengths.len(),
            min_game_length,
            max_game_length,
            mean_game_length
        )?;
        if let Some(remaining) = drain_remaining {
            write!(f, " ({} games draining)", remaining)?;
        }
        writeln!(f)?;

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
