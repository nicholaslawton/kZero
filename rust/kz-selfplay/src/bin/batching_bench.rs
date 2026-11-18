use std::borrow::Cow;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use futures::executor::ThreadPoolBuilder;
use futures::future::{AbortHandle, Abortable};
use kn_cuda_sys::wrapper::handle::CudaDevice;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use serde::Deserialize;
use serde_json::json;

use kz_core::mapping::oriflambe::repetition::{OriflambeRepetitionBoard, OriflambeRepetitionMapper};
use kz_core::network::common::PreparedEvalInput;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::dummy::{uniform_policy, uniform_values};
use kz_core::network::job_channel::job_pair;
use kz_core::network::ZeroEvaluation;
use kz_selfplay::server::executor::{batched_executor_loop_with_settings, BatchingSettings, RunCondition};
use kz_selfplay::server::generator_alphazero::generator_alphazero_main;
use kz_selfplay::server::protocol::{Evals, GeneratorUpdate, Settings};

/// Fixed-network Oriflambe self-play benchmark. Does not write games or training state.
#[derive(Debug, Parser)]
struct Args {
    /// A training generation's settings.json.
    #[arg(long)]
    settings: PathBuf,
    #[arg(long, required_unless_present = "dummy", conflicts_with = "dummy")]
    network: Option<PathBuf>,
    /// Smoke-test the harness without CUDA inference; not useful for tuning deadlines.
    #[arg(long)]
    dummy: bool,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    concurrent_games: u32,
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..))]
    seconds: u32,
    #[arg(long, default_value_t = 10)]
    warmup_seconds: u32,
    /// 0 preserves threshold-only batching; low concurrency will then stall.
    #[arg(long, default_value_t = 100)]
    max_batch_wait_ms: u32,
    #[arg(long)]
    batch_timings: bool,
}

#[derive(Deserialize)]
struct TrainingSettings {
    fixed_settings: FixedSettings,
    selfplay_settings: Settings,
}

#[derive(Deserialize)]
struct FixedSettings {
    game: GameSettings,
    muzero: bool,
    start_pos: String,
    cpu_threads_per_device: usize,
    gpu_batch_size: usize,
    search_batch_size: usize,
    eval_random_symmetries: bool,
}

#[derive(Deserialize)]
struct GameSettings {
    name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let settings: TrainingSettings = serde_json::from_reader(File::open(&args.settings)?)?;
    let fixed = settings.fixed_settings;
    if fixed.game.name != "oriflambe-repetition" || fixed.muzero || fixed.start_pos != "default" {
        return Err("benchmark supports AlphaZero oriflambe-repetition with the default start position".into());
    }
    if fixed.cpu_threads_per_device == 0
        || fixed.search_batch_size == 0
        || fixed.gpu_batch_size < fixed.search_batch_size
    {
        return Err("settings require CPU threads > 0 and GPU batch size >= search batch size > 0".into());
    }
    let max_batch_size = fixed.gpu_batch_size;
    let job_count = max_batch_size / fixed.search_batch_size;
    let batching = BatchingSettings {
        max_wait: (args.max_batch_wait_ms != 0).then(|| Duration::from_millis(args.max_batch_wait_ms.into())),
        timings: args.batch_timings,
        report: true,
    };
    let mapper = OriflambeRepetitionMapper;
    let graph = args
        .network
        .as_ref()
        .map(|path| load_graph_from_onnx_path(path, false).map(|graph| optimize_graph(&graph, Default::default())))
        .transpose()?;
    let device = if args.dummy {
        None
    } else {
        Some(
            CudaDevice::new(args.device)
                .map_err(|error| format!("CUDA device {} unavailable: {:?}", args.device, error))?,
        )
    };
    let (client, server) = job_pair::<PreparedEvalInput<OriflambeRepetitionBoard>, ZeroEvaluation<'static>>(job_count);
    let (updates, receiver) = flume::unbounded();
    let pool = ThreadPoolBuilder::new()
        .pool_size(fixed.cpu_threads_per_device)
        .create()?;
    let mut aborts = vec![];
    let mut settings_senders = vec![];

    for generator_id in 0..args.concurrent_games as usize {
        let (sender, settings_receiver) = flume::bounded(1);
        sender.send(settings.selfplay_settings.clone())?;
        settings_senders.push(sender);
        let (abort, registration) = AbortHandle::new_pair();
        aborts.push(abort);
        let generator = generator_alphazero_main(
            generator_id,
            |_| OriflambeRepetitionBoard::default(),
            settings_receiver,
            fixed.search_batch_size,
            client.clone(),
            mapper,
            fixed.eval_random_symmetries,
            updates.clone(),
            Arc::new(Mutex::new(false)),
        );
        pool.spawn_ok(async move {
            let _ = Abortable::new(generator, registration).await;
        });
    }
    drop(client);

    let result = crossbeam::scope(|scope| {
        let (graphs, graph_receiver) = flume::bounded(1);
        graphs.send(Some(graph)).unwrap();
        drop(graphs);
        let (ready_sender, ready) = flume::bounded(1);
        let eval_updates = updates.clone();
        scope
            .builder()
            .name("batching-bench".into())
            .spawn(move |_| {
                batched_executor_loop_with_settings(
                    max_batch_size,
                    RunCondition::JobCount(job_count),
                    batching,
                    graph_receiver,
                    server,
                    |graph| {
                        let network =
                            graph.map(|graph| CudaNetwork::new(mapper, &graph, max_batch_size, device.unwrap()));
                        ready_sender.send(()).unwrap();
                        network
                    },
                    |network, inputs| {
                        let result = match network {
                            Some(network) => network.evaluate_prepared_batch(inputs),
                            None => inputs
                                .iter()
                                .map(|input| ZeroEvaluation {
                                    values: uniform_values(),
                                    policy: Cow::Owned(uniform_policy(input.policy_indices.len())),
                                })
                                .collect(),
                        };
                        eval_updates
                            .send(GeneratorUpdate::ExpandEvals(Evals::new(
                                inputs.len() as u64,
                                max_batch_size as u64,
                                0,
                            )))
                            .unwrap();
                        result
                    },
                );
            })
            .unwrap();
        drop(updates);
        ready.recv().unwrap();
        let warmup_end = Instant::now() + Duration::from_secs(args.warmup_seconds.into());
        let end = warmup_end + Duration::from_secs(args.seconds.into());
        let mut moves = 0u64;
        let mut games = 0u64;
        let mut positions = 0u64;
        let mut potential = 0u64;
        loop {
            let now = Instant::now();
            if now >= end {
                break;
            }
            match receiver.recv_timeout(end - now) {
                Ok(update) if Instant::now() >= warmup_end => match update {
                    GeneratorUpdate::FinishedMove { .. } => moves += 1,
                    GeneratorUpdate::FinishedSimulation { .. } => games += 1,
                    GeneratorUpdate::ExpandEvals(evals) => {
                        positions += evals.real;
                        potential += evals.potential;
                    }
                    _ => {}
                },
                Ok(_) => {}
                Err(flume::RecvTimeoutError::Timeout) => break,
                Err(flume::RecvTimeoutError::Disconnected) => {
                    panic!("benchmark workers disconnected before measurement finished")
                }
            }
        }
        for abort in aborts {
            abort.abort();
        }
        drop(settings_senders);
        // Keep receiving until aborted generators and the executor have released their senders.
        for _ in receiver {}
        json!({
            "settings": args.settings, "network": args.network, "dummy": args.dummy,
            "device": args.device, "concurrent_games": args.concurrent_games,
            "cpu_threads": fixed.cpu_threads_per_device, "gpu_batch_size": max_batch_size,
            "search_batch_size": fixed.search_batch_size,
            "max_batch_wait_ms": args.max_batch_wait_ms, "batch_timings": args.batch_timings,
            "warmup_seconds": args.warmup_seconds, "seconds": args.seconds,
            "moves": moves, "games": games, "positions": positions,
            "moves_per_second": moves as f64 / args.seconds as f64,
            "positions_per_second": positions as f64 / args.seconds as f64,
            "fill_ratio": if potential == 0 { None } else { Some(positions as f64 / potential as f64) },
        })
    })
    .map_err(|_| "benchmark executor panicked")?;
    println!("BatchingBenchmark {}", result);
    Ok(())
}
