# kZero

A from-scratch implementation of the [AlphaGo](https://www.nature.com/articles/nature24270), [AlphaZero](https://arxiv.org/abs/1712.01815) and [MuZero](https://www.nature.com/articles/s41586-020-03051-4.epdf) papers for different board games, written in a combination of Python and Rust.

For GPU inference the [Kyanite library](https://github.com/KarelPeeters/Kyanite) is used.

## Table of contents

<!-- TOC -->
* [kZero](#kzero)
  * [Table of contents](#table-of-contents)
  * [Project Overview](#project-overview)
    * [Top level](#top-level)
    * [Selfplay server (Rust)](#selfplay-server-rust)
    * [Training (Python)](#training-python)
  * [Current status and results](#current-status-and-results)
  * [File structure](#file-structure)
    * [Python](#python)
    * [Rust](#rust)
      * [Core crates](#core-crates)
      * [Low-level crates](#low-level-crates)
      * [Utility crates](#utility-crates)
<!-- TOC -->

## Project Overview

### Top level

The top-level overview is shown in the figure above. The neural network training is implemented in Python, and Rust is used to implement the more performance-critical selfplay. The two parts communicate though a TCP connection and the file system.

![Top level system diagram](./docs/arch_overview.svg)

During a training loop the training framework connects to the selfplay TCP server running on localhost and gives it some initial settings. The selfplay server writes generated games to a file, and when it has generated enough games it signals this to the training framework. The network is then trained on these new games, and when training finishes the new network is sent to the selfplay server, closing the loop.

### Selfplay server (Rust)

The selfplay server is shown in the figure below. The orange boxes are the different threads running in the system.

![Selfplay server diagram](./docs/arch_selfplay.svg)

* The _commander_ receives commands from the TCP socket and forwards them to the appropriate place. Selfplay settings and hyperparameters are sent to the generators, and new neural networks are loaded from the filesystem and sent to the executors.
* The _generator_ thread pools run multiple simulations concurrently. Each simulation has its own MCTS tree that is grown over time. NN evaluation requests are sent to an executor.
* The _executor_ threads collect NN evaluations requests into batches and use the NN inference framework described later to run these evaluations, sending results back to the corresponding generator.
* Finally the _collector_ receives finished simulations from the generators, writes them to a file and notifies the TCP socket once enough simulations have finished.

Much effort was put into optimizing this system:

* Rust is used to get good baseline performance for the tricky code in tree search, memory transfers, thread interactions, NN input encoding, ...
* The MCTS itself uses virtual loss to collect small batches of evaluations.
* Many simulations are run concurrently on async thread pools. This allows for a second level of batching in the executors.
* Each GPU can have multiple executor threads, enabling concurrent batching, memory transfers and execution.
* Multiple GPUs can be used at full speed.

#### Bounded-wait AlphaZero batching

AlphaZero self-play retains its normal full-batch/job-count thresholds, but also
executes pending work after `max_batch_wait_ms` (provisionally 100 ms). This avoids
stalling when too few games remain to fill a batch. A value of `0` restores threshold-only batching for baseline
measurements; that mode can stall at low concurrency.

The deadline starts when a job is buffered by an executor, not when it enters the
shared channel. New arrivals do not reset it. It bounds batching wait, not channel
backpressure, CPU scheduling, inference execution, or total time to finish a move.
Network unloading/loading resets buffered-work ages, excluding intentional pauses.
Other executor callers retain their existing policies; MuZero is unchanged.

For the Oriflambe wrapper, run from the workspace:

```sh
nu alpha-oriflambe/train.nu start --arch hybrid --max-batch-wait-ms 100 --batch-timings
```

`--batch-timings` is optional and disabled by default. Configuration is passed
through Python's `FixedSelfplaySettings` / `StartupSettings` to the Rust server.
Each AlphaZero executor emits `Batching {json}` summaries approximately every
60 seconds while executing, on network changes, and at exit. Cheap counters report
execution triggers, useful positions, job participations, and useful batch fill.
There is no per-request logging.

Inspect or export reports from the workspace with the Oriflambe wrapper:

```sh
nu alpha-oriflambe/train.nu metrics batching
nu alpha-oriflambe/train.nu metrics batching --latest
nu alpha-oriflambe/train.nu metrics batching --output batching-results.json
nu alpha-oriflambe/train.nu metrics batching --log archived-selfplay.log --raw
```

The default view shows per-window throughput, fill, deadline counts, and normal
wait/evaluation timing highlights. `--latest` shows the newest full report as
expanded JSON so nested timing distributions are visible;
`--raw` returns all full records for Nushell pipelines. `--output` always saves
all complete reports, including nested timings, and refuses to overwrite a file.
An unfinished final log line is skipped with a warning if it is a batching report;
malformed complete reports are errors. Save results before starting another
session, which replaces the logs. Retain the full selfplay log as configuration
context when sharing results. Exporting does not stop or change training.

Detailed timing mode adds monotonic timestamps around evaluation and retains at
most the latest 4096 samples per executor per reporting window. Summaries give
p50/p95/p99/max wait, host-observed evaluation duration, and deadline overshoot,
separately for threshold, deadline, and disconnect executions. Evaluation duration
includes input encoding, transfers, synchronized CUDA evaluation, decoding, and
the evaluation callback's bookkeeping; it is not GPU-kernel-only time. No extra
GPU synchronization is added. Normal batch-assembly measurements may include CPU
descheduling or time spent finishing earlier batches.

Partial batches are supported but padded to the configured CUDA batch size.
Useful fill and positions/second therefore matter more than raw execution count.
Summary throughput windows can include idle time; use the benchmark's explicit
measurement window for comparisons.

##### Measuring and selecting a deadline

Build once, then run `target/release/batching_bench` from `rust/`. The harness uses
the actual Oriflambe-repetition self-play generator, one GPU executor, a fixed
network, and the search settings from a saved training `settings.json`. It keeps
games in memory and writes neither training state nor self-play data. Use a
settings file and network from the same experiment. It does not reproduce
multi-executor or multi-GPU contention; validate those configurations in training.

```sh
cargo build --release -p kz-selfplay --bin batching_bench

target/release/batching_bench \
  --settings ../data/loop/oriflambe/training/gen_0/settings.json \
  --network ../data/loop/oriflambe/training/gen_8/network.onnx \
  --concurrent-games 32 --warmup-seconds 30 --seconds 120 \
  --max-batch-wait-ms 0 --batch-timings
```

Use the actual file paths for your experiment: settings are saved at the first
generation of each session, not necessarily every generation. `--device` selects
the GPU (default 0); the harness does not run other GPUs. Do not benchmark alongside
training or other GPU workloads.

1. Record GPU/CPU, software versions, network, settings, and thermal/power state.
   At normal concurrency, measure the threshold-only baseline above.
2. Repeat with deadlines around the measured normal wait distribution, for
   example 10, 25, 50, 100, and 250 ms. Repeat each case at least three times,
   alternating order, with the same warm-up and measurement durations.
3. Compare `BatchingBenchmark {json}` final results: useful positions/second,
   moves/second, and fill ratio. Games/second is secondary because games vary in
   length. Search randomness means runs are comparable, not identical trajectories.
4. Repeat with detailed timings disabled to quantify their overhead. Also compare
   the threshold-only timings-off baseline to bound scheduler/counter overhead.
   Measuring the entire change against the old executor requires running the same
   workload on the pre-change revision on the same hardware.
5. Test `--concurrent-games 1`, then a few games, with bounded waiting enabled.
   These must make move progress. Threshold-only low-concurrency runs deliberately
   stall but the harness still ends at its time limit.

The final benchmark line excludes network loading and warm-up. Periodic executor
summaries include warm-up, so do not pool their samples blindly with the final
throughput window. The benchmark reports zero progress explicitly if stalled.
Use longer windows when a slow model cannot finish a move in a short run.
`--dummy` instead of `--network` smoke-tests the harness without CUDA inference;
those timings cannot tune GPU batching.

Choose a deadline that avoids low-concurrency stalls while rarely forcing partial
batches at normal load. Target no more than a provisional 2% throughput regression
across repeated runs, accounting for noise. If no fixed value works across actual
workloads, use these measurements to justify bounded adaptation rather than
introducing it prematurely. The 100 ms default is not a hardware-tuned result.

Development tests use fake evaluation and do not execute in normal training:

```sh
cargo test -p kz-selfplay --lib server::executor
```

#### Draining self-play for a stop

For Oriflambe training started from the workspace, run:

```sh
nu alpha-oriflambe/train.nu stop --drain
```

No new games start after the drain request is accepted. Games already in progress
finish, and the selfplay log reports how many remain. Complete generation files
are trained normally, including publishing the new network. The remaining partial
file is saved and continued on the next `start` with compatible self-play settings.
If draining starts during optimization, it also waits for optimization to finish.
The usual generation-boundary `stop` and destructive `stop --now` are unchanged.
Start a new training session after updating the code before using `--drain`;
already-running processes do not acquire the new behavior.

Drain requires AlphaZero, a nonzero batching deadline, and a session started
without `--target-gen`. It is not a time guarantee: long games, optimization, or
inefficient partially filled GPU batches can make a drain take longer than a few
minutes. Changing the game, game count per generation, search batch size,
symmetry, or self-play settings before resuming a partial file is rejected.

This is a graceful-stop checkpoint, not crash recovery. An interruption or failed
drain may leave incomplete or inconsistent files; the stop command reports
failure instead of confirming success. Inspect both logs and back up the
experiment before attempting recovery. A partial checkpoint whose data lengths
do not match, or which coexists with a completed file, is rejected rather than
truncated. Optimization itself is not checkpointed.

### Training (Python)

Finally the training component is shown in the figure below.

![Training diagram](./docs/arch_training.svg)

The python framework manages the user configuration, and sends the relevant parameters and the current neural network to the selfplay server. 

Incoming data files are added to the replay buffer. The sampler uses multiple threads to load training data from those data files, building batches. These batches are used in NN training. Finally newly trained networks are sent to the selfplay server, closing the loop.

The status of the replay buffer and training statistics are plotted in realtime using a Qt GUI.


## Current status and results

The AlphaZero implementation fully works. The specific board games that have had training runs are Chess, Ataxx and Go, all relatively successful. These training runs typically last for a couple of days on modest hardware ~2x GTX 3090.

Almost all the code is generic over the specific game being played, so adding new games or even more exotic RL environments should be easy.

MuZero is not working yet, there is some training instability that still has to be fixed.

## File structure

The basic file structure is as follows:

### Python

The `python` folder contains the training code, we use the PyTorch framework. 

* `lib` contains the training framework
* `lib/data` contains data file loading, the replay buffer, the sampler, ...
* `lib/mapping` contains metadata for building the NN architecture for various games
* `lib/lib` contains metadata for building the NN architecture for various games
* `lib/model` contains building blocks for NN architectures
* `main` contains the entry points for starting supervised training or selfplay

### Rust

The `rust` folder is a workspace consisting of a bunch of crates:

#### Core crates

* `kz-core` is the most important crate, it contains the AlphaZero and MuZero tree search implementations, game-specific details and NN wrappers.
* `kz-selfplay` is the selfplay server used during training loops. It runs a bunch of games in parallel on an async thread pool.
* `kz-misc` contains various utilities for evaluating network performance, converting between file formats, and is the crate where I do most of my experimenting in.
* `kz-lichess` is a lichess bot wrapper, you can occasionally play against it here: https://lichess.org/@/kZero-bot

#### Low-level crates

* `cuda-sys` contains Rust wrappers for Cuda and related frameworks, most importantly CuDNN. These wrappers are generated based on the system headers at build time.
* `nn-graph` is a graph format for NN inference, along with an ONNX parser and simple CPU executor.
* `cuda-nn-eval` is a Cuda/CuDNN executor for NN inference, based on the previously mentioned I

#### Utility crates

* `kz-util`: Boring utility functions that don't fit anywhere else.
* `pgn-reader`: A streaming pgn file parser. This was initially useful to pretrain networks on the lichess and CCRL databases.
* `licorice`: A "fork" of https://gitlab.com/hyperchessbotauthor/licorice with some small updates to get it working on lichess again.
