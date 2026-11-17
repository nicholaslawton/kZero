use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const REPORT_INTERVAL: Duration = Duration::from_secs(60);
const MAX_SAMPLES: usize = 4096;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Trigger {
    Threshold = 0,
    Deadline = 1,
    Disconnect = 2,
}

struct Sample {
    trigger: Trigger,
    wait_ms: f64,
    evaluation_ms: f64,
    overshoot_ms: f64,
}

pub(super) struct BatchingStats {
    enabled: bool,
    timings: bool,
    max_batch_size: usize,
    start: Instant,
    batches: [u64; 3],
    positions: u64,
    jobs: u64,
    samples: VecDeque<Sample>,
}

impl BatchingStats {
    pub fn new(enabled: bool, timings: bool, max_batch_size: usize) -> Self {
        Self {
            enabled,
            timings: enabled && timings,
            max_batch_size,
            start: Instant::now(),
            batches: [0; 3],
            positions: 0,
            jobs: 0,
            samples: VecDeque::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn timings(&self) -> bool {
        self.timings
    }

    pub fn restart_window(&mut self) {
        self.start = Instant::now();
    }

    pub fn record(
        &mut self,
        trigger: Trigger,
        positions: usize,
        jobs: usize,
        wait: Option<Duration>,
        evaluation: Option<Duration>,
        max_wait: Option<Duration>,
    ) {
        if !self.enabled {
            return;
        }
        self.batches[trigger as usize] += 1;
        self.positions += positions as u64;
        self.jobs += jobs as u64;
        if let (Some(wait), Some(evaluation)) = (wait, evaluation) {
            if self.samples.len() == MAX_SAMPLES {
                self.samples.pop_front();
            }
            self.samples.push_back(Sample {
                trigger,
                wait_ms: wait.as_secs_f64() * 1000.0,
                evaluation_ms: evaluation.as_secs_f64() * 1000.0,
                overshoot_ms: max_wait.map_or(0.0, |limit| wait.saturating_sub(limit).as_secs_f64() * 1000.0),
            });
        }
    }

    pub fn report_if_due(&mut self) {
        if self.enabled && self.start.elapsed() >= REPORT_INTERVAL {
            self.report(std::thread::current().name().unwrap_or("unnamed"), "interval");
        }
    }

    pub fn report(&mut self, executor: &str, reason: &str) {
        let batches: u64 = self.batches.iter().sum();
        if !self.enabled || batches == 0 {
            self.restart_window();
            return;
        }
        let seconds = self.start.elapsed().as_secs_f64();
        let mut report = json!({
            "executor": executor,
            "reason": reason,
            "seconds": seconds,
            "batches": batches,
            "threshold_batches": self.batches[0],
            "deadline_batches": self.batches[1],
            "disconnect_batches": self.batches[2],
            "positions": self.positions,
            "jobs": self.jobs,
            "fill_ratio": self.positions as f64 / (batches as f64 * self.max_batch_size as f64),
            "positions_per_second": self.positions as f64 / seconds,
        });
        if self.timings {
            for (name, trigger) in [
                ("threshold", Trigger::Threshold),
                ("deadline", Trigger::Deadline),
                ("disconnect", Trigger::Disconnect),
            ] {
                let samples: Vec<_> = self.samples.iter().filter(|sample| sample.trigger == trigger).collect();
                report[name] = json!({
                    "samples": samples.len(),
                    "wait_ms": distribution(samples.iter().map(|s| s.wait_ms).collect()),
                    "evaluation_ms": distribution(samples.iter().map(|s| s.evaluation_ms).collect()),
                    "overshoot_ms": distribution(samples.iter().map(|s| s.overshoot_ms).collect()),
                });
            }
        }
        println!("Batching {}", report);
        self.batches = [0; 3];
        self.positions = 0;
        self.jobs = 0;
        self.samples.clear();
        self.restart_window();
    }
}

fn distribution(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_unstable_by(f64::total_cmp);
    let percentile = |p: f64| values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1)];
    json!({
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": values[values.len() - 1],
    })
}
