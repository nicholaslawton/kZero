use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::time::{Duration, Instant};

use board_game::board::Board;
use flume::{Receiver, RecvError, Selector, Sender, TryRecvError};
use futures::never::Never;
use itertools::Itertools;
use kn_cuda_sys::wrapper::handle::CudaDevice;
use kn_graph::graph::Graph;
use superluminal_perf::{begin_event_with_color, end_event};

use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::{Job, JobServer};
use kz_core::network::{Network, ZeroEvaluation};

use crate::server::executor_stats::{BatchingStats, Trigger};
use crate::superluminal::{CL_BLUE, CL_GREEN, CL_YELLOW};

#[derive(Debug, Copy, Clone)]
pub enum RunCondition {
    FullBatch,
    JobCount(usize),
    Any,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct BatchingSettings {
    /// None preserves the original threshold-only policy, for baseline measurements.
    pub max_wait: Option<Duration>,
    pub timings: bool,
    pub report: bool,
}

pub fn batched_executor_loop<G, N, X, Y>(
    max_batch_size: usize,
    run_condition: RunCondition,
    graph_receiver: Receiver<Option<G>>,
    server: JobServer<X, Y>,
    load_network: impl FnMut(G) -> N,
    evaluate_batch: impl FnMut(&mut N, &[X]) -> Vec<Y>,
) {
    batched_executor_loop_with_settings(
        max_batch_size,
        run_condition,
        BatchingSettings::default(),
        graph_receiver,
        server,
        load_network,
        evaluate_batch,
    )
}

pub fn batched_executor_loop_with_settings<G, N, X, Y>(
    max_batch_size: usize,
    run_condition: RunCondition,
    batching: BatchingSettings,
    graph_receiver: Receiver<Option<G>>,
    server: JobServer<X, Y>,
    mut load_network: impl FnMut(G) -> N,
    mut evaluate_batch: impl FnMut(&mut N, &[X]) -> Vec<Y>,
) {
    let thread_name = std::thread::current().name().unwrap_or("unnamed").to_owned();
    assert_ne!(max_batch_size, 0, "Got batch size 0 for {}", thread_name);
    assert!(
        batching.max_wait != Some(Duration::ZERO),
        "Batch wait must be positive or disabled"
    );

    let job_receiver = server.into_receiver();

    let mut state = State::new();
    let mut network: Option<N> = None;
    let mut stats = BatchingStats::new(batching.report, batching.timings, max_batch_size);
    let track_age = batching.max_wait.is_some() || batching.timings;

    // this is a separate flag and not just `graph_receiver.is_disconnected`
    //   to ensure we have properly handled the disconnection event itself
    let mut graph_disconnected = false;

    loop {
        if let Some(network) = network.as_mut() {
            let trigger = if state.should_eval(run_condition, max_batch_size) {
                Some(Trigger::Threshold)
            } else if state
                .deadline(batching.max_wait)
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                Some(Trigger::Deadline)
            } else {
                None
            };
            if let Some(trigger) = trigger {
                run_eval(
                    &mut state,
                    network,
                    &mut evaluate_batch,
                    max_batch_size,
                    &mut stats,
                    trigger,
                    batching.max_wait,
                );
                continue;
            }
        }

        // check that there is at least one channel to wait on
        assert!(network.is_some() || !graph_disconnected);
        let mut selector = Selector::new();

        // only wait for graphs if the channel is still open
        if !graph_disconnected {
            selector = selector.recv(&graph_receiver, Message::Graph);
        }

        // only wait for jobs if we already have a network, otherwise we're just pointlessly filling our buffers
        if network.is_some() {
            selector = selector.recv(&job_receiver, Message::Job);
        }

        // block until we get a new message
        begin_event_with_color("wait", CL_YELLOW);
        let deadline = network.as_ref().and_then(|_| state.deadline(batching.max_wait));
        let message = match deadline {
            Some(deadline) => selector.wait_deadline(deadline).ok(),
            None => Some(selector.wait()),
        };
        end_event();
        let Some(message) = message else { continue };

        let _: Never = match message {
            Message::Graph(Ok(graph)) => {
                stats.report(&thread_name, "network_change");
                handle_new_graph(&mut network, graph, &mut load_network, &thread_name);
                // Intentional pauses and network loading are not batch-assembly time.
                state.reset_age(track_age && network.is_some());
                stats.restart_window();
                continue;
            }
            Message::Job(job) => {
                // safe because we only listen to the job channel if we have a network
                let network = network.as_mut().unwrap();

                match job {
                    Ok(job) => {
                        // we've got a new job, great
                        state.push_job(job, track_age.then(Instant::now));

                        // check for additional jobs (non-blocking)
                        //   we could loop again but we might as well just fall back to the outer loop
                        while state.x.len() < max_batch_size {
                            match job_receiver.try_recv() {
                                // yay, we've got an additional job
                                Ok(job) => state.push_job(job, track_age.then(Instant::now)),
                                // we've run out of non-blocking jobs
                                Err(TryRecvError::Empty) => break,
                                // let the outer match deal with it
                                Err(TryRecvError::Disconnected) => break,
                            }
                        }

                        continue;
                    }
                    Err(RecvError::Disconnected) => {
                        // the job channel has disconnected

                        // evaluate all remaining jobs if any
                        assert!(state.items_to_eval() < max_batch_size);
                        if state.items_to_eval() > 0 {
                            run_eval(
                                &mut state,
                                network,
                                &mut evaluate_batch,
                                max_batch_size,
                                &mut stats,
                                Trigger::Disconnect,
                                batching.max_wait,
                            );
                        }
                        assert!(state.items_to_eval() == 0 && state.items_to_send() == 0);

                        // at this point we can safely exit, we won't get any more jobs
                        //  this also closes the graph receiver
                        stats.report(&thread_name, "disconnect");
                        return;
                    }
                }
            }
            Message::Graph(Err(RecvError::Disconnected)) => {
                // we'll never get a new graph again
                match &network {
                    Some(_) => {
                        // .. but we do have a final network to continue evaluating batches with
                        graph_disconnected = true;
                        continue;
                    }
                    None => {
                        // ... and we don't even have a network, make sure that that's okay
                        // leftover jobs?
                        if state.items_to_eval() > 0 {
                            panic!(
                                "Executor {}: graph disconnected by we still have items pending: {:?}",
                                thread_name, state
                            );
                        }
                        // wait for job receiver disconnection
                        match job_receiver.recv() {
                            Ok(_) => panic!("Executor {}: got new job after graph disconnection", thread_name),
                            Err(RecvError::Disconnected) => {}
                        }
                        // we can safely exit this executor now that both channels are disconnected
                        stats.report(&thread_name, "disconnect");
                        return;
                    }
                }
            }
        };
    }
}

pub fn alphazero_batched_executor_loop<B: Board, M: BoardMapper<B>>(
    max_batch_size: usize,
    device: CudaDevice,
    mapper: M,
    run_condition: RunCondition,
    graph: Graph,
    server: JobServer<B, ZeroEvaluation<'static>>,
) {
    let (graph_sender, graph_receiver) = flume::bounded(1);
    graph_sender.send(Some(graph)).unwrap();
    drop(graph_sender);

    batched_executor_loop(
        max_batch_size,
        run_condition,
        graph_receiver,
        server,
        |graph| CudaNetwork::new(mapper, &graph, max_batch_size, device),
        |network, batch_x| network.evaluate_batch(&batch_x),
    )
}

#[derive(Debug)]
enum Message<G, J> {
    Graph(G),
    Job(J),
}

struct State<X, Y> {
    x: VecDeque<X>,
    senders: VecDeque<(usize, Sender<Vec<Y>>, Option<Instant>)>,
    leftover_y: VecDeque<Y>,
}

impl<X, Y> Debug for State<X, Y> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("items_to_eval", &self.items_to_eval())
            .field("items_waiting_for_send", &self.items_waiting_for_send())
            .field("items_to_send", &self.items_to_send())
            .field("senders", &self.senders.len())
            .finish()
    }
}

impl<X, Y> State<X, Y> {
    pub fn new() -> Self {
        Self {
            x: VecDeque::new(),
            senders: VecDeque::new(),
            leftover_y: VecDeque::new(),
        }
    }

    fn check_invariants(&self) {
        assert!(!self.can_fill_next_sender());
        assert_eq!(
            self.items_to_eval() + self.items_waiting_for_send(),
            self.items_to_send()
        );
    }

    fn items_to_eval(&self) -> usize {
        self.x.len()
    }

    fn items_waiting_for_send(&self) -> usize {
        self.leftover_y.len()
    }

    fn items_to_send(&self) -> usize {
        self.senders.iter().map(|&(len, _, _)| len).sum::<usize>()
    }

    fn oldest_pending(&self) -> Option<Instant> {
        self.senders.front().and_then(|entry| entry.2)
    }

    fn deadline(&self, max_wait: Option<Duration>) -> Option<Instant> {
        self.oldest_pending().zip(max_wait).map(|(start, wait)| start + wait)
    }

    fn reset_age(&mut self, track_age: bool) {
        let now = track_age.then(Instant::now);
        for entry in &mut self.senders {
            entry.2 = now;
        }
    }

    fn jobs_in_batch(&self, batch_size: usize) -> usize {
        let mut positions = self.leftover_y.len() + batch_size;
        let mut jobs = 0;
        for &(len, _, _) in &self.senders {
            jobs += 1;
            positions = positions.saturating_sub(len);
            if positions == 0 {
                break;
            }
        }
        jobs
    }

    fn push_job(&mut self, job: Job<X, Y>, queued_at: Option<Instant>) {
        // TODO we could avoid an extra vec copy if all batches we receive "happen to" have exact size max_batch_size
        //   by keeping the single leading batch as a separate vec and falling back to deque if necessary
        //   this is pretty complicated and shouldn't matter too much though

        let Job { x, sender } = job;

        if x.len() == 0 {
            // avoid ever putting empty senders in the queue since that introduces tricky edge cases
            let _ = sender.send(vec![]);
        } else {
            self.senders.push_back((x.len(), sender, queued_at));
            self.x.extend(x.into_iter());
        }

        self.check_invariants();
    }

    fn should_eval(&self, cond: RunCondition, max_batch_size: usize) -> bool {
        if self.x.len() == 0 {
            return false;
        }
        if self.x.len() >= max_batch_size {
            return true;
        }

        match cond {
            RunCondition::FullBatch => false,
            RunCondition::JobCount(count) => self.senders.len() >= count,
            RunCondition::Any => self.x.len() > 0,
        }
    }

    fn get_batch(&mut self, max_batch_size: usize) -> &[X] {
        let batch_size = min(self.x.len(), max_batch_size);
        assert_ne!(batch_size, 0);

        if self.x.as_slices().0.len() < batch_size {
            self.x.make_contiguous();
        }

        &self.x.as_slices().0[0..batch_size]
    }

    fn can_fill_next_sender(&self) -> bool {
        match self.senders.get(0) {
            None => {
                assert_eq!(self.leftover_y.len(), 0);
                false
            }
            Some(&(count, _, _)) => self.leftover_y.len() >= count,
        }
    }

    // Distribute the given response over the appropriate senders.
    // We ignore SendError here, since it's not our fault that the receiver stopped caring about the response.
    fn respond_batch(&mut self, batch_y: Vec<Y>) {
        let batch_size = batch_y.len();

        // remove just-evaluated x-values
        assert!(batch_size <= self.x.len());
        drop(self.x.drain(0..batch_size));

        if self.leftover_y.is_empty() && self.senders[0].0 == batch_size {
            // shortcut to avoid extra copies, just send the entire vec immediately
            let _ = self.senders.pop_front().unwrap().1.send(batch_y);
        } else {
            // add results to temporary storage
            self.leftover_y.extend(batch_y.into_iter());

            // send as many values out as possible
            while self.can_fill_next_sender() {
                let (count, sender, _) = self.senders.pop_front().unwrap();
                let block_y = self.leftover_y.drain(0..count).collect_vec();
                let _ = sender.send(block_y);
            }
        }

        self.check_invariants();
    }
}

fn run_eval<X, Y, N>(
    state: &mut State<X, Y>,
    network: &mut N,
    mut evaluate_batch: impl FnMut(&mut N, &[X]) -> Vec<Y>,
    max_batch_size: usize,
    stats: &mut BatchingStats,
    trigger: Trigger,
    max_wait: Option<Duration>,
) {
    let size = state.items_to_eval().min(max_batch_size);
    let jobs = if stats.enabled() { state.jobs_in_batch(size) } else { 0 };
    let start = stats.timings().then(Instant::now);
    let wait = start.zip(state.oldest_pending()).map(|(now, queued)| now - queued);
    begin_event_with_color("run", CL_GREEN);
    let batch_x = state.get_batch(max_batch_size);
    let batch_y = evaluate_batch(network, batch_x);
    end_event();
    let evaluation = start.map(|start| start.elapsed());
    stats.record(trigger, size, jobs, wait, evaluation, max_wait);

    begin_event_with_color("reply", CL_YELLOW);
    state.respond_batch(batch_y);
    end_event();
    stats.report_if_due();
}

fn handle_new_graph<N, G>(
    network: &mut Option<N>,
    graph: Option<G>,
    mut load_network: impl FnMut(G) -> N,
    thread_name: &str,
) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        begin_event_with_color("drop", CL_BLUE);
        println!("{} dropping network", thread_name);
        drop(network);
        end_event();
    }

    // load the new network if any
    *network = graph.map(|graph| {
        begin_event_with_color("load", CL_BLUE);
        println!("{} loading new network", thread_name);
        let network = load_network(graph);
        end_event();
        network
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kz_core::network::job_channel::{job_pair, JobClient};
    use std::thread::{self, JoinHandle};

    struct Executor {
        client: JobClient<u32, u32>,
        graphs: Sender<Option<()>>,
        evaluations: Receiver<(usize, Instant)>,
        thread: JoinHandle<()>,
    }

    fn spawn(condition: RunCondition, wait: Option<Duration>) -> Executor {
        let (client, server) = job_pair(256);
        let (graphs, receiver) = flume::unbounded();
        let (events, evaluations) = flume::unbounded();
        graphs.send(Some(())).unwrap();
        let thread = thread::spawn(move || {
            batched_executor_loop_with_settings(
                16,
                condition,
                BatchingSettings {
                    max_wait: wait,
                    ..BatchingSettings::default()
                },
                receiver,
                server,
                |_| (),
                move |_, inputs| {
                    events.send((inputs.len(), Instant::now())).unwrap();
                    inputs.iter().map(|x| x * 2).collect()
                },
            );
        });
        Executor {
            client,
            graphs,
            evaluations,
            thread,
        }
    }

    impl Executor {
        fn finish(self) {
            drop(self.client);
            drop(self.graphs);
            self.thread.join().unwrap();
        }
    }

    #[test]
    fn threshold_executes_without_waiting_for_deadline() {
        for condition in [RunCondition::FullBatch, RunCondition::JobCount(2)] {
            let executor = spawn(condition, Some(Duration::from_secs(30)));
            let a = executor.client.map(vec![1; 8]);
            let b = executor.client.map(vec![2; 8]);
            assert_eq!(a.recv_timeout(Duration::from_secs(2)).unwrap(), vec![2; 8]);
            assert_eq!(b.recv_timeout(Duration::from_secs(2)).unwrap(), vec![4; 8]);
            assert_eq!(executor.evaluations.recv().unwrap().0, 16);
            executor.finish();
        }
        let executor = spawn(RunCondition::JobCount(2), Some(Duration::from_secs(30)));
        let a = executor.client.map(vec![1]);
        let b = executor.client.map(vec![2]);
        assert_eq!(a.recv_timeout(Duration::from_secs(2)).unwrap(), vec![2]);
        assert_eq!(b.recv_timeout(Duration::from_secs(2)).unwrap(), vec![4]);
        assert_eq!(executor.evaluations.recv().unwrap().0, 2);
        executor.finish();
    }

    #[test]
    fn sparse_and_split_jobs_finish_while_client_stays_connected() {
        let executor = spawn(RunCondition::FullBatch, Some(Duration::from_millis(20)));
        let response = executor.client.map((0..19).collect());
        assert_eq!(
            response.recv_timeout(Duration::from_secs(2)).unwrap(),
            (0..19).map(|x| x * 2).collect::<Vec<_>>(),
        );
        assert_eq!(executor.evaluations.recv().unwrap().0, 16);
        assert_eq!(executor.evaluations.recv().unwrap().0, 3);
        let response = executor.client.map(vec![99]);
        assert_eq!(response.recv_timeout(Duration::from_secs(2)).unwrap(), vec![198]);
        executor.finish();
    }

    #[test]
    fn arrivals_do_not_reset_the_oldest_deadline() {
        let mut state = State::<u32, u32>::new();
        let (sender, _) = flume::unbounded();
        let start = Instant::now();
        state.push_job(
            Job {
                x: vec![1],
                sender: sender.clone(),
            },
            Some(start),
        );
        state.push_job(Job { x: vec![2], sender }, Some(start + Duration::from_secs(1)));
        assert_eq!(
            state.deadline(Some(Duration::from_secs(2))),
            Some(start + Duration::from_secs(2))
        );
        assert_eq!(state.jobs_in_batch(1), 1);
        assert_eq!(state.jobs_in_batch(2), 2);
    }

    #[test]
    fn sparse_arrivals_cannot_postpone_execution_until_the_stream_stops() {
        let executor = spawn(RunCondition::FullBatch, Some(Duration::from_millis(40)));
        let client = executor.client.clone();
        let producer = thread::spawn(move || {
            for value in 0..12 {
                let _ = client.map(vec![value]);
                thread::sleep(Duration::from_millis(20));
            }
        });
        executor.evaluations.recv_timeout(Duration::from_millis(200)).unwrap();
        producer.join().unwrap();
        executor.finish();
    }

    #[test]
    fn pause_suspends_deadline_and_resume_restarts_assembly_clock() {
        let executor = spawn(RunCondition::FullBatch, Some(Duration::from_millis(100)));
        executor.graphs.send(None).unwrap();
        let response = executor.client.map(vec![3]);
        assert!(response.recv_timeout(Duration::from_millis(200)).is_err());
        executor.graphs.send(Some(())).unwrap();
        assert_eq!(response.recv_timeout(Duration::from_secs(2)).unwrap(), vec![6]);
        executor.finish();
    }

    #[test]
    fn resuming_preserves_buffered_work_but_excludes_pause_age() {
        let mut state = State::<u32, u32>::new();
        let (sender, receiver) = flume::unbounded();
        state.push_job(
            Job { x: vec![7], sender },
            Some(Instant::now() - Duration::from_secs(60)),
        );
        state.reset_age(false);
        assert!(state.deadline(Some(Duration::from_millis(10))).is_none());
        let resumed = Instant::now();
        state.reset_age(true);
        assert!(state.oldest_pending().unwrap() >= resumed);
        state.respond_batch(vec![14]);
        assert_eq!(receiver.recv().unwrap(), vec![14]);
    }

    #[test]
    fn disconnect_flushes_partial_work_without_a_deadline() {
        let executor = spawn(RunCondition::FullBatch, None);
        let response = executor.client.map(vec![5]);
        let Executor {
            client,
            graphs,
            evaluations: _,
            thread,
        } = executor;
        drop(client);
        assert_eq!(response.recv_timeout(Duration::from_secs(2)).unwrap(), vec![10]);
        drop(graphs);
        thread.join().unwrap();
    }
}
