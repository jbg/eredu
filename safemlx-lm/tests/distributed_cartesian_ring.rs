#![cfg(unix)]

use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    Array, Device, DeviceType, Stream,
};
use safemlx_lm::{
    architectures::distributed::expert::{AllToAllVPlan, RoutedTransport},
    core::{CollectiveScope, DistributedSession},
    DeviceAssignment, MlxBackend, ParallelTopology,
};

const WORKER_ENV: &str = "SAFEMLX_CARTESIAN_RING_WORKER";
const TRIPLE_WORKER_ENV: &str = "SAFEMLX_CARTESIAN_TRIPLE_RING_WORKER";

fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> ParallelTopology {
    ParallelTopology::from_rank(
        4,
        rank,
        tp,
        pp,
        ep,
        DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap()
}

fn scalar(value: i32) -> Array {
    Array::from_slice(&[value], &[1])
}

fn values(value: &Array) -> Vec<i32> {
    value.evaluated().unwrap().as_slice::<i32>().to_vec()
}

fn native_all_to_all_counts() -> [[usize; 4]; 4] {
    [[1, 2, 1, 0], [1, 0, 0, 2], [0, 0, 0, 0], [1, 1, 0, 1]]
}

fn compact_rows(source: usize, counts: &[usize], columns: i32, stream: &Stream) -> Array {
    let mut rows = Vec::new();
    for (destination, &count) in counts.iter().enumerate() {
        for row in 0..count {
            let value = (source * 100 + destination * 10 + row) as i32;
            rows.extend([value, -value]);
        }
    }
    Array::from_slice(&rows, &[rows.len() as i32 / columns, columns])
        .copy(stream)
        .unwrap()
}

fn expected_rows(counts: &[[usize; 4]; 4], destination: usize) -> Vec<i32> {
    let mut expected = Vec::new();
    for (source, source_counts) in counts.iter().enumerate() {
        for row in 0..source_counts[destination] {
            let value = (source * 100 + destination * 10 + row) as i32;
            expected.extend([value, -value]);
        }
    }
    expected
}

#[test]
fn cartesian_ring_worker() {
    let Some(expected_rank) = std::env::var_os(WORKER_ENV) else {
        return;
    };
    let expected_rank: usize = expected_rank.to_string_lossy().parse().unwrap();
    let world = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(world.rank(), expected_rank);
    assert_eq!(world.size(), 4);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    // Native four-rank exchange: rank 2 sends no routes, counts are uneven,
    // traffic is bidirectional, and source order is visible in two-column rows.
    let count_matrix = native_all_to_all_counts();
    let send_counts = count_matrix[expected_rank];
    let recv_counts = count_matrix.map(|source| source[expected_rank]);
    let plan = AllToAllVPlan::new(&send_counts, &world, &stream).unwrap();
    for _ in 0..2 {
        let input = compact_rows(expected_rank, &send_counts, 2, &stream);
        let exchange = plan.exchange(&input, &world, &stream).unwrap();
        assert_eq!(exchange.source_counts, recv_counts);
        assert_eq!(
            values(&exchange.received),
            expected_rows(&count_matrix, expected_rank)
        );
        assert_eq!(exchange.statistics.padding_bytes, 0);
        assert_eq!(
            exchange.statistics.routed_transport,
            RoutedTransport::Native
        );
        let legacy_padded_bytes = 4 * 4 * 2 * 2 * std::mem::size_of::<i32>();
        assert!(
            exchange.statistics.payload_allocation_upper_bound_bytes < legacy_padded_bytes,
            "rank {expected_rank} compact bound {} was not below legacy {legacy_padded_bytes}",
            exchange.statistics.payload_allocation_upper_bound_bytes
        );
    }
    let empty = safemlx::ops::zeros_dtype(&[0, 3], safemlx::Dtype::Float32, &stream).unwrap();
    let empty_counts = [0usize; 4];
    let empty =
        distributed::all_to_all_v(&empty, &empty_counts, &empty_counts, &world, &stream).unwrap();
    assert_eq!(empty.shape(), &[0, 3]);
    empty.evaluated().unwrap();
    let after_exchange =
        distributed::all_sum(&scalar(expected_rank as i32 + 1), &world, &stream).unwrap();
    assert_eq!(values(&after_exchange), vec![10]);

    // TP+PP: TP groups are [0, 1] and [2, 3]; pipeline lanes are [0, 2]
    // and [1, 3]. Both axes are logical subgroups under Ring.
    {
        let execution = MlxBackend::new(&stream)
            .create_communication_session(topology(expected_rank, 2, 2, 1), &world)
            .unwrap();
        let input = scalar(expected_rank as i32 + 1);
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Tensor),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );
        if execution.topology().pipeline_parallel_rank == 0 {
            execution
                .send_pipeline(&scalar(expected_rank as i32 + 10))
                .unwrap()
                .synchronize()
                .unwrap();
        } else {
            let received = execution
                .receive_pipeline(&[1], safemlx_lm::core::checkpoint::TensorDtype::I32)
                .unwrap()
                .into_value()
                .unwrap();
            assert_eq!(values(&received), vec![expected_rank as i32 + 8]);
        }
    }

    // TP+EP: TP groups are [0, 2] and [1, 3]; EP groups are [0, 1] and [2, 3].
    {
        let execution = MlxBackend::new(&stream)
            .create_communication_session(topology(expected_rank, 2, 1, 2), &world)
            .unwrap();
        let input = scalar(expected_rank as i32 + 1);
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Tensor),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank.is_multiple_of(2) {
                4
            } else {
                6
            }]
        );
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Expert),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );

        // Ring cannot split these EP pairs natively. Exercise the topology-
        // planned logical route with asymmetric counts in both directions.
        let expert_scope = CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Expert);
        assert!(execution.scope_is_logical(expert_scope).unwrap());
        let expert_subgroup = execution
            .topology()
            .subgroup(safemlx_lm::ParallelAxis::Expert)
            .unwrap();
        let local_rank = expert_subgroup.rank;
        let logical_send = if local_rank == 0 { [0, 2] } else { [1, 0] };
        let logical_recv = if local_rank == 0 { [0, 1] } else { [2, 0] };
        let logical_input = compact_rows(expected_rank, &logical_send, 2, &stream);
        let logical_received = DistributedSession::all_to_all_v(
            &execution,
            expert_scope,
            &logical_input,
            &logical_send,
            &logical_recv,
        )
        .unwrap()
        .wait()
        .unwrap();
        let peer_global_rank = expert_subgroup.global_ranks[1 - local_rank];
        let destination = local_rank;
        let expected = (0..logical_recv[1 - local_rank])
            .flat_map(|row| {
                let value = (peer_global_rank * 100 + destination * 10 + row) as i32;
                [value, -value]
            })
            .collect::<Vec<_>>();
        assert_eq!(values(&logical_received), expected);
    }

    // PP+EP: stage-local EP reduction followed by matching-EP pipeline transport.
    {
        let execution = MlxBackend::new(&stream)
            .create_communication_session(topology(expected_rank, 1, 2, 2), &world)
            .unwrap();
        let input = scalar(expected_rank as i32 + 1);
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Expert),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );
        if execution.topology().pipeline_parallel_rank == 0 {
            execution
                .send_pipeline(&scalar(expected_rank as i32 + 20))
                .unwrap()
                .synchronize()
                .unwrap();
        } else {
            let received = execution
                .receive_pipeline(&[1], safemlx_lm::core::checkpoint::TensorDtype::I32)
                .unwrap()
                .into_value()
                .unwrap();
            assert_eq!(values(&received), vec![expected_rank as i32 + 18]);
        }
        let (failed, cancelled) = execution
            .operation_consensus(expected_rank == 1, expected_rank == 2)
            .unwrap();
        assert!(failed);
        assert!(cancelled);
    }
}

#[test]
fn cartesian_triple_ring_worker() {
    let Some(expected_rank) = std::env::var_os(TRIPLE_WORKER_ENV) else {
        return;
    };
    let expected_rank: usize = expected_rank.to_string_lossy().parse().unwrap();
    let world = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!((world.rank(), world.size()), (expected_rank, 8));
    let topology = ParallelTopology::from_rank(
        8,
        expected_rank,
        2,
        2,
        2,
        DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let execution = MlxBackend::new(&stream)
        .create_communication_session(topology, &world)
        .unwrap();
    let input = scalar(expected_rank as i32 + 1);

    let tp = DistributedSession::all_reduce_sum(
        &execution,
        CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Tensor),
        &input,
    )
    .unwrap()
    .wait()
    .unwrap();
    let expected_tp = topology
        .subgroup(safemlx_lm::ParallelAxis::Tensor)
        .unwrap()
        .global_ranks
        .iter()
        .map(|rank| *rank as i32 + 1)
        .sum::<i32>();
    assert_eq!(values(&tp), vec![expected_tp]);

    let ep = DistributedSession::all_reduce_sum(
        &execution,
        CollectiveScope::Axis(safemlx_lm::core::topology::ParallelAxis::Expert),
        &input,
    )
    .unwrap()
    .wait()
    .unwrap();
    let expected_ep = topology
        .subgroup(safemlx_lm::ParallelAxis::Expert)
        .unwrap()
        .global_ranks
        .iter()
        .map(|rank| *rank as i32 + 1)
        .sum::<i32>();
    assert_eq!(values(&ep), vec![expected_ep]);

    if topology.pipeline_parallel_rank == 0 {
        execution
            .send_pipeline(&scalar(expected_rank as i32 + 100))
            .unwrap()
            .synchronize()
            .unwrap();
    } else {
        let received = execution
            .receive_pipeline(&[1], safemlx_lm::core::checkpoint::TensorDtype::I32)
            .unwrap()
            .into_value()
            .unwrap();
        assert_eq!(values(&received), vec![expected_rank as i32 + 96]);
    }
    let consensus = execution
        .operation_consensus(expected_rank == 1, expected_rank == 6)
        .unwrap();
    assert_eq!(consensus, (true, true));
}

struct ChildGuard {
    children: Vec<Child>,
}

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.children
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "Cartesian Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_cartesian_ring ring_four_process_pairwise_topologies -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns four Ring workers and opens loopback sockets; run explicitly"]
fn ring_four_process_pairwise_topologies() {
    run_cartesian_ring_workers(4, "cartesian_ring_worker", WORKER_ENV);
}

/// Proves TP, EP, matching-coordinate PP lanes, and global failure/cancellation
/// consensus over one eight-rank TP=2 × PP=2 × EP=2 Ring topology.
#[test]
#[ignore = "spawns eight Ring workers and opens loopback sockets; run explicitly"]
fn ring_eight_process_triple_axis_topology() {
    run_cartesian_ring_workers(8, "cartesian_triple_ring_worker", TRIPLE_WORKER_ENV);
}

fn run_cartesian_ring_workers(world_size: usize, worker: &str, worker_env: &str) {
    assert!(distributed::is_available(Backend::Ring));
    let sockets = (0..world_size)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("ring-hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);

    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(world_size),
    };
    for rank in 0..world_size {
        children.children.push(
            Command::new(&executable)
                .args(["--exact", worker, "--nocapture"])
                .env(worker_env, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let timeout = if world_size > 4 { 120 } else { 60 };
    let deadline = Instant::now() + Duration::from_secs(timeout);
    let mut timed_out = false;
    loop {
        let statuses = children
            .children
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.children {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    let outputs = children.finish();
    let failures = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{world_size}-process Cartesian Ring integration test failed:\n{}",
        if timed_out {
            format!(
                "timed out after {timeout} seconds\n\n{}",
                failures.join("\n\n")
            )
        } else {
            failures.join("\n\n")
        }
    );
}
