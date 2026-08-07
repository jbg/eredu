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
use safemlx_lm::{CartesianExecution, DeviceAssignment, ParallelTopology};

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

    // TP+PP: TP groups are [0, 1] and [2, 3]; pipeline lanes are [0, 2]
    // and [1, 3]. Both axes are logical subgroups under Ring.
    {
        let execution =
            CartesianExecution::new(topology(expected_rank, 2, 2, 1), Some(2), None, &world)
                .unwrap();
        let reduced = execution
            .tensor_context(&stream)
            .unwrap()
            .all_sum(&scalar(expected_rank as i32 + 1))
            .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );
        if execution.topology().pipeline_parallel_rank == 0 {
            execution
                .send_pipeline(&scalar(expected_rank as i32 + 10), &stream)
                .unwrap();
        } else {
            let received = execution
                .receive_pipeline(&[1], safemlx::Dtype::Int32, &stream)
                .unwrap();
            assert_eq!(values(&received), vec![expected_rank as i32 + 8]);
        }
    }

    // TP+EP: TP groups are [0, 2] and [1, 3]; EP groups are [0, 1] and [2, 3].
    {
        let execution =
            CartesianExecution::new(topology(expected_rank, 2, 1, 2), None, Some(2), &world)
                .unwrap();
        let reduced = execution
            .tensor_context(&stream)
            .unwrap()
            .all_sum(&scalar(expected_rank as i32 + 1))
            .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank.is_multiple_of(2) {
                4
            } else {
                6
            }]
        );
        let reduced = distributed::all_sum(
            &scalar(expected_rank as i32 + 1),
            execution.expert_group().unwrap(),
            &stream,
        )
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );
    }

    // PP+EP: stage-local EP reduction followed by matching-EP pipeline transport.
    {
        let execution =
            CartesianExecution::new(topology(expected_rank, 1, 2, 2), Some(2), Some(2), &world)
                .unwrap();
        let reduced = distributed::all_sum(
            &scalar(expected_rank as i32 + 1),
            execution.expert_group().unwrap(),
            &stream,
        )
        .unwrap();
        assert_eq!(
            values(&reduced),
            vec![if expected_rank < 2 { 3 } else { 7 }]
        );
        if execution.topology().pipeline_parallel_rank == 0 {
            execution
                .send_pipeline(&scalar(expected_rank as i32 + 20), &stream)
                .unwrap();
        } else {
            let received = execution
                .receive_pipeline(&[1], safemlx::Dtype::Int32, &stream)
                .unwrap();
            assert_eq!(values(&received), vec![expected_rank as i32 + 18]);
        }
        let (failed, cancelled) = execution
            .operation_consensus(expected_rank == 1, expected_rank == 2, &stream)
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
    let execution = CartesianExecution::new(topology, Some(4), Some(4), &world).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = scalar(expected_rank as i32 + 1);

    let tp = execution
        .tensor_context(&stream)
        .unwrap()
        .all_sum(&input)
        .unwrap();
    let expected_tp = execution
        .preflight()
        .tensor_subgroup
        .global_ranks
        .iter()
        .map(|rank| *rank as i32 + 1)
        .sum::<i32>();
    assert_eq!(values(&tp), vec![expected_tp]);

    let ep = distributed::all_sum(&input, execution.expert_group().unwrap(), &stream).unwrap();
    let expected_ep = execution
        .preflight()
        .expert_subgroup
        .global_ranks
        .iter()
        .map(|rank| *rank as i32 + 1)
        .sum::<i32>();
    assert_eq!(values(&ep), vec![expected_ep]);

    if topology.pipeline_parallel_rank == 0 {
        execution
            .send_pipeline(&scalar(expected_rank as i32 + 100), &stream)
            .unwrap();
    } else {
        let received = execution
            .receive_pipeline(&[1], safemlx::Dtype::Int32, &stream)
            .unwrap();
        assert_eq!(values(&received), vec![expected_rank as i32 + 96]);
    }
    let consensus = execution
        .operation_consensus(expected_rank == 1, expected_rank == 6, &stream)
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
