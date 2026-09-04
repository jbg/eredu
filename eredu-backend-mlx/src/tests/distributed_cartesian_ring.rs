#![cfg(unix)]

use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::composition::expert_dispatch::{AllToAllVPlan, RoutedTransport};
use crate::MlxTensor;
use eredu_core::{
    checkpoint::TensorDtype, BoundedSubmissionOutcome, CollectiveScope, CompletionCancellationMode,
    DistributedSession, ParallelAxis,
};
use eredu_runtime::{
    project_all_communication_manifests, CommunicationCompletionPolicy,
    CommunicationGroupRequirements, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationPeerCounts, CommunicationTensorLimits, FailureAgreementBackend,
    TopologyCommunicationPlan, VariableAllToAllBackend,
};
use safemlx::{
    distributed::{self, Backend},
    Array, Device, DeviceType, Stream,
};

const WORKER_ENV: &str = "EREDU_CARTESIAN_RING_WORKER";
const TRIPLE_WORKER_ENV: &str = "EREDU_CARTESIAN_TRIPLE_RING_WORKER";

fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> eredu_core::ParallelRankTopology {
    crate::test_parallel_rank(rank, tp, pp, ep)
}

fn requirement(operation: CommunicationOperation) -> CommunicationOperationRequirement {
    let max_count_per_peer =
        (operation == CommunicationOperation::VariableAllToAll).then_some(4096);
    let dtypes = match operation {
        CommunicationOperation::AllReduceSum => vec![TensorDtype::F32],
        CommunicationOperation::SendReceive => vec![TensorDtype::I32],
        CommunicationOperation::VariableAllToAll => {
            vec![TensorDtype::I32, TensorDtype::F32]
        }
        _ => unreachable!("Cartesian fixture requests only tensor-carrying operations"),
    };
    CommunicationOperationRequirement::tensors(
        operation,
        dtypes,
        CommunicationTensorLimits::new(1, 4, 16_384, max_count_per_peer).unwrap(),
        true,
    )
    .unwrap()
}

fn requirements(
    operations: impl IntoIterator<Item = CommunicationOperation>,
) -> CommunicationGroupRequirements {
    CommunicationGroupRequirements::new(operations.into_iter().map(requirement)).unwrap()
}

fn completion_policy() -> CommunicationCompletionPolicy {
    CommunicationCompletionPolicy::new(
        Duration::from_secs(5),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap()
}

fn manifest_execution(
    topology: eredu_core::ParallelRankTopology,
    world: &safemlx::distributed::Group,
    stream: &Stream,
) -> (
    crate::backend::distributed::MlxDistributedSession,
    Option<eredu_core::CollectiveGroupId>,
    Option<eredu_core::CollectiveGroupId>,
    Option<eredu_core::CollectiveGroupId>,
    eredu_core::CollectiveGroupId,
) {
    let plan = TopologyCommunicationPlan::new()
        .with_completion_policy(completion_policy())
        .with_session_group(
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .with_tensor_groups(requirements([CommunicationOperation::AllReduceSum]))
        .with_pipeline_groups(requirements([CommunicationOperation::SendReceive]))
        .with_expert_groups(requirements([
            CommunicationOperation::AllReduceSum,
            CommunicationOperation::VariableAllToAll,
        ]));
    let session_group = plan.session_group_id().unwrap();
    let manifest = project_all_communication_manifests(topology.topology(), &plan)
        .unwrap()
        .swap_remove(topology.global_rank());
    let find = |axis| {
        let members = topology.subgroup(axis).unwrap().global_ranks().to_vec();
        manifest
            .groups()
            .iter()
            .find(|descriptor| descriptor.members() == members)
            .map(|descriptor| descriptor.id())
    };
    let tensor = find(ParallelAxis::Tensor);
    let pipeline = find(ParallelAxis::Pipeline);
    let expert = find(ParallelAxis::Expert);
    let execution =
        crate::backend::distributed::MlxDistributedSession::from_manifest(&manifest, world, stream)
            .unwrap();
    (execution, tensor, pipeline, expert, session_group)
}

fn operation_consensus(
    execution: &crate::backend::distributed::MlxDistributedSession,
    group_id: eredu_core::CollectiveGroupId,
    local_failed: bool,
    local_cancelled: bool,
    stream: &Stream,
) -> (bool, bool) {
    let group = execution.selected_group(group_id).unwrap();
    let agree_any = |local: bool| {
        let submission = <crate::backend::nn::shared::MlxNeuralBackend as FailureAgreementBackend>::agree_success(
            !local,
            group,
            stream,
        )
        .unwrap();
        let BoundedSubmissionOutcome::Completed(output) = submission
            .wait_bounded(completion_policy().bounded_wait())
            .unwrap()
        else {
            panic!("Cartesian failure agreement exceeded its selected completion deadline")
        };
        !<crate::backend::nn::shared::MlxNeuralBackend as FailureAgreementBackend>::resolve_failure_agreement(output)
            .unwrap()
    };
    (agree_any(local_failed), agree_any(local_cancelled))
}

fn scalar(value: i32) -> Array {
    Array::from_slice(&[value], &[1])
}

fn scalar_f32(value: f32) -> Array {
    Array::from_slice(&[value], &[1])
}

fn values(value: &Array) -> Vec<i32> {
    value.evaluated().unwrap().as_slice::<i32>().to_vec()
}

fn float_values(value: &Array) -> Vec<f32> {
    value.evaluated().unwrap().as_slice::<f32>().to_vec()
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
    let world = crate::backend::runtime::distributed::Group::uncontracted(
        &distributed::init(true, Backend::Ring).unwrap(),
    );
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
        let full_matrix_padded_bytes = 4 * 4 * 2 * 2 * std::mem::size_of::<i32>();
        assert!(
            exchange.statistics.payload_allocation_upper_bound_bytes < full_matrix_padded_bytes,
            "rank {expected_rank} compact bound {} was not below the full-matrix padded bound {full_matrix_padded_bytes}",
            exchange.statistics.payload_allocation_upper_bound_bytes
        );
    }
    let peer_counts =
        CommunicationPeerCounts::new(send_counts.to_vec(), recv_counts.to_vec(), world.size())
            .unwrap();
    let integer =
        <crate::backend::nn::shared::MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
            MlxTensor::from_array(compact_rows(expected_rank, &send_counts, 2, &stream)),
            &peer_counts,
            0,
            &world,
            &stream,
        )
        .unwrap();
    assert_eq!(integer.completion.retained_arrays(), 2);
    assert_eq!(integer.completion.retained_count_buffers(), 2);
    assert_eq!(integer.completion.retained_groups(), 1);
    assert_eq!(integer.completion.retained_streams(), 1);
    assert_eq!(
        values(integer.wait().unwrap().as_array()),
        expected_rows(&count_matrix, expected_rank)
    );
    let floating_input = compact_rows(expected_rank, &send_counts, 2, &stream)
        .as_dtype(safemlx::Dtype::Float32, &stream)
        .unwrap();
    let floating =
        <crate::backend::nn::shared::MlxNeuralBackend as VariableAllToAllBackend>::variable_all_to_all(
            MlxTensor::from_array(floating_input),
            &peer_counts,
            0,
            &world,
            &stream,
        )
        .unwrap()
        .wait()
        .unwrap();
    let expected_floating = expected_rows(&count_matrix, expected_rank)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    assert_eq!(
        floating.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected_floating
    );
    let empty = safemlx::ops::zeros_dtype(&[0, 3], safemlx::Dtype::Float32, &stream).unwrap();
    let empty_counts = [0usize; 4];
    let empty = crate::backend::runtime::distributed::all_to_all_v(
        &empty,
        &empty_counts,
        &empty_counts,
        &world,
        &stream,
    )
    .unwrap();
    assert_eq!(empty.shape(), &[0, 3]);
    empty.evaluated().unwrap();
    let after_exchange = crate::backend::runtime::distributed::all_sum(
        &scalar(expected_rank as i32 + 1),
        &world,
        &stream,
    )
    .unwrap();
    assert_eq!(values(&after_exchange), vec![10]);

    // TP+PP: TP groups are [0, 1] and [2, 3]; pipeline lanes are [0, 2]
    // and [1, 3]. Both axes are logical subgroups under Ring.
    {
        let (execution, tensor_group, pipeline_group, _, _) = manifest_execution(
            topology(expected_rank, 2, 2, 1),
            world.native_group(),
            &stream,
        );
        let input = MlxTensor::from_array(scalar_f32(expected_rank as f32 + 1.0));
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Group(tensor_group.unwrap()),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            float_values(reduced.as_array()),
            vec![if expected_rank < 2 { 3.0 } else { 7.0 }]
        );
        if expected_rank < 2 {
            let value = MlxTensor::from_array(scalar(expected_rank as i32 + 10));
            execution
                .send_selected(pipeline_group.unwrap(), 1, &value)
                .unwrap()
                .synchronize()
                .unwrap();
        } else {
            let received = execution
                .receive_selected(
                    pipeline_group.unwrap(),
                    0,
                    &[1],
                    eredu_core::checkpoint::TensorDtype::I32,
                )
                .unwrap()
                .into_value()
                .unwrap();
            assert_eq!(values(received.as_array()), vec![expected_rank as i32 + 8]);
        }
    }

    // TP+EP: TP groups are [0, 2] and [1, 3]; EP groups are [0, 1] and [2, 3].
    {
        let (execution, tensor_group, _, expert_group, _) = manifest_execution(
            topology(expected_rank, 2, 1, 2),
            world.native_group(),
            &stream,
        );
        let input = MlxTensor::from_array(scalar_f32(expected_rank as f32 + 1.0));
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Group(tensor_group.unwrap()),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            float_values(reduced.as_array()),
            vec![if expected_rank.is_multiple_of(2) {
                4.0
            } else {
                6.0
            }]
        );
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Group(expert_group.unwrap()),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            float_values(reduced.as_array()),
            vec![if expected_rank < 2 { 3.0 } else { 7.0 }]
        );

        // Ring cannot split these EP pairs natively. Exercise the topology-
        // planned logical route with asymmetric counts in both directions.
        let expert_scope = CollectiveScope::Group(expert_group.unwrap());
        assert!(execution.scope_is_logical(expert_scope).unwrap());
        let local_rank = expected_rank % 2;
        let logical_send = if local_rank == 0 { [0, 2] } else { [1, 0] };
        let logical_recv = if local_rank == 0 { [0, 1] } else { [2, 0] };
        let logical_input =
            MlxTensor::from_array(compact_rows(expected_rank, &logical_send, 2, &stream));
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
        let peer_global_rank = expected_rank ^ 1;
        let destination = local_rank;
        let expected = (0..logical_recv[1 - local_rank])
            .flat_map(|row| {
                let value = (peer_global_rank * 100 + destination * 10 + row) as i32;
                [value, -value]
            })
            .collect::<Vec<_>>();
        assert_eq!(values(logical_received.as_array()), expected);
    }

    // PP+EP: stage-local EP reduction followed by matching-EP pipeline transport.
    {
        let (execution, _, pipeline_group, expert_group, session_group) = manifest_execution(
            topology(expected_rank, 1, 2, 2),
            world.native_group(),
            &stream,
        );
        let input = MlxTensor::from_array(scalar_f32(expected_rank as f32 + 1.0));
        let reduced = DistributedSession::all_reduce_sum(
            &execution,
            CollectiveScope::Group(expert_group.unwrap()),
            &input,
        )
        .unwrap()
        .wait()
        .unwrap();
        assert_eq!(
            float_values(reduced.as_array()),
            vec![if expected_rank < 2 { 3.0 } else { 7.0 }]
        );
        if expected_rank < 2 {
            let value = MlxTensor::from_array(scalar(expected_rank as i32 + 20));
            execution
                .send_selected(pipeline_group.unwrap(), 1, &value)
                .unwrap()
                .synchronize()
                .unwrap();
        } else {
            let received = execution
                .receive_selected(
                    pipeline_group.unwrap(),
                    0,
                    &[1],
                    eredu_core::checkpoint::TensorDtype::I32,
                )
                .unwrap()
                .into_value()
                .unwrap();
            assert_eq!(values(received.as_array()), vec![expected_rank as i32 + 18]);
        }
        let (failed, cancelled) = operation_consensus(
            &execution,
            session_group,
            expected_rank == 1,
            expected_rank == 2,
            &stream,
        );
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
    let world = crate::backend::runtime::distributed::Group::uncontracted(
        &distributed::init(true, Backend::Ring).unwrap(),
    );
    assert_eq!((world.rank(), world.size()), (expected_rank, 8));
    let topology = topology(expected_rank, 2, 2, 2);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let (execution, tensor_group, pipeline_group, expert_group, session_group) =
        manifest_execution(topology, world.native_group(), &stream);
    let input = MlxTensor::from_array(scalar_f32(expected_rank as f32 + 1.0));

    let tp = DistributedSession::all_reduce_sum(
        &execution,
        CollectiveScope::Group(tensor_group.unwrap()),
        &input,
    )
    .unwrap()
    .wait()
    .unwrap();
    let expected_tp = topology
        .subgroup(eredu_core::ParallelAxis::Tensor)
        .unwrap()
        .global_ranks()
        .iter()
        .map(|rank| *rank as i32 + 1)
        .map(|value| value as f32)
        .sum::<f32>();
    assert_eq!(float_values(tp.as_array()), vec![expected_tp]);

    let ep = DistributedSession::all_reduce_sum(
        &execution,
        CollectiveScope::Group(expert_group.unwrap()),
        &input,
    )
    .unwrap()
    .wait()
    .unwrap();
    let expected_ep = topology
        .subgroup(eredu_core::ParallelAxis::Expert)
        .unwrap()
        .global_ranks()
        .iter()
        .map(|rank| *rank as i32 + 1)
        .map(|value| value as f32)
        .sum::<f32>();
    assert_eq!(float_values(ep.as_array()), vec![expected_ep]);

    if topology.pipeline_parallel_rank() == 0 {
        let value = MlxTensor::from_array(scalar(expected_rank as i32 + 100));
        execution
            .send_selected(pipeline_group.unwrap(), 1, &value)
            .unwrap()
            .synchronize()
            .unwrap();
    } else {
        let received = execution
            .receive_selected(
                pipeline_group.unwrap(),
                0,
                &[1],
                eredu_core::checkpoint::TensorDtype::I32,
            )
            .unwrap()
            .into_value()
            .unwrap();
        assert_eq!(values(received.as_array()), vec![expected_rank as i32 + 96]);
    }
    let consensus = operation_consensus(
        &execution,
        session_group,
        expected_rank == 1,
        expected_rank == 6,
        &stream,
    );
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
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_cartesian_ring::ring_four_process_pairwise_topologies -- --ignored --exact --nocapture`
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
    let worker = format!("tests::distributed_cartesian_ring::{worker}");
    let mut children = ChildGuard {
        children: Vec::with_capacity(world_size),
    };
    for rank in 0..world_size {
        children.children.push(
            Command::new(&executable)
                .args(["--exact", worker.as_str(), "--nocapture"])
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
