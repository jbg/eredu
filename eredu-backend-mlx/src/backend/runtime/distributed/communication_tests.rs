use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use eredu_core::{
    checkpoint::TensorDtype, BoundedSubmissionOutcome, CollectiveGroupId, Completion as _,
    ParallelTopology,
};
use eredu_runtime::{
    project_all_communication_manifests, BarrierBackend, BoundaryRoleContract, BroadcastBackend,
    CommunicationGroupDescriptor, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationRouteDescriptor,
    CommunicationRouteId, CommunicationTensorLimits, DistributedExecutionPhase, EvenGatherBackend,
    FailureAgreementBackend, OpaqueFailureAgreement, PartitionCommitAgreement,
    PartitionCommunication, PointToPointBackend, RealizedCommunicationGroup,
    RoleExactBoundaryContract, TopologyCommunicationPlan,
};
use safemlx::{
    distributed::{self, Backend},
    Array, Device, DeviceType, Stream,
};

use crate::{
    backend::nn::shared::{MlxCommunicationTensorMetadata, MlxNeuralBackend},
    MlxTensor,
};

use super::topology::{CommunicationRouteEndpoint, ParallelCommunicators};
use super::Group;

const WORKER_RANK: &str = "EREDU_FINE_COMMUNICATION_RING_WORKER";
const MISMATCH_WORKER_RANK: &str = "EREDU_MANIFEST_MISMATCH_RING_WORKER";
const MISMATCH_COMPLETION_POLICY: &str = "EREDU_MANIFEST_MISMATCH_COMPLETION_POLICY";
const FAILURE_AGREEMENT_WORKER_RANK: &str = "EREDU_FAILURE_AGREEMENT_RING_WORKER";
const SUBGROUP_WAVE_WORKER_RANK: &str = "EREDU_SUBGROUP_WAVE_RING_WORKER";
const VARIABLE_SUBGROUP_WORKER_RANK: &str = "EREDU_VARIABLE_SUBGROUP_RING_WORKER";

fn completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
    eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_secs(30),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap()
}

#[test]
fn selected_setup_deadline_rejects_busy_runtime_before_native_submission() {
    use std::sync::{Arc, Barrier};

    let native = distributed::init(false, Backend::Ring).unwrap();
    let policy = eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_millis(5),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let descriptor = CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(41),
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap(),
    )
    .unwrap();
    let group = Group::uncontracted(&native)
        .with_manifest_contract(&descriptor, policy)
        .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    std::thread::scope(|scope| {
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        scope.spawn(move || {
            let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(1))
                .unwrap()
                .enter()
                .unwrap();
            worker_entered.wait();
            worker_release.wait();
        });
        entered.wait();
        let error = <MlxNeuralBackend as BarrierBackend>::barrier(&group, &stream)
            .expect_err("busy runtime must fail before collective graph submission");
        assert!(error.what().contains("selected deadline"));
        assert_eq!(
            crate::backend::runtime::distributed::group::native_collective_submissions(),
            0
        );
        release.wait();
    });
}

#[test]
fn setup_deadline_poisons_exact_authority_and_retry_makes_no_native_call() {
    use std::sync::{Arc, Barrier};

    let native = distributed::init(false, Backend::Ring).unwrap();
    let policy = eredu_runtime::CommunicationCompletionPolicy::new(
        Duration::from_millis(5),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let id = CollectiveGroupId::new(42);
    let descriptor = CommunicationGroupDescriptor::new(
        id,
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap();
    let group = Group::uncontracted(&native)
        .with_manifest_contract(&descriptor, policy)
        .unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![descriptor], Vec::new())
        .unwrap()
        .with_completion_policy(policy);
    let communication = PartitionCommunication::<MlxNeuralBackend, _, _, _>::new(
        manifest,
        vec![RealizedCommunicationGroup::new(id, group)],
        Vec::<
            eredu_runtime::RealizedCommunicationRoute<
                super::topology::CommunicationRouteRealization,
            >,
        >::new(),
        MlxCommunicationTensorMetadata,
    )
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    std::thread::scope(|scope| {
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        scope.spawn(move || {
            let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(1))
                .unwrap()
                .enter()
                .unwrap();
            worker_entered.wait();
            worker_release.wait();
        });
        entered.wait();
        let first = OpaqueFailureAgreement
            .agree_phase(
                &communication,
                id,
                DistributedExecutionPhase::Execution,
                true,
                &stream,
            )
            .expect_err("setup deadline must poison the selected communication authority");
        let retry = OpaqueFailureAgreement
            .agree_phase(
                &communication,
                id,
                DistributedExecutionPhase::Execution,
                true,
                &stream,
            )
            .expect_err("poisoned communication must reject retry before backend entry");
        assert!(matches!(
            first,
            eredu_runtime::PartitionExecutionError::CommunicationSubmissionFailed { .. }
        ));
        assert!(matches!(
            retry,
            eredu_runtime::PartitionExecutionError::CommunicationPoisoned { .. }
        ));
        assert_eq!(
            crate::backend::runtime::distributed::group::native_collective_submissions(),
            0
        );
        release.wait();
    });
}

fn route_requirement() -> CommunicationOperationRequirement {
    CommunicationOperationRequirement::tensors(
        CommunicationOperation::SendReceive,
        [TensorDtype::F32, TensorDtype::I32, TensorDtype::U32],
        CommunicationTensorLimits::new(3, 2, 8, None).unwrap(),
        true,
    )
    .unwrap()
}

fn subgroup_requirement() -> CommunicationOperationRequirement {
    CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
        true,
    )
    .unwrap()
}

fn variable_subgroup_requirement() -> CommunicationOperationRequirement {
    CommunicationOperationRequirement::tensors(
        CommunicationOperation::VariableAllToAll,
        [TensorDtype::F32, TensorDtype::I32],
        CommunicationTensorLimits::new(1, 2, 64, Some(8)).unwrap(),
        true,
    )
    .unwrap()
}

#[test]
fn subgroup_wave_worker() {
    let Some(rank) = std::env::var_os(SUBGROUP_WAVE_WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!((native.rank(), native.size()), (rank, 4));
    let (tensor_id, tensor_members, tensor_local) = match rank {
        0 => (CollectiveGroupId::new(101), vec![0, 1], 0),
        1 => (CollectiveGroupId::new(101), vec![0, 1], 1),
        2 => (CollectiveGroupId::new(102), vec![2, 3], 0),
        3 => (CollectiveGroupId::new(102), vec![2, 3], 1),
        _ => unreachable!(),
    };
    let (pipeline_id, pipeline_members, pipeline_local) = match rank {
        0 => (CollectiveGroupId::new(103), vec![0, 2], 0),
        2 => (CollectiveGroupId::new(103), vec![0, 2], 1),
        1 => (CollectiveGroupId::new(104), vec![1, 3], 0),
        3 => (CollectiveGroupId::new(104), vec![1, 3], 1),
        _ => unreachable!(),
    };
    let requirements = || CommunicationGroupRequirements::new([subgroup_requirement()]).unwrap();
    let manifest = CommunicationManifest::new(
        4,
        rank,
        vec![
            CommunicationGroupDescriptor::new(
                tensor_id,
                0,
                tensor_members,
                Some(tensor_local),
                requirements(),
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                pipeline_id,
                1,
                pipeline_members,
                Some(pipeline_local),
                requirements(),
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communication = ParallelCommunicators::from_manifest(&manifest, &native, &stream).unwrap();
    let tensor = communication.communication_group(tensor_id).unwrap();
    let pipeline = communication.communication_group(pipeline_id).unwrap();
    assert!(tensor.is_logical());
    assert!(pipeline.is_logical());
    assert_eq!((tensor.size(), pipeline.size()), (2, 2));
    assert_eq!(
        (tensor.native_group().size(), pipeline.native_group().size()),
        (4, 4)
    );

    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let local = Array::from_slice(&[rank as f32 + 1.0], &[1]);
    let tensor_sum =
        crate::backend::runtime::distributed::group::all_sum(&local, tensor, &stream).unwrap();
    let total =
        crate::backend::runtime::distributed::group::all_sum(&tensor_sum, pipeline, &stream)
            .unwrap();
    assert_eq!(total.evaluated().unwrap().as_slice::<f32>(), &[10.0]);
    assert_eq!(
        crate::backend::runtime::distributed::group::native_collective_submissions(),
        2
    );
}

#[test]
fn variable_subgroup_wave_worker() {
    let Some(rank) = std::env::var_os(VARIABLE_SUBGROUP_WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!((native.rank(), native.size()), (rank, 4));
    let topology = ParallelTopology::new(2, 1, 2, 1).unwrap();
    let plan = TopologyCommunicationPlan::new()
        .with_completion_policy(completion_policy())
        .with_tensor_groups(CommunicationGroupRequirements::new([subgroup_requirement()]).unwrap())
        .with_expert_groups(
            CommunicationGroupRequirements::new([variable_subgroup_requirement()]).unwrap(),
        );
    let manifests = project_all_communication_manifests(topology, &plan).unwrap();
    let manifest = &manifests[rank];
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communicators = ParallelCommunicators::from_manifest(manifest, &native, &stream).unwrap();
    let tensor_descriptor = &manifest.groups()[0];
    let expert_descriptor = &manifest.groups()[1];
    let tensor = communicators
        .communication_group(tensor_descriptor.id())
        .unwrap();
    let expert = communicators
        .communication_group(expert_descriptor.id())
        .unwrap();
    assert!(tensor.is_logical());
    assert!(expert.is_logical());
    assert_eq!(tensor.native_group().size(), 4);
    assert_eq!(expert.native_group().size(), 4);

    let local = Array::from_slice(&[rank as f32 + 1.0], &[1]);
    let tensor_sum = super::group::all_sum(&local, tensor, &stream).unwrap();
    let expected_tensor_sum = tensor_descriptor
        .members()
        .iter()
        .map(|member| *member as f32 + 1.0)
        .sum::<f32>();
    assert_eq!(
        tensor_sum.evaluated().unwrap().as_slice::<f32>(),
        &[expected_tensor_sum]
    );

    // One member sends two rows while its peer submits a genuinely empty input.
    // In a safe world wave, the other two ranks are represented by exact zero
    // counts rather than skipping the native operation.
    let local_rank = expert_descriptor.local_index().unwrap();
    let forward_matrix = [[0_usize, 2], [0, 0]];
    let send = forward_matrix[local_rank];
    let receive = forward_matrix.map(|source| source[local_rank]);
    let peer = expert_descriptor.members()[1 - local_rank];
    let forward_values = if local_rank == 0 {
        vec![
            rank as i32 * 100 + peer as i32 * 10,
            rank as i32 * 100 + peer as i32 * 10 + 1,
        ]
    } else {
        Vec::new()
    };
    let forward = Array::from_slice(&forward_values, &[forward_values.len() as i32, 1]);
    let received = super::group::all_to_all_v(&forward, &send, &receive, expert, &stream).unwrap();
    safemlx::transforms::async_eval_with_event([&received])
        .unwrap()
        .synchronize()
        .unwrap();
    let expected = if local_rank == 0 {
        Vec::new()
    } else {
        vec![
            (rank ^ 1) as i32 * 100 + rank as i32 * 10,
            (rank ^ 1) as i32 * 100 + rank as i32 * 10 + 1,
        ]
    };
    let received_values = if received.size() == 0 {
        Vec::new()
    } else {
        received.evaluated().unwrap().as_slice::<i32>().to_vec()
    };
    assert_eq!(received_values, expected);

    // Return the same rows through the inverse count matrix, then repeat with
    // floating payloads to match the metadata/data sequence used by routed work.
    let returned = super::group::all_to_all_v(&received, &receive, &send, expert, &stream).unwrap();
    safemlx::transforms::async_eval_with_event([&returned])
        .unwrap()
        .synchronize()
        .unwrap();
    let returned_values = if returned.size() == 0 {
        Vec::new()
    } else {
        returned.evaluated().unwrap().as_slice::<i32>().to_vec()
    };
    assert_eq!(returned_values, forward_values);
    let floating = forward.as_dtype(safemlx::Dtype::Float32, &stream).unwrap();
    let floating_received =
        super::group::all_to_all_v(&floating, &send, &receive, expert, &stream).unwrap();
    safemlx::transforms::async_eval_with_event([&floating_received])
        .unwrap()
        .synchronize()
        .unwrap();
    let floating_returned =
        super::group::all_to_all_v(&floating_received, &receive, &send, expert, &stream).unwrap();
    safemlx::transforms::async_eval_with_event([&floating_returned])
        .unwrap()
        .synchronize()
        .unwrap();
    let floating_returned_values = if floating_returned.size() == 0 {
        Vec::new()
    } else {
        floating_returned
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    };
    assert_eq!(
        floating_returned_values,
        forward_values
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>()
    );
}

fn framed_values(
    values: Vec<MlxTensor>,
    specs: Vec<(TensorDtype, Vec<usize>)>,
) -> Vec<eredu_runtime::RoleExactBoundaryValue<MlxTensor>> {
    let roles = specs
        .into_iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            BoundaryRoleContract::new(format!("role.{index}"), dtype, shape).unwrap()
        })
        .collect::<Vec<_>>();
    RoleExactBoundaryContract::new("communication.test", roles.clone())
        .unwrap()
        .frame_values(CommunicationRouteId::new(41), &roles, values)
        .unwrap()
}

#[test]
fn point_to_point_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(native.rank(), rank);
    let count_group_id = CollectiveGroupId::new(43);
    let count_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllGatherEven,
        [TensorDtype::I32],
        CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
        true,
    )
    .unwrap();
    let session_group_id = CollectiveGroupId::new(1);
    let session_requirements = CommunicationGroupRequirements::new([
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 2, None).unwrap(),
            true,
        )
        .unwrap(),
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 2, None).unwrap(),
            true,
        )
        .unwrap(),
        CommunicationOperationRequirement::barrier(true),
    ])
    .unwrap();
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![
            CommunicationGroupDescriptor::new(
                session_group_id,
                0,
                vec![0, 1],
                Some(rank),
                session_requirements,
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                count_group_id,
                1,
                vec![0, 1],
                Some(rank),
                CommunicationGroupRequirements::new([count_requirement]).unwrap(),
            )
            .unwrap(),
        ],
        vec![CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(41),
            0,
            0,
            1,
            route_requirement(),
        )
        .unwrap()],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communicators = ParallelCommunicators::from_manifest(&manifest, &native, &stream).unwrap();
    let session_group = communicators.communication_group(session_group_id).unwrap();
    let count_group = communicators.communication_group(count_group_id).unwrap();
    assert_eq!(session_group.opaque_id(), Some(session_group_id));
    assert_eq!(count_group.opaque_id(), Some(count_group_id));
    super::group::reset_native_collective_submissions();
    let error = super::group::all_sum(
        &Array::from_slice(&[rank as f32], &[1]),
        count_group,
        &stream,
    )
    .expect_err("full-world opaque IDs with equal members must not substitute contracts");
    assert!(error.what().contains("does not select operation"));
    assert_eq!(super::group::native_collective_submissions(), 0);
    let route = communicators.route(CommunicationRouteId::new(41)).unwrap();
    assert_eq!(
        route.endpoint(),
        Some(if rank == 0 {
            CommunicationRouteEndpoint::Source
        } else {
            CommunicationRouteEndpoint::Destination
        })
    );
    assert_eq!(route.peer_rank(), Some(1 - rank));
    assert_eq!(route.group().unwrap().rank(), rank);
    assert_eq!(route.group().unwrap().size(), 2);
    assert!(route.group().unwrap().is_logical());
    let counts = <MlxNeuralBackend as EvenGatherBackend>::all_gather_even(
        MlxTensor::from_array(Array::from_slice(&[i32::try_from(rank).unwrap() + 1], &[1])),
        0,
        count_group,
        &stream,
    )
    .unwrap();
    assert_eq!(counts.completion.retained_arrays(), 2);
    assert_eq!(
        counts
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    let lazy_predecessor = crate::backend::runtime::distributed::all_sum(
        &Array::from_slice(&[rank as f32 + 1.0], &[1]),
        session_group,
        &stream,
    )
    .unwrap();
    let published = <MlxNeuralBackend as BroadcastBackend>::broadcast(
        MlxTensor::from_array(lazy_predecessor),
        0,
        session_group,
        &stream,
    )
    .unwrap()
    .wait()
    .unwrap();
    assert_eq!(
        published.as_array().evaluated().unwrap().as_slice::<f32>(),
        &[3.0]
    );
    let wrong_dtype = framed_values(
        vec![MlxTensor::from_array(Array::from_slice(&[1_i64], &[1]))],
        vec![(TensorDtype::I64, vec![1])],
    );
    let error =
        <MlxNeuralBackend as PointToPointBackend>::send_receive(wrong_dtype, route, &stream)
            .expect_err("route dtype must be checked before native submission");
    assert!(error.what().contains("does not advertise dtype"));
    let oversized = framed_values(
        vec![MlxTensor::from_array(Array::from_slice(
            &[0.0_f32; 9],
            &[3, 3],
        ))],
        vec![(TensorDtype::F32, vec![3, 3])],
    );
    let error = <MlxNeuralBackend as PointToPointBackend>::send_receive(oversized, route, &stream)
        .expect_err("route placeholder shape must be checked before native submission");
    assert!(error.what().contains("exceeds route limits"));
    let values = if rank == 0 {
        vec![
            MlxTensor::from_array(Array::from_slice(&[1.0_f32, 2.0], &[2])),
            MlxTensor::from_array(Array::from_slice(&[-7_i32, 10], &[1, 2])),
            MlxTensor::from_array(Array::from_slice(&[23_u32], &[1])),
        ]
    } else {
        vec![
            MlxTensor::from_array(Array::zeros::<f32>(&[2], &stream).unwrap()),
            MlxTensor::from_array(Array::zeros::<i32>(&[1, 2], &stream).unwrap()),
            MlxTensor::from_array(Array::zeros::<u32>(&[1], &stream).unwrap()),
        ]
    };
    let values = framed_values(
        values,
        vec![
            (TensorDtype::F32, vec![2]),
            (TensorDtype::I32, vec![1, 2]),
            (TensorDtype::U32, vec![1]),
        ],
    );
    let submission =
        <MlxNeuralBackend as PointToPointBackend>::send_receive(values, route, &stream).unwrap();
    assert_eq!(submission.completion.retained_arrays(), 12);
    assert_eq!(
        submission.completion.submitted_outputs(),
        if rank == 0 { 6 } else { 9 }
    );
    assert_eq!(submission.completion.retained_count_buffers(), 0);
    assert_eq!(submission.completion.retained_groups(), 1);
    assert_eq!(submission.completion.retained_routes(), 1);
    assert_eq!(submission.completion.retained_streams(), 1);
    let output = submission.wait().unwrap();
    assert_eq!(
        output[0].as_array().evaluated().unwrap().as_slice::<f32>(),
        &[1.0, 2.0]
    );
    assert_eq!(
        output[1].as_array().evaluated().unwrap().as_slice::<i32>(),
        &[-7, 10]
    );
    assert_eq!(
        output[2].as_array().evaluated().unwrap().as_slice::<u32>(),
        &[23]
    );

    let publication_input = if rank == 0 {
        Array::from_slice(&[31.0_f32, 37.0], &[2])
    } else {
        Array::zeros::<f32>(&[2], &stream).unwrap()
    };
    let publication = <MlxNeuralBackend as BroadcastBackend>::broadcast(
        MlxTensor::from_array(publication_input),
        0,
        session_group,
        &stream,
    )
    .unwrap();
    assert_eq!(publication.completion.retained_arrays(), 3);
    assert_eq!(
        publication
            .wait()
            .unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[31.0, 37.0]
    );
    let commit = <MlxNeuralBackend as BarrierBackend>::barrier(session_group, &stream).unwrap();
    assert_eq!(commit.retained_arrays(), 2);
    assert_eq!(commit.retained_groups(), 1);
    assert_eq!(commit.retained_streams(), 1);
    commit.wait().unwrap();
}

#[test]
fn failure_agreement_worker() {
    let Some(rank) = std::env::var_os(FAILURE_AGREEMENT_WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(native.rank(), rank);
    let group_id = CollectiveGroupId::new(53);
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![CommunicationGroupDescriptor::new(
            group_id,
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        Vec::new(),
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communicators = ParallelCommunicators::from_manifest(&manifest, &native, &stream).unwrap();
    let group = communicators.communication_group(group_id).unwrap();

    let unanimous =
        <MlxNeuralBackend as FailureAgreementBackend>::agree_success(true, group, &stream).unwrap();
    assert_eq!(unanimous.completion.retained_arrays(), 2);
    assert_eq!(unanimous.completion.retained_groups(), 1);
    assert_eq!(unanimous.completion.retained_streams(), 1);
    let BoundedSubmissionOutcome::Completed(unanimous) = unanimous
        .wait_bounded(completion_policy().bounded_wait())
        .unwrap()
    else {
        panic!("failure agreement exceeded test deadline")
    };
    assert!(
        <MlxNeuralBackend as FailureAgreementBackend>::resolve_failure_agreement(unanimous)
            .unwrap()
    );

    let mixed =
        <MlxNeuralBackend as FailureAgreementBackend>::agree_success(rank == 0, group, &stream)
            .unwrap();
    assert_eq!(mixed.completion.retained_arrays(), 2);
    assert_eq!(mixed.completion.retained_groups(), 1);
    assert_eq!(mixed.completion.retained_streams(), 1);
    let BoundedSubmissionOutcome::Completed(mixed) = mixed
        .wait_bounded(completion_policy().bounded_wait())
        .unwrap()
    else {
        panic!("failure agreement exceeded test deadline")
    };
    assert!(
        !<MlxNeuralBackend as FailureAgreementBackend>::resolve_failure_agreement(mixed).unwrap()
    );
}

#[test]
fn manifest_mismatch_worker() {
    let Some(rank) = std::env::var_os(MISMATCH_WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let completion_policy_mismatch = std::env::var_os(MISMATCH_COMPLETION_POLICY).is_some();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(native.rank(), rank);
    let dtype = if rank == 0 || completion_policy_mismatch {
        TensorDtype::F32
    } else {
        // U32 all-sum is intentionally outside MLX's advertised mechanism
        // surface. Consensus must still complete on both ranks before that
        // rank-local capability check is allowed to run.
        TensorDtype::U32
    };
    let requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        [dtype],
        CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
        true,
    )
    .unwrap();
    let manifest = CommunicationManifest::new(
        2,
        rank,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(47),
            0,
            vec![0, 1],
            Some(rank),
            CommunicationGroupRequirements::new([requirement]).unwrap(),
        )
        .unwrap()],
        Vec::new(),
    )
    .unwrap();
    let manifest = if completion_policy_mismatch && rank == 1 {
        manifest
    } else {
        manifest.with_completion_policy(completion_policy())
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    super::group::reset_native_collective_submissions();
    super::topology::reset_manifest_group_realizations();
    let error = ParallelCommunicators::from_manifest(&manifest, &native, &stream)
        .expect_err("different rank-local capability requirements must fail before splitting");
    assert_eq!(
        error.to_string(),
        "parallel placement error: communication manifest consensus failed: communication projection for rank 1 is incompatible with its peers"
    );
    assert_eq!(
        super::topology::manifest_group_realizations(),
        0,
        "manifest corruption reached subgroup realization"
    );
    assert_eq!(
        super::group::native_collective_submissions(),
        0,
        "manifest corruption reached subgroup or payload collective work"
    );
}

struct Children(Vec<Child>);

impl Children {
    fn finish(mut self) -> Vec<Output> {
        self.0
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_overlapping_tp_pp_subgroups_use_exact_logical_membership() {
    assert!(distributed::is_available(Backend::Ring));
    let sockets = (0..4)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let ports = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port())
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!(
            "[[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"]]",
            ports[0], ports[1], ports[2], ports[3]
        ),
    )
    .unwrap();
    drop(sockets);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(4));
    for rank in 0..4 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::subgroup_wave_worker",
                    "--nocapture",
                ])
                .env(SUBGROUP_WAVE_WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let deadline = Instant::now() + Duration::from_secs(45);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "overlapping subgroup Ring rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "four-process overlapping subgroup Ring failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_overlapping_tp_ep_variable_all_to_all_uses_one_world_wave() {
    assert!(distributed::is_available(Backend::Ring));
    let sockets = (0..4)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let ports = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port())
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!(
            "[[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"]]",
            ports[0], ports[1], ports[2], ports[3]
        ),
    )
    .unwrap();
    drop(sockets);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(4));
    for rank in 0..4 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::variable_subgroup_wave_worker",
                    "--nocapture",
                ])
                .env(VARIABLE_SUBGROUP_WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let deadline = Instant::now() + Duration::from_secs(45);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "overlapping TP/EP VariableAllToAll rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "four-process overlapping TP/EP VariableAllToAll failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_point_to_point_bundle_retains_exact_resources() {
    assert!(distributed::is_available(Backend::Ring));
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::point_to_point_worker",
                    "--nocapture",
                ])
                .env(WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "fine communication Ring rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process point-to-point bundle failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_failure_agreement_propagates_one_false_status() {
    assert!(distributed::is_available(Backend::Ring));
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::failure_agreement_worker",
                    "--nocapture",
                ])
                .env(FAILURE_AGREEMENT_WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "failure-agreement Ring rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process failure agreement failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

fn run_manifest_mismatch_ring(completion_policy_mismatch: bool) {
    assert!(distributed::is_available(Backend::Ring));
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(2));
    for rank in 0..2 {
        let mut child = Command::new(&executable);
        child
            .args([
                "--exact",
                "backend::runtime::distributed::communication_tests::manifest_mismatch_worker",
                "--nocapture",
            ])
            .env(MISMATCH_WORKER_RANK, rank.to_string())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if completion_policy_mismatch {
            child.env(MISMATCH_COMPLETION_POLICY, "1");
        } else {
            child.env_remove(MISMATCH_COMPLETION_POLICY);
        }
        children.0.push(child.spawn().unwrap());
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "manifest mismatch Ring rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process manifest mismatch failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_manifest_mismatch_fails_consistently_without_blocking() {
    run_manifest_mismatch_ring(false);
    run_manifest_mismatch_ring(true);
}
