use super::*;
use eredu_core::ParallelTopology;
use eredu_runtime::{
    project_all_communication_manifests, CommunicationGroupRequirements, TensorSlice,
    TopologyCommunicationPlan,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

fn completion_policy() -> eredu_runtime::CommunicationCompletionPolicy {
    eredu_runtime::CommunicationCompletionPolicy::new(
        std::time::Duration::from_secs(30),
        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap()
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn write_index(dir: &Path, mappings: &[(&str, &str)]) {
    let weight_map = mappings
        .iter()
        .map(|(key, file)| ((*key).to_string(), serde_json::json!(file)))
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_i32_tensor(path: &Path, name: &str, values: &[i32], shape: Vec<usize>) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::I32, shape, &bytes).unwrap();
    serialize_to_file([(name, view)], None, path).unwrap();
}

fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> MlxRankContext {
    let topology = crate::test_parallel_rank(rank, tp, pp, ep);
    MlxRankContext::new(
        topology.world_size(),
        topology.global_rank(),
        DeviceAssignment::new(DeviceType::Cpu, 0),
    )
    .unwrap()
}

fn communication_requirement(
    operation: CommunicationOperation,
) -> CommunicationOperationRequirement {
    CommunicationOperationRequirement::tensors(
        operation,
        [TensorDtype::F32, TensorDtype::Bf16],
        CommunicationTensorLimits::new(
            1,
            4,
            16_384,
            (operation == CommunicationOperation::VariableAllToAll).then_some(4096),
        )
        .unwrap(),
        true,
    )
    .unwrap()
}

fn communication_group_requirements(
    operation: CommunicationOperation,
) -> CommunicationGroupRequirements {
    CommunicationGroupRequirements::new([communication_requirement(operation)]).unwrap()
}

#[test]
fn manifest_preparation_preserves_opaque_groups_and_routes() {
    let manifest = CommunicationManifest::new(
        6,
        2,
        vec![
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(7),
                0,
                vec![0, 2, 4],
                Some(1),
                communication_group_requirements(CommunicationOperation::AllReduceSum),
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(11),
                1,
                vec![2, 3],
                Some(0),
                communication_group_requirements(CommunicationOperation::AllGatherEven),
            )
            .unwrap(),
        ],
        vec![CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(19),
            0,
            2,
            5,
            communication_requirement(CommunicationOperation::SendReceive),
        )
        .unwrap()],
    )
    .unwrap();

    assert_eq!(manifest.world_size(), 6);
    assert_eq!(manifest.rank(), 2);
    assert_eq!(manifest.groups().len(), 2);
    assert_eq!(manifest.groups()[0].id(), CollectiveGroupId::new(7));
    assert_eq!(manifest.groups()[0].members(), [0, 2, 4]);
    assert_eq!(manifest.groups()[0].local_index(), Some(1));
    assert_eq!(manifest.groups()[1].id(), CollectiveGroupId::new(11));
    assert_eq!(manifest.routes().len(), 1);
    assert_eq!(manifest.routes()[0].id(), CommunicationRouteId::new(19));
    assert_eq!(manifest.routes()[0].source(), 2);
    assert_eq!(manifest.routes()[0].destination(), 5);
}

#[test]
fn manifest_constructor_preserves_singleton_group_mechanics() {
    let all_reduce_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
        true,
    )
    .unwrap();
    let gather_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllGatherEven,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
        true,
    )
    .unwrap();
    let variable_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::VariableAllToAll,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 4, Some(1)).unwrap(),
        true,
    )
    .unwrap();
    let inexact_requirement = CommunicationOperationRequirement::tensors(
        CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],
        CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
        false,
    )
    .unwrap();
    let manifest = CommunicationManifest::new(
        1,
        0,
        vec![
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(23),
                0,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([all_reduce_requirement]).unwrap(),
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(29),
                1,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([gather_requirement]).unwrap(),
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(31),
                2,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([variable_requirement]).unwrap(),
            )
            .unwrap(),
            CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(37),
                3,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([inexact_requirement]).unwrap(),
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communicators = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    assert!(communicators.group(CollectiveGroupId::new(23)).is_none());
    let all_reduce = communicators
        .communication_group(CollectiveGroupId::new(23))
        .unwrap();
    let gather = communicators
        .communication_group(CollectiveGroupId::new(29))
        .unwrap();
    let variable = communicators
        .communication_group(CollectiveGroupId::new(31))
        .unwrap();
    let inexact = communicators
        .communication_group(CollectiveGroupId::new(37))
        .unwrap();
    assert_eq!(all_reduce.opaque_id(), Some(CollectiveGroupId::new(23)));
    assert_eq!(gather.opaque_id(), Some(CollectiveGroupId::new(29)));
    assert_eq!(
        all_reduce.native_group().size(),
        gather.native_group().size()
    );

    super::super::group::reset_native_collective_submissions();
    let bf16 = Array::from_slice(&[1.0_f32], &[1])
        .as_dtype(safemlx::Dtype::Bfloat16, &stream)
        .unwrap();
    let error = super::super::group::all_sum(&bf16, all_reduce, &stream)
        .expect_err("F32-only manifest must reject BF16 before native submission");
    assert!(error.what().contains("does not admit dtype"));
    assert_eq!(super::super::group::native_collective_submissions(), 0);

    let input = Array::from_slice(&[1.0_f32], &[1]);
    let error = super::super::group::all_sum(&input, gather, &stream)
        .expect_err("one opaque ID must not borrow another ID's operation contract");
    assert!(error.what().contains("does not select operation"));
    assert_eq!(super::super::group::native_collective_submissions(), 0);

    let oversized = Array::from_slice(&[0.0_f32; 5], &[5]);
    let error = super::super::group::all_sum(&oversized, all_reduce, &stream)
        .expect_err("manifest tensor limit must reject before native submission");
    assert!(error.what().contains("rejects tensor shape"));
    assert_eq!(super::super::group::native_collective_submissions(), 0);

    let variable_input = Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let error = super::super::group::all_to_all_v(&variable_input, &[2], &[2], variable, &stream)
        .expect_err("per-peer contract must reject before native submission");
    assert!(error.what().contains("peer counts exceed"));
    assert_eq!(super::super::group::native_collective_submissions(), 0);

    let error = super::super::group::all_sum(&input, inexact, &stream)
        .expect_err("inexact manifest operation must not reach exact native entry point");
    assert!(error.what().contains("selects inexact operation"));
    assert_eq!(super::super::group::native_collective_submissions(), 0);

    let output = super::super::group::all_sum(&input, all_reduce, &stream).unwrap();
    assert_eq!(super::super::group::native_collective_submissions(), 1);
    assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), &[1.0]);
    assert!(communicators.route(CommunicationRouteId::new(1)).is_none());
}

#[test]
fn route_handle_rejects_endpoints_outside_its_owned_world() {
    let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), 1);
    let world = Group::uncontracted(&world);
    let descriptor = CommunicationRouteDescriptor::new(
        CommunicationRouteId::new(29),
        0,
        0,
        1,
        communication_requirement(CommunicationOperation::SendReceive),
    )
    .unwrap();
    let error = CommunicationRouteRealization::from_descriptor(&descriptor, &world, false)
        .expect_err("route must be bound to a world containing both endpoints");
    assert!(error.to_string().contains("owned world size 1"));
}

#[test]
fn projected_manifests_prepare_one_local_group_per_creation_batch_on_every_rank() {
    let topology = ParallelTopology::new(2, 2, 2, 1).unwrap();
    let plan = TopologyCommunicationPlan::new()
        .with_tensor_groups(communication_group_requirements(
            CommunicationOperation::AllReduceSum,
        ))
        .with_pipeline_groups(communication_group_requirements(
            CommunicationOperation::AllGatherEven,
        ))
        .with_expert_groups(communication_group_requirements(
            CommunicationOperation::VariableAllToAll,
        ));
    let manifests = project_all_communication_manifests(topology, &plan).unwrap();
    eredu_runtime::validate_compatible_communication_manifests(&manifests).unwrap();

    for manifest in &manifests {
        assert_eq!(manifest.groups().len(), 3);
        for (creation_order, group) in manifest.groups().iter().enumerate() {
            assert_eq!(
                manifest.groups()[creation_order].creation_order(),
                creation_order
            );
            assert_eq!(group.id(), manifest.groups()[creation_order].id());
            assert_eq!(
                group.members()[group.local_index().unwrap()],
                manifest.rank()
            );
            assert_eq!(group.members().len(), 2);
        }
    }
}

#[test]
fn mlx_manifest_capabilities_admit_publication_but_fail_closed_on_unsupported_dtype() {
    let publication = CommunicationManifest::new(
        2,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(1),
            0,
            vec![0, 1],
            Some(0),
            CommunicationGroupRequirements::new([
                communication_requirement(CommunicationOperation::Broadcast),
                CommunicationOperationRequirement::barrier(true),
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    mlx_communication_capabilities()
        .validate_manifest(&publication)
        .unwrap();

    let unsupported = CommunicationManifest::new(
        1,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(1),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::I32],
                CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let error = mlx_communication_capabilities()
        .validate_manifest(&unsupported)
        .expect_err("integer broadcast must be rejected");
    assert!(error.to_string().contains("I32"));
}

#[test]
fn mlx_failure_agreement_capability_is_exact_and_distinct_from_barrier() {
    let failure_agreement = CommunicationManifest::new(
        1,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(3),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([
                CommunicationOperationRequirement::failure_agreement(true),
            ])
            .unwrap(),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    mlx_communication_capabilities()
        .validate_manifest(&failure_agreement)
        .unwrap();

    let capabilities_without_agreement =
        CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap()
            .with_completion_capabilities(
                CommunicationCompletionCapabilities::new([
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                ])
                .unwrap(),
            );
    assert!(capabilities_without_agreement
        .validate_manifest(&failure_agreement)
        .is_err());

    let inexact_agreement =
        CommunicationCapabilities::new([CommunicationOperationRequirement::failure_agreement(
            false,
        )])
        .unwrap()
        .with_completion_capabilities(
            CommunicationCompletionCapabilities::new([
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
        );
    assert!(inexact_agreement
        .validate_manifest(&failure_agreement)
        .is_err());
}

#[test]
fn mlx_capabilities_are_operation_specific_and_exclude_encoded_payloads() {
    let capabilities = mlx_communication_capabilities();
    let group_manifest = |operation, dtype| {
        let max_peer_count = (operation == CommunicationOperation::VariableAllToAll).then_some(8);
        CommunicationManifest::new(
            1,
            0,
            vec![CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(1),
                0,
                vec![0],
                Some(0),
                CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                    operation,
                    [dtype],
                    CommunicationTensorLimits::new(1, 2, 8, max_peer_count).unwrap(),
                    true,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap()],
            vec![],
        )
        .unwrap()
        .with_completion_policy(completion_policy())
    };

    capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::AllGatherEven,
            TensorDtype::I32,
        ))
        .unwrap();
    assert!(capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::AllReduceSum,
            TensorDtype::I32,
        ))
        .is_err());
    assert!(capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::AllGatherUneven,
            TensorDtype::I32,
        ))
        .is_err());
    capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::VariableAllToAll,
            TensorDtype::I32,
        ))
        .unwrap();
    assert!(capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::VariableAllToAll,
            TensorDtype::U32,
        ))
        .is_err());
    assert!(capabilities
        .validate_manifest(&group_manifest(
            CommunicationOperation::AllGatherEven,
            TensorDtype::U32,
        ))
        .is_err());

    let route_manifest = |dtype| {
        CommunicationManifest::new(
            2,
            0,
            vec![],
            vec![CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(1),
                0,
                0,
                1,
                CommunicationOperationRequirement::tensors(
                    CommunicationOperation::SendReceive,
                    [dtype],
                    CommunicationTensorLimits::new(1, 2, 8, None).unwrap(),
                    true,
                )
                .unwrap(),
            )
            .unwrap()],
        )
        .unwrap()
        .with_completion_policy(completion_policy())
    };
    for dtype in [TensorDtype::I32, TensorDtype::U32] {
        capabilities
            .validate_manifest(&route_manifest(dtype))
            .unwrap();
    }
    assert!(capabilities
        .validate_manifest(&route_manifest(TensorDtype::I64))
        .is_err());

    let requirements =
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::Encoded("packed".into())],
            CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
            true,
        )
        .unwrap()])
        .unwrap();
    let manifest = CommunicationManifest::new(
        1,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(1),
            0,
            vec![0],
            Some(0),
            requirements,
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    assert!(capabilities.validate_manifest(&manifest).is_err());

    let route_manifest = CommunicationManifest::new(
        2,
        0,
        vec![],
        vec![CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(1),
            0,
            0,
            1,
            communication_requirement(CommunicationOperation::SendReceive),
        )
        .unwrap()],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    mlx_communication_capabilities()
        .validate_manifest(&route_manifest)
        .unwrap();

    let exact_manifest = CommunicationManifest::new(
        1,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(1),
            0,
            vec![0],
            Some(0),
            communication_group_requirements(CommunicationOperation::AllReduceSum),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    mlx_communication_capabilities()
        .validate_manifest(&exact_manifest)
        .unwrap();

    let barrier = CommunicationManifest::new(
        1,
        0,
        vec![CommunicationGroupDescriptor::new(
            CollectiveGroupId::new(2),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap(),
        )
        .unwrap()],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    mlx_communication_capabilities()
        .validate_manifest(&barrier)
        .unwrap();
}

#[test]
fn validates_tensor_slices() {
    let slice = TensorSlice::for_shape(&[4, 12], 1, 2, 3).unwrap();
    assert_eq!(slice.start(), 8);
    assert_eq!(slice.end(), 12);
    assert_eq!(slice.local_shape(&[4, 12]), [4, 4]);
    assert!(TensorSlice::for_shape(&[4, 11], 1, 0, 3).is_err());
    assert!(TensorSlice::for_shape(&[4, 12], 2, 0, 3).is_err());
    assert!(TensorSlice::for_shape(&[4, 12], 1, 3, 3).is_err());
}

#[test]
fn validates_explicit_execution_stream_device() {
    let stream = stream();
    topology(0, 1, 1, 1)
        .validate_execution_stream(&stream)
        .unwrap();
    let other_assignment =
        MlxRankContext::new(1, 0, DeviceAssignment::new(DeviceType::Cpu, 1)).unwrap();
    assert!(other_assignment.validate_execution_stream(&stream).is_err());
}

#[test]
fn plan_exposes_replicated_omitted_and_quantized_companions() {
    let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
    plan.insert("replicated", TensorPlacement::Replicated);
    plan.insert("remote", TensorPlacement::Omit);
    plan.insert_quantized_companions("projection", TensorPlacement::Local, true);
    assert_eq!(
        plan.placement("replicated"),
        Some(&TensorPlacement::Replicated)
    );
    assert_eq!(plan.placement("remote"), Some(&TensorPlacement::Omit));
    assert_eq!(
        plan.placement("projection.weight"),
        Some(&TensorPlacement::Local)
    );
    assert_eq!(
        plan.placement("projection.scales"),
        Some(&TensorPlacement::Local)
    );
    assert_eq!(
        plan.placement("projection.biases"),
        Some(&TensorPlacement::Local)
    );

    let mut invalid = PlacementPlan::new(topology(0, 1, 1, 1));
    invalid.insert("bad_owner", TensorPlacement::Rank { rank: 1 });
    assert!(invalid.validate().is_err());
}

#[test]
fn plan_supports_balanced_uneven_ranges() {
    let mut plan = PlacementPlan::new(topology(2, 3, 1, 1));
    let range = eredu_core::balanced_contiguous_range(11, 3, 2, false).unwrap();
    plan.insert(
        "embedding.weight",
        TensorPlacement::Range {
            axis: 0,
            start: range.start,
            end: range.end,
        },
    );
    assert_eq!(range, 8..11);
    assert_eq!(
        plan.placement("embedding.weight"),
        Some(&TensorPlacement::Range {
            axis: 0,
            start: 8,
            end: 11,
        })
    );
    plan.insert_expected(
        "head.weight",
        vec![11, 4],
        TensorPlacement::Range {
            axis: 0,
            start: 8,
            end: 11,
        },
    )
    .unwrap();
    plan.validate().unwrap();
}

#[test]
fn typed_rank_ownership_resolves_locally() {
    let mut rank_zero = PlacementPlan::new(topology(0, 2, 2, 1));
    rank_zero.insert("owned", TensorPlacement::Rank { rank: 3 });
    let mut rank_three = PlacementPlan::new(topology(3, 2, 2, 1));
    rank_three.insert("owned", TensorPlacement::Rank { rank: 3 });
    assert!(matches!(
        rank_zero.logical.resolve("owned", &[2]).unwrap(),
        eredu_runtime::ResolvedTensorPlacement::Omit
    ));
    assert!(matches!(
        rank_three.logical.resolve("owned", &[2]).unwrap(),
        eredu_runtime::ResolvedTensorPlacement::Materialize
    ));
}

#[test]
fn selective_loader_skips_remote_shards_and_reconstructs_tp_slices() {
    let dir = tempfile::tempdir().unwrap();
    let stream = stream();
    write_i32_tensor(
        &dir.path().join("local.safetensors"),
        "model.projection.weight",
        &[0, 1, 2, 3, 10, 11, 12, 13],
        vec![2, 4],
    );
    // This is deliberately not a safetensors file. Correct index-level
    // selection must never open it for either rank.
    std::fs::write(dir.path().join("remote.safetensors"), b"must not be opened").unwrap();
    write_index(
        dir.path(),
        &[
            ("model.projection.weight", "local.safetensors"),
            ("model.remote.weight", "remote.safetensors"),
        ],
    );
    let local_shard = dir.path().join("local.safetensors").canonicalize().unwrap();

    let mut reconstructed = Vec::new();
    for rank in 0..2 {
        let topology = topology(rank, 2, 1, 1);
        let mut plan = PlacementPlan::new(topology);
        plan.insert_expected(
            "model.projection.weight",
            vec![2, 4],
            TensorPlacement::Shard {
                axis: 1,
                index: rank,
                parts: 2,
            },
        )
        .unwrap();
        plan.insert("model.remote.weight", TensorPlacement::Omit);
        let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
        assert_eq!(partition.len(), 1);
        assert_eq!(
            partition.opened_shards(),
            std::slice::from_ref(&local_shard)
        );
        assert!(partition.get("model.remote.weight").is_none());
        let local = partition
            .get("model.projection.weight")
            .unwrap()
            .evaluated()
            .unwrap();
        assert_eq!(local.as_array().shape(), &[2, 2]);
        reconstructed.push(local.as_slice::<i32>().to_vec());
    }
    // Slices are axis-1 contiguous views, so reconstruct each row from
    // the corresponding rows of both rank-local tensors.
    assert_eq!(reconstructed[0], [0, 1, 10, 11]);
    assert_eq!(reconstructed[1], [2, 3, 12, 13]);
    let union = [
        reconstructed[0][0],
        reconstructed[0][1],
        reconstructed[1][0],
        reconstructed[1][1],
        reconstructed[0][2],
        reconstructed[0][3],
        reconstructed[1][2],
        reconstructed[1][3],
    ];
    assert_eq!(union, [0, 1, 2, 3, 10, 11, 12, 13]);
}

#[test]
fn selective_loader_materializes_only_ordered_noncontiguous_indices() {
    let dir = tempfile::tempdir().unwrap();
    let stream = stream();
    write_i32_tensor(
        &dir.path().join("model.safetensors"),
        "experts",
        &[0, 1, 10, 11, 20, 21, 30, 31, 40, 41],
        vec![5, 2],
    );
    let mut plan = PlacementPlan::new(topology(1, 1, 1, 2));
    plan.insert_expected(
        "experts",
        vec![5, 2],
        TensorPlacement::Indices {
            axis: 0,
            indices: vec![3, 1],
        },
    )
    .unwrap();
    let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
    let local = partition.get("experts").unwrap().evaluated().unwrap();
    assert_eq!(local.as_array().shape(), &[2, 2]);
    assert_eq!(local.as_slice::<i32>(), &[30, 31, 10, 11]);

    for indices in [vec![], vec![1, 1], vec![1, 5]] {
        let mut invalid = PlacementPlan::new(topology(1, 1, 1, 2));
        assert!(invalid
            .insert_expected(
                "experts",
                vec![5, 2],
                TensorPlacement::Indices { axis: 0, indices },
            )
            .is_err());
    }
}

#[test]
fn replicated_default_loads_the_original_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let stream = stream();
    write_i32_tensor(
        &dir.path().join("model.safetensors"),
        "weight",
        &[3, 5, 7, 9],
        vec![2, 2],
    );
    let plan = PlacementPlan::replicated(topology(0, 1, 1, 1));
    let partition = load_safetensors_partition(dir.path(), &plan, &stream).unwrap();
    let loaded = partition.get("weight").unwrap().evaluated().unwrap();
    assert_eq!(loaded.as_slice::<i32>(), &[3, 5, 7, 9]);
}

#[test]
fn omitted_unsupported_tensor_is_never_materialized() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = [0u8; 4];
    let unsupported = TensorView::new(Dtype::F8_E5M2, vec![4], &bytes).unwrap();
    serialize_to_file(
        [("remote", unsupported)],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
    plan.insert("remote", TensorPlacement::Omit);
    let partition = load_safetensors_partition(dir.path(), &plan, &stream()).unwrap();
    assert!(partition.is_empty());
}

#[test]
fn remote_only_index_shard_is_never_opened() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
    write_index(dir.path(), &[("remote.weight", "remote.safetensors")]);
    let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
    plan.insert("remote.weight", TensorPlacement::Omit);
    let partition = load_safetensors_partition(dir.path(), &plan, &stream()).unwrap();
    assert!(partition.is_empty());
    assert!(partition.opened_shards().is_empty());
}

#[test]
fn selective_loader_rejects_an_incomplete_opened_shard() {
    let dir = tempfile::tempdir().unwrap();
    write_i32_tensor(
        &dir.path().join("local.safetensors"),
        "requested.weight",
        &[1],
        vec![1],
    );
    std::fs::write(dir.path().join("remote.safetensors"), b"not safetensors").unwrap();
    write_index(
        dir.path(),
        &[
            ("requested.weight", "local.safetensors"),
            ("missing.weight", "local.safetensors"),
            ("remote.weight", "remote.safetensors"),
        ],
    );
    let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
    plan.insert("requested.weight", TensorPlacement::Local);
    plan.insert("missing.weight", TensorPlacement::Omit);
    plan.insert("remote.weight", TensorPlacement::Omit);

    assert!(matches!(
        load_safetensors_partition(dir.path(), &plan, &stream()),
        Err(Error::CheckpointStore(
            eredu_checkpoint::store::StoreError::ContradictoryIndexMapping { key, .. }
        )) if key == "missing.weight"
    ));
}

#[test]
fn strict_partition_rejects_missing_malformed_and_unexpected_local_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let stream = stream();
    write_i32_tensor(
        &dir.path().join("model.safetensors"),
        "present",
        &[1, 2, 3, 4],
        vec![2, 2],
    );
    let topology = topology(0, 1, 1, 1);

    let mut malformed = PlacementPlan::new(topology);
    malformed
        .insert_expected("present", vec![4, 2], TensorPlacement::Local)
        .unwrap();
    assert!(matches!(
        load_safetensors_partition(dir.path(), &malformed, &stream),
        Err(Error::Parallel(_))
    ));

    let mut missing = PlacementPlan::new(topology);
    missing.insert("present", TensorPlacement::Omit);
    missing.insert("required", TensorPlacement::Local);
    let error = load_safetensors_partition(dir.path(), &missing, &stream).unwrap_err();
    match error {
        Error::StrictLoadValidation { missing, unused } => {
            assert_eq!(missing, ["required"]);
            assert!(unused.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }

    let strict_empty = PlacementPlan::new(topology);
    let error = load_safetensors_partition(dir.path(), &strict_empty, &stream).unwrap_err();
    match error {
        Error::StrictLoadValidation { missing, unused } => {
            assert!(missing.is_empty());
            assert_eq!(unused, ["present"]);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn late_partition_preflight_failure_performs_zero_payload_reads_or_native_work() {
    let dir = tempfile::tempdir().unwrap();
    let early_bytes = [1_i32, 2]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let late_bytes = [3_i32, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let early = TensorView::new(Dtype::I32, vec![2], &early_bytes).unwrap();
    let late = TensorView::new(Dtype::I32, vec![2], &late_bytes).unwrap();
    serialize_to_file(
        [("a.valid", early), ("z.invalid", late)],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();

    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let mut plan = PlacementPlan::new(topology(0, 1, 1, 1));
    plan.insert_expected("a.valid", vec![2], TensorPlacement::Local)
        .unwrap();
    plan.insert_expected("z.invalid", vec![3], TensorPlacement::Local)
        .unwrap();
    PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.store(0, Ordering::Relaxed);
    let source_stream = stream();
    let execution_stream = stream();

    assert!(matches!(
        load_partition_from_store_on_streams(&store, &plan, &source_stream, &execution_stream,),
        Err(Error::Parallel(_))
    ));
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
    assert_eq!(
        PARTITION_NATIVE_MATERIALIZATION_ATTEMPTS.load(Ordering::Relaxed),
        0
    );
}
