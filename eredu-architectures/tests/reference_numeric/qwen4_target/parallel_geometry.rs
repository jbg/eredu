//! Encoded physical selections and literal-aware rank identity without materialization.
use super::*;
use eredu_checkpoint::{
    store::TensorSelection, BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat,
};
use eredu_nn::{GatedProductGroupLayout, GroupedProjectionSpec, LinearFormatSpec, LinearRowLayout};

fn geometry() -> TargetSpec {
    let mut config = configuration();
    config.attention.heads = 4;
    config.attention.kv_heads = 2;
    config.recurrent.key_heads = 4;
    config.recurrent.value_heads = 8;
    config.experts.intermediate = 4;
    specification_for(config)
}
fn range(axis: usize, start: usize, end: usize) -> TensorSelection {
    TensorSelection::Range { axis, start, end }
}

#[test]
fn qwen4_tensor_geometry_retains_fused_fp8_scale_coordinates() {
    let mut spec = geometry();
    let UnitSpec::Decoder { feed_forward, .. } = &mut spec.units[0] else {
        panic!("decoder")
    };
    let original = &feed_forward.feed_forward.experts;
    let GatedProductGroupLayout::Packed { gate_up, down } = original.layout() else {
        panic!("packed")
    };
    let encoded = |projection: &GroupedProjectionSpec, fused| {
        let format = LinearFormatSpec::scaled(
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(2, 2, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
            ),
            ParameterSpec::trainable(format!("{}_scales", projection.weight().id)).unwrap(),
        )
        .unwrap();
        let format = if fused {
            format
                .with_row_layout(LinearRowLayout::equal_partitions(2).unwrap())
                .unwrap()
        } else {
            format
        };
        GroupedProjectionSpec::new(
            projection.weight().clone(),
            projection.bias().cloned(),
            format,
        )
        .unwrap()
    };
    let gate_name = gate_up.weight().id.as_str().to_owned();
    let down_name = down.weight().id.as_str().to_owned();
    feed_forward.feed_forward.experts = GroupedGatedProductSpec::new(
        original.group_count(),
        original.input_dimensions(),
        original.intermediate_dimensions(),
        original.output_dimensions(),
        original.policy(),
        GatedProductGroupLayout::Packed {
            gate_up: encoded(gate_up, true),
            down: encoded(down, false),
        },
    )
    .unwrap()
    .with_reduction(original.reduction());
    let GatedProductGroupLayout::Packed { gate_up, down } =
        feed_forward.feed_forward.experts.layout()
    else {
        panic!("packed")
    };
    let formats = BTreeMap::from([
        (gate_name.clone(), gate_up.format().clone()),
        (down_name.clone(), down.format().clone()),
    ]);
    let partition = spec.tensor_partition(1, 2).unwrap();
    assert_eq!(
        partition.parameter_selections(&gate_name).unwrap(),
        [range(1, 2, 4), range(1, 6, 8)]
    );
    assert_eq!(
        partition
            .parameter_selections(&format!("{gate_name}_scales"))
            .unwrap(),
        [range(1, 1, 2), range(1, 3, 4)]
    );
    assert_eq!(
        partition.parameter_selections(&down_name).unwrap(),
        [range(2, 2, 4)]
    );
    assert_eq!(
        partition
            .parameter_selections(&format!("{down_name}_scales"))
            .unwrap(),
        [range(2, 1, 2)]
    );
    let (global, local) = encoded_placement_descriptions(&spec, &partition, &formats);
    let layout = partition.local_layout(&global, &local).unwrap();
    let down = layout.tensor(&down_name).unwrap();
    assert_eq!(down.logical_units(), Some(2));
    assert_eq!(down.logical_range(), Some(&(1..2)));
    let scalar_units = eredu_core::component::ComponentCoordinateMap::partition_units(
        4,
        down.logical_units().unwrap(),
        down.logical_range().unwrap().clone(),
    )
    .unwrap();
    assert_eq!(scalar_units.contiguous_range(), Some(2..4));
    let scales = layout.tensor(&format!("{gate_name}_scales")).unwrap();
    assert_eq!(scales.global_shape(), [3, 4, 1]);
    assert_eq!(scales.local_shape(), [3, 2, 1]);
    assert_eq!(
        scales.placement(),
        &eredu_runtime::TensorPlacement::Indices {
            axis: 1,
            indices: vec![1, 3]
        }
    );
    // A four-way split would cut each two-row inverse-scale block in half.
    assert!(spec.tensor_partition(0, 4).is_err());
}

#[test]
fn qwen4_tensor_geometry_rejects_partial_gguf_blocks_and_preserves_tp1() {
    let mut config = geometry().configuration().clone();
    config.experts.shared_intermediate = 64;
    let mut spec = specification_for(config);
    let UnitSpec::Decoder { feed_forward, .. } = &mut spec.units[0] else {
        panic!("decoder")
    };
    let down = &mut feed_forward.feed_forward.shared[2];
    down.format = LinearFormatSpec::unscaled(LinearFormat::GgufIQuant {
        ggml_type: eredu_gguf::GgmlType::Q8_0,
        endian: eredu_gguf::Endian::Little,
    })
    .unwrap();
    let name = down.weight.id.as_str().to_owned();
    let formats = BTreeMap::from([(name.clone(), down.format.clone())]);
    // Two encoded blocks of 32 values / 34 bytes: TP2 owns one complete block.
    let partition = spec.tensor_partition(1, 2).unwrap();
    assert_eq!(
        partition.parameter_selections(&name).unwrap(),
        [range(1, 34, 68)]
    );
    let (global, local) = encoded_placement_descriptions(&spec, &partition, &formats);
    let layout = partition.local_layout(&global, &local).unwrap();
    let down = layout.tensor(&name).unwrap();
    assert_eq!(down.global_shape(), [2, 68]);
    assert_eq!(down.local_shape(), [2, 34]);
    assert_eq!(down.logical_units(), Some(2));
    assert_eq!(down.logical_range(), Some(&(1..2)));
    // The 34-byte physical block contains 32 logical expert units.
    let scalar_units = eredu_core::component::ComponentCoordinateMap::partition_units(
        64,
        down.logical_units().unwrap(),
        down.logical_range().unwrap().clone(),
    )
    .unwrap();
    assert_eq!(scalar_units.contiguous_range(), Some(32..64));
    assert_eq!(
        down.placement(),
        &eredu_runtime::TensorPlacement::Range {
            axis: 1,
            start: 34,
            end: 68
        }
    );
    // 68 physical bytes divide by four, but sixteen values do not form a block.
    assert!(spec
        .tensor_partition(0, 4)
        .unwrap_err()
        .to_string()
        .contains("GGUF encoding block"));
    let identity = spec.tensor_partition(0, 1).unwrap();
    assert_eq!(
        identity.local_spec().geometry_fingerprint(),
        spec.geometry_fingerprint()
    );
    assert!(identity.parameter_selections(&name).is_none());
}

#[test]
fn qwen4_tensor_geometry_rejects_rank_and_source_drift() {
    let spec = geometry();
    for (rank, ranks) in [(0, 0), (2, 2), (usize::MAX, 2)] {
        assert!(spec.tensor_partition(rank, ranks).is_err());
    }
    let partition = spec.tensor_partition(0, 2).unwrap();
    let bound = bind_spec(spec.clone());
    let local = bound.tensor_partition(&partition).unwrap();
    let other_rank = bound
        .tensor_partition(&spec.tensor_partition(1, 2).unwrap())
        .unwrap();
    assert_ne!(local.state_fingerprint(), other_rank.state_fingerprint());
    assert_eq!(
        local.geometry().configuration().recurrent.key_heads,
        spec.configuration().recurrent.key_heads
    );
    assert_eq!(
        local.geometry().configuration().vocabulary,
        spec.configuration().vocabulary
    );
    let mut other = spec;
    other.limits.history_tokens -= 1;
    assert!(bind_spec(other).tensor_partition(&partition).is_err());
}

fn placement_descriptions(
    spec: &TargetSpec,
    partition: &eredu_architectures::qwen4_exp::target::TargetTensorPartition,
) -> (
    eredu_runtime::ArchitectureParameterDescription,
    eredu_runtime::ArchitectureParameterDescription,
) {
    let ctx = NumericContext::default();
    let bound = bind_spec(spec.clone());
    let global = TargetModel::<NumericBackend>::new(bound.clone(), &ctx).unwrap();
    let local =
        TargetModel::<NumericBackend>::new_tensor_parallel(bound, partition.clone(), &ctx).unwrap();
    (
        global.parameter_description(&ctx).unwrap(),
        local.parameter_description(&ctx).unwrap(),
    )
}

// The scalar backend keeps FP8/GGUF parameters decoded and omits their physical
// companions. Expand those logical descriptions through the neutral format contract
// for these layout-only fixtures; native encoded modules expose these physical shapes.
fn encoded_placement_descriptions(
    spec: &TargetSpec,
    partition: &eredu_architectures::qwen4_exp::target::TargetTensorPartition,
    formats: &BTreeMap<String, LinearFormatSpec>,
) -> (
    eredu_runtime::ArchitectureParameterDescription,
    eredu_runtime::ArchitectureParameterDescription,
) {
    let (global, local) = placement_descriptions(spec, partition);
    (
        expand_description(global, formats),
        expand_description(local, formats),
    )
}

fn expand_description(
    description: eredu_runtime::ArchitectureParameterDescription,
    formats: &BTreeMap<String, LinearFormatSpec>,
) -> eredu_runtime::ArchitectureParameterDescription {
    use eredu_runtime::{ArchitectureParameterDescription, OwnedParameterGroupSpec};
    let groups = eredu_runtime::expand_linear_format_parameter_groups(
        description
            .groups()
            .iter()
            .map(|g| g.group().clone())
            .collect(),
        |member| Ok(formats.get(member.target()).cloned()),
    )
    .unwrap();
    let owned = description
        .groups()
        .iter()
        .zip(&groups)
        .map(|(original, group)| {
            OwnedParameterGroupSpec::new(original.owner().clone(), group.clone())
        })
        .collect::<Vec<_>>();
    ArchitectureParameterDescription::new(
        description.graph(),
        description.unit_layout(),
        groups,
        owned,
    )
    .unwrap()
}

#[test]
fn qwen4_tensor_layout_preserves_partial_kv_replication_and_fused_order() {
    use eredu_runtime::TensorPlacement as P;
    let spec = geometry();
    for rank in 0..4 {
        let partition = spec.tensor_partition(rank, 4).unwrap();
        let (global, local) = placement_descriptions(&spec, &partition);
        let layout = partition.local_layout(&global, &local).unwrap();
        let rank_topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(4, 1, 1, 1).unwrap(),
            rank,
        )
        .unwrap();
        assert_eq!(
            partition.projected_layout(&global, rank_topology).unwrap(),
            layout
        );
        let wrong_rank =
            eredu_core::ParallelRankTopology::new(rank_topology.topology(), (rank + 1) % 4)
                .unwrap();
        assert!(partition.projected_layout(&global, wrong_rank).is_err());
        let kv = layout
            .tensor("model.layers.1.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(kv.global_shape(), [4, 2]);
        assert_eq!(kv.local_shape(), [2, 2]);
        assert_eq!(kv.logical_units(), Some(2));
        assert_eq!(kv.logical_range(), Some(&(rank / 2..rank / 2 + 1)));
        assert_eq!(
            kv.placement(),
            &P::Range {
                axis: 0,
                start: 2 * (rank / 2),
                end: 2 * (rank / 2 + 1)
            }
        );
        let qkv = layout
            .tensor("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap();
        assert_eq!(qkv.global_shape(), [32, 2]);
        assert_eq!(qkv.local_shape(), [8, 2]);
        assert_eq!(
            qkv.placement(),
            &P::Indices {
                axis: 0,
                indices: [
                    2 * rank..2 * rank + 2,
                    8 + 2 * rank..10 + 2 * rank,
                    16 + 4 * rank..20 + 4 * rank
                ]
                .into_iter()
                .flatten()
                .collect()
            }
        );
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.1.ple.key_proj.weight",
        ] {
            assert_eq!(layout.tensor(name).unwrap().placement(), &P::Replicated);
        }
        for (_, tensor) in layout.tensors() {
            assert!(!tensor.fell_back_to_replication());
            match tensor.placement() {
                P::Range { axis, start, end } => {
                    let width = end - start;
                    assert_eq!(
                        tensor.logical_units(),
                        Some(tensor.global_shape()[*axis] / width)
                    );
                    assert_eq!(
                        tensor.logical_range().cloned(),
                        Some(start / width..end / width)
                    );
                }
                _ => {
                    assert!(tensor.logical_units().is_none());
                    assert!(tensor.logical_range().is_none());
                }
            }
        }
        assert!(
            partition.local_layout(&global, &global).is_err(),
            "unpartitioned local shapes reject"
        );
    }
    let partition = spec.tensor_partition(0, 1).unwrap();
    let (global, local) = placement_descriptions(&spec, &partition);
    let layout = partition.local_layout(&global, &local).unwrap();
    assert!(layout
        .tensors()
        .all(|(_, tensor)| tensor.placement() == &P::Replicated
            && tensor.global_shape() == tensor.local_shape()));
}

#[test]
fn qwen4_tensor_layout_rejects_missing_selected_parameters_and_changed_ownership() {
    use eredu_runtime::{
        ArchitectureParameterDescription, OwnedParameterGroupSpec, ParameterGroupOwner,
        ParameterGroupSpec,
    };
    let spec = geometry();
    let partition = spec.tensor_partition(0, 2).unwrap();
    let (global, local) = placement_descriptions(&spec, &partition);
    let remove = "model.layers.0.linear_attn.in_proj_qkv.weight";
    let omitted = |description: &ArchitectureParameterDescription| {
        let groups: Vec<_> = description
            .groups()
            .iter()
            .map(|owned| {
                let group = owned.group();
                OwnedParameterGroupSpec::new(
                    owned.owner().clone(),
                    ParameterGroupSpec::new(
                        group.logical_name(),
                        group.role(),
                        group
                            .members()
                            .iter()
                            .filter(|member| member.target() != remove)
                            .cloned(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        ArchitectureParameterDescription::new(
            description.graph(),
            description.unit_layout(),
            groups.iter().map(|g| g.group().clone()),
            groups.clone(),
        )
        .unwrap()
    };
    assert!(partition
        .local_layout(&omitted(&global), &omitted(&local))
        .unwrap_err()
        .to_string()
        .contains("retained selection has no described parameter"));
    let groups: Vec<_> = local
        .groups()
        .iter()
        .enumerate()
        .map(|(i, owned)| {
            OwnedParameterGroupSpec::new(
                if i == 0 {
                    ParameterGroupOwner::static_role("norm")
                } else {
                    owned.owner().clone()
                },
                owned.group().clone(),
            )
        })
        .collect();
    let changed = ArchitectureParameterDescription::new(
        local.graph(),
        local.unit_layout(),
        groups.iter().map(|g| g.group().clone()),
        groups.clone(),
    )
    .unwrap();
    assert!(partition
        .local_layout(&global, &changed)
        .unwrap_err()
        .to_string()
        .contains("changed its logical owner or role"));
}

#[test]
fn qwen4_expert_layout_preserves_released_fp8_scale_blocks_across_tp2_ep2() {
    use eredu_runtime::TensorPlacement as P;
    let directory = tempfile::tempdir().unwrap();
    let prepared = prepared_tensor_parallel::small_safetensors(directory.path(), true);
    let before = prepared
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let spec = prepared.spec();
    for rank in 0..4 {
        let (layout, ep, tp) = encoded_expert_layout(spec, rank);
        let count = if ep == 0 { 2 } else { 1 };
        let start = if ep == 0 { 0 } else { 2 };
        for (suffix, shape, selection) in [
            (
                "gate_up_proj",
                vec![count, 256, 2],
                P::Indices {
                    axis: 1,
                    indices: (tp * 128..(tp + 1) * 128)
                        .chain(256 + tp * 128..256 + (tp + 1) * 128)
                        .collect(),
                },
            ),
            (
                "gate_up_proj_scales",
                vec![count, 2, 1],
                P::Indices {
                    axis: 1,
                    indices: vec![tp, 2 + tp],
                },
            ),
            (
                "down_proj",
                vec![count, 2, 128],
                P::Range {
                    axis: 2,
                    start: tp * 128,
                    end: (tp + 1) * 128,
                },
            ),
            (
                "down_proj_scales",
                vec![count, 1, 1],
                P::Range {
                    axis: 2,
                    start: tp,
                    end: tp + 1,
                },
            ),
        ] {
            let tensor = layout
                .tensor(&format!("model.layers.0.mlp.experts.{suffix}"))
                .unwrap();
            assert_eq!(tensor.local_shape(), shape);
            assert_eq!(tensor.placement(), &selection);
            assert_eq!(
                tensor.additional_placements(),
                [P::Range {
                    axis: 0,
                    start,
                    end: start + count
                }]
            );
        }
        let table = layout.tensor("model.layers.1.ple.key_proj.weight").unwrap();
        assert!(table.additional_placements().is_empty());
        assert_eq!(table.placement(), &P::Replicated);
    }
    assert_eq!(
        prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        before,
        "encoded rank geometry never acquires weights or scales"
    );
}

#[test]
fn qwen4_expert_layout_preserves_gguf_blocks_and_uneven_ep_members() {
    use eredu_runtime::TensorPlacement as P;
    let mut config = geometry().configuration().clone();
    config.experts.intermediate = 64;
    let mut spec = specification_for(config);
    for unit in &mut spec.units {
        let UnitSpec::Decoder { feed_forward, .. } = unit else {
            continue;
        };
        let original = &feed_forward.feed_forward.experts;
        let GatedProductGroupLayout::Packed { gate_up, down } = original.layout() else {
            panic!();
        };
        let down = GroupedProjectionSpec::new(
            down.weight().clone(),
            down.bias().cloned(),
            LinearFormatSpec::unscaled(LinearFormat::GgufIQuant {
                ggml_type: eredu_gguf::GgmlType::Q8_0,
                endian: eredu_gguf::Endian::Little,
            })
            .unwrap(),
        )
        .unwrap();
        feed_forward.feed_forward.experts = GroupedGatedProductSpec::new(
            original.group_count(),
            original.input_dimensions(),
            original.intermediate_dimensions(),
            original.output_dimensions(),
            original.policy(),
            GatedProductGroupLayout::Packed {
                gate_up: gate_up.clone(),
                down,
            },
        )
        .unwrap()
        .with_reduction(original.reduction());
    }
    for rank in 0..4 {
        let (layout, ep, tp) = encoded_expert_layout(&spec, rank);
        let tensor = layout
            .tensor("model.layers.0.mlp.experts.down_proj")
            .unwrap();
        assert_eq!(tensor.global_shape(), [3, 2, 68]);
        assert_eq!(tensor.local_shape(), [if ep == 0 { 2 } else { 1 }, 2, 34]);
        assert_eq!(
            tensor.placement(),
            &P::Range {
                axis: 2,
                start: tp * 34,
                end: (tp + 1) * 34
            }
        );
        assert_eq!(
            tensor.additional_placements(),
            [P::Range {
                axis: 0,
                start: if ep == 0 { 0 } else { 2 },
                end: if ep == 0 { 2 } else { 3 },
            }]
        );
        assert_eq!(tensor.logical_units(), Some(2));
        assert_eq!(tensor.logical_range(), Some(&(tp..tp + 1)));
    }
    assert!(spec
        .tensor_partition(0, 4)
        .unwrap_err()
        .to_string()
        .contains("GGUF encoding block"));
}

fn encoded_expert_layout(
    spec: &TargetSpec,
    global_rank: usize,
) -> (eredu_runtime::LocalModelLayout, usize, usize) {
    let rank = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 2, 1).unwrap(),
        global_rank,
    )
    .unwrap();
    let tp = rank.tensor_parallel_rank();
    let ep = rank.expert_parallel_rank();
    let partition = spec.tensor_partition(tp, 2).unwrap();
    let mut formats = BTreeMap::new();
    for unit in &spec.units {
        let UnitSpec::Decoder { feed_forward, .. } = unit else {
            continue;
        };
        let GatedProductGroupLayout::Packed { gate_up, down } =
            feed_forward.feed_forward.experts.layout()
        else {
            panic!();
        };
        for projection in [gate_up, down] {
            formats.insert(
                projection.weight().id.to_string(),
                projection.format().clone(),
            );
        }
    }
    let (global, tensor_local) = encoded_placement_descriptions(spec, &partition, &formats);
    let ctx = NumericContext::default();
    let mut model = TargetModel::<NumericBackend>::new_tensor_parallel(
        bind_spec(spec.clone()),
        partition.clone(),
        &ctx,
    )
    .unwrap();
    model
        .set_expert_realization(&partition.local_spec().expert_realization(rank).unwrap())
        .unwrap();
    let expert_local = expand_description(model.parameter_description(&ctx).unwrap(), &formats);
    let layout = partition
        .local_expert_layout(&global, &tensor_local, &expert_local, ep, 2)
        .unwrap();
    assert_eq!(
        partition.projected_layout(&global, rank).unwrap(),
        layout,
        "pure projection must preserve encoded TP companions and uneven EP ownership"
    );
    let router = layout.tensor("model.layers.0.mlp.gate.weight").unwrap();
    assert_eq!(
        router.placement(),
        &eredu_runtime::TensorPlacement::Replicated
    );
    assert!(router.additional_placements().is_empty());
    (layout, ep, tp)
}
