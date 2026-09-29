//! EP narrows parameter ownership while global routing and local TP arithmetic survive.
use super::*;
use eredu_nn::{GatedProductGroupLayout, GatedProductGroupParameters, GroupedProjectionSpec};

#[test]
fn qwen4_expert_realization_preserves_router_and_uneven_owner_parameters() {
    for independent in [false, true] {
        let mut config = configuration();
        config.attention.heads = 4;
        config.attention.kv_heads = 2;
        config.recurrent.key_heads = 4;
        config.recurrent.value_heads = 8;
        config.experts.intermediate = 4;
        let mut spec = specification_for(config);
        if independent {
            for unit in &mut spec.units {
                let UnitSpec::Decoder {
                    layer,
                    feed_forward,
                    ..
                } = unit
                else {
                    continue;
                };
                let expert = &feed_forward.feed_forward.experts;
                let groups = (0..expert.group_count())
                    .map(|group| {
                        let projection = |part| {
                            GroupedProjectionSpec::new(
                                ParameterSpec::trainable(format!(
                                    "model.layers.{layer}.mlp.experts.{group}.{part}"
                                ))
                                .unwrap(),
                                None,
                                dense_linear_format(),
                            )
                            .unwrap()
                        };
                        GatedProductGroupParameters::new(
                            projection("gate"),
                            projection("up"),
                            projection("down"),
                        )
                    })
                    .collect();
                feed_forward.feed_forward.experts = GroupedGatedProductSpec::new(
                    expert.group_count(),
                    expert.input_dimensions(),
                    expert.intermediate_dimensions(),
                    expert.output_dimensions(),
                    expert.policy(),
                    GatedProductGroupLayout::Independent(groups),
                )
                .unwrap()
                .with_reduction(expert.reduction());
            }
        }
        let topology = eredu_core::ParallelTopology::new(2, 1, 2, 1).unwrap();
        for rank in 0..4 {
            let topology = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
            let tensor = spec
                .tensor_partition(topology.tensor_parallel_rank(), 2)
                .unwrap();
            let realization = tensor.local_spec().expert_realization(topology).unwrap();
            let expected = if topology.expert_parallel_rank() == 0 {
                vec![0, 1]
            } else {
                vec![2]
            };
            assert_eq!(realization.local_global_group_indices(), expected);
            let ctx = NumericContext::default();
            let mut model = TargetModel::<NumericBackend>::new_tensor_parallel(
                bind_spec(spec.clone()),
                tensor.clone(),
                &ctx,
            )
            .unwrap();
            let tp_parameters = model.parameter_description(&ctx).unwrap();
            let original_state = model.state_layout().unwrap();
            model.set_expert_realization(&realization).unwrap();
            assert_eq!(model.state_layout().unwrap(), original_state);
            if !independent {
                use eredu_runtime::TensorPlacement;
                let global = TargetModel::<NumericBackend>::new(bind_spec(spec.clone()), &ctx)
                    .unwrap()
                    .parameter_description(&ctx)
                    .unwrap();
                let local = model.parameter_description(&ctx).unwrap();
                let layout = tensor
                    .local_expert_layout(
                        &global,
                        &tp_parameters,
                        &local,
                        topology.expert_parallel_rank(),
                        2,
                    )
                    .unwrap();
                let begin = expected[0];
                let end = begin + expected.len();
                let gate = layout
                    .tensor("model.layers.0.mlp.experts.gate_up_proj")
                    .unwrap();
                assert_eq!(gate.local_shape(), [expected.len(), 4, 2]);
                assert_eq!(
                    gate.additional_placements(),
                    [TensorPlacement::Range {
                        axis: 0,
                        start: begin,
                        end
                    }]
                );
                let tp = topology.tensor_parallel_rank();
                assert_eq!(
                    gate.placement(),
                    &TensorPlacement::Indices {
                        axis: 1,
                        indices: vec![2 * tp, 2 * tp + 1, 4 + 2 * tp, 5 + 2 * tp]
                    }
                );
                let down = layout
                    .tensor("model.layers.0.mlp.experts.down_proj")
                    .unwrap();
                assert_eq!(down.local_shape(), [expected.len(), 2, 2]);
                assert_eq!(
                    down.placement(),
                    &TensorPlacement::Range {
                        axis: 2,
                        start: 2 * tp,
                        end: 2 * tp + 2
                    }
                );
                assert_eq!(down.additional_placements(), gate.additional_placements());
                let router = layout.tensor("model.layers.0.mlp.gate.weight").unwrap();
                assert_eq!(router.placement(), &TensorPlacement::Replicated);
                assert!(router.additional_placements().is_empty());
            }
            for (ordinal, unit) in tensor.local_spec().units.iter().enumerate() {
                let UnitSpec::Decoder {
                    layer,
                    feed_forward,
                    ..
                } = unit
                else {
                    continue;
                };
                let local = realization
                    .unit_spec(
                        eredu_architectures::decoder::TARGET_EXECUTION_GROUP,
                        ordinal,
                    )
                    .unwrap();
                assert_eq!(local.group_count() as usize, expected.len());
                assert_eq!(local.intermediate_dimensions(), 2);
                assert_eq!(
                    local.reduction(),
                    eredu_nn::GroupReduction::SequentialGroupOrder
                );
                let mut parameters = Parameters::default();
                model
                    .construct_unit(ordinal, &ctx)
                    .unwrap()
                    .visit_parameters_mut(&mut parameters);
                assert_eq!(
                    parameters.0[&format!("model.layers.{layer}.mlp.gate.weight")].shape(),
                    [3, 2]
                );
                match local.layout() {
                    GatedProductGroupLayout::Independent(groups) => {
                        for (parameters, global) in groups.iter().zip(&expected) {
                            assert_eq!(
                                parameters.gate().weight().id.as_str(),
                                format!("model.layers.{layer}.mlp.experts.{global}.gate")
                            );
                        }
                    }
                    GatedProductGroupLayout::Packed { gate_up, .. } => {
                        assert_eq!(
                            parameters.0[gate_up.weight().id.as_str()].shape(),
                            [expected.len() as i32, 4, 2]
                        );
                    }
                    _ => panic!("unexpected expert layout"),
                }
                assert_eq!(
                    feed_forward.feed_forward.router.selection().group_count(),
                    3
                );
            }
            assert!(model.set_expert_realization(&realization).is_err());
        }
    }
}

#[test]
fn qwen4_expert_realization_rejects_equation_drift_without_mutating_model() {
    let spec = specification();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 2, 1).unwrap(),
        0,
    )
    .unwrap();
    let expected = spec.expert_realization(topology).unwrap();
    let altered = expected
        .unit_specs()
        .iter()
        .map(|(address, spec)| {
            (
                address.clone(),
                spec.clone().with_reduction(eredu_nn::GroupReduction::Sum),
            )
        })
        .collect();
    let altered =
        eredu_architectures::ExpertRealizationPlan::balanced(3, topology, altered).unwrap();
    let ctx = NumericContext::default();
    let mut model = TargetModel::<NumericBackend>::new(bind_spec(spec), &ctx).unwrap();
    assert!(model.set_expert_realization(&altered).is_err());
    model.set_expert_realization(&expected).unwrap();
}

#[test]
fn qwen4_expert_waves_declare_lexical_status_rows_and_replicated_vocabulary() {
    use eredu_architectures::partitioned_execution::{
        RoutedTensorDimension as Dimension, RoutedTensorReduction as Reduction,
        RoutedTensorReductions, TextPartitionArchitecture,
    };
    let spec = specification();
    let ctx = NumericContext::default();
    let model = TargetModel::<NumericBackend>::new(bind_spec(spec.clone()), &ctx).unwrap();
    assert!(!<TargetModel<NumericBackend> as TextPartitionArchitecture<NumericBackend, State>>::partition_tensor_vocabulary_sharded(&model));
    for (ordinal, unit) in spec.units.iter().enumerate() {
        let routed = matches!(unit, UnitSpec::Decoder { .. });
        let actual = <TargetModel<NumericBackend> as TextPartitionArchitecture<
            NumericBackend,
            State,
        >>::partition_routed_tensor_reductions(&model, ordinal, routed)
        .unwrap();
        match unit {
            UnitSpec::Decoder { .. } => assert_eq!(actual, RoutedTensorReductions::hidden(1, 1)),
            UnitSpec::Lexical { spec, .. } => {
                let table = spec.embedding.lookup_spec();
                assert_eq!(
                    actual,
                    RoutedTensorReductions {
                        before: vec![
                            Reduction::Status,
                            Reduction::Tensor {
                                shape: vec![Dimension::Tokens(2), Dimension::Fixed(2)],
                                dtype: eredu_nn::TensorElementType::F32,
                            },
                        ],
                        after: vec![],
                    }
                );
                assert_eq!(table.dimensions, 2);
                assert!(<TargetModel<NumericBackend> as TextPartitionArchitecture<
                    NumericBackend,
                    State,
                >>::partition_routed_tensor_reductions(
                    &model, ordinal, true
                )
                .is_err());
            }
        }
    }
}
