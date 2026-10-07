//! Pinned HF equations through rank-local resident and bounded transformer units.
use super::voicechat_transformer::{geometry, CausalCache};
use super::voicechat_tts_generator::{compare, compare_cache, tensor};
use super::*;
use eredu_architectures::voicechat::tts_transformer::TransformerConfig;
use eredu_architectures::voicechat::tts_transformer_layered::{
    tts_transformer_parameter_description, LayerwiseTransformer, TransformerArchitecture,
    TransformerUnit,
};
use serde_json::Value;

type Stack =
    LayerwiseTransformer<NumericBackend, NumericReplicatedPolicy<TransformerUnit<NumericBackend>>>;

fn groups(config: &TransformerConfig) -> Vec<ParameterGroupSpec> {
    tts_transformer_parameter_description(config)
        .unwrap()
        .groups()
        .iter()
        .map(|owned| owned.group().clone())
        .collect()
}

// The tiny oracle has Q2/KV1, while the released stacks have Q16/KV16.
// An explicitly retained physical layout exercises KV replication without
// claiming that the current KV-group planner selects it automatically.
fn replicated_kv_layout(groups: &[ParameterGroupSpec], rank: usize) -> LocalModelLayout {
    let mut layout = numeric_local_layout(groups, 2, rank).unwrap();
    for group in groups {
        if group.role() != ParameterRole::AttentionHeads {
            continue;
        }
        for member in group.members() {
            let name = member.target();
            let axis = if name.ends_with(".q_proj.weight") {
                Some(0)
            } else if name.ends_with(".o_proj.weight") {
                Some(1)
            } else {
                None
            };
            let mut local = member.global_shape().to_vec();
            let placement = if let Some(axis) = axis {
                assert_eq!(local[axis], 8);
                local[axis] = 4;
                TensorPlacement::Range {
                    axis,
                    start: rank * 4,
                    end: (rank + 1) * 4,
                }
            } else {
                TensorPlacement::Replicated
            };
            layout.insert(
                name.to_owned(),
                LocalTensorLayout::new(
                    group.logical_name(),
                    group.role(),
                    member.global_shape().to_vec(),
                    local,
                    placement,
                    axis.map(|_| 2),
                    axis.map(|_| rank..rank + 1),
                    false,
                ),
            );
        }
    }
    layout
}

fn local_weights(
    parameters: &Value,
    layout: &LocalModelLayout,
    duplicate_kv: bool,
) -> BTreeMap<String, NumericTensor> {
    parameters
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let mut value = tensor(value);
            if duplicate_kv
                && (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight"))
            {
                value = NumericTensor::concatenate(
                    &[value.clone(), value],
                    0,
                    &NumericContext::default(),
                )
                .unwrap();
            }
            let selected = layout.tensor(name).unwrap();
            assert_eq!(
                value
                    .shape()
                    .iter()
                    .map(|&n| n as usize)
                    .collect::<Vec<_>>(),
                selected.global_shape(),
                "{name} checkpoint shape"
            );
            (
                name.clone(),
                select_parameter(&value, selected.placement()).unwrap(),
            )
        })
        .collect()
}

fn stack(
    config: TransformerConfig,
    layout: &LocalModelLayout,
    values: &BTreeMap<String, NumericTensor>,
    resident: bool,
    context: &NumericContext,
) -> Stack {
    let architecture =
        TransformerArchitecture::<NumericBackend>::new_parallel(config, layout).unwrap();
    let policy = if resident {
        let mut units = Vec::new();
        for group in 0..2 {
            for index in 0..architecture.group_unit_count(group).unwrap() {
                let mut unit = architecture.build_unit(group, index, context).unwrap();
                struct Bind<'a>(&'a BTreeMap<String, NumericTensor>);
                impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
                    fn visit_mut(
                        &mut self,
                        metadata: ParameterMetadata,
                        target: &'a mut NumericTensor,
                    ) {
                        let value = &self.0[metadata.id.as_str()];
                        assert_eq!(target.shape(), value.shape(), "{}", metadata.id.as_str());
                        *target = value.clone();
                    }
                }
                unit.visit_parameters_mut(&mut Bind(values));
                units.push(unit);
            }
        }
        NumericReplicatedPolicy::resident(units)
    } else {
        let mut policy = NumericReplicatedPolicy::bounded();
        policy.parameter_overrides = values.clone();
        policy
    };
    LayerwiseTransformer::new(architecture, policy)
}

fn tp2_fixture(causal: bool) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let data = &fixture[if causal { "gemma3" } else { "t5gemma" }];
    let dimensions = geometry(
        if causal {
            &data["config"]
        } else {
            &data["config"]["encoder"]
        },
        causal,
    );
    for duplicate_kv in [false, true] {
        let mut dimensions = dimensions.clone();
        if duplicate_kv {
            // Identical KV projections leave the oracle equation unchanged and
            // exercise ordinary symbolic head-group partitioning at TP2.
            dimensions.num_key_value_heads = 2;
        }
        let config = TransformerConfig::new(&dimensions).unwrap();
        let declarations = groups(&config);
        for resident in [true, false] {
            let group = NumericParallelGroup::new(2);
            std::thread::scope(|scope| {
                let handles = (0..2)
                    .map(|rank| {
                        let layout = if duplicate_kv {
                            numeric_local_layout(&declarations, 2, rank).unwrap()
                        } else {
                            replicated_kv_layout(&declarations, rank)
                        };
                        let config = config.clone();
                        let group = Arc::clone(&group);
                        let fixture = &fixture;
                        scope.spawn(move || {
                            let context = NumericContext::with_local_layout(layout.clone());
                            let parallel = NumericParallelContext::new(rank, group);
                            let values = local_weights(&data["parameters"], &layout, duplicate_kv);
                            let mut model = stack(config, &layout, &values, resident, &context);
                            if causal {
                                let mut caches = [
                                    CausalCache(NumericCache::new(Some(3))),
                                    CausalCache(NumericCache::new(None)),
                                ];
                                for step in data["chunks"].as_array().unwrap() {
                                    let output = model
                                        .forward_causal_parallel(
                                            &tensor(&step["input"]),
                                            &mut caches,
                                            &parallel,
                                            &context,
                                        )
                                        .unwrap();
                                    compare(&output, &step["output"], fixture);
                                    compare_cache(&caches, &step["cache"], fixture);
                                }
                            } else {
                                let mask = tensor(&data["attention_mask"])
                                    .equal_i32(1, &context)
                                    .unwrap();
                                for batch in data["batches"].as_array().unwrap() {
                                    let output = model
                                        .forward_encoder_parallel(
                                            &tensor(&batch["input"]),
                                            &mask,
                                            &parallel,
                                            &context,
                                        )
                                        .unwrap();
                                    compare(&output, &batch["output"], fixture);
                                }
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        }
    }
}

#[test]
fn voicechat_transformer_tp2_resident_and_bounded_cached_outputs_match_pinned_hf() {
    tp2_fixture(true);
}

#[test]
fn voicechat_transformer_tp2_resident_and_bounded_character_masks_match_pinned_hf() {
    tp2_fixture(false);
}
