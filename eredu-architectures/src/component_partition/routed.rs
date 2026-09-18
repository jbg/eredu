//! Exact routed coordinates from retained bank ownership and physical write layout.
use super::*;
use eredu_core::component::{
    RoutedComponentCoordinateMap, RoutedComponentGroup, RoutedComponentParameter,
};

/// Derives one selected bank's global expert and scalar-unit coordinates. The
/// grouped realization supplies execution ownership; storing an independent
/// expert matrix does not establish that this rank executes that expert.
/// Invocation ownership, capture admission and distributed receipts stay separate.
pub fn derive_routed_component_coordinates(
    component: &RoutedComponentGroup,
    layout: &LocalModelLayout,
    plan: &crate::routed_text::RoutedGroupedPlan,
) -> Result<RoutedComponentCoordinateMap, ComponentPartitionError> {
    let invalid = || ComponentPartitionError::InvalidPlacement(component.id.clone());
    if component.expert_count != plan.global_group_count()
        || component.units_per_expert == 0
        || matches!(plan, crate::routed_text::RoutedGroupedPlan::Linear(_))
    {
        return Err(invalid());
    }
    derive_coordinates_for_experts(component, layout, plan.local_global_group_indices())
}

pub(crate) fn derive_coordinates_for_experts(
    component: &RoutedComponentGroup,
    layout: &LocalModelLayout,
    local_groups: &[usize],
) -> Result<RoutedComponentCoordinateMap, ComponentPartitionError> {
    experts_worker(component, layout, local_groups, Destination(None))
}
fn packed_coordinates(
    name: &str,
    expert_count: usize,
    units_per_expert: usize,
    output_width: usize,
    tensor: &LocalTensorLayout,
) -> Result<(ComponentCoordinateMap, ComponentCoordinateMap), ComponentPartitionError> {
    packed_worker(
        name,
        expert_count,
        units_per_expert,
        output_width,
        tensor,
        Destination(None),
    )
}
pub(in crate::component_partition) fn experts_worker(
    component: &RoutedComponentGroup,
    layout: &LocalModelLayout,
    local_groups: &[usize],
    allocation: Destination<'_>,
) -> Result<RoutedComponentCoordinateMap, ComponentPartitionError> {
    allocation.controls::<(
        &RoutedComponentGroup,
        &LocalModelLayout,
        &[usize],
        RoutedComponentCoordinateMap,
        ComponentCoordinateMap,
        Option<ComponentCoordinateMap>,
        &LocalTensorLayout,
    )>()?;
    let invalid = || allocation.invalid(&component.id);
    let experts = allocation.indices(component.expert_count, allocation.copy(local_groups)?)?;
    if experts.local_count() == 0 {
        return Ok(RoutedComponentCoordinateMap::new(
            experts,
            ComponentCoordinateMap::range(component.units_per_expert, 0..0)?,
        ));
    }
    let units = match &component.write_weight {
        RoutedComponentParameter::Packed { name } => {
            let tensor = layout
                .tensor(&name.parameter)
                .ok_or_else(|| allocation.missing(&name.parameter))?;
            let (stored_experts, units) = packed_worker(
                &component.id,
                component.expert_count,
                component.units_per_expert,
                component.output_width,
                tensor,
                allocation,
            )?;
            if local_groups
                .iter()
                .any(|expert| stored_experts.global_to_local(*expert).is_none())
            {
                return Err(invalid());
            }
            units
        }
        RoutedComponentParameter::Independent { names } => {
            if names.len() != component.expert_count {
                return Err(invalid());
            }
            let mut units = None;
            for expert in local_groups {
                let name = names[*expert].as_ref().ok_or_else(invalid)?;
                let tensor = layout
                    .tensor(&name.parameter)
                    .ok_or_else(|| allocation.missing(&name.parameter))?;
                if tensor.global_shape().first() != Some(&component.output_width) {
                    return Err(invalid());
                }
                let next = coordinates::write_worker(
                    &name.parameter,
                    component.units_per_expert,
                    tensor,
                    allocation,
                )?;
                if units.as_ref().is_some_and(|units| *units != next) {
                    return Err(invalid());
                }
                units = Some(next);
            }
            units.ok_or_else(invalid)?
        }
    };
    Ok(RoutedComponentCoordinateMap::new(experts, units))
}

fn packed_worker(
    name: &str,
    expert_count: usize,
    units_per_expert: usize,
    output_width: usize,
    tensor: &LocalTensorLayout,
    allocation: Destination<'_>,
) -> Result<(ComponentCoordinateMap, ComponentCoordinateMap), ComponentPartitionError> {
    allocation.controls::<(
        &str,
        usize,
        usize,
        usize,
        &LocalTensorLayout,
        Option<ComponentCoordinateMap>,
        Option<TensorPlacement>,
        LocalTensorLayout,
        ComponentCoordinateMap,
        ComponentCoordinateMap,
    )>()?;
    let invalid = || allocation.invalid(name);
    let global = tensor.global_shape();
    let local = tensor.local_shape();
    if global.len() != 3
        || local.len() != 3
        || global.contains(&0)
        || local.contains(&0)
        || global[0] != expert_count
        || global[1] != output_width
        || global[1] != local[1]
    {
        return Err(invalid());
    }
    let mut expert_map = None;
    let mut unit_placement = None;
    for placement in tensor
        .additional_placements()
        .iter()
        .chain(std::iter::once(tensor.placement()))
    {
        let (axis, normalized) = match placement {
            TensorPlacement::Replicated | TensorPlacement::Local => continue,
            TensorPlacement::Range { axis, start, end } => (
                *axis,
                TensorPlacement::Range {
                    axis: 1,
                    start: *start,
                    end: *end,
                },
            ),
            TensorPlacement::Shard { axis, index, parts } => (
                *axis,
                TensorPlacement::Shard {
                    axis: 1,
                    index: *index,
                    parts: *parts,
                },
            ),
            TensorPlacement::Indices { axis, indices } => (
                *axis,
                TensorPlacement::Indices {
                    axis: 1,
                    indices: allocation.copy(indices)?,
                },
            ),
            _ => return Err(invalid()),
        };
        match axis {
            0 if expert_map.is_none() => {
                expert_map = Some(match normalized {
                    TensorPlacement::Range { start, end, .. } => {
                        ComponentCoordinateMap::range(global[0], start..end)?
                    }
                    TensorPlacement::Indices { indices, .. } => {
                        allocation.indices(global[0], indices)?
                    }
                    TensorPlacement::Shard { index, parts, .. }
                        if parts > 0 && index < parts && global[0].is_multiple_of(parts) =>
                    {
                        let width = global[0] / parts;
                        ComponentCoordinateMap::range(
                            global[0],
                            index * width..(index + 1) * width,
                        )?
                    }
                    _ => return Err(invalid()),
                });
            }
            2 if unit_placement.is_none() => unit_placement = Some(normalized),
            _ => return Err(invalid()),
        }
    }
    let experts = expert_map.unwrap_or(ComponentCoordinateMap::range(global[0], 0..global[0])?);
    if experts.local_count() != local[0] {
        return Err(invalid());
    }
    // Reuse the dense write-axis proof, including packed-column scaling and
    // retained semantic ranges. No encoding factor is guessed from dtype/names.
    let matrix = LocalTensorLayout::new(
        allocation.text(tensor.logical_name())?,
        tensor.role(),
        allocation.copy(&global[1..])?,
        allocation.copy(&local[1..])?,
        unit_placement.unwrap_or(TensorPlacement::Replicated),
        tensor.logical_units(),
        tensor.logical_range().cloned(),
        tensor.fell_back_to_replication(),
    )
    .with_partition_chunk_size(tensor.partition_chunk_size());
    let units = coordinates::write_worker(name, units_per_expert, &matrix, allocation)?;
    Ok((experts, units))
}

mod bank;
pub(crate) use bank::derive_bank_unit_coordinates;

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;
    fn fixture() -> (crate::qwen::ModelArgs, RoutedComponentGroup) {
        let config = serde_json::json!({"model_type":"qwen3_moe","hidden_size":8,"intermediate_size":0,
            "num_hidden_layers":2,"moe_intermediate_size":10,"num_experts":5,"num_experts_per_tok":2,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":2,"vocab_size":16,"rms_norm_eps":1e-5,
            "max_position_embeddings":64,"tie_word_embeddings":false});
        let args = crate::qwen::model_args_from_config_value(&config).unwrap();
        let group = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor()
            .routed_components
            .remove(0);
        (args, group)
    }
    fn plan(
        mut args: crate::qwen::ModelArgs,
        rank: usize,
        ep: usize,
        width: usize,
    ) -> crate::routed_text::RoutedGroupedPlan {
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(1, 1, ep, 1).unwrap(),
            rank,
        )
        .unwrap();
        let global = args.num_experts as usize;
        args.num_experts = eredu_core::balanced_contiguous_range(global, ep, rank, false)
            .unwrap()
            .len() as i32;
        args.moe_intermediate_size = width as i32;
        crate::routed_text::RoutedGroupedPlan::Gated(
            crate::ExpertRealizationPlan::balanced(
                global,
                topology,
                BTreeMap::from([(
                    (
                        eredu_runtime::ExecutionGroupId::new("text_decoder").unwrap(),
                        0,
                    ),
                    crate::qwen::expert_bank_spec(&args, 0).unwrap(),
                )]),
            )
            .unwrap(),
        )
    }
    #[test]
    fn packed_routed_coordinates_preserve_expert_ownership_and_semantic_unit_ranges() {
        let (args, group) = fixture();
        let RoutedComponentParameter::Packed { name } = &group.write_weight else {
            panic!()
        };
        for (rank, range) in [(0, 0..3), (1, 3..5)] {
            let realization = plan(args.clone(), rank, 2, 4);
            let tensor = LocalTensorLayout::new(
                "bank",
                eredu_runtime::ParameterRole::ExpertIntermediate,
                vec![5, 8, 5],
                vec![range.len(), 8, 2],
                TensorPlacement::Range {
                    axis: 2,
                    start: 1,
                    end: 3,
                },
                Some(5),
                Some(1..3),
                false,
            )
            .with_additional_placement(TensorPlacement::Range {
                axis: 0,
                start: range.start,
                end: range.end,
            });
            let mut layout = LocalModelLayout::default();
            layout.insert(name.parameter.clone(), tensor);
            let map = derive_routed_component_coordinates(&group, &layout, &realization).unwrap();
            assert_eq!(map.units().contiguous_range(), Some(2..6));
            assert_eq!(
                (0..map.experts().local_count())
                    .map(|n| map.experts().local_to_global(n).unwrap())
                    .collect::<Vec<_>>(),
                range.collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn routed_index_permutations_and_independent_storage_do_not_invent_expert_ownership() {
        let (args, mut group) = fixture();
        let realization = plan(args, 1, 2, 3);
        let RoutedComponentParameter::Packed { name } = group.write_weight.clone() else {
            panic!()
        };
        let tensor = LocalTensorLayout::new(
            "bank",
            eredu_runtime::ParameterRole::ExpertIntermediate,
            vec![5, 8, 10],
            vec![2, 8, 3],
            TensorPlacement::Indices {
                axis: 2,
                indices: vec![8, 1, 4],
            },
            None,
            None,
            false,
        )
        .with_additional_placement(TensorPlacement::Indices {
            axis: 0,
            indices: vec![4, 3],
        });
        let mut layout = LocalModelLayout::default();
        layout.insert(name.parameter.clone(), tensor);
        let map = derive_routed_component_coordinates(&group, &layout, &realization).unwrap();
        assert_eq!(map.local_to_global(0, 0), Some((3, 8)));
        let names: Vec<_> = (0..5)
            .map(|expert| {
                Some(eredu_core::component::RoutedComponentParameterName {
                    parameter: format!("expert.{expert}"),
                    ..name.clone()
                })
            })
            .collect();
        for name in names.iter().flatten() {
            layout.insert(
                name.parameter.clone(),
                LocalTensorLayout::new(
                    "bank",
                    eredu_runtime::ParameterRole::ExpertIntermediate,
                    vec![8, 10],
                    vec![8, 3],
                    TensorPlacement::Indices {
                        axis: 1,
                        indices: vec![8, 1, 4],
                    },
                    None,
                    None,
                    false,
                ),
            );
        }
        group.write_weight = RoutedComponentParameter::Independent { names };
        let independent =
            derive_routed_component_coordinates(&group, &layout, &realization).unwrap();
        assert_eq!(independent, map);
        assert!(
            independent.global_to_local(0, 8).is_none(),
            "stored remote expert must not become owned"
        );
    }
    #[test]
    fn routed_coordinate_derivation_rejects_ambiguous_axes_or_foreign_storage() {
        let (args, group) = fixture();
        let realization = plan(args, 1, 2, 4);
        let RoutedComponentParameter::Packed { name } = &group.write_weight else {
            panic!()
        };
        for fault in 0..4 {
            let mut tensor = LocalTensorLayout::new(
                "bank",
                eredu_runtime::ParameterRole::ExpertIntermediate,
                vec![5, 8, 5],
                vec![2, 8, 2],
                TensorPlacement::Range {
                    axis: 2,
                    start: 1,
                    end: 3,
                },
                Some(5),
                Some(if fault == 0 { 0..2 } else { 1..3 }),
                false,
            )
            .with_additional_placement(TensorPlacement::Range {
                axis: 0,
                start: if fault == 1 { 0 } else { 3 },
                end: if fault == 1 { 2 } else { 5 },
            });
            if fault == 2 {
                tensor = tensor.with_additional_placement(TensorPlacement::Range {
                    axis: 2,
                    start: 1,
                    end: 3,
                });
            }
            if fault == 3 {
                tensor = tensor.with_additional_placement(TensorPlacement::Range {
                    axis: 1,
                    start: 0,
                    end: 8,
                });
            }
            let mut layout = LocalModelLayout::default();
            layout.insert(name.parameter.clone(), tensor);
            assert!(derive_routed_component_coordinates(&group, &layout, &realization).is_err());
        }
    }
}

pub(super) mod placement;
pub use placement::PartitionedRoutedObservation;
