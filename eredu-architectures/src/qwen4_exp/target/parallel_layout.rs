//! Exact physical placement over retained tensor-rank selections.
use super::*;
use eredu_runtime::{
    ArchitectureParameterDescription, LocalModelLayout, LocalTensorLayout, TensorPlacement,
};

impl TargetTensorPartition {
    /// Projects global parameters into the independently described local module shapes.
    /// K/V sharing, packed bytes and scale companions use the retained physical ranges;
    /// this does not infer balanced sharding or reinterpret physical axes as logical heads.
    pub fn local_layout(
        &self,
        global: &ArchitectureParameterDescription,
        local: &ArchitectureParameterDescription,
    ) -> Result<LocalModelLayout, Error> {
        local_layout(&self.selections, global, local)
    }
}

impl TargetTensorPartition {
    /// Composes exact tensor placement with balanced packed-expert ownership.
    /// The independently constructed local module description validates every
    /// physical companion after both selections; K/V replication stays unchanged.
    pub fn local_expert_layout(
        &self,
        global: &ArchitectureParameterDescription,
        tensor_local: &ArchitectureParameterDescription,
        expert_local: &ArchitectureParameterDescription,
        expert_rank: usize,
        expert_ranks: usize,
    ) -> Result<LocalModelLayout, Error> {
        let tensor = self.local_layout(global, tensor_local)?;
        let topology = eredu_core::ParallelTopology::new(1, 1, expert_ranks, 1).map_err(invalid)?;
        let rank = eredu_core::ParallelRankTopology::new(topology, expert_rank).map_err(invalid)?;
        let realization = self.local_spec().expert_realization(rank)?;
        let groups = realization.local_global_group_indices();
        let first = *groups
            .first()
            .ok_or_else(|| invalid("expert owner is empty"))?;
        let range = first..first + groups.len();
        local_expert_layout(
            tensor,
            expert_local,
            self.local_spec()
                .units
                .iter()
                .filter_map(|unit| match unit {
                    UnitSpec::Decoder { feed_forward, .. } => {
                        Some(&feed_forward.feed_forward.experts)
                    }
                    _ => None,
                }),
            realization.global_expert_count(),
            range,
            expert_ranks,
        )
    }
}

impl TargetTensorPartition {
    /// Projects exact physical target coordinates without modules or native resources.
    pub fn projected_layout(
        &self,
        global: &ArchitectureParameterDescription,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<LocalModelLayout, Error> {
        if rank.tensor_parallel_rank() != self.rank() || rank.tensor_parallel_size() != self.ranks()
        {
            return Err(invalid(
                "projection topology differs from retained tensor coordinates",
            ));
        }
        let tensor = project_local_layout(&self.selections, global, None)?;
        let experts = self.local_spec().expert_realization(rank)?;
        let groups = experts.local_global_group_indices();
        let first = *groups
            .first()
            .ok_or_else(|| invalid("expert owner is empty"))?;
        project_expert_layout(
            tensor,
            None,
            self.local_spec()
                .units
                .iter()
                .filter_map(|unit| match unit {
                    UnitSpec::Decoder { feed_forward, .. } => {
                        Some(&feed_forward.feed_forward.experts)
                    }
                    _ => None,
                }),
            experts.global_expert_count(),
            first..first + groups.len(),
            rank.expert_parallel_size(),
        )
    }
}

pub(crate) fn local_layout(
    selections: &BTreeMap<String, Vec<TensorSelection>>,
    global: &ArchitectureParameterDescription,
    local: &ArchitectureParameterDescription,
) -> Result<LocalModelLayout, Error> {
    project_local_layout(selections, global, Some(local))
}

pub(crate) fn project_local_layout(
    selections: &BTreeMap<String, Vec<TensorSelection>>,
    global: &ArchitectureParameterDescription,
    local: Option<&ArchitectureParameterDescription>,
) -> Result<LocalModelLayout, Error> {
    if local.is_some_and(|local| {
        global.graph() != local.graph() || global.unit_layout() != local.unit_layout()
    }) {
        return Err(invalid(
            "global and local parameter descriptions have different execution owners",
        ));
    }
    let members = |description: &ArchitectureParameterDescription| {
        description
            .groups()
            .iter()
            .flat_map(|owned| {
                owned.group().members().iter().map(move |member| {
                    (
                        member.target().to_owned(),
                        (
                            owned.owner().clone(),
                            owned.group().logical_name().to_owned(),
                            owned.group().role(),
                            member.global_shape().to_vec(),
                        ),
                    )
                })
            })
            .collect::<BTreeMap<_, _>>()
    };
    let global = members(global);
    let local = local.map(members);
    if local
        .as_ref()
        .is_some_and(|local| global.keys().ne(local.keys()))
    {
        return Err(invalid(
            "global and local parameter descriptions have different targets",
        ));
    }
    if let Some(name) = selections.keys().find(|name| !global.contains_key(*name)) {
        return Err(invalid(format!(
            "retained selection has no described parameter {name}"
        )));
    }
    let mut layout = LocalModelLayout::default();
    for (name, (owner, group, role, global_shape)) in global {
        let expected_local = local.as_ref().map(|local| &local[&name]);
        if expected_local.is_some_and(|(local_owner, local_group, local_role, _)| {
            &owner != local_owner || &group != local_group || &role != local_role
        }) {
            return Err(invalid(format!(
                "parameter {name} changed its logical owner or role"
            )));
        }
        let mut expected_shape = global_shape.clone();
        let placement = if let Some(selections) = selections.get(&name).map(Vec::as_slice) {
            let Some(TensorSelection::Range { axis, start, end }) = selections.first() else {
                return Err(invalid(format!(
                    "parameter {name} has no retained physical range"
                )));
            };
            let axis = *axis;
            let width = global_shape
                .get(axis)
                .copied()
                .ok_or_else(|| invalid(format!("parameter {name} has an invalid physical axis")))?;
            let mut selected_width = 0usize;
            for selection in selections {
                let TensorSelection::Range {
                    axis: selected_axis,
                    start,
                    end,
                } = selection
                else {
                    return Err(invalid(format!(
                        "parameter {name} has a non-range physical selection"
                    )));
                };
                if *selected_axis != axis || start >= end || *end > width {
                    return Err(invalid(format!(
                        "parameter {name} has an invalid physical range"
                    )));
                }
                selected_width = selected_width.checked_add(end - start).ok_or_else(|| {
                    invalid(format!("parameter {name} selected width overflowed"))
                })?;
            }
            expected_shape[axis] = selected_width;
            if selections.len() == 1 {
                TensorPlacement::Range {
                    axis,
                    start: *start,
                    end: *end,
                }
            } else {
                let indices: Vec<_> = selections
                    .iter()
                    .flat_map(|selection| {
                        let TensorSelection::Range { start, end, .. } = selection else {
                            unreachable!()
                        };
                        *start..*end
                    })
                    .collect();
                if indices
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != indices.len()
                {
                    return Err(invalid(format!(
                        "parameter {name} repeats physical indices"
                    )));
                }
                TensorPlacement::Indices { axis, indices }
            }
        } else {
            TensorPlacement::Replicated
        };
        if expected_local.is_some_and(|(_, _, _, local_shape)| &expected_shape != local_shape) {
            return Err(invalid(format!(
                "parameter {name} local shape differs from retained selection {expected_shape:?}"
            )));
        }
        // Retain the exact equal-chunk coordinate domain. These units are
        // partition chunks, not physical scalar columns or attention heads:
        // encoded columns and their semantic components share the same chunks.
        let (logical_units, logical_range) = match &placement {
            TensorPlacement::Range { axis, start, end } => {
                let width = end - start;
                if global_shape[*axis].is_multiple_of(width) && start.is_multiple_of(width) {
                    (
                        Some(global_shape[*axis] / width),
                        Some(start / width..end / width),
                    )
                } else {
                    (None, None)
                }
            }
            _ => (None, None),
        };
        layout.insert(
            name,
            LocalTensorLayout::new(
                group,
                role,
                global_shape,
                expected_shape,
                placement,
                logical_units,
                logical_range,
                false,
            ),
        );
    }
    Ok(layout)
}

pub(crate) fn local_expert_layout<'a>(
    tensor: LocalModelLayout,
    expert_local: &ArchitectureParameterDescription,
    expert_specs: impl Iterator<Item = &'a GroupedGatedProductSpec>,
    global_experts: usize,
    range: Range<usize>,
    expert_ranks: usize,
) -> Result<LocalModelLayout, Error> {
    project_expert_layout(
        tensor,
        Some(expert_local),
        expert_specs,
        global_experts,
        range,
        expert_ranks,
    )
}

fn project_expert_layout<'a>(
    tensor: LocalModelLayout,
    expert_local: Option<&ArchitectureParameterDescription>,
    expert_specs: impl Iterator<Item = &'a GroupedGatedProductSpec>,
    global_experts: usize,
    range: Range<usize>,
    expert_ranks: usize,
) -> Result<LocalModelLayout, Error> {
    let mut experts = std::collections::BTreeSet::new();
    for spec in expert_specs {
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
            if expert_ranks == 1 {
                continue;
            }
            return Err(invalid(
                "expert placement requires canonical packed parameters",
            ));
        };
        for projection in [gate_up, down] {
            for parameter in std::iter::once(projection.weight())
                .chain(projection.bias())
                .chain(projection.format().scale())
                .chain(projection.format().affine_bias())
            {
                if !tensor.contains(parameter.id.as_str()) {
                    return Err(invalid(format!(
                        "expert parameter {} is not described",
                        parameter.id
                    )));
                }
                experts.insert(parameter.id.as_str().to_owned());
            }
        }
    }
    let local: Option<BTreeMap<_, _>> = expert_local.map(|expert_local| {
        expert_local
            .groups()
            .iter()
            .flat_map(|group| {
                group
                    .members()
                    .iter()
                    .map(move |member| (member.target(), (group, member)))
            })
            .collect()
    });
    if local
        .as_ref()
        .is_some_and(|local| local.len() != tensor.len())
    {
        return Err(invalid(
            "expert placement changed the canonical parameter identities",
        ));
    }
    let mut layout = LocalModelLayout::default();
    for (name, tp) in tensor.tensors() {
        let member = if let Some(local) = &local {
            let (group, member) = local
                .get(name)
                .ok_or_else(|| invalid("local expert parameter absent"))?;
            if group.group().logical_name() != tp.logical_name()
                || group.group().role() != tp.role()
            {
                return Err(invalid(format!(
                    "expert placement changed the owner of {name}"
                )));
            }
            Some(*member)
        } else {
            None
        };
        let mut shape = tp.local_shape().to_vec();
        let ep_range = if expert_ranks > 1 && experts.contains(name) {
            if tp.global_shape().first().copied() != Some(global_experts) {
                return Err(invalid(format!(
                    "expert parameter {name} has an invalid group axis"
                )));
            }
            Some(range.clone())
        } else {
            None
        };
        if let Some(range) = &ep_range {
            shape[0] = range.len();
        }
        if member.is_some_and(|member| shape != member.global_shape()) {
            return Err(invalid(format!(
                "parameter {name} expert-local shape differs from retained selection {shape:?}"
            )));
        }
        let mut placed = LocalTensorLayout::new(
            tp.logical_name(),
            tp.role(),
            tp.global_shape().to_vec(),
            shape,
            tp.placement().clone(),
            tp.logical_units(),
            tp.logical_range().cloned(),
            false,
        );
        if let Some(range) = ep_range {
            placed = placed.with_additional_placement(TensorPlacement::Range {
                axis: 0,
                start: range.start,
                end: range.end,
            });
        }
        layout.insert(name.to_owned(), placed);
    }
    Ok(layout)
}

/// The runtime traverses lexical injections and decoder layers as distinct units.
/// Join only already-declared boundaries to those ordinals, using the same paths
/// as actual traversal, rather than interpreting checkpoint layer numbers as units.
pub(crate) fn partition_unit_observations(
    spec: &TargetSpec,
    descriptor: &eredu_core::ArchitectureDescriptor,
) -> Result<Vec<crate::partitioned_execution::PartitionedUnitObservation>, String> {
    use eredu_core::{SymbolicDimension as D, TensorAxis, UnitObservation};
    let expected = vec![
        TensorAxis {
            name: "batch".into(),
            dimension: D::Batch,
        },
        TensorAxis {
            name: "sequence".into(),
            dimension: D::Sequence,
        },
        TensorAxis {
            name: "stream".into(),
            dimension: D::Known(spec.boundary.geometry.streams() as usize),
        },
        TensorAxis {
            name: "hidden".into(),
            dimension: D::Known(spec.boundary.geometry.hidden_size() as usize),
        },
    ];
    let group = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut points = Vec::new();
    for (unit, specification) in spec.units.iter().enumerate() {
        for seam in [UnitObservation::Input, UnitObservation::Output] {
            let path = seam.path(&specification.path());
            let Some(point) = descriptor.observations.get(&path) else {
                continue;
            };
            if point.axes.as_ref() != Some(&expected)
                || point.value_type != eredu_core::ObservationValueType::Tensor
                || descriptor.node(&point.node_id).is_none()
            {
                return Err(format!(
                    "unit boundary {path} differs from retained residual geometry"
                ));
            }
            points.push(crate::partitioned_execution::PartitionedUnitObservation {
                group: group.clone(),
                unit,
                path,
            });
        }
    }
    Ok(points)
}
