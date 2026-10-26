//! Exact physical parameter placement from retained family-authored selections.
use std::collections::BTreeMap;

use eredu_checkpoint::store::TensorSelection;

use crate::{
    ArchitectureParameterDescription, LocalModelLayout, LocalTensorLayout, OwnedParameterGroupSpec,
    ParameterMemberSpec, TensorPlacement,
};

/// A retained physical selection disagrees with its declared parameter resources.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParameterSelectionLayoutError {
    /// Global and local declarations describe different invocation graphs.
    #[error("global and local parameter descriptions have different execution owners")]
    ExecutionOwners,
    /// The independent local declaration changes the set of parameter targets.
    #[error("global and local parameter descriptions have different targets")]
    Targets,
    /// A selection names a target absent from the canonical declaration.
    #[error("retained selection has no described parameter {0}")]
    UnknownTarget(String),
    /// A local target changes its canonical owner, group or semantic role.
    #[error("parameter {0} changed its logical owner or role")]
    ChangedOwner(String),
    /// A selected parameter has no first contiguous physical range.
    #[error("parameter {0} has no retained physical range")]
    MissingRange(String),
    /// The physical axis is absent from the declared tensor geometry.
    #[error("parameter {0} has an invalid physical axis")]
    InvalidAxis(String),
    /// A selection mixes physical ranges with another selection representation.
    #[error("parameter {0} has a non-range physical selection")]
    NonRange(String),
    /// Physical ranges split axes, are empty or exceed the declared shape.
    #[error("parameter {0} has an invalid physical range")]
    InvalidRange(String),
    /// The complete selected physical width cannot be represented.
    #[error("parameter {0} selected width overflowed")]
    WidthOverflow(String),
    /// Fused physical ranges select the same scalar coordinate twice.
    #[error("parameter {0} repeats physical indices")]
    RepeatedIndices(String),
    /// Retained selection and independent local construction have different shapes.
    #[error("parameter {target} local shape differs from retained selection {expected:?}")]
    LocalShape {
        /// Canonical physical parameter target.
        target: String,
        /// Shape produced by the retained physical ranges.
        expected: Vec<usize>,
    },
}

impl From<ParameterSelectionLayoutError> for eredu_nn::Error {
    fn from(error: ParameterSelectionLayoutError) -> Self {
        Self::backend_source(error)
    }
}

/// Projects retained physical ranges without choosing a partition or applying transforms.
/// An optional independent local declaration validates exact targets, owners and shapes.
/// Fused segments retain their declared order, and equal physical chunks retain their
/// coordinate domain independently of semantic heads, packed columns or scalar indices.
pub fn project_parameter_selection_layout(
    selections: &BTreeMap<String, Vec<TensorSelection>>,
    global: &ArchitectureParameterDescription,
    local: Option<&ArchitectureParameterDescription>,
) -> Result<LocalModelLayout, ParameterSelectionLayoutError> {
    if local.is_some_and(|local| {
        global.graph() != local.graph() || global.unit_layout() != local.unit_layout()
    }) {
        return Err(ParameterSelectionLayoutError::ExecutionOwners);
    }
    let global = member_index(global);
    let local = local.map(member_index);
    if local
        .as_ref()
        .is_some_and(|local| global.keys().ne(local.keys()))
    {
        return Err(ParameterSelectionLayoutError::Targets);
    }
    if let Some(name) = selections
        .keys()
        .find(|name| !global.contains_key(name.as_str()))
    {
        return Err(ParameterSelectionLayoutError::UnknownTarget(
            name.to_owned(),
        ));
    }
    let mut layout = LocalModelLayout::default();
    for (name, (group, member)) in global {
        let global_shape = member.global_shape();
        let expected_local = local.as_ref().map(|local| &local[name]);
        if expected_local.is_some_and(|(local_group, _)| {
            group.owner() != local_group.owner()
                || group.group().logical_name() != local_group.group().logical_name()
                || group.group().role() != local_group.group().role()
        }) {
            return Err(ParameterSelectionLayoutError::ChangedOwner(name.to_owned()));
        }
        let mut expected_shape = global_shape.to_vec();
        let placement = if let Some(selections) = selections.get(name).map(Vec::as_slice) {
            let Some(TensorSelection::Range { axis, start, end }) = selections.first() else {
                return Err(ParameterSelectionLayoutError::MissingRange(name.to_owned()));
            };
            let axis = *axis;
            let width = global_shape
                .get(axis)
                .copied()
                .ok_or_else(|| ParameterSelectionLayoutError::InvalidAxis(name.to_owned()))?;
            let mut selected_width = 0usize;
            for selection in selections {
                let TensorSelection::Range {
                    axis: selected_axis,
                    start,
                    end,
                } = selection
                else {
                    return Err(ParameterSelectionLayoutError::NonRange(name.to_owned()));
                };
                if *selected_axis != axis || start >= end || *end > width {
                    return Err(ParameterSelectionLayoutError::InvalidRange(name.to_owned()));
                }
                selected_width = selected_width
                    .checked_add(end - start)
                    .ok_or_else(|| ParameterSelectionLayoutError::WidthOverflow(name.to_owned()))?;
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
                    return Err(ParameterSelectionLayoutError::RepeatedIndices(
                        name.to_owned(),
                    ));
                }
                TensorPlacement::Indices { axis, indices }
            }
        } else {
            TensorPlacement::Replicated
        };
        if expected_local.is_some_and(|(_, member)| expected_shape != member.global_shape()) {
            return Err(ParameterSelectionLayoutError::LocalShape {
                target: name.to_owned(),
                expected: expected_shape,
            });
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
            name.to_owned(),
            LocalTensorLayout::new(
                group.group().logical_name(),
                group.group().role(),
                global_shape.to_vec(),
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

fn member_index(
    description: &ArchitectureParameterDescription,
) -> BTreeMap<&str, (&OwnedParameterGroupSpec, &ParameterMemberSpec)> {
    description
        .groups()
        .iter()
        .flat_map(|group| {
            group
                .members()
                .iter()
                .map(move |member| (member.target(), (group, member)))
        })
        .collect()
}
