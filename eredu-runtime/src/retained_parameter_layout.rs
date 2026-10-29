//! Checks the exact layout retained by architecture preparation without replanning it.
use crate::{ArchitectureParameterDescription, LocalModelLayout, TensorPlacement};

/// Invalid retained physical geometry, checked without replanning rank ownership.
#[derive(Debug, thiserror::Error)]
pub enum RetainedParameterLayoutError {
    /// A declared canonical target has no retained tensor.
    #[error("retained layout has no parameter {0}")]
    MissingParameter(String),
    /// Global shape, semantic role or parameter group differs from the declaration.
    #[error("retained layout changed the global declaration of {0}")]
    Declaration(String),
    /// A physical selection is invalid for the preceding shape.
    #[error("retained layout for {target}: {reason}")]
    Placement {
        /// Canonical parameter target.
        target: String,
        /// Rejected selection constraint.
        reason: &'static str,
    },
    /// The ordered placements do not produce the retained local shape.
    #[error("retained layout for {target} produces shape {actual:?}, expected {expected:?}")]
    Shape {
        /// Canonical parameter target.
        target: String,
        /// Shape produced by the ordered placements.
        actual: Vec<usize>,
        /// Retained materialized shape.
        expected: Vec<usize>,
    },
    /// The retained catalog includes targets outside the authoritative declaration.
    #[error("retained layout contains undeclared parameters")]
    UndeclaredParameters,
}

/// Checks a complete retained layout against its canonical parameter declaration.
/// Additional placements precede the primary placement; their exact local shape
/// must agree. This validates physical consistency without rebalancing placement,
/// interpreting family geometry or acquiring checkpoint/native resources.
pub fn validate_retained_parameter_layout(
    parameters: &ArchitectureParameterDescription,
    layout: &LocalModelLayout,
) -> Result<(), RetainedParameterLayoutError> {
    let mut count = 0;
    for group in parameters.groups() {
        for member in group.members() {
            count += 1;
            let name = member.target();
            let tensor = layout
                .tensor(name)
                .ok_or_else(|| RetainedParameterLayoutError::MissingParameter(name.into()))?;
            if tensor.global_shape() != member.global_shape()
                || tensor.logical_name() != group.group().logical_name()
                || tensor.role() != group.group().role()
            {
                return Err(RetainedParameterLayoutError::Declaration(name.into()));
            }
            let mut shape = tensor.global_shape().to_vec();
            for placement in tensor
                .additional_placements()
                .iter()
                .chain(std::iter::once(tensor.placement()))
            {
                apply(&mut shape, placement).map_err(|reason| {
                    RetainedParameterLayoutError::Placement {
                        target: name.into(),
                        reason,
                    }
                })?;
            }
            if shape != tensor.local_shape() {
                return Err(RetainedParameterLayoutError::Shape {
                    target: name.into(),
                    actual: shape,
                    expected: tensor.local_shape().to_vec(),
                });
            }
        }
    }
    if count != layout.len() {
        return Err(RetainedParameterLayoutError::UndeclaredParameters);
    }
    Ok(())
}

fn apply(shape: &mut [usize], placement: &TensorPlacement) -> Result<(), &'static str> {
    let (axis, width) = match placement {
        TensorPlacement::Replicated | TensorPlacement::Local | TensorPlacement::Rank { .. } => {
            return Ok(());
        }
        TensorPlacement::Omit => {
            return Err("a complete parameter layout cannot omit a tensor");
        }
        TensorPlacement::Shard { axis, index, parts } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("shard axis is outside the source")?;
            if *parts == 0
                || index >= parts
                || !extent.is_multiple_of(*parts)
                || extent / parts == 0
            {
                return Err("invalid equal shard");
            }
            (*axis, extent / parts)
        }
        TensorPlacement::Range { axis, start, end } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("range axis is outside the source")?;
            if start >= end || *end > extent {
                return Err("invalid physical range");
            }
            (*axis, end - start)
        }
        TensorPlacement::Indices { axis, indices } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("index axis is outside the source")?;
            if indices.is_empty()
                || indices.iter().any(|index| *index >= extent)
                || indices
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != indices.len()
            {
                return Err("invalid physical indices");
            }
            (*axis, indices.len())
        }
    };
    shape[axis] = width;
    Ok(())
}
