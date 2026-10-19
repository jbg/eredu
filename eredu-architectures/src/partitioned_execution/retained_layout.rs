//! Checks the exact layout retained by architecture preparation without replanning it.
use eredu_runtime::{ArchitectureParameterDescription, LocalModelLayout, TensorPlacement};

pub(super) fn validate(
    parameters: &ArchitectureParameterDescription,
    layout: &LocalModelLayout,
) -> Result<(), String> {
    let mut count = 0;
    for group in parameters.groups() {
        for member in group.members() {
            count += 1;
            let name = member.target();
            let tensor = layout
                .tensor(name)
                .ok_or_else(|| format!("retained layout has no parameter {name}"))?;
            if tensor.global_shape() != member.global_shape()
                || tensor.logical_name() != group.group().logical_name()
                || tensor.role() != group.group().role()
            {
                return Err(format!(
                    "retained layout changed the global declaration of {name}"
                ));
            }
            let mut shape = tensor.global_shape().to_vec();
            for placement in tensor
                .additional_placements()
                .iter()
                .chain(std::iter::once(tensor.placement()))
            {
                apply(&mut shape, placement)
                    .map_err(|error| format!("retained layout for {name}: {error}"))?;
            }
            if shape != tensor.local_shape() {
                return Err(format!(
                    "retained layout for {name} produces shape {shape:?}, expected {:?}",
                    tensor.local_shape()
                ));
            }
        }
    }
    if count != layout.len() {
        return Err("retained layout contains undeclared parameters".into());
    }
    Ok(())
}

fn apply(shape: &mut [usize], placement: &TensorPlacement) -> Result<(), String> {
    let (axis, width) = match placement {
        TensorPlacement::Replicated | TensorPlacement::Local | TensorPlacement::Rank { .. } => {
            return Ok(());
        }
        TensorPlacement::Omit => {
            return Err("a complete parameter layout cannot omit a tensor".into());
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
                return Err("invalid equal shard".into());
            }
            (*axis, extent / parts)
        }
        TensorPlacement::Range { axis, start, end } => {
            let extent = shape
                .get(*axis)
                .copied()
                .ok_or("range axis is outside the source")?;
            if start >= end || *end > extent {
                return Err("invalid physical range".into());
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
                return Err("invalid physical indices".into());
            }
            (*axis, indices.len())
        }
    };
    shape[axis] = width;
    Ok(())
}
