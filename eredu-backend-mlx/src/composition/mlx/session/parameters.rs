//! Native loaded-slot operations. Architecture names and geometry come from traversal.
use super::*;
use eredu_core::{capture::*, intervention::InterventionDtype, parameters::*};

use safemlx::ops::indexing::{ArrayIndex, ArrayIndexOp, TryIndexMutOp};
use std::collections::BTreeMap;
#[path = "parameters/encoding.rs"]
mod encoding;
#[path = "parameters/partition.rs"]
pub(super) mod partition;
use eredu_core::parameters::native::EffectiveParameterLayout;

fn dtype(value: Dtype) -> Option<InterventionDtype> {
    match value {
        Dtype::Float32 => Some(InterventionDtype::Float32),
        Dtype::Float16 => Some(InterventionDtype::Float16),
        Dtype::Bfloat16 => Some(InterventionDtype::Bfloat16),
        _ => None,
    }
}
fn native_indices(region: &ParameterRegion) -> Vec<ArrayIndexOp<'static>> {
    region
        .starts
        .iter()
        .zip(&region.shape)
        .map(|(start, count)| (*start as i32..(start + count) as i32).index_op())
        .collect()
}

fn apply_effective_update(
    result: &mut Array,
    region: &ParameterRegion,
    update: &ParameterUpdate,
    stream: &Stream,
) -> Result<Result<(), ParameterError>, Error> {
    let shape: Vec<_> = region.shape.iter().map(|n| *n as i32).collect();
    let indices = native_indices(region);
    let values = Array::from_slice(update.values(), &shape);
    let effective = match update {
        ParameterUpdate::Replace { .. } => values,
        ParameterUpdate::Add { .. } => result
            .try_index_device(indices.as_slice(), stream)?
            .as_dtype(Dtype::Float32, stream)?
            .add(values, stream)?,
    }
    .as_dtype(result.dtype(), stream)?;
    if !effective
        .is_finite(stream)?
        .all(false, stream)?
        .evaluated()?
        .as_slice::<bool>()[0]
    {
        return Ok(Err(ParameterError::Invalid(
            "parameter edit overflows target dtype".into(),
        )));
    }
    result.try_index_mut_device(indices.as_slice(), &effective, stream)?;
    result.evaluated()?;
    Ok(Ok(()))
}
