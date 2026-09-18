//! Physical scalar evidence from the pinned fast::layer_norm constructor.
use super::*;

pub(super) fn scalar(operation: WorkspaceOperationView<'_>) -> Option<F> {
    let K::LayerNorm { weight, bias } = operation.kind else {
        return None;
    };
    if operation.outputs.len() != 1
        || operation.inputs.len() != 1 + usize::from(weight) + usize::from(bias)
    {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let width = *input.shape().last()?;
    if width <= 0
        || input.dtype() != WorkspaceDtype::Float32
        || input.shape() != operation.outputs.get(0)?.shape()
        || operation
            .inputs
            .iter()
            .skip(1)
            .any(|value| value.dtype() != WorkspaceDtype::Float32 || value.shape() != [width])
    {
        return None;
    }
    let source = dtype(operation, 0)?;
    // MLX fast.cpp chooses x.dtype when no weight is present, including its
    // bias-only branch (the bias is cast). Otherwise it selects the exact
    // result_type(x, weight, bias) before either the native or fallback worker.
    if !weight {
        return Some(source);
    }
    let result = promote(source, dtype(operation, 1)?);
    Some(if bias {
        promote(result, dtype(operation, 2)?)
    } else {
        result
    })
}

#[cfg(test)]
#[path = "layer_norm/tests.rs"]
mod tests;
