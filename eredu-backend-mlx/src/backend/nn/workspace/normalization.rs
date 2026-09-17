//! Bounds for the actual MLX normalization equations. Counts include every
//! intermediate until completion, with no assumption of fusion or donation.

use super::{
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};

use super::facts::{self, add, mul, Aliases, Emitter, FactResult, Output};

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, sink))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use eredu_nn::NormalizationScale;
    let (name, groups) = match &operation.kind {
        WorkspaceOperationKindView::Normalization(name, groups) => (*name, *groups),
        WorkspaceOperationKindView::ConstructedNormalization(spec) => {
            spec.validate_fixed()?;
            ("constructed_rms", spec.groups)
        }
        _ => return Ok(None),
    };
    if !matches!(
        name,
        "rms"
            | "l2"
            | "layer_norm"
            | "constructed_rms"
            | "gated_group_rms_norm"
            | "silu_gated_group_rms_norm"
    ) {
        return Ok(None);
    }
    if operation.inputs.is_empty() || operation.outputs.len() != 1 {
        return invalid();
    }
    if operation
        .inputs
        .iter()
        .chain(operation.outputs.iter())
        .any(|input| input.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    let input = operation.inputs.get(0).unwrap();
    let output = operation.outputs.get(0).unwrap();
    let width = input
        .shape()
        .last()
        .copied()
        .filter(|width| *width > 0)
        .ok_or_else(|| MlxWorkspaceFactError::descriptor("invalid Metal normalization width"))?;
    if input.shape() != output.shape()
        || groups.is_some_and(|groups| groups <= 0 || width % groups != 0)
    {
        return invalid();
    }
    let gated = matches!(name, "gated_group_rms_norm" | "silu_gated_group_rms_norm");
    let weights = if gated {
        if operation.inputs.len() != 3
            || operation.inputs.get(1).unwrap().shape() != input.shape()
            || groups.is_none()
        {
            return invalid();
        }
        operation.inputs.slice(2..operation.inputs.len()).unwrap()
    } else {
        operation.inputs.slice(1..operation.inputs.len()).unwrap()
    };
    if weights.iter().any(|weight| weight.shape() != [width]) {
        return invalid();
    }
    let dimensions = width as u64;
    let elements = input.elements()?;
    let group_width = dimensions / groups.unwrap_or(1) as u64;
    let rows = elements / group_width;
    // Feature-vector capacity also covers casts of learned scales when the
    // input has zero rows. Nonempty norm inputs have at least one full row.
    let full = capacity(allocation, elements.max(dimensions))?;
    let vector = capacity(allocation, dimensions)?;
    let row = capacity(allocation, rows)?;
    let scalar = capacity(allocation, 1)?;
    let reduction = sum_cost(allocation, elements, rows, group_width)?;
    let custom_weightless = add(mul(4, full)?, scalar)?;
    // Weightless fallback: square; sum/divide; epsilon add; rsqrt;
    // multiply input; final cast. Binary/unary primitives include possible
    // casts. The mean count and epsilon are separate scalar arrays.
    let fallback_weightless = add(
        add(mul(5, full)?, mul(8, row)?)?,
        add(reduction, mul(2, scalar)?)?,
    )?;
    let weightless = custom_weightless.max(fallback_weightless);
    // Matching F16/BF16 learned RMS widens, performs custom weightless RMS,
    // casts down, then multiplies. Other type pairs use fast RMS: input and
    // weight casts plus an output (which can itself be the contiguous copy).
    let learned = add(mul(7, full)?, scalar)?.max(add(mul(2, full)?, vector)?);
    let total = match name {
        "layer_norm" => {
            if weights.len() > 2 || groups.is_some() {
                return invalid();
            }
            // Native Metal fast LayerNorm uses one output/contiguous copy,
            // input cast, each present parameter cast and absent-parameter
            // scalar placeholders. Its reductions stay in threadgroup memory.
            add(
                add(mul(2, full)?, mul(weights.len() as u64, vector)?)?,
                mul((2 - weights.len()) as u64, scalar)?,
            )?
        }
        "rms" => {
            if groups.is_some() || weights.len() > 1 {
                return invalid();
            }
            if weights.is_empty() {
                weightless
            } else {
                add(learned, full)?
            }
        }
        "l2" => {
            if groups.is_some() || !weights.is_empty() {
                return invalid();
            }
            // square + sum + epsilon add + rsqrt + input multiply.
            add(add(mul(4, full)?, mul(5, row)?)?, add(reduction, scalar)?)?
        }
        "gated_group_rms_norm" | "silu_gated_group_rms_norm" => {
            // Both supported orders contain input/gate widening, sigmoid and
            // gate multiplication, two reshapes, square/mean/add/rsqrt,
            // normalization and scale/gate products, and final conversion.
            // Count each at most-F32 intermediate, even when orders differ.
            add(
                add(mul(20, full)?, mul(8, row)?)?,
                add(reduction, mul(2, scalar)?)?,
            )?
        }
        "constructed_rms" => {
            let WorkspaceOperationKindView::ConstructedNormalization(spec) = &operation.kind else {
                unreachable!()
            };
            let weighted = !matches!(spec.scale, NormalizationScale::Unit);
            if spec.dimensions != width || weights.len() != usize::from(weighted) {
                return invalid();
            }
            let offset = if matches!(spec.scale, NormalizationScale::LearnedOffset { .. }) {
                add(mul(3, vector)?, scalar)?
            } else {
                0
            };
            if groups.is_some() {
                // Grouped construction widens before reshaping, so nonempty
                // inputs definitely use the custom F32 weightless kernel.
                let norm = if elements != 0 {
                    custom_weightless
                } else {
                    weightless
                };
                let base = add(mul(4, full)?, norm)?; // widen, two reshapes, final cast
                if weighted {
                    add(base, add(add(vector, offset)?, mul(3, full)?)?)?
                } else {
                    base
                }
            } else if weighted {
                add(learned, offset)?
            } else {
                weightless
            }
        }
        _ => unreachable!(),
    };
    sink.output(Output::AllocateOrAliasInputs {
        bytes: full,
        inputs: Aliases::Range {
            start: 0,
            end: operation.inputs.len(),
        },
    })?;
    sink.finish(total.checked_sub(full).ok_or(MlxWorkspaceFactError::descriptor("normalization bound omits its result"))?, format_args!("vendored MLX Metal {name}: exact normalization policy, at most F32 tensor buffers, all casts/reshape/contiguous copies and equation intermediates retained; bounded native reduction accumulator or custom/fast kernel; groups={groups:?}, page={} with oversized cache reuse; active tensor buffers only", allocation.page_size())).map(Some)
}

fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal normalization workspace descriptor",
    ))
}

#[cfg(test)]
mod tests;
