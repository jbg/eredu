//! Rotary application, including first-use frequency construction. No device or
//! parameter values are needed to price the selected equation.

use super::facts::{self, add, mul, Aliases, Emitter, FactResult, Output};
use super::{reduction::capacity_fixed as capacity, *};
use eredu_nn::{RotaryAlgorithm, RotaryArithmetic};

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary_with(
        |sink| emit(operation.as_view(), allocation, sink),
        |error| ordinary_error(operation, error),
    )
}

pub(super) fn ordinary_error(
    operation: &WorkspaceOperation,
    error: MlxWorkspaceFactError,
) -> Error {
    if matches!(error.cause(), MlxWorkspaceFactCause::RotaryAlgorithm(_)) {
        if let WorkspaceOperationKind::Rotary(spec, _) = operation.kind {
            if let Err(original) = spec.algorithm.validate() {
                return original;
            }
        }
    }
    error.ordinary()
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use WorkspaceOperationKindView as K;
    let (dimensions, offset, constructed) = match operation.kind {
        K::TensorRotary(d, _, base, scale, offset) => {
            if !base.is_finite() || base <= 0.0 || !scale.is_finite() {
                return Err(MlxWorkspaceFactError::descriptor(
                    "invalid Metal rotary base or scale",
                ));
            }
            (d, Some(offset), None)
        }
        K::RotaryFrequencies(d, _, offset) => (d, Some(offset), None),
        K::Rotary(spec, offset) => {
            spec.algorithm.validate_fixed()?;
            if !spec.base.is_finite() || spec.base <= 0.0 {
                return Err(MlxWorkspaceFactError::descriptor(
                    "invalid Metal rotary base",
                ));
            }
            (spec.dimensions, offset, Some(spec))
        }
        _ => return Ok(None),
    };
    let explicit_frequencies = matches!(operation.kind, K::RotaryFrequencies(..));
    let expected_inputs = if offset.is_none() {
        3
    } else if explicit_frequencies {
        2
    } else {
        1
    };
    if operation.inputs.len() != expected_inputs || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal rotary descriptor",
        ));
    }
    let input = operation.inputs.get(0).unwrap();
    if input.dtype() != WorkspaceDtype::Float32 {
        return Ok(None);
    }
    let shape = input.shape();
    if shape.len() < 2
        || shape
            .iter()
            .enumerate()
            .any(|(axis, n)| *n < 0 || (*n == 0 && axis + 2 != shape.len()))
        || dimensions <= 0
        || dimensions % 2 != 0
        || dimensions > shape[shape.len() - 1]
        || operation.outputs.get(0).unwrap() != input
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal rotary geometry",
        ));
    }
    let width = shape[shape.len() - 1] as u64;
    let length = shape[shape.len() - 2] as u64;
    let count = input.elements()?;
    let leading = shape[..shape.len() - 2]
        .iter()
        .try_fold(1_u64, |product, &extent| mul(product, extent as u64))?;
    i32::try_from(leading).map_err(|_| {
        MlxWorkspaceFactError::descriptor("Metal rotary leading extent overflows i32")
    })?;
    if let Some(offset) = offset {
        offset.checked_add(length as i32).ok_or_else(|| {
            MlxWorkspaceFactError::descriptor("Metal rotary position endpoint overflows i32")
        })?;
    }
    let cap = |n| capacity(allocation, n);
    let full = cap(count)?;
    let half_dims = dimensions as u64 / 2;
    let scalar = cap(1)?;
    let frequency_state = if let Some(spec) = constructed {
        match spec.algorithm {
            // Include the complete lazy construction graph even on first use:
            // powers, wavelength comparisons, interpolation and reciprocal.
            RotaryAlgorithm::Llama3 { .. } => add(mul(20, cap(half_dims)?)?, mul(16, scalar)?)?,
            RotaryAlgorithm::Yarn { .. } | RotaryAlgorithm::Proportional { .. } => {
                mul(2, cap(half_dims)?)?
            }
            _ if spec.arithmetic == RotaryArithmetic::InputProducts => cap(half_dims)?,
            _ => 0,
        }
    } else {
        0
    };
    let mut aliases_input = false;
    let total = if offset.is_none() {
        if shape.len() != 4 {
            return Err(MlxWorkspaceFactError::descriptor(
                "explicit rotary embeddings require a four-axis input",
            ));
        }
        let mut embeddings = 0;
        for layout in operation
            .inputs
            .slice(1..operation.inputs.len())
            .unwrap()
            .iter()
        {
            let s = layout.shape();
            if layout.dtype() != WorkspaceDtype::Float32
                || !matches!(s.len(), 2 | 3)
                || (s.len() == 3 && s[0] != 1 && s[0] != shape[0])
                || (s[s.len() - 2] != 1 && s[s.len() - 2] != shape[2])
                || (s[s.len() - 1] != 1 && s[s.len() - 1] != shape[3])
            {
                return Err(MlxWorkspaceFactError::descriptor(
                    "explicit rotary embedding broadcast differs",
                ));
            }
            embeddings = add(embeddings, cap(layout.elements()?)?)?;
        }
        // Embedding dtype casts, negative second half, rotate-half concat,
        // both full products and sum. Include possible F32 scalar promotion
        // casts of the two halves and one product before the final sum.
        let first = cap(mul(count / width, width / 2)?)?;
        let second = cap(mul(count / width, width - width / 2)?)?;
        add(
            add(mul(5, full)?, add(first, mul(2, second)?)?)?,
            add(embeddings, scalar)?,
        )?
    } else if let Some(spec) = constructed {
        let explicit = spec.arithmetic == RotaryArithmetic::InputProducts;
        let wavelength =
            matches!(spec.algorithm, RotaryAlgorithm::Llama3 { .. }) && !spec.traditional;
        let equation = if explicit || wavelength {
            let angles = cap(mul(length, half_dims)?)?;
            let half = cap(mul(mul(leading, length)?, half_dims)?)?;
            let rotated = cap(mul(mul(leading, length)?, dimensions as u64)?)?;
            // Input/final reshapes; integer positions and F32 conversion;
            // angle construction and trigonometric buffers; six products/sums.
            // InputProducts additionally casts/amplifies cosine and sine.
            // Wavelength native arithmetic can cast both input halves to F32
            // for each multiplication instead.
            let mut cost = add(
                add(mul(2, full)?, mul(2, cap(length)?)?)?,
                add(
                    add(mul(if explicit { 7 } else { 4 }, angles)?, scalar)?,
                    add(mul(if explicit { 6 } else { 10 }, half)?, rotated)?,
                )?,
            )?;
            if explicit && spec.traditional {
                // Pair reshape, two integer-index reshape copies, and the
                // interleaved concatenation's final reshape.
                cost = add(cost, add(mul(2, rotated)?, mul(2, half)?)?)?;
            }
            if dimensions as u64 != width {
                cost = add(cost, full)?;
                if !explicit {
                    cost = add(cost, cap(mul(count / width, width - dimensions as u64)?)?)?;
                }
            }
            cost
        } else {
            let flattened = !matches!(
                spec.algorithm,
                RotaryAlgorithm::Default | RotaryAlgorithm::Linear { .. }
            );
            if !flattened && shape.len() < 3 {
                return Err(MlxWorkspaceFactError::descriptor(
                    "native rotary requires at least three axes",
                ));
            }
            let batches = if flattened { leading } else { shape[0] as u64 };
            let mut cost = fused_cost(
                count,
                batches,
                mul(length, width)?,
                flattened,
                half_dims,
                allocation,
            )?;
            aliases_input = batches == 1;
            if flattened {
                cost = add(cost, mul(2, full)?)?;
            }
            if matches!(spec.algorithm, RotaryAlgorithm::Yarn { .. }) {
                // Input amplitude multiplication, potential input/scalar casts.
                cost = add(cost, add(mul(2, full)?, mul(2, scalar)?)?)?;
                aliases_input = false;
            }
            cost
        };
        equation
    } else {
        if shape.len() < 3 {
            return Err(MlxWorkspaceFactError::descriptor(
                "native rotary requires at least three axes",
            ));
        }
        if explicit_frequencies && operation.inputs.get(1).unwrap().shape() != [dimensions / 2] {
            return Err(MlxWorkspaceFactError::descriptor(
                "Metal rotary frequency dimensions differ",
            ));
        }
        let batches = shape[0] as u64;
        aliases_input = batches == 1;
        fused_cost(
            count,
            batches,
            mul(length, width)?,
            explicit_frequencies,
            half_dims,
            allocation,
        )?
    };
    let total = add(total, frequency_state)?;
    let output = if aliases_input {
        Output::AllocateOrAliasInputs {
            bytes: full,
            inputs: Aliases::Slice(&[0]),
        }
    } else {
        Output::Allocate(full)
    };
    let scratch = total
        .checked_sub(full)
        .ok_or_else(|| MlxWorkspaceFactError::descriptor("rotary workspace omits output"))?;
    sink.output(output)?;
    sink.finish(scratch, format_args!("vendored MLX Metal rotary: batch outputs/concatenation, scalar offsets, frequency casts, explicit products and possible reshape/promotion copies; lazy constructed frequency graph included on first use; all buffers retained through completion; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size())).map(Some)
}

fn fused_cost(
    count: u64,
    batches: u64,
    matrix_elements: u64,
    frequencies: bool,
    half_dims: u64,
    allocation: MetalAllocationFacts,
) -> FactResult<u64> {
    // The native dispatcher forms T*D and (N+3) in signed native integers,
    // including when its element-addressing kernel uses 64-bit offsets.
    i32::try_from(matrix_elements).map_err(|_| {
        MlxWorkspaceFactError::descriptor("Metal rotary matrix extent overflows i32")
    })?;
    // Empty pooled history is a valid sequence. Its scalar/frequency graph
    // still contributes workspace; no head grid is launched for zero elements.
    i32::try_from(if matrix_elements == 0 {
        0
    } else {
        count / batches / matrix_elements
    })
    .ok()
    .and_then(|heads| heads.checked_add(3))
    .ok_or_else(|| MlxWorkspaceFactError::descriptor("Metal rotary head grid overflows i32"))?;
    // safemlx submits one native RoPE per first-axis batch and retains all
    // outputs before concatenating. Metal copies into/reuses its result buffer,
    // even for a partial rotation or non-contiguous input; there is no extra
    // full-sized internal copy. Each invocation creates an integer offset and
    // may cast the frequency vector to F32.
    let mut per_batch = add(
        capacity(allocation, count / batches)?,
        capacity(allocation, 1)?,
    )?;
    if frequencies {
        per_batch = add(per_batch, capacity(allocation, half_dims)?)?;
    }
    let total = mul(batches, per_batch)?;
    if batches > 1 {
        add(total, capacity(allocation, count)?)
    } else {
        Ok(total)
    }
}

#[cfg(test)]
mod tests;
