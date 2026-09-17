//! Tensor buffers for the selected Metal convolution kernels. Native dispatch
//! uses geometry, so depthwise/implicit kernels do not inherit an im2col charge.

use super::facts::{self, add, mul, Emitter, FactResult, Output};
use super::{reduction::capacity_fixed as capacity, *};

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
    let WorkspaceOperationKindView::Convolution {
        stride,
        padding,
        dilation,
        groups,
        transposed,
    } = &operation.kind
    else {
        return Ok(None);
    };
    let Some([input, weight]) = operation.inputs.array() else {
        return invalid();
    };
    let Some([output]) = operation.outputs.array() else {
        return invalid();
    };
    let dims = stride.len();
    if !(1..=2).contains(&dims)
        || input.shape().len() != dims + 2
        || weight.shape().len() != dims + 2
        || padding.len() != dims
        || dilation.len() != dims
        || *groups <= 0
        || stride.iter().chain(dilation.iter()).any(|&v| v <= 0)
        || padding.iter().any(|&v| v < 0)
        || transposed
            .as_ref()
            .is_some_and(|extra| dims != 1 || extra.len() != dims || extra.iter().any(|&v| v < 0))
    {
        return invalid();
    }
    if input.dtype() != WorkspaceDtype::Float32 || weight.dtype() != WorkspaceDtype::Float32 {
        return Ok(None);
    }
    if input.shape().iter().chain(weight.shape()).any(|&v| v <= 0) {
        return invalid();
    }
    let channels = input.shape()[dims + 1] as u64;
    let outputs = weight.shape()[0] as u64;
    let groups = *groups as u64;
    if mul(weight.shape()[dims + 1] as u64, groups)? != channels || outputs % groups != 0 {
        return invalid();
    }
    let mut expected = [0; 4];
    expected[0] = input.shape()[0];
    for axis in 0..dims {
        let n = input.shape()[axis + 1] as i128;
        let kernel = weight.shape()[axis + 1] as i128;
        let s = stride[axis] as i128;
        let p = padding[axis] as i128;
        let d = dilation[axis] as i128;
        let extent = if let Some(extra) = transposed {
            (n - 1) * s - 2 * p + d * (kernel - 1) + extra[axis] as i128 + 1
        } else {
            (n + 2 * p - d * (kernel - 1) - 1).div_euclid(s) + 1
        };
        expected[axis + 1] = i32::try_from(extent)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                MlxWorkspaceFactError::descriptor("invalid Metal convolution output extent")
            })?;
    }
    expected[dims + 1] = outputs as i32;
    if output.shape() != &expected[..dims + 2] || output.dtype() != WorkspaceDtype::Float32 {
        return invalid();
    }

    // MLX's negative transpose padding crops the undilated input. The adapter
    // instead computes the unpadded result and selects its correct output view.
    let cropped_transpose = transposed.is_some()
        && crate::backend::nn::convolution::transpose_1d_needs_output_crop(
            weight.shape(),
            stride[0],
            padding[0],
            dilation[0],
        );
    let mut native_output = expected;
    if cropped_transpose {
        native_output[1] = native_output[1]
            .checked_add(padding[0].checked_mul(2).ok_or_else(|| {
                MlxWorkspaceFactError::descriptor("convolution crop extent overflow")
            })?)
            .ok_or_else(|| MlxWorkspaceFactError::descriptor("convolution crop extent overflow"))?;
    }
    let output_elements = elements(&native_output[..dims + 2])?;
    let result = capacity(allocation, output_elements)?;
    // Two original dtype casts and two row-contiguous materializations. All
    // floating values are bounded by four bytes; no donation is assumed.
    let mut scratch = mul(
        2,
        add(
            capacity(allocation, input.elements()?)?,
            capacity(allocation, weight.elements()?)?,
        )?,
    )?;
    let input_dilation_one = transposed.is_none() || stride.iter().all(|&v| v == 1);
    let aligned = (channels / groups <= 4 || (channels / groups) % 16 == 0)
        && (outputs / groups <= 16 || (outputs / groups) % 16 == 0);
    let rows = output_elements / outputs;
    let winograd = dims == 2
        && transposed.is_none()
        && crate::backend::nn::convolution::uses_metal_winograd_2d(
            input.shape(),
            weight.shape(),
            (stride[0], stride[1]),
            (dilation[0], dilation[1]),
            groups as i32,
        );
    let depthwise = dims == 1
        && transposed.is_none()
        && groups == channels
        && groups == outputs
        && stride[0] == 1
        && dilation[0] == 1
        && padding[0] == 0;
    let implicit = if dims == 1 || groups > 1 {
        input_dilation_one && aligned
    } else {
        input_dilation_one && aligned || channels % 16 == 0 && outputs % 16 == 0 || rows >= 256
    };
    let mechanism = if winograd {
        let h = native_output[1] as u64;
        let w = native_output[2] as u64;
        let batch = native_output[0] as u64;
        let tiles = mul(batch, mul(h.div_ceil(6), w.div_ceil(6))?)?;
        let padded = mul(
            mul(batch, channels)?,
            mul(
                add(mul(6, h.div_ceil(6))?, 2)?,
                add(mul(6, w.div_ceil(6))?, 2)?,
            )?,
        )?;
        for count in [
            1,
            padded,
            mul(64, mul(channels, outputs)?)?,
            mul(64, mul(tiles, channels)?)?,
            mul(64, mul(tiles, outputs)?)?,
            output_elements, // F32 result before restoring reduced precision
        ] {
            scratch = add(scratch, capacity(allocation, count)?)?;
        }
        // The 64 independent Winograd GEMMs have non-broadcast weights. They
        // cannot collapse to unbatched split-K and allocate no partial tensor.
        "FP32 Winograd padded input, fill scalar, three transformed buffers and possible final dtype restoration"
    } else if original_layout(operation).is_some() || depthwise || implicit {
        "depthwise/implicit convolution with no unfolded matrix or tensor scratch"
    } else {
        let reduction = weight.elements()? / outputs;
        if rows > i32::MAX as u64 || reduction > i32::MAX as u64 {
            return invalid(); // native explicit GEMM requires these signed extents
        }
        scratch = add(
            scratch,
            capacity(allocation, mul(mul(rows, reduction)?, groups)?)?,
        )?;
        if groups > 1 {
            scratch = add(scratch, capacity(allocation, weight.elements()?)?)?;
            "grouped unfolded matrix and transposed weights; direct regular batched GEMM"
        } else {
            let partitions = matrix::steel_split_partitions_fixed(rows, outputs, reduction)?;
            scratch = add(
                scratch,
                capacity(allocation, mul(partitions, output_elements)?)?,
            )?;
            "unfolded matrix and geometry-selected direct Steel split-K, including single-row outputs"
        }
    };
    sink.output(Output::Allocate(result))?;
    sink.finish(scratch, format_args!("vendored MLX Metal {mechanism}; possible floating casts and contiguous input/weight copies; cropped transpose retains full unpadded output={cropped_transpose}; page={} with bounded oversized reuse; active tensor buffers only, host workspace separately required", allocation.page_size())).map(Some)
}
fn elements(shape: &[i32]) -> FactResult<u64> {
    shape
        .iter()
        .try_fold(1, |count, &extent| mul(count, extent as u64))
}
fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal convolution workspace descriptor",
    ))
}

#[cfg(test)]
mod tests;

// This is the same compiled native dispatch/constructor query used before
// original execution. Physical fallback algorithms retain their independent
// conservative storage facts; only this complete profile supplies a receipt.
pub(super) fn original_layout(
    operation: WorkspaceOperationView<'_>,
) -> Option<safemlx::ops::OriginalConvolutionLayout> {
    let WorkspaceOperationKindView::Convolution {
        stride,
        padding,
        dilation,
        groups,
        transposed: None,
    } = operation.kind
    else {
        return None;
    };
    let [input, weight] = operation.inputs.array()?;
    let [output] = operation.outputs.array()?;
    if [input.dtype(), weight.dtype(), output.dtype()]
        .iter()
        .any(|dtype| *dtype != WorkspaceDtype::Float32)
    {
        return None;
    }
    let layout = safemlx::ops::OriginalConvolutionLayout::inspect(
        input.shape(),
        weight.shape(),
        stride,
        padding,
        dilation,
        groups,
    )?;
    (layout.output_shape() == output.shape()).then_some(layout)
}
