//! The selected native GGUF decoder: a completed byte source and one eager F32 seed.
use super::*;
use crate::backend::nn::native_quantization::NativeQuantizationFormat;

#[derive(Clone, Copy)]
pub(in super::super) struct Geometry {
    pub rank: usize,
    pub values: usize,
}

pub(in super::super) fn inspect(
    op: WorkspaceOperationView<'_>,
) -> facts::FactResult<Option<Geometry>> {
    let WorkspaceOperationKindView::ParameterDecode(decoding) = op.kind else {
        return Ok(None);
    };
    let LinearFormat::GgufIQuant { ggml_type, .. } = decoding.format else {
        return Ok(None);
    };
    let Some(format) = NativeQuantizationFormat::from_ggml_type(ggml_type) else {
        return Ok(None);
    };
    let bad = || MlxWorkspaceFactError::descriptor("standalone GGUF decoder geometry differs");
    let ([input], [output]) = (
        op.inputs.array().ok_or_else(bad)?,
        op.outputs.array().ok_or_else(bad)?,
    );
    let rank = output.shape().len();
    if !(2..=3).contains(&rank) {
        return Ok(None);
    }
    if input.dtype() != WorkspaceDtype::Uint8
        || output.dtype() != WorkspaceDtype::Float32
        || output.shape().iter().any(|&n| n <= 0)
    {
        return Err(bad());
    }
    let (block_values, block_bytes) = format.block_geometry();
    let width = output.shape()[rank - 1];
    if width % block_values != 0 {
        return Err(bad());
    }
    let values = output.elements()?;
    let packed_bytes = facts::mul(values / block_values as u64, block_bytes as u64)?;
    if input.bytes()? != packed_bytes {
        return Err(bad());
    }
    // Array::try_from_slice uses the native checked i32 element count.
    if values > i32::MAX as u64 || packed_bytes > i32::MAX as u64 {
        return Ok(None);
    }
    Ok(Some(Geometry {
        rank,
        values: usize::try_from(values)?,
    }))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
    let Some(g) = inspect(op)? else {
        return Ok(None);
    };
    sink.output(facts::Output::Allocate(
        allocation.fixed_buffer_capacity(facts::mul(g.values as u64, 4)?)?,
    ))?;
    sink.finish(0, format_args!("selected GGUF host decoder creates one eager F32 backing; Copy shares that backing and both source/result completions are separately recorded")).map(Some)
}

pub(in super::super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Geometry>(),
        size_of::<Option<Geometry>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<NativeQuantizationFormat>(),
        size_of::<usize>() * 4,
        size_of::<u64>() * 3,
        super::super::host_array::slice_control_bytes()?,
        safemlx::original_scoped_evaluation_control_bytes()?,
        crate::backend::runtime::cache::value_completion_control_bytes(1)?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

pub(in super::super) fn emit_host(
    op: WorkspaceOperationView<'_>,
    sink: &mut facts::HostEmitter<'_>,
) -> facts::FactResult<Option<WorkspaceHostFacts>> {
    let Some(_) = inspect(op)? else {
        return Ok(None);
    };
    let WorkspaceOperationKindView::ParameterDecode(
        eredu_nn::parameter_values::ParameterDecoding {
            format: LinearFormat::GgufIQuant { ggml_type, endian },
            ..
        },
    ) = op.kind
    else {
        unreachable!("qualified GGUF operation");
    };
    let Some(bytes) =
        crate::native_quantization::NativeQuantizedTensor::standalone_dequantization_control_bytes(
            op.outputs.get(0).expect("qualified GGUF output").shape(),
            ggml_type,
            endian,
        )
    else {
        return Ok(None);
    };
    sink.finish(u64::try_from(bytes)?, format_args!("selected GGUF parameter decoder: exact retained native storage owner, byte-reader controls, shape vectors and the selected bounded host decoding worker")).map(Some)
}
