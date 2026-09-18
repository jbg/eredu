//! The pinned MaskedScatter primitive, including its exclusive U32 mask scan.
use super::facts::{Emitter, FactResult, Output, add, buffer_capacity, mul};
use super::*;

pub(super) fn geometry(operation: WorkspaceOperationView<'_>) -> FactResult<Option<()>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("masked_scatter")
    ) {
        return Ok(None);
    }
    let invalid = || MlxWorkspaceFactError::descriptor("invalid Metal masked row-scatter geometry");
    let Some([input, mask, source]) = operation.inputs.array() else {
        return Err(invalid());
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(invalid());
    };
    if mask.dtype() != WorkspaceDtype::Bool
        || output.shape() != input.shape()
        || output.dtype() != input.dtype()
    {
        return Err(invalid());
    }
    validate_masked_scatter_shapes(input.shape(), mask.shape(), source.shape())
        .map_err(|_| invalid())?;
    // The unchanged Metal worker indexes the flattened mask and prefix sum
    // with uint. A missing source-row axis is synthesized as mask.size() in
    // MLX's signed Shape representation. No wrapping geometry is admitted.
    if input.elements()? > u64::from(u32::MAX)
        || (source.shape().len() <= input.shape().len() - mask.shape().len()
            && mask.elements()? > i32::MAX as u64)
    {
        return Ok(None);
    }
    Ok(Some(()))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    if geometry(operation)?.is_none() {
        return Ok(None);
    }
    let input = operation.inputs.get(0).unwrap();
    let source = operation.inputs.get(2).unwrap();
    let elements = input.elements()?;
    let output = buffer_capacity(allocation, input.bytes()?)?;
    // Source conversion precedes broadcasting. Its complete original extent
    // matters even when unused rows exceed the selected mask population.
    let cast = buffer_capacity(allocation, mul(source.elements()?, input.dtype().bytes())?)?;
    let scratch = if elements == 0 {
        cast
    } else {
        // flatten_in_eval either copies to row-contiguous storage or returns
        // a view. Only a still-strided view enters contiguous_copy_gpu, so at
        // most one complete mask copy is born. Adding singleton dimensions
        // and broadcasts are metadata-only views.
        let mask_copy = buffer_capacity(allocation, elements)?;
        let offsets = buffer_capacity(allocation, mul(elements, 4)?)?;
        add(cast, add(mask_copy, offsets)?)?
    };
    sink.output(Output::Allocate(output))?;
    sink.finish(scratch, format_args!("MLX Metal MaskedScatter: destination copy, original source cast, flattened/contiguous Bool mask alternatives and U32 exclusive-scan offsets; scan writes its supplied output without extra numerical scratch, then the unchanged masked assign kernel; singleton/broadcast/squeeze views alias; page={} with bounded oversized reuse", allocation.page_size())).map(Some)
}

#[cfg(test)]
mod tests;
