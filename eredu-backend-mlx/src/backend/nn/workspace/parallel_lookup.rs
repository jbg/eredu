//! Selected-row storage of the existing vocabulary mask/lookup native worker.
use super::facts::{Emitter, FactResult, Output, add, mul};
use super::*;
use eredu_checkpoint::LinearFormat;

/// Borrow the same actual local parameters through the existing embedding
/// descriptor. Global IDs are validated by the wrapper; the child row kernel
/// receives masked local IDs. No native source or table is manufactured.
pub(super) fn embedding(
    operation: WorkspaceOperationView<'_>,
) -> FactResult<Option<WorkspaceOperationView<'_>>> {
    let WorkspaceOperationKindView::VocabularyParallelLookup {
        format,
        range,
        policy,
    } = operation.kind
    else {
        return Ok(None);
    };
    policy.validate_fixed()?;
    range
        .validate_fixed()
        .map_err(|_| MlxWorkspaceFactError::descriptor("invalid vocabulary ownership"))?;
    if i32::try_from(range.global_vocabulary).is_err()
        || operation.inputs.len() < 2
        || operation.outputs.len() != 1
        || operation
            .inputs
            .get(1)
            .and_then(|weight| weight.shape().first().copied())
            != i32::try_from(range.local.len()).ok()
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "vocabulary lookup source rows differ",
        ));
    }
    Ok(Some(WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Embedding(
            format,
            eredu_nn::EmbeddingLookupPolicy::Strict,
        ),
        inputs: operation.inputs,
        outputs: operation.outputs,
    }))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(child) = embedding(operation)? else {
        return Ok(None);
    };
    let WorkspaceOperationKindView::VocabularyParallelLookup { format, policy, .. } =
        operation.kind
    else {
        unreachable!()
    };
    if !matches!(
        format.encoding(),
        LinearFormat::Dense | LinearFormat::Affine(_) | LinearFormat::MxFp4
    ) {
        return Ok(None);
    }
    let mut rows = Emitter::count();
    let result = if format.encoding() == LinearFormat::Dense {
        indexing::emit(child, allocation, &mut rows)?
    } else {
        packed::emit(child, allocation, &mut rows)?
    };
    let Some(result) = result else {
        return Ok(None);
    };
    let Some(WorkspaceOutputEffect::Allocate(output)) = rows.first_output() else {
        return Ok(None);
    };
    if result.layout.outputs != 1 || result.layout.aliases != 0 {
        return Ok(None);
    }
    let count = operation
        .inputs
        .first()
        .expect("checked lookup IDs")
        .elements()?;
    let scalar = reduction::capacity_fixed(allocation, 1)?;
    let index = reduction::capacity_fixed(allocation, count)?;
    let mask = facts::buffer_capacity(allocation, count.max(1))?;
    // Reuse the complete strict validation/row storage. Ownership adds local
    // subtraction, one final masked output and zeros of its physical dtype;
    // count their possible retained intermediates without source-table copies.
    let extra = add(add(index, mul(2, output)?)?, mul(2, scalar)?)?;
    let extra = if matches!(policy, eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(_)) {
        add(extra, add(mul(2, mask)?, mul(2, scalar)?)?)?
    } else {
        extra
    };
    // The actual rank wrapper replaces the ordinary embedding checks: global
    // validation, two local range seeds, safe-index zero and masked-output zero.
    let (default, births) = indexing::token_validation_default_scratch(count, policy, allocation)?;
    sink.default_scratch(add(default, mul(4, scalar)?)?, births + 4)?;
    sink.output(Output::Allocate(output))?;
    sink.finish(add(result.scratch_bytes,extra)?,format_args!(
        "same selected local embedding rows plus global token validation, ownership subtraction/masking and dtype-preserving zero output; no complete table expansion; actual rank Sum is a separate source-bound equation"))
        .map(Some)
}
