//! Storage-transform bounds for the retained Metal realization. CPU compaction
//! and host-data constructors use the same shared Metal allocator, so their
//! tensor buffers are counted once here, independently of host staging facts.

use super::facts::{self, buffer_capacity as capacity, mul, Aliases, Emitter, FactResult, Output};
use super::*;

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, sink))
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use WorkspaceOperationKindView as Kind;
    if !matches!(
        operation.kind,
        Kind::SliceUpdate { .. } | Kind::StaticSliceUpdate { .. } | Kind::Contiguous | Kind::DeepCopy
    ) {
        return Ok(None);
    }
    let Some([output]) = operation.outputs.array() else {
        return invalid();
    };
    let Some(source) = operation.inputs.first() else {
        return invalid();
    };
    if source != output {
        return invalid();
    }
    let bytes = capacity(allocation, output.bytes()?)?;
    let (storage, scratch_bytes, equation) = match &operation.kind {
        Kind::SliceUpdate { starts } => {
            let Some([_, update]) = operation.inputs.array() else {
                return invalid();
            };
            if starts.len() != source.shape().len()
                || update.shape().len() != source.shape().len()
                || update.dtype() != source.dtype()
                || starts.iter().zip(update.shape()).zip(source.shape()).any(
                    |((&start, &extent), &limit)| {
                        start < 0 || start.checked_add(extent).is_none_or(|end| end > limit)
                    },
                )
            {
                return invalid();
            }
            // SliceUpdate copies the full destination then updates its slice
            // directly, reading arbitrary strides. The safemlx indexing path
            // may strip/reinsert leading singleton axes with two reshapes.
            // Price both copies and the update dtype cast independently: the
            // floating descriptor represents F16, BF16 and F32 execution.
            (
                Output::AllocateOrAliasInputs {
                    bytes,
                    inputs: Aliases::Slice(&[0, 1]),
                },
                mul(3, capacity(allocation, update.bytes()?)?)?,
                "full destination copy, two possible update reshape copies and update cast; empty/full replacement may alias",
            )
        }
        Kind::StaticSliceUpdate { starts, ends, strides } => {
            let Some([_, update]) = operation.inputs.array() else { return invalid(); };
            if starts.len()!=source.shape().len() || ends.len()!=starts.len()
                || strides.len()!=starts.len() || update.shape().len()!=starts.len()
                || update.dtype()!=source.dtype()
                || starts.iter().zip(*ends).zip(*strides).zip(source.shape()).zip(update.shape())
                    .any(|((((&a,&b),&step),&n),&width)| a<0 || b<=a || b>n || step<=0
                        || b.checked_sub(a).and_then(|d|d.checked_add(step-1)).map(|d|d/step)!=Some(width)) {
                return invalid();
            }
            (Output::AllocateOrAliasInputs { bytes, inputs:Aliases::Slice(&[0,1]) },
                capacity(allocation, update.bytes()?)?,
                "exact positive-stride SliceUpdate: full destination copy and possible update cast; no host indexing reshapes")
        }
        Kind::Contiguous => {
            if operation.inputs.len() != 1 {
                return invalid();
            }
            (
                Output::AllocateOrAliasInputs {
                    bytes,
                    inputs: Aliases::Slice(&[0]),
                },
                0,
                "row-contiguous materialization: at most one result, otherwise shared input; arbitrary strides and oversized backing",
            )
        }
        Kind::DeepCopy => {
            if operation.inputs.len() != 1 {
                return invalid();
            }
            // EvaluatedArray::deep_clone first compacts non-row-major data on
            // the reused default CPU stream, then copies directly from that
            // storage into a new shared native buffer. No input donation or
            // early release of the compact buffer is assumed.
            (
                Output::Allocate(bytes),
                bytes,
                "independent host-data result plus possible CPU stride-compaction buffer, both in the shared Metal allocator",
            )
        }
        _ => unreachable!(),
    };
    sink.output(storage)?;
    sink.finish(scratch_bytes, format_args!("vendored MLX Metal/safemlx {equation}; page={} with bounded oversized reuse; active tensor buffers only; host workspace separately required", allocation.page_size())).map(Some)
}

fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal storage-transform descriptor",
    ))
}

#[cfg(test)]
mod tests;
