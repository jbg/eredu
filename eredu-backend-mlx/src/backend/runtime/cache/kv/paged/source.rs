//! Current paged state described from one exact manager/block/tail loan.
use super::*;
mod admission;
use crate::backend::runtime::cache::residency::{
    CacheBlockSource, CacheBlockSourceLoan, CacheSourceError, CacheSourceFailure,
};
use eredu_nn::{Error, workspace::WorkspaceContext};
use eredu_runtime::{CacheBlockSelection, CacheStoragePhase, MutableCacheTail};
use std::mem::{size_of, size_of_val};

/// Fixed actual native layout, including the key-only persistence sentinel.
/// This has no source, native ownership or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedCacheArrayGeometry {
    pub(crate) shape: [i32; 4],
    pub(crate) dtype: Dtype,
    pub(crate) logical_bytes: u64,
}
/// One immutable sealed block's current descriptive metadata.
#[derive(Debug)]
pub(crate) struct PagedCacheBlockGeometry {
    pub(crate) id: CacheBlockId,
    pub(crate) phase: CacheStoragePhase,
    pub(crate) arrays: [PagedCacheArrayGeometry; 2],
    pub(crate) imported: bool,
}
/// Context-paid geometry retaining its metadata owner after the lexical source
/// loan ends. It is intentionally not accepted as a source-copy/role receipt.
pub(crate) struct PagedCacheSourceGeometry {
    pub(crate) session_id: u64,
    pub(crate) generation: u64,
    pub(crate) pool_id: u64,
    pub(crate) global_layer: usize,
    pub(crate) rank: Option<CacheRankIdentity>,
    pub(crate) offset: i64,
    pub(crate) tail_start: i64,
    pub(crate) sliding_window: Option<i32>,
    pub(crate) prefix_tokens: i32,
    pub(crate) key_only: bool,
    pub(crate) blocks: Vec<PagedCacheBlockGeometry>,
    pub(crate) tail: Option<[PagedCacheArrayGeometry; 2]>,
    _context: WorkspaceContext,
}

/// Source geometry and exact native resources borrowed together. The current
/// pager plus manager guard retain every tail, block, backing and pool owner.
pub(crate) struct PagedKeyValueSource<'a> {
    cache: &'a PagedKeyValueCache,
    blocks: CacheBlockSourceLoan<'a>,
    tail: Option<[PagedCacheArrayGeometry; 2]>,
}
impl PagedKeyValueCache {
    /// Exact fixed controls preceding the source loan. The manager query and
    /// returned geometry are separate producers because their counts differ.
    pub(crate) fn workspace_source_control_bytes<R, F>(_: &F) -> Option<usize> {
        Self::workspace_source_control_for::<R, F>()
    }
    fn workspace_source_fixed_bytes(callback_bytes: usize) -> Option<usize> {
        let parts = [
            size_of::<PagedKeyValueSource<'_>>(),
            size_of::<Option<[PagedCacheArrayGeometry; 2]>>(),
            size_of::<[Option<&Array>; 2]>(),
            size_of::<Option<[&Array; 2]>>(),
            size_of::<Option<[&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>; 2]>>(),
            size_of::<Option<&safemlx::HostTransferMetadataSnapshot>>(),
            size_of::<CacheBlockSelection>(),
            size_of::<MutableCacheTail>(),
            size_of::<CacheSourceError>(),
            size_of::<CacheSourceFailure>(),
            size_of::<(i64, i64, i64, i64, i64)>(),
            size_of::<Option<[PagedCacheArrayGeometry; 2]>>(),
            callback_bytes,
            // The source validation visits each actual record, but all fixed
            // descriptor/geometry temporaries are reused sequentially.
            source_inspection_controls()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Loans current canonical state without evaluating tails, reading shards,
    /// polling completions, or manufacturing a resident concatenation state.
    /// The callback cannot return references to catalog-owned resources.
    pub(crate) fn with_workspace_source<R>(
        &self,
        context: &WorkspaceContext,
        consume: impl for<'loan> FnOnce(PagedKeyValueSource<'loan>) -> Result<R, CacheSourceFailure>,
    ) -> Result<R, CacheSourceFailure> {
        let continuation = SourceContinuation {
            cache: self,
            context,
            consume,
        };
        context
            .charge_metadata(
                Self::workspace_source_fixed_bytes(size_of_val(&continuation)).ok_or_else(
                    || CacheSourceFailure::source(CacheSourceError::Overflow, context),
                )?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let selection = CacheBlockSelection::new(
            self.global_layer,
            CacheRepresentation::KeyValue,
            0,
            i64::MAX,
            0,
        );
        self.manager
            .with_source_loan(selection, context, move |blocks| continuation.run(blocks))
    }
    /// Same fixed geometry validator under a short read-only source loan.
    /// Native references never escape; this creates no source or work authority.
    pub(crate) fn inspect_copy_source<R, E>(
        &self,
        fail: impl Fn(CacheSourceError) -> E,
        consume: impl for<'loan> FnOnce(PagedKeyValueSource<'loan>) -> Result<R, E>,
    ) -> Result<R, E> {
        let selection = CacheBlockSelection::new(
            self.global_layer,
            CacheRepresentation::KeyValue,
            0,
            i64::MAX,
            0,
        );
        self.manager
            .inspect_copy_source(selection, &fail, |blocks| {
                let source = Self::validate_source(self, blocks).map_err(&fail)?;
                consume(source)
            })
    }
    pub(crate) fn workspace_source_control_for<R, F>() -> Option<usize> {
        Self::workspace_source_fixed_bytes(size_of::<SourceContinuation<'_, F>>())?.checked_add(
            CacheResidencyManager::source_loan_control_bytes::<R>(size_of::<
                SourceContinuation<'_, F>,
            >())?,
        )
    }
    fn validate_source<'a>(
        cache: &'a Self,
        blocks: CacheBlockSourceLoan<'a>,
    ) -> Result<PagedKeyValueSource<'a>, CacheSourceError> {
        if !std::ptr::eq(cache.manager(), blocks.manager())
            || cache.offset < 0
            || cache.tail_start < 0
            || cache.tail_start > cache.offset
            || cache.prefix_tokens < 0
            || cache.sliding_window.is_some_and(|window| window <= 0)
        {
            return Err(CacheSourceError::Identity);
        }
        let tail = match (&cache.tail_keys, &cache.tail_values) {
            (None, None) if cache.tail_start == cache.offset => None,
            (Some(keys), Some(values)) if cache.tail_start < cache.offset => {
                let pair = [array_geometry(keys)?, array_geometry(values)?];
                validate_pair(pair, cache.offset - cache.tail_start, cache.key_only)?;
                Some(pair)
            }
            _ => return Err(CacheSourceError::Geometry),
        };
        let tail_bytes = tail.map_or(Ok(0), pair_bytes)?;
        match blocks.tail() {
            Some(canonical) if canonical.end == cache.offset && canonical.bytes == tail_bytes => {}
            None if tail.is_none() => {} // sealed-only state has no mutable-tail owner
            _ => return Err(CacheSourceError::Identity),
        }
        let mut previous_end = 0;
        let mut prefix_end = 0;
        let required_prefix = i64::from(cache.prefix_tokens).min(cache.tail_start);
        let history_start = cache
            .sliding_window
            .map_or(0, |window| (cache.offset - i64::from(window)).max(0))
            .min(cache.tail_start);
        let mut history_end = history_start;
        let mut expected = tail;
        for block in blocks.blocks() {
            let id = block.id();
            if id.rank != cache.rank || id.end > cache.tail_start || id.start < previous_end {
                return Err(CacheSourceError::Identity);
            }
            previous_end = id.end;
            let pair = block_geometry(block, cache.key_only)?;
            if let Some(expected) = expected {
                same_widths(expected, pair)?;
            } else {
                expected = Some(pair);
            }
            cover(id.start, id.end, 0, required_prefix, &mut prefix_end)?;
            cover(
                id.start,
                id.end,
                history_start,
                cache.tail_start,
                &mut history_end,
            )?;
        }
        if prefix_end != required_prefix || history_end != cache.tail_start {
            return Err(CacheSourceError::MissingHistory);
        }
        Ok(PagedKeyValueSource {
            cache,
            blocks,
            tail,
        })
    }
}
// Named callback frame so the public query and actual manager continuation
// consume the same layout even for a caller closure with extended alignment.
struct SourceContinuation<'a, F> {
    cache: &'a PagedKeyValueCache,
    context: &'a WorkspaceContext,
    consume: F,
}
impl<F> SourceContinuation<'_, F> {
    fn run<R>(self, blocks: CacheBlockSourceLoan<'_>) -> Result<R, CacheSourceFailure>
    where
        F: for<'loan> FnOnce(PagedKeyValueSource<'loan>) -> Result<R, CacheSourceFailure>,
    {
        let source = PagedKeyValueCache::validate_source(self.cache, blocks)
            .map_err(|cause| CacheSourceFailure::source(cause, self.context))?;
        (self.consume)(source)
    }
}
impl<'a> PagedKeyValueSource<'a> {
    pub(crate) fn pin_copy_source(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<crate::backend::runtime::cache::residency::PinnedCacheSource, CacheSourceFailure>
    {
        self.blocks.pin_selected(context)
    }
    pub(crate) fn manager(&self) -> &'a CacheResidencyManager {
        self.blocks.manager()
    }
    pub(crate) fn manager_source(&self) -> &CacheBlockSourceLoan<'a> {
        &self.blocks
    }
    pub(crate) fn tail_arrays(&self) -> Option<[&'a Array; 2]> {
        self.cache
            .tail_keys
            .as_ref()
            .zip(self.cache.tail_values.as_ref())
            .map(|(keys, values)| [keys, values])
    }
    pub(crate) fn block_count(&self) -> usize {
        self.blocks.blocks().count()
    }
    /// Actual fixed controls and one exact block-metadata Vec. No native handle
    /// cloning, serialized source inventory or guessed block allowance appears.
    pub(crate) fn geometry_control_bytes(&self) -> Option<usize> {
        geometry_fixed_bytes()?.checked_add(WorkspaceContext::metadata_vec_bytes::<
            PagedCacheBlockGeometry,
        >(self.block_count())?)
    }
    pub(crate) fn prepare_geometry(
        &self,
        context: &WorkspaceContext,
    ) -> Result<PagedCacheSourceGeometry, CacheSourceFailure> {
        context
            .charge_metadata(
                geometry_fixed_bytes().ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Overflow, context)
                })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let mut blocks = context
            .metadata_vec(self.block_count())
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for block in self.blocks.blocks() {
            let arrays = block_geometry(block, self.cache.key_only)
                .map_err(|cause| CacheSourceFailure::source(cause, context))?;
            blocks.push(PagedCacheBlockGeometry {
                id: block.id().clone(),
                phase: block.phase(),
                arrays,
                imported: block.imported(),
            });
        }
        Ok(PagedCacheSourceGeometry {
            session_id: self.blocks.session_id(),
            generation: self.blocks.generation(),
            pool_id: self.blocks.pool().id(),
            global_layer: self.cache.global_layer,
            rank: self.cache.rank,
            offset: self.cache.offset,
            tail_start: self.cache.tail_start,
            sliding_window: self.cache.sliding_window,
            prefix_tokens: self.cache.prefix_tokens,
            key_only: self.cache.key_only,
            blocks,
            tail: self.tail,
            _context: context.clone(),
        })
    }
}
fn source_inspection_controls() -> Option<usize> {
    let parts = [
        size_of::<safemlx::OwnedArrayDescriptorLoan<'_>>(),
        size_of::<safemlx::ArrayDescriptorFacts>(),
        size_of::<[PagedCacheArrayGeometry; 2]>(),
        size_of::<CacheBlockSource<'_>>(),
        size_of::<[[i32; 4]; 2]>(),
        size_of::<[&str; 2]>(),
        size_of::<[Option<&Array>; 2]>(),
        size_of::<Option<[&Array; 2]>>(),
        size_of::<Option<[&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>; 2]>>(),
        size_of::<Option<&safemlx::HostTransferMetadataSnapshot>>(),
        size_of::<Result<PagedCacheArrayGeometry, CacheSourceError>>(),
        size_of::<Result<[PagedCacheArrayGeometry; 2], CacheSourceError>>(),
        size_of::<Result<(), CacheSourceError>>(),
        size_of::<(u64, usize, i64)>(),
        size_of::<Option<[PagedCacheArrayGeometry; 2]>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn geometry_fixed_bytes() -> Option<usize> {
    let parts = [
        size_of::<PagedCacheSourceGeometry>(),
        size_of::<CacheSourceFailure>(),
        size_of::<PagedCacheBlockGeometry>(),
        size_of::<CacheBlockId>(),
        size_of::<Vec<PagedCacheBlockGeometry>>(),
        size_of::<Result<PagedCacheSourceGeometry, CacheSourceFailure>>(),
        source_inspection_controls()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn cover(
    start: i64,
    end: i64,
    required_start: i64,
    required_end: i64,
    cursor: &mut i64,
) -> Result<(), CacheSourceError> {
    let first = start.max(required_start);
    let last = end.min(required_end);
    if first < last {
        if first > *cursor {
            return Err(CacheSourceError::MissingHistory);
        }
        *cursor = (*cursor).max(last);
    }
    Ok(())
}
fn array_geometry(array: &Array) -> Result<PagedCacheArrayGeometry, CacheSourceError> {
    let source = array.try_descriptor()?;
    let facts = source.facts();
    if facts.allocation().is_none() {
        return Err(CacheSourceError::PendingStorage);
    }
    let value = geometry(source.shape(), facts.dtype())?;
    if value.logical_bytes
        != u64::try_from(facts.logical_bytes()).map_err(|_| CacheSourceError::Overflow)?
    {
        return Err(CacheSourceError::Geometry);
    }
    Ok(value)
}
fn geometry(shape: &[i32], dtype: Dtype) -> Result<PagedCacheArrayGeometry, CacheSourceError> {
    let shape = <[i32; 4]>::try_from(shape).map_err(|_| CacheSourceError::Geometry)?;
    if shape.iter().any(|&value| value <= 0) {
        return Err(CacheSourceError::Geometry);
    }
    let width = match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
    };
    let logical_bytes = shape
        .iter()
        .try_fold(width, |bytes: u64, &value| bytes.checked_mul(value as u64))
        .ok_or(CacheSourceError::Overflow)?;
    Ok(PagedCacheArrayGeometry {
        shape,
        dtype,
        logical_bytes,
    })
}
fn dtype(name: &str) -> Result<Dtype, CacheSourceError> {
    Ok(match name {
        "Bool" => Dtype::Bool,
        "Uint8" => Dtype::Uint8,
        "Uint16" => Dtype::Uint16,
        "Uint32" => Dtype::Uint32,
        "Uint64" => Dtype::Uint64,
        "Int8" => Dtype::Int8,
        "Int16" => Dtype::Int16,
        "Int32" => Dtype::Int32,
        "Int64" => Dtype::Int64,
        "Float16" => Dtype::Float16,
        "Float32" => Dtype::Float32,
        "Float64" => Dtype::Float64,
        "Bfloat16" => Dtype::Bfloat16,
        "Complex64" => Dtype::Complex64,
        _ => return Err(CacheSourceError::Geometry),
    })
}
fn pair_bytes(pair: [PagedCacheArrayGeometry; 2]) -> Result<u64, CacheSourceError> {
    pair[0]
        .logical_bytes
        .checked_add(pair[1].logical_bytes)
        .ok_or(CacheSourceError::Overflow)
}
fn validate_pair(
    pair: [PagedCacheArrayGeometry; 2],
    tokens: i64,
    key_only: bool,
) -> Result<(), CacheSourceError> {
    if pair[0].dtype != pair[1].dtype
        || pair[0].shape[..3] != pair[1].shape[..3]
        || i64::from(pair[0].shape[2]) != tokens
        || if key_only {
            pair[1].shape[3] != 1
        } else {
            pair[0].shape[3] != pair[1].shape[3]
        }
    {
        return Err(CacheSourceError::Geometry);
    }
    Ok(())
}
fn same_widths(
    a: [PagedCacheArrayGeometry; 2],
    b: [PagedCacheArrayGeometry; 2],
) -> Result<(), CacheSourceError> {
    for index in 0..2 {
        if a[index].dtype != b[index].dtype
            || [a[index].shape[0], a[index].shape[1], a[index].shape[3]]
                != [b[index].shape[0], b[index].shape[1], b[index].shape[3]]
        {
            return Err(CacheSourceError::Geometry);
        }
    }
    Ok(())
}
fn block_geometry(
    block: CacheBlockSource<'_>,
    key_only: bool,
) -> Result<[PagedCacheArrayGeometry; 2], CacheSourceError> {
    let shapes = block.shapes();
    let dtypes = block.dtypes();
    let pair = [
        geometry(shapes[0], dtype(dtypes[0])?)?,
        geometry(shapes[1], dtype(dtypes[1])?)?,
    ];
    validate_pair(pair, block.id().end - block.id().start, key_only)?;
    if pair_bytes(pair)? != block.logical_bytes() {
        return Err(CacheSourceError::Geometry);
    }
    if let Some(arrays) = block.device() {
        for index in 0..2 {
            if array_geometry(arrays[index])? != pair[index] {
                return Err(CacheSourceError::Geometry);
            }
        }
    }
    if let Some(host) = block.host() {
        for index in 0..2 {
            if let Some(metadata) = host[index].prepared_metadata() {
                if metadata.shape() != pair[index].shape
                    || metadata.dtype() != pair[index].dtype
                    || metadata.nbytes() as u64 != pair[index].logical_bytes
                {
                    return Err(CacheSourceError::Geometry);
                }
            }
        }
    }
    Ok(pair)
}

#[cfg(test)]
mod tests;

mod projection;
pub(crate) use projection::{
    PagedAppendWorkspaceFailure, PagedWorkspaceProjectionFailure, ProjectedPagedAppendState,
    ProjectedPagedCacheSource, ProjectedPagedSource,
};
