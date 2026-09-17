//! Physical snapshot operands under their actual paged source loans.
use super::prepared_copy::KvCopySource;
use super::*;
use crate::backend::{
    nn::workspace::OwnedArrayProjection,
    runtime::cache::{
        residency::{CacheSourceError, CacheSourceFailure, PinnedCacheSource},
        state::{SnapshotArraySources, SnapshotOperand, SnapshotProjectionCause},
    },
};
use eredu_nn::workspace::WorkspaceContext;
use std::mem::size_of;

/// Borrowed physical attention cell. Fixed-role tables are retained by the
/// enclosing decoder source; this view never converts their outer representation.
pub(in crate::backend::runtime::cache::state) enum PagedSnapshotLayer<'a> {
    Absent,
    Paged(&'a PagedKeyValueCache),
    Other,
}
impl<'a> PagedSnapshotLayer<'a> {
    fn pager(self) -> Option<&'a PagedKeyValueCache> {
        match self {
            Self::Paged(cache) => Some(cache),
            _ => None,
        }
    }
}
/// Exact retained outer owner, borrowed without a temporary table or clone.
/// It grants no allocation, source registration or execution authority.
pub(in crate::backend::runtime::cache::state) trait PagedSnapshotState {
    fn layer_count(&self) -> usize;
    fn global_start(&self) -> usize;
    fn layout(&self) -> &eredu_runtime::SharedStateLayout;
    fn layer(&self, index: usize) -> Option<PagedSnapshotLayer<'_>>;
}
fn kv_layer(layer: &MlxKeyValueLayerState) -> PagedSnapshotLayer<'_> {
    match layer {
        MlxKeyValueLayerState::Stateless => PagedSnapshotLayer::Absent,
        MlxKeyValueLayerState::Paged(cache) => PagedSnapshotLayer::Paged(cache),
        MlxKeyValueLayerState::Device(_) => PagedSnapshotLayer::Other,
    }
}
impl PagedSnapshotState for MlxKeyValueState {
    fn layout(&self) -> &eredu_runtime::SharedStateLayout {
        &self.layout
    }
    fn layer_count(&self) -> usize {
        self.layers.len()
    }
    fn global_start(&self) -> usize {
        self.global_layer_start
    }
    fn layer(&self, index: usize) -> Option<PagedSnapshotLayer<'_>> {
        self.layers.slots().get(index).map(kv_layer)
    }
}
impl PagedSnapshotState for super::prepared_copy::SavedResidentKvCopy {
    fn layout(&self) -> &eredu_runtime::SharedStateLayout {
        self.shared_layout()
    }
    fn layer_count(&self) -> usize {
        KvCopySource::Saved(self).len()
    }
    fn global_start(&self) -> usize {
        self.global_layer_start()
    }
    fn layer(&self, index: usize) -> Option<PagedSnapshotLayer<'_>> {
        KvCopySource::Saved(self).layer(index).map(kv_layer)
    }
}
impl<'a> KvCopySource<'a> {
    pub(super) fn paged_snapshot_state(self) -> &'a dyn PagedSnapshotState {
        match self {
            Self::Live(source) => source,
            Self::Saved(source) => source,
        }
    }
}
/// Borrows the actual full selected state and its validated shared manager.
/// No grant, clone, fallback store or converted layer table is constructed.
pub(crate) struct PagedSnapshotSource<'a> {
    source: &'a dyn PagedSnapshotState,
    manager: &'a CacheResidencyManager,
}
impl MlxKeyValueState {
    pub(crate) fn snapshot_paged_source(
        &self,
    ) -> Result<Option<PagedSnapshotSource<'_>>, SnapshotProjectionCause> {
        if self.paged_transaction_branch
            && self
                .layers
                .slots()
                .iter()
                .any(|layer| matches!(layer, MlxKeyValueLayerState::Paged(_)))
        {
            return Err(SnapshotProjectionCause::ChangedAt("live paged transaction branch"));
        }
        PagedSnapshotSource::prepare(self)
    }
}
impl<'a> PagedSnapshotSource<'a> {
    pub(in crate::backend::runtime::cache::state) fn prepare(
        source: &'a dyn PagedSnapshotState,
    ) -> Result<Option<Self>, SnapshotProjectionCause> {
        let manager = (0..source.layer_count()).find_map(|index| {
            source
                .layer(index)
                .and_then(PagedSnapshotLayer::pager)
                .map(PagedKeyValueCache::manager)
        });
        let Some(manager) = manager else {
            return Ok(None);
        };
        if source.layer_count() != source.layout().layout().len() {
            return Err(SnapshotProjectionCause::ChangedAt("outer layer count and declared layout"));
        }
        for index in 0..source.layer_count() {
            match (source.layer(index), source.layout().layout().layer(index)) {
                (
                    Some(PagedSnapshotLayer::Absent),
                    Some(LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }),
                ) => {}
                (
                    Some(PagedSnapshotLayer::Paged(cache)),
                    Some(
                        LayerCachePolicy::KeyValue { attention, .. }
                        | LayerCachePolicy::KeyValueWithFixedState { attention, .. },
                    ),
                ) => {
                    if !manager.same_catalog(cache.manager()) {
                        return Err(SnapshotProjectionCause::ChangedAt("paged layer manager identity"));
                    }
                    if source.global_start().checked_add(index) != Some(cache.global_layer()) {
                        return Err(SnapshotProjectionCause::ChangedAt("paged global layer coordinate"));
                    }
                    if !attention.sliding_window_i32().is_ok_and(|window| {
                        cache.matches_original_snapshot_policy(window)
                    }) {
                        return Err(SnapshotProjectionCause::ChangedAt("paged attention policy or retained history"));
                    }
                }
                _ => return Err(SnapshotProjectionCause::ChangedAt("physical layer kind and declared policy")),
            }
        }
        let mut actual_tails = 0usize;
        for index in 0..source.layer_count() {
            if let Some(PagedSnapshotLayer::Paged(cache)) = source.layer(index) {
                let has_tail = cache
                    .inspect_copy_source(SnapshotProjectionCause::Paged, |source| {
                        // Canonical tail entries include an empty end marker
                        // after sealing. The shared source validator already
                        // matches its exact bytes/end to this physical pager;
                        // array presence is not the canonical entry population.
                        Ok(source.manager_source().tail().is_some())
                    })?;
                actual_tails = actual_tails
                    .checked_add(usize::from(has_tail))
                    .ok_or(SnapshotProjectionCause::Overflow)?;
            }
        }
        manager.inspect_copy_source(
            eredu_runtime::CacheBlockSelection::new(
                0,
                eredu_core::cache::CacheRepresentation::KeyValue,
                0,
                i64::MAX,
                0,
            ),
            SnapshotProjectionCause::Paged,
            |loan| {
                if loan.catalog_population().1 != actual_tails {
                    return Err(SnapshotProjectionCause::ChangedAt("catalog and physical tail population"));
                }
                for block in loan.all_blocks() {
                    let index = block
                        .id()
                        .global_layer
                        .checked_sub(source.global_start())
                        .ok_or(SnapshotProjectionCause::ChangedAt("catalog block precedes global layer range"))?;
                    match source.layer(index) {
                        Some(PagedSnapshotLayer::Paged(cache))
                            if cache.global_layer() == block.id().global_layer
                                && block.id().representation
                                    == eredu_core::cache::CacheRepresentation::KeyValue => {}
                        _ => return Err(SnapshotProjectionCause::ChangedAt("catalog block layer or representation")),
                    }
                }
                Ok(())
            },
        )?;
        Ok(Some(Self { source, manager }))
    }
    pub(in crate::backend::runtime::cache::state) fn manager(&self) -> &'a CacheResidencyManager {
        self.manager
    }
    pub(in crate::backend::runtime::cache::state) fn visit_pagers<
        E: From<SnapshotProjectionCause>,
    >(
        &self,
        visitor: &mut dyn FnMut(&'a PagedKeyValueCache) -> Result<(), E>,
    ) -> Result<(), E> {
        for index in 0..self.source.layer_count() {
            match self.source.layer(index) {
                Some(PagedSnapshotLayer::Absent) => {}
                Some(PagedSnapshotLayer::Paged(cache)) => visitor(cache)?,
                _ => return Err(SnapshotProjectionCause::Changed.into()),
            }
        }
        Ok(())
    }
}
// Concrete lexical capture frame used by the source-loan and pin debit.
struct Pin<'a, 'b> {
    projection: &'a mut OwnedArrayProjection<'b>,
    context: &'b WorkspaceContext,
}
impl SnapshotArraySources for PagedSnapshotSource<'_> {
    fn visit_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        for index in 0..self.source.layer_count() {
            let layer = self
                .source
                .layer(index)
                .ok_or(SnapshotProjectionCause::Changed)?;
            if let PagedSnapshotLayer::Paged(cache) = layer {
                cache.visit_snapshot_arrays(visitor)?;
            }
        }
        Ok(())
    }
    fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        for index in 0..self.source.layer_count() {
            let layer = self
                .source
                .layer(index)
                .ok_or(SnapshotProjectionCause::Changed)?;
            if let PagedSnapshotLayer::Paged(cache) = layer {
                cache.inspect_copy_source(SnapshotProjectionCause::Paged, |source| {
                    for block in source.manager_source().blocks() {
                        let disk = block.retained_disk_types()?;
                        if let Some(arrays) = block.device() {
                            for array in arrays {
                                visitor(SnapshotOperand::Array(array))?;
                            }
                        } else if let Some(host) = block.host() {
                            for buffer in host {
                                visitor(SnapshotOperand::Host(buffer))?;
                            }
                        } else if disk.is_none() {
                            return Err(CacheSourceError::PromotionRequired.into());
                        }
                        // Disk-only payloads have no numerical copy operand.
                        // The exact file stays in the pinned source manager and
                        // is retained in the independent canonical publication.
                    }
                    if let Some(arrays) = source.tail_arrays() {
                        for array in arrays {
                            visitor(SnapshotOperand::Array(array))?;
                        }
                    }
                    Ok(())
                })?;
            }
        }
        Ok(())
    }
    fn copy_source_count(&self) -> usize {
        (0..self.source.layer_count())
            .filter(|index| {
                matches!(
                    self.source.layer(*index),
                    Some(PagedSnapshotLayer::Paged(_))
                )
            })
            .count()
    }
    fn source_controls(&self) -> Result<usize, SnapshotProjectionCause> {
        let mut bytes =
            crate::backend::runtime::cache::residency::CacheBlockSource::retained_disk_control_bytes(
            ) + size_of::<Self>()
                + size_of::<Pin<'_, '_>>()
                + size_of::<&eredu_runtime::SharedStateLayout>()
                + size_of::<Option<&LayerCachePolicy>>()
                + size_of::<Option<i32>>()
                + size_of::<std::ops::Range<usize>>()
                + size_of::<SnapshotOperand<'_>>()
                + size_of::<
                    &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
                >()
                + size_of::<Option<[&Array; 2]>>()
                + size_of::<Option<[&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>; 2]>>();
        for index in 0..self.source.layer_count() {
            let layer = self
                .source
                .layer(index)
                .ok_or(SnapshotProjectionCause::Changed)?;
            if let PagedSnapshotLayer::Paged(cache) = layer {
                bytes = cache.inspect_copy_source(SnapshotProjectionCause::Paged, |source| {
                    PagedKeyValueCache::workspace_source_control_for::<(), Pin<'_, '_>>()
                        .and_then(|n| {
                            n.checked_add(PinnedCacheSource::control_bytes(source.block_count())?)
                        })
                        .and_then(|n| n.checked_add(size_of::<Result<(), CacheSourceFailure>>()))
                        .and_then(|n| n.checked_add(bytes))
                        .ok_or(SnapshotProjectionCause::Overflow)
                })?;
            }
        }
        Ok(bytes)
    }
    fn pin_sources(
        &self,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<(), SnapshotProjectionCause> {
        let context = projection.context();
        for index in 0..self.source.layer_count() {
            let layer = self
                .source
                .layer(index)
                .ok_or(SnapshotProjectionCause::Changed)?;
            if let PagedSnapshotLayer::Paged(cache) = layer {
                let pin = Pin {
                    projection,
                    context,
                };
                cache.with_workspace_source(context, move |mut source| {
                    pin.projection
                        .retain_copy_source(|| source.pin_copy_source(pin.context))
                })?;
            }
        }
        Ok(())
    }
}

impl PagedKeyValueCache {
    pub(in crate::backend::runtime::cache::state::key_value) fn visit_snapshot_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        self.inspect_copy_source(SnapshotProjectionCause::Paged, |source| {
            for block in source.manager_source().blocks() {
                let arrays = block.device().ok_or(CacheSourceError::PromotionRequired)?;
                for array in arrays {
                    visitor(array)?;
                }
            }
            if let Some(arrays) = source.tail_arrays() {
                for array in arrays {
                    visitor(array)?;
                }
            }
            Ok(())
        })
    }
}
