//! One actual pager source joined to a complete cross-layer storage inventory.
use super::*;
use crate::backend::nn::workspace::OwnedArrayProjection;
use crate::backend::runtime::cache::residency::PreparedCacheHostPromotion;
use eredu_core::cache::LayerCachePolicy;
use eredu_runtime::working_memory::{
    WorkspacePagedAppendState, WorkspacePagedBlock, WorkspacePagedLayerState,
};
use std::cell::RefCell;
use std::num::NonZeroU32;
#[path = "layer/disk.rs"]
mod disk;
#[path = "layer/promotion.rs"]
mod promotion;
use disk::RetainedPagedDisk;

/// Exact canonical source and descriptive state retained after its manager
/// loan. Native arrays live in the enclosing shared physical-root inventory.
pub(crate) struct ProjectedPagedSource {
    // Native transfer prefixes retire before original host pins and H.
    promotions: RefCell<Vec<Option<PreparedCacheHostPromotion>>>,
    promotion_capacity: usize,
    files: Vec<RetainedPagedDisk>,
    host_trace: Option<eredu_runtime::working_memory::WorkspacePagedHostTrace>,
    geometry: PagedCacheSourceGeometry,
    append_geometry: eredu_runtime::working_memory::WorkspacePagedGeometry,
    pins: PinnedCacheSource,
}
impl ProjectedPagedSource {
    /// Metadata-only handoff after the enclosing complete storage collector has
    /// authenticated both independent managers and every actual backing.
    pub(crate) fn inherit_host_trace(&mut self, saved: &Self) {
        self.host_trace = saved.host_trace.clone();
    }
    pub(crate) fn host_trace(
        &self,
    ) -> Option<&eredu_runtime::working_memory::WorkspacePagedHostTrace> {
        self.host_trace.as_ref()
    }
    pub(crate) fn append_geometry(&self) -> eredu_runtime::working_memory::WorkspacePagedGeometry {
        self.append_geometry
    }
    pub(crate) fn selection(&self) -> eredu_runtime::CacheBlockSelection {
        self.pins.selection()
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        self.pins.manager()
    }
    pub(crate) fn catalog_validation_control_bytes() -> Option<usize> {
        ProjectedPagedCacheSource::validation_control_bytes()?
            .checked_add(size_of::<(&Self, &CacheBlockSourceLoan<'_>)>())?
            .checked_add(size_of::<(Option<MutableCacheTail>, u64)>())?
            .checked_add(size_of::<(
                std::slice::Iter<'_, RetainedPagedDisk>,
                &CacheBlockSourceLoan<'_>,
                Option<crate::backend::runtime::cache::residency::CacheFileSource>,
            )>())?
            .checked_add(size_of::<
                Result<[PagedCacheArrayGeometry; 2], CacheSourceError>,
            >())
    }
    pub(crate) fn validate_catalog(
        &self,
        loan: &CacheBlockSourceLoan<'_>,
    ) -> Result<(), CacheSourceError> {
        self.pins.validate_loan(loan)?;
        self.validate_files(loan)?;
        let geometry = &self.geometry;
        if geometry.session_id != loan.session_id()
            || geometry.generation != loan.generation()
            || geometry.pool_id != loan.pool().id()
        {
            return Err(CacheSourceError::Identity);
        }
        let bytes = geometry.tail.map_or(Ok(0), pair_bytes)?;
        match loan.tail() {
            Some(tail) if tail.end == geometry.offset && tail.bytes == bytes => {}
            None if geometry.tail.is_none() => {}
            _ => return Err(CacheSourceError::Identity),
        }
        if geometry.blocks.len() != loan.blocks().count() {
            return Err(CacheSourceError::Identity);
        }
        for (expected, actual) in geometry.blocks.iter().zip(loan.blocks()) {
            if &expected.id != actual.id()
                || expected.phase != actual.phase()
                || expected.imported != actual.imported()
                || expected.arrays != block_geometry(actual, geometry.key_only)?
            {
                return Err(CacheSourceError::Identity);
            }
        }
        Ok(())
    }
    pub(crate) fn owned_source_pin_count(&self, id: &eredu_core::cache::CacheBlockId) -> usize {
        usize::from(self.pins.ids().contains(id))
    }
    pub(crate) fn geometry(&self) -> &PagedCacheSourceGeometry {
        &self.geometry
    }
    /// Same source check as standalone projection, against the actual current
    /// pager. Source identity alone never authorizes mutation or submission.
    pub(crate) fn validate_source(
        &self,
        cache: &PagedKeyValueCache,
        storage: &ProjectedNativeStorage,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        context
            .charge_metadata(
                ProjectedPagedCacheSource::validation_control_bytes()
                    .and_then(|bytes| {
                        bytes.checked_add(size_of::<(
                            &Self,
                            &PagedKeyValueCache,
                            &ProjectedNativeStorage,
                            &WorkspaceContext,
                        )>())
                    })
                    .ok_or_else(|| {
                        CacheSourceFailure::source(CacheSourceError::Overflow, context)
                    })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        cache.with_workspace_source(context, |source| {
            validate_paged_source(&self.pins, &self.geometry, storage, &source, Some(self))
                .map_err(|cause| CacheSourceFailure::source(cause, context))
        })
    }
}

struct LayerProjection<'a, 'storage> {
    global_layer: usize,
    policy: &'a LayerCachePolicy,
    batch: NonZeroU32,
    projection: &'a mut OwnedArrayProjection<'storage>,
}
impl LayerProjection<'_, '_> {
    fn run(
        self,
        mut source: PagedKeyValueSource<'_>,
    ) -> Result<WorkspacePagedAppendState, CacheSourceFailure> {
        let context = self.projection.context();
        if source.cache.global_layer != self.global_layer {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                context,
            ));
        }
        // Retain every authenticated file/version, including backings of hot
        // rows. A path without a live or persistent source cannot substitute.
        let mut files = context
            .metadata_vec(
                source
                    .blocks
                    .blocks()
                    .filter(|block| block.disk().is_some())
                    .count(),
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for block in source.blocks.blocks() {
            if block.disk().is_some() {
                files.push(RetainedPagedDisk::prepare(
                    block,
                    source.cache.key_only,
                    context,
                )?);
            } else if block.device().is_none() && block.host().is_none() {
                return Err(CacheSourceFailure::source(
                    CacheSourceError::PromotionRequired,
                    context,
                ));
            }
        }
        let promotion_capacity = source
            .blocks
            .blocks()
            .filter(|block| block.device().is_none())
            .count();
        let promotions = context
            .metadata_vec(promotion_capacity)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let host_trace = if source.manager().options().host_budget_bytes() != 0 {
            Some(
                eredu_runtime::working_memory::WorkspacePagedHostTrace::new(context)
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
            )
        } else {
            None
        };
        let geometry = source.prepare_geometry(context)?;
        let paged_geometry = append::append_geometry(
            &geometry,
            source.manager(),
            self.policy,
            self.batch,
            context,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let mut blocks = context
            .metadata_vec(source.block_count())
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        // The collector checks its final source slot before invoking this
        // acquisition. On success it immediately retains the pin outside this
        // manager callback. All subsequent partial imports use that same owner.
        self.projection.retain_paged_source(|| {
            let pins = source.blocks.pin_selected(context)?;
            Ok(ProjectedPagedSource {
                promotions: RefCell::new(promotions),
                promotion_capacity,
                files,
                host_trace: host_trace.clone(),
                geometry,
                append_geometry: paged_geometry,
                pins,
            })
        })?;
        let floating_type = |dtype| {
            Ok(match dtype {
                Dtype::Float32 => eredu_nn::workspace::WorkspaceFloatingType::Float32,
                Dtype::Float16 => eredu_nn::workspace::WorkspaceFloatingType::Float16,
                Dtype::Bfloat16 => eredu_nn::workspace::WorkspaceFloatingType::Bfloat16,
                _ => {
                    return Err(CacheSourceFailure::source(
                        CacheSourceError::Geometry,
                        context,
                    ));
                }
            })
        };
        for block in source.blocks.blocks() {
            let value = if let Some(arrays) = block.device() {
                let values = project_owned_pair(self.projection, arrays, context)?;
                if block.disk().is_some() {
                    let pair = block_geometry(block, source.cache.key_only)
                        .map_err(|cause| CacheSourceFailure::source(cause, context))?;
                    let read = [
                        context
                            .declare_host_read_value(&pair[0].shape, floating_type(pair[0].dtype)?)
                            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
                        context
                            .declare_host_read_value(&pair[1].shape, floating_type(pair[1].dtype)?)
                            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
                    ];
                    WorkspacePagedBlock::backed(block.id().start, block.id().end, values, read)
                } else {
                    WorkspacePagedBlock::new(block.id().start, block.id().end, values, true)
                }
            } else if let Some(host) = block.host() {
                let types = [
                    floating_type(
                        host[0]
                            .try_fixed_descriptor::<4>()
                            .map_err(|cause| CacheSourceFailure::source(cause.into(), context))?
                            .dtype(),
                    )?,
                    floating_type(
                        host[1]
                            .try_fixed_descriptor::<4>()
                            .map_err(|cause| CacheSourceFailure::source(cause.into(), context))?
                            .dtype(),
                    )?,
                ];
                let values = [
                    self.projection
                        .project_host::<4>(host[0])
                        .map_err(|cause| CacheSourceFailure::projection(cause, context))?,
                    self.projection
                        .project_host::<4>(host[1])
                        .map_err(|cause| CacheSourceFailure::projection(cause, context))?,
                ];
                let id = block.id().clone();
                WorkspacePagedBlock::host(id.start, id.end, values, types)
            } else {
                if block.phase() != CacheStoragePhase::DiskReady
                    || block.disk().and_then(|disk| disk.file_source()).is_none()
                {
                    return Err(CacheSourceFailure::source(
                        CacheSourceError::PendingStorage,
                        context,
                    ));
                }
                let pair = block_geometry(block, source.cache.key_only)
                    .map_err(|cause| CacheSourceFailure::source(cause, context))?;
                let values = [
                    context
                        .declare_host_read_value(&pair[0].shape, floating_type(pair[0].dtype)?)
                        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
                    context
                        .declare_host_read_value(&pair[1].shape, floating_type(pair[1].dtype)?)
                        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
                ];
                WorkspacePagedBlock::read_source(block.id().start, block.id().end, values)
            };
            blocks.push(value);
        }
        // The finite Host itinerary owns per-pass destinations. The standalone
        // source path retains its existing one-time destination producer.
        // Prepare each final destination after the immutable traversal ended.
        // The output is inserted before any subsequent fallible operation.
        for index in 0..blocks.len() {
            if host_trace.is_some() || !blocks[index].requires_host_transfer() {
                continue;
            }
            let range = blocks[index].range();
            let id = source
                .blocks
                .blocks()
                .find(|block| block.id().start == range.start && block.id().end == range.end)
                .map(|block| block.id().clone())
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, context))?;
            self.projection
                .retain_paged_promotion(self.global_layer, &id, || {
                    source.blocks.prepare_host_promotion(&id, context)
                })?;
        }
        let tail = source
            .tail_arrays()
            .map(|arrays| project_owned_pair(self.projection, arrays, context))
            .transpose()?;
        let state =
            WorkspacePagedAppendState::project(paged_geometry, blocks.into_iter(), tail, context)
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        match host_trace {
            Some(trace) => state
                .with_host_transfers(trace)
                .map_err(|cause| CacheSourceFailure::metadata(cause, context)),
            None => Ok(state),
        }
    }
}
impl PagedKeyValueCache {
    /// Pooling's DeviceState retains physical module identity inside each actual
    /// pager. Read that identity here; the local table index is not a replacement.
    pub(crate) fn project_workspace_pooling_append_into(
        &self,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspacePagedAppendState, CacheSourceFailure> {
        let context = projection.context();
        context
            .charge_metadata(size_of::<(
                &Self,
                &LayerCachePolicy,
                NonZeroU32,
                &mut OwnedArrayProjection<'_>,
                Result<WorkspacePagedAppendState, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        self.project_workspace_append_into(self.global_layer, policy, batch, projection)
    }
    pub(crate) fn project_workspace_layer_into(
        &self,
        layer: usize,
        global_layer: usize,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspacePagedLayerState, CacheSourceFailure> {
        let context = projection.context();
        let paged = self.project_workspace_append_into(global_layer, policy, batch, projection)?;
        WorkspacePagedLayerState::project(layer, policy, paged, [], context).map_err(|cause| {
            CacheSourceFailure::metadata(cause.into_workspace_error(context), context)
        })
    }
    pub(crate) fn project_workspace_append_into(
        &self,
        global_layer: usize,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspacePagedAppendState, CacheSourceFailure> {
        let context = projection.context();
        context
            .charge_metadata(
                layer_control_bytes().ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Overflow, context)
                })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let work = LayerProjection {
            global_layer,
            policy,
            batch,
            projection,
        };
        self.with_workspace_source(context, |source| work.run(source))
    }
}
fn project_owned_pair(
    projection: &mut OwnedArrayProjection<'_>,
    arrays: [&Array; 2],
    context: &WorkspaceContext,
) -> Result<[WorkspaceTensor; 2], CacheSourceFailure> {
    context
        .charge_metadata(size_of::<(
            &mut OwnedArrayProjection<'_>,
            [&Array; 2],
            &WorkspaceContext,
            [WorkspaceTensor; 2],
            Result<[WorkspaceTensor; 2], CacheSourceFailure>,
        )>())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let first = projection
        .project_prepared(arrays[0])
        .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
    let second = projection
        .project_prepared(arrays[1])
        .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
    Ok([first, second])
}
fn layer_control_bytes() -> Option<usize> {
    let controls = [
        size_of::<Vec<RetainedPagedDisk>>(),
        size_of::<Option<crate::backend::runtime::cache::residency::CacheFileSource>>(),
        size_of::<[eredu_nn::workspace::WorkspaceHostReadValue; 2]>(),
        size_of::<Result<eredu_nn::workspace::WorkspaceHostReadValue, eredu_nn::Error>>(),
        size_of::<[PagedCacheArrayGeometry; 2]>(),
        size_of::<Option<eredu_runtime::working_memory::WorkspacePagedHostTrace>>(),
        size_of::<Result<eredu_runtime::working_memory::WorkspacePagedHostTrace, eredu_nn::Error>>(
        ),
        size_of::<RefCell<Vec<Option<PreparedCacheHostPromotion>>>>(),
        size_of::<[safemlx::HostTransferDescriptor<4>; 2]>(),
        safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
        size_of::<[eredu_nn::workspace::WorkspaceFloatingType; 2]>(),
        size_of::<[&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>; 2]>(),
        size_of::<(usize, std::ops::Range<i64>, CacheBlockId)>(),
        size_of::<Result<eredu_nn::workspace::WorkspaceFloatingType, CacheSourceFailure>>(),
        size_of::<LayerProjection<'_, '_>>(),
        size_of::<(
            &PagedKeyValueCache,
            usize,
            &LayerCachePolicy,
            NonZeroU32,
            &mut OwnedArrayProjection<'_>,
            &WorkspaceContext,
        )>(),
        size_of::<Result<WorkspacePagedAppendState, CacheSourceFailure>>(),
        size_of::<(
            &PagedKeyValueCache,
            usize,
            usize,
            &LayerCachePolicy,
            NonZeroU32,
            &mut OwnedArrayProjection<'_>,
        )>(),
        size_of::<PagedCacheSourceGeometry>(),
        size_of::<WorkspacePagedAppendState>(),
        size_of::<WorkspacePagedLayerState>(),
        size_of::<eredu_runtime::working_memory::WorkspacePagedGeometry>(),
        size_of::<Vec<WorkspacePagedBlock>>(),
        size_of::<std::vec::IntoIter<WorkspacePagedBlock>>(),
        size_of::<Option<[WorkspaceTensor; 2]>>(),
        size_of::<Result<WorkspacePagedLayerState, CacheSourceFailure>>(),
        size_of::<Result<WorkspacePagedAppendState, Error>>(),
        size_of::<Result<WorkspacePagedLayerState, eredu_runtime::StateError>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
