//! Actual completed block/tail projection retained by canonical source pins.
use super::*;
mod append;
mod layer;
use crate::backend::{
    nn::workspace::{
        ExistingArrayProjection, ProjectedNativeStorage, ProjectionInventoryLayout,
        ProjectionSourceError, ProjectionSourceLayout,
    },
    runtime::cache::residency::PinnedCacheSource,
};
pub(crate) use append::{PagedAppendWorkspaceFailure, ProjectedPagedAppendState};
use eredu_nn::workspace::WorkspaceTensor;
pub(crate) use layer::ProjectedPagedSource;

/// Paged state metadata and its exact retained native roots. This is a source
/// for a future paged equation/transfer binder, not resident-concatenation state
/// or a model-role receipt. Native handles retire before canonical pins and H.
pub(crate) struct ProjectedPagedCacheSource {
    geometry: PagedCacheSourceGeometry,
    blocks: Vec<[WorkspaceTensor; 2]>,
    tail: Option<[WorkspaceTensor; 2]>,
    storage: ProjectedNativeStorage,
    pins: PinnedCacheSource,
    _context: WorkspaceContext,
}
impl ProjectedPagedCacheSource {
    pub(crate) fn geometry(&self) -> &PagedCacheSourceGeometry {
        &self.geometry
    }
    pub(crate) fn blocks(&self) -> &[[WorkspaceTensor; 2]] {
        &self.blocks
    }
    pub(crate) fn tail(&self) -> Option<&[WorkspaceTensor; 2]> {
        self.tail.as_ref()
    }
    pub(crate) fn storage(&self) -> &ProjectedNativeStorage {
        &self.storage
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        self.pins.manager()
    }
    /// Rechecks actual current local state and the same canonical pinned IDs.
    /// No caller-provided geometry or physical identity can establish a source.
    pub(crate) fn validation_control_bytes() -> Option<usize> {
        let parts = [
            // The source descriptor and its retained witness are simultaneously
            // borrowed; those same two loan frames are reused for each operand.
            Array::descriptor_control_bytes()?,
            Array::descriptor_control_bytes()?,
            safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            size_of::<[Option<&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>>; 2]>(),
            size_of::<[safemlx::HostTransferDescriptor<4>; 2]>(),
            size_of::<Result<(), CacheSourceError>>(),
            size_of::<Result<(), CacheSourceFailure>>(),
            size_of::<Option<safemlx::ArrayAllocationInfo>>(),
            size_of::<safemlx::ArrayAllocationInfo>(),
            size_of::<Option<&Array>>(),
            size_of::<std::slice::Iter<'_, CacheBlockId>>(),
            size_of::<(usize, usize)>(),
            size_of::<Option<&ProjectedPagedSource>>(),
            size_of::<CacheSourceFailure>(),
            size_of::<(&Self, &PagedKeyValueSource<'_>, &WorkspaceContext)>(),
            size_of::<(
                &PinnedCacheSource,
                &PagedCacheSourceGeometry,
                &ProjectedNativeStorage,
                &PagedKeyValueSource<'_>,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn validate_source(
        &self,
        source: &PagedKeyValueSource<'_>,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        context
            .charge_metadata(
                Self::validation_control_bytes().ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Overflow, context)
                })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        self.validate_source_inner(source)
            .map_err(|cause| CacheSourceFailure::source(cause, context))
    }
    fn validate_source_inner(
        &self,
        source: &PagedKeyValueSource<'_>,
    ) -> Result<(), CacheSourceError> {
        validate_paged_source(&self.pins, &self.geometry, &self.storage, source, None)
    }
}

/// On failure the accepted source pins, metadata and native clone prefix remain
/// intact. The enclosing caller controls their retirement after the manager
/// guard is gone; a refusal never releases the paying host owner early.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct PagedWorkspaceProjectionFailure {
    #[source]
    cause: CacheSourceFailure,
    retained: PendingProjection,
}
impl std::fmt::Debug for PagedWorkspaceProjectionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedWorkspaceProjectionFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl PagedWorkspaceProjectionFailure {
    pub(crate) fn cause(&self) -> &CacheSourceFailure {
        &self.cause
    }
}
struct PendingProjection {
    geometry: Option<PagedCacheSourceGeometry>,
    blocks: Vec<[WorkspaceTensor; 2]>,
    tail: Option<[WorkspaceTensor; 2]>,
    storage: Option<ProjectedNativeStorage>,
    pins: Option<PinnedCacheSource>,
    context: WorkspaceContext,
}
struct ProjectionWork<'a> {
    pending: &'a mut PendingProjection,
    context: &'a WorkspaceContext,
}
impl ProjectionWork<'_> {
    fn run(self, mut source: PagedKeyValueSource<'_>) -> Result<(), CacheSourceFailure> {
        let context = self.context;
        let pending = self.pending;
        // Validate every selected tier before acquiring a pin or cloning any
        // handle. Host/disk cached shapes cannot impersonate resident storage.
        source
            .visit_device_arrays(|_| Ok(()))
            .map_err(|cause| CacheSourceFailure::source(cause, context))?;
        pending.geometry = Some(source.prepare_geometry(context)?);
        pending.blocks = context
            .metadata_vec(source.block_count())
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let count = source
            .array_count()
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let mut projection = ExistingArrayProjection::with_source_count(context, count)
            .map_err(|cause| CacheSourceFailure::projection(cause.into(), context))?;
        // This destination is owned outside the lexical manager callback.
        pending.pins = Some(source.blocks.pin_selected(context)?);
        for block in source.blocks.blocks() {
            let arrays = block.device().ok_or_else(|| {
                CacheSourceFailure::source(CacheSourceError::PromotionRequired, context)
            })?;
            pending
                .blocks
                .push(project_pair(&mut projection, arrays, context)?);
        }
        if let Some(arrays) = source.tail_arrays() {
            pending.tail = Some(project_pair(&mut projection, arrays, context)?);
        }
        projection
            .retain_prepared_storage(&mut pending.storage)
            .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
        Ok(())
    }
}
impl PagedKeyValueCache {
    /// Exact outer callback/loan controls, separate from the source-derived
    /// block/pin/import query below. Both are consumed by the same worker.
    pub(crate) fn device_projection_loan_control_bytes() -> Option<usize> {
        let frame = size_of::<SourceContinuation<'_, ProjectionWork<'_>>>();
        Self::workspace_source_fixed_bytes(frame)?.checked_add(
            CacheResidencyManager::source_loan_control_bytes::<()>(frame)?,
        )
    }
    pub(crate) fn project_device_workspace(
        &self,
        context: &WorkspaceContext,
    ) -> Result<ProjectedPagedCacheSource, PagedWorkspaceProjectionFailure> {
        let mut pending = PendingProjection {
            geometry: None,
            blocks: Vec::new(),
            tail: None,
            storage: None,
            pins: None,
            context: context.clone(),
        };
        let admission = projection_fixed_bytes()
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))
            .and_then(|fixed| {
                context
                    .charge_metadata(fixed)
                    .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))
            });
        if let Err(cause) = admission {
            return Err(PagedWorkspaceProjectionFailure {
                cause,
                retained: pending,
            });
        }
        let work = ProjectionWork {
            pending: &mut pending,
            context,
        };
        let result = self.with_workspace_source(context, move |source| work.run(source));
        // No manager guard exists from this point, including on error/unwind.
        if let Err(cause) = result {
            return Err(PagedWorkspaceProjectionFailure {
                cause,
                retained: pending,
            });
        }
        Ok(ProjectedPagedCacheSource {
            geometry: pending.geometry.take().expect("successful source geometry"),
            blocks: std::mem::take(&mut pending.blocks),
            tail: pending.tail.take(),
            storage: pending.storage.take().expect("successful native inventory"),
            pins: pending
                .pins
                .take()
                .expect("successful canonical source pins"),
            _context: pending.context,
        })
    }
}
impl PagedKeyValueSource<'_> {
    pub(crate) fn array_count(&self) -> Option<usize> {
        let mut count = usize::from(self.tail.is_some());
        for block in self.blocks.blocks() {
            if block.device().is_some() || block.host().is_some() {
                count = count.checked_add(1)?;
            }
        }
        count.checked_mul(2)
    }
    /// Exact visited-operand imports and final containers. Aliases are counted
    /// conservatively once per operand by the existing projection producer and
    /// share one actual physical root/retained native handle during construction.
    pub(crate) fn device_projection_control_bytes(
        &self,
        context: &WorkspaceContext,
    ) -> Result<usize, CacheSourceFailure> {
        let overflow = || CacheSourceFailure::source(CacheSourceError::Overflow, context);
        let count = self.array_count().ok_or_else(overflow)?;
        let mut bytes = projection_fixed_bytes().ok_or_else(overflow)?;
        for value in [
            self.geometry_control_bytes(),
            PinnedCacheSource::control_bytes(self.block_count()),
            WorkspaceContext::metadata_vec_bytes::<[WorkspaceTensor; 2]>(self.block_count()),
            ProjectionInventoryLayout::new(count).map(|layout| layout.requested_bytes()),
        ] {
            bytes = bytes
                .checked_add(value.ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let mut add = |array: &Array| -> Result<(), CacheSourceFailure> {
            let layout = ProjectionSourceLayout::inspect(array)
                .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
            bytes = bytes
                .checked_add(layout.requested_bytes())
                .ok_or_else(overflow)?;
            Ok(())
        };
        for block in self.blocks.blocks() {
            let arrays = block.device().ok_or_else(|| {
                CacheSourceFailure::source(CacheSourceError::PromotionRequired, context)
            })?;
            for array in arrays {
                add(array)?;
            }
        }
        if let Some(arrays) = self.tail_arrays() {
            for array in arrays {
                add(array)?;
            }
        }
        Ok(bytes)
    }
    fn visit_device_arrays(
        &self,
        mut visit: impl FnMut(&Array) -> Result<(), CacheSourceError>,
    ) -> Result<(), CacheSourceError> {
        for block in self.blocks.blocks() {
            let arrays = block.device().ok_or(CacheSourceError::PromotionRequired)?;
            for array in arrays {
                visit(array)?;
            }
        }
        if let Some(arrays) = self.tail_arrays() {
            for array in arrays {
                visit(array)?;
            }
        }
        Ok(())
    }
}
fn project_pair<'a>(
    projection: &mut ExistingArrayProjection<'a>,
    arrays: [&'a Array; 2],
    context: &WorkspaceContext,
) -> Result<[WorkspaceTensor; 2], CacheSourceFailure> {
    let first = projection
        .project_prepared(arrays[0])
        .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
    let second = projection
        .project_prepared(arrays[1])
        .map_err(|cause| CacheSourceFailure::projection(cause, context))?;
    Ok([first, second])
}
fn projection_fixed_bytes() -> Option<usize> {
    let parts = [
        size_of::<ProjectedPagedCacheSource>(),
        size_of::<PagedWorkspaceProjectionFailure>(),
        size_of::<PendingProjection>(),
        size_of::<ProjectionWork<'_>>(),
        size_of::<Result<ProjectedPagedCacheSource, PagedWorkspaceProjectionFailure>>(),
        size_of::<[WorkspaceTensor; 2]>(),
        size_of::<Option<[WorkspaceTensor; 2]>>(),
        size_of::<Vec<[WorkspaceTensor; 2]>>(),
        size_of::<ProjectionSourceError>(),
        size_of::<Result<[WorkspaceTensor; 2], CacheSourceFailure>>(),
        size_of::<Result<(), CacheSourceFailure>>(),
        size_of::<(usize, usize)>(),
        size_of::<Option<ProjectedNativeStorage>>(),
        size_of::<Option<PinnedCacheSource>>(),
        size_of::<WorkspaceContext>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn validate_paged_source(
    pins: &PinnedCacheSource,
    geometry: &PagedCacheSourceGeometry,
    storage: &ProjectedNativeStorage,
    source: &PagedKeyValueSource<'_>,
    retained: Option<&ProjectedPagedSource>,
) -> Result<(), CacheSourceError> {
    pins.validate_loan(&source.blocks)?;
    if let Some(retained) = retained {
        retained.validate_files(&source.blocks)?;
    }
    let cache = source.cache;
    if geometry.offset != cache.offset
        || geometry.tail_start != cache.tail_start
        || geometry.global_layer != cache.global_layer
        || geometry.rank != cache.rank
        || geometry.sliding_window != cache.sliding_window
        || geometry.prefix_tokens != cache.prefix_tokens
        || geometry.key_only != cache.key_only
        || geometry.tail != source.tail
    {
        return Err(CacheSourceError::Identity);
    }
    let validate_array = |array: &Array| {
        let descriptor = array.try_descriptor()?;
        let actual = descriptor
            .facts()
            .allocation()
            .ok_or(CacheSourceError::PendingStorage)?;
        let retained = storage
            .native_array(actual.identity())
            .ok_or(CacheSourceError::Identity)?;
        let retained = retained.try_descriptor()?;
        if retained.facts().allocation() != Some(actual) {
            return Err(CacheSourceError::Identity);
        }
        Ok(())
    };
    for block in source.blocks.blocks() {
        if let Some(arrays) = block.device() {
            for array in arrays {
                validate_array(array)?;
            }
        } else if let Some(buffers) = block.host() {
            for buffer in buffers {
                let descriptor = buffer.try_fixed_descriptor::<4>()?;
                let retained = storage
                    .native_host(descriptor.allocation().identity())
                    .ok_or(CacheSourceError::Identity)?;
                if !std::sync::Arc::ptr_eq(buffer, retained)
                    || retained.try_fixed_descriptor::<4>()?.allocation() != descriptor.allocation()
                {
                    return Err(CacheSourceError::Identity);
                }
            }
        } else {
            let retained = retained.ok_or(CacheSourceError::PromotionRequired)?;
            let file = retained
                .retained_file(block.id())
                .ok_or(CacheSourceError::Identity)?;
            if block.phase() != CacheStoragePhase::DiskReady
                || block
                    .disk()
                    .and_then(|disk| disk.live_file())
                    .is_none_or(|actual| !file.same_source(actual))
            {
                return Err(CacheSourceError::Identity);
            }
        }
    }
    if let Some(arrays) = source.tail_arrays() {
        for array in arrays {
            validate_array(array)?;
        }
    }
    Ok(())
}
