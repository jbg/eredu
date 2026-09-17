//! One owned physical-identity collector across short-lived source loans.
use super::*;
use eredu_nn::workspace::WorkspaceLayoutView;

/// Borrowed control of an externally retained destination. A cache manager loan
/// may end immediately after `project_prepared`: only its exact prepared native
/// handle is retained. Source/pin authority remains with the enclosing owner.
pub(crate) struct OwnedArrayProjection<'a> {
    context: &'a WorkspaceContext,
    storage: &'a mut ProjectedNativeStorage,
    maximum: usize,
    paged_maximum: Option<usize>,
    copy_maximum: Option<usize>,
}
impl<'a> OwnedArrayProjection<'a> {
    fn frame_bytes() -> usize {
        size_of::<(
            Self,
            &mut Option<ProjectedNativeStorage>,
            &WorkspaceContext,
            usize,
            ProjectedNativeStorage,
            Result<Self, ProjectionSourceError>,
        )>()
    }
    /// Actual final row storage and constructor controls. No borrowed row table
    /// or second native clone population is constructed by this destination.
    pub(crate) fn construction_bytes(sources: usize) -> Option<usize> {
        WorkspaceContext::metadata_vec_bytes::<ProjectionRow<ProjectionNativeSource>>(sources)?
            .checked_add(Self::frame_bytes())
    }
    /// Exact per-source descriptor/root/clone controls; shape/root storage is
    /// supplied separately by the same ProjectionSourceLayout worker.
    pub(crate) fn import_control_bytes() -> Option<usize> {
        ExistingArrayProjection::import_frame_bytes()?
            .checked_add(Self::import_frames())?
            .checked_add(
                preparation::storage_control_bytes::<ProjectionNativeSource>(size_of::<(
                    &Array,
                    &WorkspaceContext,
                    &Array,
                    &WorkspaceContext,
                )>())?,
            )?
            .checked_add(ExistingArrayProjection::clone_control_bytes()?)
    }
    /// Same actual descriptor-derived shape/root census with this owner's
    /// import controls replacing the borrowed-to-owned adapter controls.
    pub(crate) fn source_bytes(array: &Array) -> Result<usize, ProjectionSourceError> {
        let ordinary = ExistingArrayProjection::import_control_bytes()
            .and_then(|bytes| bytes.checked_add(ExistingArrayProjection::clone_control_bytes()?))
            .ok_or(ProjectionSourceError::Overflow)?;
        ProjectionSourceLayout::inspect(array)?
            .requested_bytes()
            .checked_sub(ordinary)
            .and_then(|bytes| bytes.checked_add(Self::import_control_bytes()?))
            .ok_or(ProjectionSourceError::Overflow)
    }
    fn import_frames() -> usize {
        size_of::<(
            &mut Self,
            &Array,
            &WorkspaceContext,
            Result<WorkspaceTensor, ProjectionSourceError>,
        )>()
    }
    /// Installs the destination before entering any source loan. Every accepted
    /// native prefix remains there after a later import failure or unwind.
    pub(crate) fn prepare(
        destination: &'a mut Option<ProjectedNativeStorage>,
        context: &'a WorkspaceContext,
        sources: usize,
    ) -> Result<Self, ProjectionSourceError> {
        if destination.is_some() {
            return Err(ProjectionInventoryError::Destination.into());
        }
        context
            .charge_metadata(Self::frame_bytes())
            .map_err(Error::from)?;
        let rows = context.metadata_vec(sources)?;
        *destination = Some(ProjectedNativeStorage {
            rows,
            known: 0,
            paged_sources: Vec::new(),
            copy_sources: Vec::new(),
            _host_preparation: None,
            _funding: context.metadata_funding(),
        });
        Ok(Self {
            context,
            storage: destination.as_mut().expect("installed destination"),
            maximum: sources,
            paged_maximum: None,
            copy_maximum: None,
        })
    }
    pub(crate) fn prepare_with_host(
        destination: &'a mut Option<ProjectedNativeStorage>,
        context: &'a WorkspaceContext,
        sources: usize,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self, ProjectionSourceError> {
        let projection = Self::prepare(destination, context, sources)?;
        projection.storage._host_preparation = Some(host.clone());
        Ok(projection)
    }
    pub(crate) fn context(&self) -> &'a WorkspaceContext {
        self.context
    }
    pub(crate) fn storage(&self) -> &ProjectedNativeStorage {
        self.storage
    }
    /// Reserves the actual selected pager count before entering any manager
    /// loan. A canonical source is installed before its arrays are imported.
    pub(crate) fn prepare_paged_sources(
        &mut self,
        count: usize,
    ) -> Result<(), ProjectionSourceError> {
        self.context
            .charge_metadata(size_of::<(
                &mut Self,
                usize,
                Vec<crate::backend::runtime::cache::kv::ProjectedPagedSource>,
                Result<(), ProjectionSourceError>,
            )>())
            .map_err(Error::from)?;
        if self.paged_maximum.is_some() {
            return Err(ProjectionInventoryError::Destination.into());
        }
        self.storage.paged_sources = self.context.metadata_vec(count)?;
        self.paged_maximum = Some(count);
        Ok(())
    }
    /// Capacity is checked BEFORE acquisition. No fallible operation, callback
    /// or allocation follows the returned pin until it is in the outer owner.
    pub(crate) fn retain_paged_source<F>(
        &mut self,
        acquire: F,
    ) -> Result<(), crate::backend::runtime::cache::residency::CacheSourceFailure>
    where
        F: FnOnce() -> Result<
            crate::backend::runtime::cache::kv::ProjectedPagedSource,
            crate::backend::runtime::cache::residency::CacheSourceFailure,
        >,
    {
        use crate::backend::runtime::cache::residency::{CacheSourceError, CacheSourceFailure};
        self.context
            .charge_metadata(size_of::<(
                &mut Self,
                F,
                crate::backend::runtime::cache::kv::ProjectedPagedSource,
                Result<(), CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), self.context))?;
        if self
            .paged_maximum
            .is_none_or(|maximum| self.storage.paged_sources.len() >= maximum)
        {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Overflow,
                self.context,
            ));
        }
        let source = acquire()?;
        self.storage.paged_sources.push(source);
        Ok(())
    }
    /// Retains a prepared host promotion immediately in the already installed
    /// exact layer source. No returned pin can retire under the manager loan.
    pub(crate) fn retain_paged_promotion<F>(
        &mut self,
        layer: usize,
        id: &eredu_core::cache::CacheBlockId,
        acquire: F,
    ) -> Result<(), crate::backend::runtime::cache::residency::CacheSourceFailure>
    where
        F: FnOnce() -> Result<
            crate::backend::runtime::cache::residency::PreparedCacheHostPromotion,
            crate::backend::runtime::cache::residency::CacheSourceFailure,
        >,
    {
        use crate::backend::runtime::cache::residency::{CacheSourceError, CacheSourceFailure};
        self.context
            .charge_metadata(size_of::<(
                &mut Self,
                usize,
                &eredu_core::cache::CacheBlockId,
                F,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), self.context))?;
        let source = self
            .storage
            .paged_sources
            .last_mut()
            .filter(|source| source.geometry().global_layer == layer)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, self.context))?;
        source.retain_promotion(id, acquire, self.context)
    }

    /// Same exact shape/dtype/completed-allocation validation as the borrowed
    /// collector. The common sorted insertion worker deduplicates aliases even
    /// when their original manager loans have already ended.
    pub(crate) fn project_prepared(
        &mut self,
        array: &Array,
    ) -> Result<WorkspaceTensor, ProjectionSourceError> {
        self.context
            .charge_metadata(
                ExistingArrayProjection::import_frame_bytes()
                    .and_then(|bytes| bytes.checked_add(Self::import_frames()))
                    .ok_or(ProjectionSourceError::Overflow)?,
            )
            .map_err(Error::from)?;
        let descriptor = array.try_descriptor()?;
        let dtype = projected_dtype(descriptor.facts().dtype())?;
        let layout = WorkspaceLayoutView::new(descriptor.shape(), dtype)?
            .with_representation(representation::from_descriptor(&descriptor));
        let allocation = descriptor
            .facts()
            .allocation()
            .ok_or(ProjectionSourceError::UnknownBacking)?;
        let context = self.context;
        let storage = preparation::storage_for(
            &mut self.storage.rows,
            &mut self.storage.known,
            Some(self.maximum),
            Some(allocation),
            context,
            || preparation::clone_array(array, context).map(ProjectionNativeSource::array),
            |source| {
                if source.array.is_none() {
                    source.array = Some(preparation::clone_array(array, context)?);
                }
                Ok(())
            },
        )?;
        Ok(WorkspaceTensor::import_existing(layout, &storage, context)?)
    }
}

#[cfg(test)]
mod tests;

impl OwnedArrayProjection<'_> {
    /// Same final native owner; canonical pins retire after all array aliases.
    pub(crate) fn copy_source_control_bytes(count: usize) -> Option<usize> {
        WorkspaceContext::metadata_vec_bytes::<
            crate::backend::runtime::cache::residency::PinnedCacheSource,
        >(count)?
        .checked_add(size_of::<(&mut Self, usize)>())
    }
    pub(crate) fn prepare_copy_sources(
        &mut self,
        count: usize,
    ) -> Result<(), ProjectionSourceError> {
        if self.copy_maximum.is_some() {
            return Err(ProjectionInventoryError::Destination.into());
        }
        self.context
            .charge_metadata(size_of::<(&mut Self, usize)>())
            .map_err(Error::from)?;
        self.storage.copy_sources = self.context.metadata_vec(count)?;
        self.copy_maximum = Some(count);
        Ok(())
    }
    pub(crate) fn retain_copy_source<F>(
        &mut self,
        acquire: F,
    ) -> Result<(), crate::backend::runtime::cache::residency::CacheSourceFailure>
    where
        F: FnOnce() -> Result<
            crate::backend::runtime::cache::residency::PinnedCacheSource,
            crate::backend::runtime::cache::residency::CacheSourceFailure,
        >,
    {
        use crate::backend::runtime::cache::residency::{CacheSourceError, CacheSourceFailure};
        if self
            .copy_maximum
            .is_none_or(|count| self.storage.copy_sources.len() >= count)
        {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                self.context,
            ));
        }
        // No failure point after acquisition until the pin is outside its manager guard.
        let source = acquire()?;
        self.storage.copy_sources.push(source);
        Ok(())
    }
}

mod host;
