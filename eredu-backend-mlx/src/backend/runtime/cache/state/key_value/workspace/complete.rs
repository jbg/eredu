//! Complete mixed dense/paged projection with one physical identity collector.
use super::*;
use crate::backend::{
    nn::{
        shared::MlxNeuralBackend,
        workspace::{OwnedArrayProjection, ProjectedNativeStorage},
    },
    runtime::cache::residency::{CacheSourceError, CacheSourceFailure},
};
use eredu_runtime::RuntimeLayerState;
use std::mem::{size_of, size_of_val};

/// A rejected projection retains accepted native rows and canonical source pins
/// outside every manager loan. This native result has no submission authority.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CompleteStateProjectionFailure {
    #[source]
    cause: CacheSourceFailure,
    retained: Option<ProjectedNativeStorage>,
}
impl std::fmt::Debug for CompleteStateProjectionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompleteStateProjectionFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl CompleteStateProjectionFailure {
    pub(in crate::backend::runtime::cache::state) fn without_storage(
        cause: CacheSourceFailure,
    ) -> Self {
        Self {
            cause,
            retained: None,
        }
    }
    pub(crate) fn cause(&self) -> &CacheSourceFailure {
        &self.cause
    }
    pub(crate) fn retained_storage(&self) -> Option<&ProjectedNativeStorage> {
        self.retained.as_ref()
    }
    /// Legacy pre-submission callers may tear down a failed prefix here. No
    /// manager loan is live, no native equation has been enqueued, and the exact
    /// portable cause retains its H through this ordered teardown.
    pub(in crate::backend::runtime::cache::state) fn retire_unsubmitted(
        self,
        context: &WorkspaceContext,
    ) -> eredu_nn::Error {
        let Self { cause, retained } = self;
        let error = context.metadata_source(cause);
        drop(retained);
        error
    }
}
mod shared;
pub(in crate::backend::runtime::cache::state) use shared::{SourceCounts, project_complete};
impl MlxKeyValueState {
    /// Projects actual completed source leaves, including paged managers. Source
    /// pins and native witnesses remain distinct from metadata and from native
    /// copy/execution authority. All layers share the same physical identity set.
    pub(crate) fn project_complete_workspace_with_storage(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, CompleteStateProjectionFailure> {
        super::super::prepared_copy::KvCopySource::Live(self)
            .project_complete_workspace_with_storage(batch, context)
    }
}
impl super::super::prepared_copy::KvCopySource<'_> {
    pub(in crate::backend::runtime::cache::state::key_value) fn project_complete_workspace_with_storage(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, CompleteStateProjectionFailure> {
        project_complete(
            self.shared_layout(),
            self.len(),
            batch,
            context,
            || self.complete_source_counts(context),
            |index, policy, factory, projection| {
                self.project_complete_layer(index, policy, batch, context, factory, projection)
            },
        )
    }

    fn complete_source_counts(
        &self,
        context: &WorkspaceContext,
    ) -> Result<SourceCounts, CacheSourceFailure> {
        context
            .charge_metadata(size_of::<(
                &Self,
                &WorkspaceContext,
                SourceCounts,
                Result<SourceCounts, CacheSourceFailure>,
                usize,
                <ConcatKeyValueCache as RuntimeLayerState<MlxNeuralBackend>>::RetainedValues<'_>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let mut counts = SourceCounts::default();
        for index in 0..self.len() {
            let layer = self
                .layer(index)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, context))?;
            let arrays = match layer {
                MlxKeyValueLayerState::Stateless => 0,
                MlxKeyValueLayerState::Device(cache) => {
                    RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache).count()
                }
                MlxKeyValueLayerState::Paged(cache) => {
                    counts.pagers = counts.pagers.checked_add(1).ok_or_else(|| {
                        CacheSourceFailure::source(CacheSourceError::Overflow, context)
                    })?;
                    cache.with_workspace_source(context, |source| {
                        source.array_count().ok_or_else(|| {
                            CacheSourceFailure::source(CacheSourceError::Overflow, context)
                        })
                    })?
                }
            };
            counts.arrays = counts
                .arrays
                .checked_add(arrays)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        }
        Ok(counts)
    }
    fn project_complete_layer(
        &self,
        index: usize,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        context: &WorkspaceContext,
        factory: &WorkspaceConcatStateFactory,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspaceResidentLayerState, CacheSourceFailure> {
        context
            .charge_metadata(size_of::<(
                &Self,
                usize,
                &LayerCachePolicy,
                NonZeroU32,
                &WorkspaceContext,
                &WorkspaceConcatStateFactory,
                &mut OwnedArrayProjection<'_>,
                &MlxKeyValueLayerState,
                Result<WorkspaceResidentLayerState, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let layer = self
            .layer(index)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, context))?;
        if let MlxKeyValueLayerState::Paged(cache) = layer {
            let global = self
                .global_layer_start()
                .checked_add(index)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
            return cache
                .project_workspace_layer_into(index, global, policy, batch, projection)
                .map(WorkspaceResidentLayerState::Paged);
        }
        let (position, keys, values) = layer
            .project_workspace_attention_with(policy, context, |array| {
                projection
                    .project_prepared(array)
                    .map_err(|cause| context.metadata_source(cause))
            })
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        factory
            .project_layer(index, policy, position, keys, values, [])
            .map(WorkspaceResidentLayerState::Ordinary)
            .map_err(|cause| {
                CacheSourceFailure::metadata(cause.into_workspace_error(context), context)
            })
    }
}

#[cfg(test)]
#[path = "complete/tests.rs"]
mod tests;
