//! Pooling sources share the same complete native identity/pin collector.
use super::*;
use crate::backend::{
    nn::workspace::OwnedArrayProjection,
    runtime::cache::{
        residency::{CacheSourceError, CacheSourceFailure},
        state::{
            CompleteStateProjectionFailure,
            key_value::{SourceCounts, project_complete},
        },
    },
};
use std::mem::size_of;

impl MlxPoolingAttentionStateFactory {
    pub(crate) fn project_complete_workspace_with_storage(
        state: &MlxPoolingAttentionState,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, CompleteStateProjectionFailure> {
        context
            .charge_metadata(size_of::<(
                &MlxPoolingAttentionState,
                NonZeroU32,
                &WorkspaceContext,
                Result<ProjectedResidentState, CompleteStateProjectionFailure>,
            )>())
            .map_err(|cause| {
                CompleteStateProjectionFailure::without_storage(CacheSourceFailure::metadata(
                    cause.into(),
                    context,
                ))
            })?;
        let layout = state.shared_layout().ok_or_else(|| {
            CompleteStateProjectionFailure::without_storage(CacheSourceFailure::source(
                CacheSourceError::Geometry,
                context,
            ))
        })?;
        project_complete(
            layout,
            state.as_ref().len(),
            batch,
            context,
            || source_counts(state, context),
            |index, policy, _, projection| {
                state.as_ref()[index].project_complete_layer(index, policy, batch, projection)
            },
        )
    }
}
fn source_counts(
    state: &MlxPoolingAttentionState,
    context: &WorkspaceContext,
) -> Result<SourceCounts, CacheSourceFailure> {
    context
        .charge_metadata(size_of::<(
            &MlxPoolingAttentionState,
            &WorkspaceContext,
            SourceCounts,
            std::slice::Iter<'_, MlxPoolingAttentionCache>,
            usize,
            [Option<&Array>; 5],
            Result<SourceCounts, CacheSourceFailure>,
        )>())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut counts = SourceCounts::default();
    for layer in state.as_ref() {
        let local = match layer.local() {
            LiveKeyValueCache::Resident(cache) => {
                RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache).count()
            }
            LiveKeyValueCache::Paged(cache) => {
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
            .checked_add(local)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let streams = match layer {
            MlxPoolingAttentionCache::Local(_) => 0,
            MlxPoolingAttentionCache::Compressed { .. } => 1,
            MlxPoolingAttentionCache::Sparse { .. } => 2,
        };
        for stream in 0..streams {
            let pool = layer
                .pool(stream)
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
            counts.arrays = counts
                .arrays
                .checked_add(pool.state_arrays().into_iter().flatten().count())
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        }
    }
    Ok(counts)
}
impl MlxPoolingAttentionCache {
    fn project_complete_layer(
        &self,
        index: usize,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspaceResidentLayerState, CacheSourceFailure> {
        let context = projection.context();
        context
            .charge_metadata(size_of::<(
                &Self,
                usize,
                &LayerCachePolicy,
                NonZeroU32,
                &mut OwnedArrayProjection<'_>,
                WorkspacePoolingStateFactory,
                eredu_runtime::working_memory::WorkspacePagedAppendState,
                Result<WorkspaceResidentLayerState, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        // Validate pooling semantics before accepting a source from this layer.
        eredu_runtime::state::pooling_attention_geometry_plan(index, policy, |args| {
            context.metadata_error(args)
        })
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let factory = WorkspacePoolingStateFactory::new(batch, context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        if let LiveKeyValueCache::Paged(cache) = self.local() {
            let paged = cache.project_workspace_pooling_append_into(policy, batch, projection)?;
            let position = i32::try_from(paged.geometry().offset).map_err(|cause| {
                CacheSourceFailure::metadata(context.metadata_source(cause), context)
            })?;
            let components = self
                .project_components_with(index, policy, position, context, |array| {
                    projection
                        .project_prepared(array)
                        .map_err(|cause| context.metadata_source(cause))
                })
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
            factory
                .project_paged_layer(index, policy, paged, components)
                .map(WorkspaceResidentLayerState::Pooling)
                .map_err(|cause| {
                    CacheSourceFailure::metadata(cause.into_workspace_error(context), context)
                })
        } else {
            self.project_workspace_layer(index, policy, &factory, context, |array| {
                projection
                    .project_prepared(array)
                    .map_err(|cause| context.metadata_source(cause))
            })
            .map(WorkspaceResidentLayerState::Pooling)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))
        }
    }
}
#[cfg(test)]
#[path = "complete/tests.rs"]
mod tests;
