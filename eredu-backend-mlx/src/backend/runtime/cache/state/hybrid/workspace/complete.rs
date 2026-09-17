//! Actual hybrid fixed/attention sources share the complete native collector.
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
use eredu_runtime::working_memory::WorkspacePagedLayerState;
use std::mem::{size_of, size_of_val};

impl MlxHybridState {
    pub(crate) fn project_complete_workspace_with_storage(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedResidentState, CompleteStateProjectionFailure> {
        project_complete(
            &self.layout,
            self.layers.len(),
            batch,
            context,
            || self.complete_source_counts(context),
            |index, policy, factory, projection| {
                self.project_complete_layer(index, policy, batch, factory, projection)
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
                std::slice::Iter<'_, MlxHybridLayerState>,
                usize,
                usize,
                Result<SourceCounts, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let mut counts = SourceCounts::default();
        for layer in self.layers.slots() {
            let row = source_layer_counts(layer.attention.as_ref(), layer.fixed.iter(), context)?;
            counts.pagers = counts.pagers.checked_add(row.pagers)
                .ok_or_else(||CacheSourceFailure::source(CacheSourceError::Overflow,context))?;
            counts.arrays = counts.arrays.checked_add(row.arrays)
                .ok_or_else(||CacheSourceFailure::source(CacheSourceError::Overflow,context))?;
        }
        Ok(counts)
    }
    fn project_complete_layer(
        &self,
        index: usize,
        policy: &LayerCachePolicy,
        batch: NonZeroU32,
        factory: &WorkspaceConcatStateFactory,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<WorkspaceResidentLayerState, CacheSourceFailure> {
        let context = projection.context();
        context
            .charge_metadata(size_of::<(
                &Self,
                usize,
                &LayerCachePolicy,
                NonZeroU32,
                &WorkspaceConcatStateFactory,
                &mut OwnedArrayProjection<'_>,
                &WorkspaceContext,
                Result<WorkspaceResidentLayerState, CacheSourceFailure>,
                Vec<(
                    StateTensorRole,
                    Option<eredu_nn::workspace::WorkspaceTensor>,
                )>,
                eredu_runtime::working_memory::WorkspacePagedAppendState,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let layer = &self.layers.slots()[index];
        if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) =
            &layer.attention
        {
            return project_paged_layer(index, self.global_layer_start, policy, batch,
                cache, layer.fixed.iter(), projection);
        }
        if let Some(MlxHybridAttentionState::Compressed(cache)) = &layer.attention {
            return project_compressed_with(cache, &layer.fixed, policy, batch, context, |array| {
                projection
                    .project_prepared(array)
                    .map_err(|cause| context.metadata_source(cause))
            })
            .map_err(|cause| CacheSourceFailure::metadata(cause, context));
        }
        project_ordinary_layer_with(
            index,
            policy,
            layer.attention.as_ref(),
            layer.fixed_offset,
            layer.fixed.iter(),
            factory,
            context,
            |array| {
                projection
                    .project_prepared(array)
                    .map_err(|cause| context.metadata_source(cause))
            },
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))
    }
}

/// One actual attention/fixed source census for live and saved outer tables.
pub(in crate::backend::runtime::cache::state::hybrid) fn source_layer_counts<'a,I>(
    attention:Option<&MlxHybridAttentionState>, fixed:I, context:&WorkspaceContext,
)->Result<SourceCounts,CacheSourceFailure>
where I:Iterator<Item=(&'a StateTensorRole,&'a Option<MlxTensor>)> {
    context.charge_metadata(size_of::<(Option<&MlxHybridAttentionState>,I,&WorkspaceContext,
        SourceCounts,Result<SourceCounts,CacheSourceFailure>,usize)>())
        .map_err(|cause|CacheSourceFailure::metadata(cause.into(),context))?;
    let mut counts=SourceCounts::default();
    let attention=match attention {
        Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache)))=>{
            counts.pagers=1;
            cache.with_workspace_source(context,|source|source.array_count()
                .ok_or_else(||CacheSourceFailure::source(CacheSourceError::Overflow,context)))?
        }
        Some(MlxHybridAttentionState::KeyValue(cache))=>
            RuntimeLayerState::<MlxNeuralBackend>::retained_values(cache).count(),
        Some(MlxHybridAttentionState::Compressed(cache))=>cache.workspace_source_count(context)
            .map_err(|cause|CacheSourceFailure::metadata(cause,context))?,
        None=>0,
    };
    counts.arrays=attention.checked_add(fixed.filter(|(_,value)|value.is_some()).count())
        .ok_or_else(||CacheSourceFailure::source(CacheSourceError::Overflow,context))?;
    Ok(counts)
}
/// The same paged attention/fixed-role projection for either retained table owner.
pub(in crate::backend::runtime::cache::state::hybrid) fn project_paged_layer<'a,I>(
    index:usize,global_start:usize,policy:&LayerCachePolicy,batch:NonZeroU32,
    cache:&PagedKeyValueCache,fixed:I,projection:&mut OwnedArrayProjection<'_>,
)->Result<WorkspaceResidentLayerState,CacheSourceFailure>
where I:Iterator<Item=(&'a StateTensorRole,&'a Option<MlxTensor>)> {
    let context=projection.context();
    context.charge_metadata(size_of::<(usize,usize,&LayerCachePolicy,NonZeroU32,&PagedKeyValueCache,I,
        &mut OwnedArrayProjection<'_>,eredu_runtime::working_memory::WorkspacePagedAppendState,
        Vec<(StateTensorRole,Option<eredu_nn::workspace::WorkspaceTensor>)>,
        Result<WorkspaceResidentLayerState,CacheSourceFailure>)>())
        .map_err(|cause|CacheSourceFailure::metadata(cause.into(),context))?;
    let global=global_start.checked_add(index)
        .ok_or_else(||CacheSourceFailure::source(CacheSourceError::Overflow,context))?;
    let paged=cache.project_workspace_append_into(global,policy,batch,projection)?;
    let fixed=project_fixed_with(policy,fixed,context,|array|projection.project_prepared(array)
        .map_err(|cause|context.metadata_source(cause)))
        .map_err(|cause|CacheSourceFailure::metadata(cause,context))?;
    WorkspacePagedLayerState::project(index,policy,paged,fixed,context)
        .map(WorkspaceResidentLayerState::Paged)
        .map_err(|cause|CacheSourceFailure::metadata(cause.into_workspace_error(context),context))
}

#[cfg(test)]
#[path = "complete/tests.rs"]
mod tests;
