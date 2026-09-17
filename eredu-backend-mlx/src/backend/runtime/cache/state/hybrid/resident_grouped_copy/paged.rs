//! Joined paged storage and fixed-role state; both retain their actual owners.
use super::super::super::key_value::{PagedSnapshotLayer, PagedSnapshotSource, PagedSnapshotState};
pub(super) use super::super::super::key_value::{PagedWork, PreparedPagedStorageCopy};
use super::super::super::{SnapshotArraySources, SnapshotOperand, SnapshotProjectionCause};
use super::*;
use crate::backend::nn::workspace::{OwnedArrayProjection, ProjectedResidentState};
use crate::backend::runtime::cache::residency::{CacheSourceError, CacheSourceFailure};
use std::mem::{size_of, size_of_val};

pub(super) fn attention(source: Option<&MlxHybridAttentionState>) -> PagedSnapshotLayer<'_> {
    match source {
        None => PagedSnapshotLayer::Absent,
        Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Stateless)) => {
            PagedSnapshotLayer::Absent
        }
        Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
            PagedSnapshotLayer::Paged(cache)
        }
        _ => PagedSnapshotLayer::Other,
    }
}
impl PagedSnapshotState for MlxHybridState {
    fn layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    fn layer_count(&self) -> usize {
        self.layers.len()
    }
    fn global_start(&self) -> usize {
        self.global_layer_start
    }
    fn layer(&self, index: usize) -> Option<PagedSnapshotLayer<'_>> {
        self.layers
            .slots()
            .get(index)
            .map(|layer| attention(layer.attention.as_ref()))
    }
}
impl PagedSnapshotState for SavedHybridGroupedCopy {
    fn layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    fn layer_count(&self) -> usize {
        self.layers.len()
    }
    fn global_start(&self) -> usize {
        self.global_layer_start
    }
    fn layer(&self, index: usize) -> Option<PagedSnapshotLayer<'_>> {
        self.layers
            .get(index)
            .map(|layer| attention(layer.attention.as_ref()))
    }
}
impl<'a> Source<'a> {
    fn paged_state(self) -> &'a dyn PagedSnapshotState {
        match self {
            Self::Live(source) => source,
            Self::Saved(source) => source,
        }
    }
}
impl<'a> PreparedHybridGroupedCopy<'a> {
    pub(super) fn prepare_source(
        source: Source<'a>,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        let paged = PagedSnapshotSource::prepare(source.paged_state())
            .map_err(super::super::super::key_value::PagedKvPreparationError::from)?
            .map(PreparedPagedStorageCopy::inspect)
            .transpose()?;
        if let (Source::Live(live), Some(paged)) = (source, paged.as_ref()) {
            if live
                .manager
                .as_ref()
                .is_none_or(|manager| !manager.same_catalog(paged.snapshot().manager()))
            {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
        }
        let plan = Self { source, paged };
        plan.validate_fixed()?;
        Ok(plan)
    }
    pub(crate) fn is_paged(&self) -> bool {
        self.paged.is_some()
    }
    pub(crate) fn logical_snapshot_manager_bytes(&self) -> Option<u64> {
        self.paged.as_ref().map_or(
            Some(0),
            PreparedPagedStorageCopy::logical_snapshot_manager_bytes,
        )
    }
    fn visit_fixed_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or(SnapshotProjectionCause::Changed)?;
            for (_, value) in layer.fixed.iter() {
                if let Some(value) = value {
                    visitor(value.as_array())?;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn visit_paged_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), SnapshotProjectionCause> {
        <Self as SnapshotArraySources>::visit_arrays(self, &mut |array| {
            visitor(array);
            Ok(())
        })
    }
    pub(super) fn project_paged_workspace(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedHybridGroupedCopy, Error> {
        use eredu_runtime::RuntimeLayerState;
        let projected = super::super::super::key_value::project_complete(
            self.shared_layout(),
            self.len(),
            batch,
            context,
            || self.paged_source_counts(context),
            |index, policy, factory, projection| {
                let layer = self.layer(index).ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Identity, context)
                })?;
                if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(
                    cache,
                ))) = layer.attention
                {
                    return super::super::workspace::complete::project_paged_layer(
                        index,
                        self.global_layer_start(),
                        policy,
                        batch,
                        cache,
                        layer.fixed.iter(),
                        projection,
                    );
                }
                super::super::workspace::project_ordinary_layer_with(
                    index,
                    policy,
                    layer.attention,
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
            },
        )
        .map_err(|failure| Error::Neural(failure.retire_unsubmitted(context)))?;
        let layers: &[WorkspaceResidentLayerState] = projected.state.as_ref();
        let count = layers
            .iter()
            .try_fold(0usize, |n, layer| {
                n.checked_add(layer.retained_values().count())
            })
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        context
            .charge_metadata(size_of::<(
                &Self,
                NonZeroU32,
                &WorkspaceContext,
                ProjectedResidentState,
                Vec<eredu_nn::workspace::WorkspaceTensor>,
                Result<ProjectedHybridGroupedCopy, Error>,
                usize,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut retained = context.metadata_vec(count).map_err(Error::Neural)?;
        for layer in layers {
            retained.extend(layer.retained_values().cloned());
        }
        context.begin_state_span(&retained).map_err(Error::Neural)?;
        let state=DeviceState::<WorkspaceBackend,WorkspaceResidentLayerState>::create_workspace_with_shared_layout_result(
            self.shared_layout().clone(),context,|index,policy|->Result<_,eredu_nn::Error>{
                let copied=match layers.get(index) {
                    Some(WorkspaceResidentLayerState::Paged(paged))=>WorkspaceResidentLayerState::Paged(paged.copy_for_resume(context)?),
                    // Actual fixed-only siblings use the existing ordinary state copy.
                    Some(WorkspaceResidentLayerState::Ordinary(ordinary))=> {
                        use eredu_runtime::RuntimeStateComponents;
                        if !matches!(policy,LayerCachePolicy::NoState|LayerCachePolicy::FixedState{..}) {
                            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
                        }
                        let mut copied=ordinary.clone();
                        for declaration in policy.fixed_state() {
                            let value=copied.fixed_component(declaration.role)
                                .map_err(|cause|cause.into_workspace_error(context))?;
                            if let Some(source)=value {
                                *source=eredu_nn::isolated_copy(eredu_nn::WorkspaceIsolatedCopy::new(source.clone(),context))?;
                            }
                        }
                        WorkspaceResidentLayerState::Ordinary(copied)
                    },
                    _=>return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch)),
                };
                for value in copied.retained_values(){
                    if retained.len()==count{return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));}
                    retained.push(value.clone());
                }
                Ok(copied)
            }).map_err(Error::Neural)?;
        if retained.len() != count {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let copy = context.finish_report(&retained).map_err(Error::Neural)?;
        Ok(ProjectedHybridGroupedCopy {
            state,
            source_storage: projected.storage,
            copy,
        })
    }
    fn paged_source_counts(
        &self,
        context: &WorkspaceContext,
    ) -> Result<super::super::super::key_value::SourceCounts, CacheSourceFailure> {
        let mut counts = super::super::super::key_value::SourceCounts::default();
        context
            .charge_metadata(size_of::<(
                &Self,
                &WorkspaceContext,
                super::super::super::key_value::SourceCounts,
                std::ops::Range<usize>,
                Result<super::super::super::key_value::SourceCounts, CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        for i in 0..self.len() {
            let layer = self
                .layer(i)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Identity, context))?;
            let row = super::super::workspace::complete::source_layer_counts(
                layer.attention,
                layer.fixed.iter(),
                context,
            )?;
            counts.pagers = counts
                .pagers
                .checked_add(row.pagers)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
            counts.arrays = counts
                .arrays
                .checked_add(row.arrays)
                .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        }
        Ok(counts)
    }
}
impl SnapshotArraySources for PreparedHybridGroupedCopy<'_> {
    fn visit_arrays(
        &self,
        visitor: &mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        let paged = self
            .paged
            .as_ref()
            .ok_or(SnapshotProjectionCause::Changed)?;
        paged.snapshot().visit_arrays(visitor)?;
        self.visit_fixed_arrays(visitor)
    }
    fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(SnapshotOperand<'_>) -> Result<(), SnapshotProjectionCause>,
    ) -> Result<(), SnapshotProjectionCause> {
        let paged = self
            .paged
            .as_ref()
            .ok_or(SnapshotProjectionCause::Changed)?;
        paged.snapshot().visit_operands(visitor)?;
        self.visit_fixed_arrays(&mut |array| visitor(SnapshotOperand::Array(array)))
    }
    fn copy_source_count(&self) -> usize {
        self.paged
            .as_ref()
            .map_or(0, |paged| paged.snapshot().copy_source_count())
    }
    fn source_controls(&self) -> Result<usize, SnapshotProjectionCause> {
        let frames = [
            size_of::<&Self>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Layer<'_>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<(), SnapshotProjectionCause>>(),
            size_of::<Result<(), SnapshotProjectionCause>>(),
        ];
        let source = self
            .paged
            .as_ref()
            .ok_or(SnapshotProjectionCause::Changed)?
            .snapshot()
            .source_controls()?;
        frames
            .into_iter()
            .try_fold(
                source
                    .checked_add(size_of_val(&frames))
                    .ok_or(SnapshotProjectionCause::Overflow)?,
                usize::checked_add,
            )
            .ok_or(SnapshotProjectionCause::Overflow)
    }
    fn pin_sources(
        &self,
        projection: &mut OwnedArrayProjection<'_>,
    ) -> Result<(), SnapshotProjectionCause> {
        self.paged
            .as_ref()
            .ok_or(SnapshotProjectionCause::Changed)?
            .snapshot()
            .pin_sources(projection)
    }
}

impl PreparedHybridGroupedCopy<'_> {
    pub(super) fn paged_caller_controls(&self) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<SavedHybridGroupedCopy>(),
            size_of::<Result<SavedHybridGroupedCopy, Error>>(),
            size_of::<InitializedHybridGroupCopy>(),
            size_of::<Option<PagedWork>>(),
            size_of::<Layer<'_>>(),
            size_of::<Fixed<'_>>(),
            size_of::<(&Self, Option<&eredu_core::HostPreparationAuthority>)>(),
            size_of::<(
                &Self,
                usize,
                &mut Option<PagedWork>,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
            )>(),
            size_of::<Result<Option<MlxHybridAttentionState>, Error>>(),
            size_of::<Option<MlxHybridAttentionState>>(),
            size_of::<Option<Error>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Result<(), Error>>(),
            size_of::<PreparedHybridGroupHostCopy<'_>>(),
            size_of::<Result<Option<PagedWork>, Error>>(),
            size_of::<super::dense::PreparedHybridDenseGroup<'_>>(),
            size_of::<super::dense::InitializedHybridDenseGroup<'_>>(),
            size_of::<super::dense::PreparedDenseHybridGroupedState<'_>>(),
            size_of::<Option<CacheResidencyManager>>(),
            size_of::<Result<super::dense::PreparedDenseHybridGroupedState<'_>, Error>>(),
            size_of::<(
                &Self,
                InitializedHybridGroupCopy,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
                &mut dyn FnMut(
                    &crate::backend::array_copy::PreparedSavedHostCopy,
                ) -> Result<(), Error>,
            )>(),
            size_of::<(
                &Self,
                super::dense::InitializedHybridDenseGroup<'_>,
                &Stream,
                &RefCell<Vec<Array>>,
                &mut dyn FnMut(&Array) -> Result<(), Error>,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(super) fn prepare_paged_work(
        &self,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<Option<PagedWork>, Error> {
        self.paged
            .as_ref()
            .map(|source| {
                source.prepare_work(
                    host.ok_or_else(unknown)?,
                    self.paged_caller_controls()
                        .ok_or_else(|| other(WorkingMemoryError::Overflow))?,
                )
            })
            .transpose()
    }
    pub(super) fn copy_attention_with_paged(
        &self,
        i: usize,
        paged: &mut Option<PagedWork>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe: &mut dyn FnMut(&Array) -> Result<(), Error>,
    ) -> Result<Option<MlxHybridAttentionState>, Error> {
        match self.layer(i).ok_or_else(mismatch)?.attention {
            Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache))) => {
                let work = paged.as_mut().ok_or_else(mismatch)?;
                work.copy_tail(cache, stream, roots, observe).map(|cache| {
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Paged(cache),
                    ))
                })
            }
            _ => self.copy_attention(i, stream, roots),
        }
    }
}
