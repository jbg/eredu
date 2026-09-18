//! Actual two-level Hybrid payloads: one outer table and every fixed-role table.
use super::fixed_slots::{Slot, copy_slot_retained};
use super::*;
use crate::backend::runtime::cache::state::ResidentDecoderPreparationError;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    error::Error,
    nn::workspace::{ExistingArrayProjection, ProjectedNativeStorage},
    runtime::residency::storage::StorageIdentity,
};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTraceReport};
use eredu_runtime::{
    HostSlotMetadata, SharedStateLayout,
    working_memory::{
        FundedDecoderSlots, InferenceRetention, WorkingMemoryError, WorkingMemoryPool,
        WorkspaceConcatStateFactory, WorkspaceResidentLayerState,
    },
};
use std::{cell::RefCell, num::NonZeroU32};

mod dense;
mod frozen;
mod paged;
use paged::PreparedPagedStorageCopy;
pub(crate) use dense::{
    InitializedHybridDenseGroup, PreparedDenseHybridGroupedState, PreparedHybridDenseGroup,
    PublishedDenseHybridGroupedState,
};
pub(crate) use frozen::{InitializedHybridGroupCopy, PreparedHybridGroupHostCopy};

fn other(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(e))
}
fn mismatch() -> Error {
    other(WorkingMemoryError::IdentityMismatch)
}
fn unknown() -> Error {
    other(WorkingMemoryError::UnknownBound)
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Live(&'a MlxHybridState),
    Saved(&'a SavedHybridGroupedCopy),
}
#[derive(Clone, Copy)]
enum Fixed<'a> {
    Live(&'a FixedStateSlots),
    Saved(&'a FundedDecoderSlots<Slot>),
}
impl<'a> Fixed<'a> {
    fn len(self) -> usize {
        match self {
            Self::Live(s) => s.values().len(),
            Self::Saved(s) => s.len(),
        }
    }
    fn slot(self, i: usize) -> Option<&'a Slot> {
        match self {
            Self::Live(s) => s.slot(i),
            Self::Saved(s) => s.get(i),
        }
    }
    fn iter(self) -> impl ExactSizeIterator<Item = (&'a StateTensorRole, &'a Option<MlxTensor>)> {
        (0..self.len()).map(move |i| {
            let s = self.slot(i).expect("complete retained role table");
            (&s.0, &s.1)
        })
    }
}
struct Layer<'a> {
    attention: Option<&'a MlxHybridAttentionState>,
    fixed: Fixed<'a>,
    fixed_offset: i32,
}

/// Frozen layers hold their own funded child payload; no live-state/old-request
/// wrapper survives. Attention contains copied values and scalar cache controls.
pub(crate) struct SavedHybridGroupedLayer {
    attention: Option<MlxHybridAttentionState>,
    fixed: FundedDecoderSlots<Slot>,
    fixed_offset: i32,
}
pub(crate) struct SavedHybridGroupedCopy {
    layers: FundedDecoderSlots<SavedHybridGroupedLayer>,
    layout: SharedStateLayout,
    global_layer_start: usize,
    retained: u64,
    protected: u64,
}
impl SavedHybridGroupedCopy {
    pub(crate) fn prepare_copy(&self) -> Result<PreparedHybridGroupedCopy<'_>, Error> {
        self.prepare_copy_fixed()
            .map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(crate) fn prepare_copy_fixed(
        &self,
    ) -> Result<PreparedHybridGroupedCopy<'_>, ResidentDecoderPreparationError> {
        PreparedHybridGroupedCopy::prepare_source(Source::Saved(self))
    }
    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }
    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        self.retained
    }
    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        self.protected
    }
}

pub(crate) struct PreparedHybridGroupedCopy<'a> {
    source: Source<'a>,
    paged: Option<PreparedPagedStorageCopy<'a>>,
}
impl<'a> PreparedHybridGroupedCopy<'a> {
    pub(crate) fn prepare(source: &'a MlxHybridState) -> Result<Self, Error> {
        Self::prepare_fixed(source).map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(crate) fn prepare_fixed(
        source: &'a MlxHybridState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        Self::prepare_source(Source::Live(source))
    }
    fn len(&self) -> usize {
        match self.source {
            Source::Live(s) => s.layers.len(),
            Source::Saved(s) => s.layers.len(),
        }
    }
    fn layer(&self, i: usize) -> Option<Layer<'a>> {
        match self.source {
            Source::Live(s) => s.layers.slots().get(i).map(|s| Layer {
                attention: s.attention.as_ref(),
                fixed: Fixed::Live(&s.fixed),
                fixed_offset: s.fixed_offset,
            }),
            Source::Saved(s) => s.layers.get(i).map(|s| Layer {
                attention: s.attention.as_ref(),
                fixed: Fixed::Saved(&s.fixed),
                fixed_offset: s.fixed_offset,
            }),
        }
    }
    pub(crate) fn shared_layout(&self) -> &'a SharedStateLayout {
        match self.source {
            Source::Live(s) => &s.layout,
            Source::Saved(s) => &s.layout,
        }
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        match self.source {
            Source::Live(s) => s.global_layer_start,
            Source::Saved(s) => s.global_layer_start,
        }
    }
    fn validate(&self) -> Result<(), Error> {
        self.validate_fixed()
            .map_err(ResidentDecoderPreparationError::into_error)
    }
    fn validate_fixed(&self) -> Result<(), ResidentDecoderPreparationError> {
        if matches!(self.source,Source::Live(s) if s.manager.is_some() != self.paged.is_some())
            || self.len() != self.shared_layout().layout().len()
        {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or(WorkingMemoryError::IdentityMismatch)?;
            let policy = self
                .shared_layout()
                .layout()
                .layer(i)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            let compatible = match (policy, layer.attention) {
                (LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. }, None) => true,
                (
                    LayerCachePolicy::CompressedLatentRotary { .. },
                    Some(MlxHybridAttentionState::Compressed(cache)),
                ) => {
                    cache.prepare_isolated_copy_fixed()?;
                    true
                }
                (
                    LayerCachePolicy::KeyValue { .. }
                    | LayerCachePolicy::KeyValueWithFixedState { .. },
                    Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(_))),
                ) => self.paged.is_some(),
                (
                    LayerCachePolicy::KeyValue { .. }
                    | LayerCachePolicy::KeyValueWithFixedState { .. }
                    | LayerCachePolicy::KeyOnly { .. }
                    | LayerCachePolicy::KeyOnlyWithFixedState { .. },
                    Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(_))),
                ) => true,
                _ => false,
            };
            if !compatible || layer.fixed.len() != policy.fixed_state().len() {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
            let mut previous = None;
            for (role, _) in layer.fixed.iter() {
                if previous.is_some_and(|p| p >= *role)
                    || !policy.fixed_state().iter().any(|d| d.role == *role)
                {
                    return Err(WorkingMemoryError::IdentityMismatch.into());
                }
                previous = Some(*role);
            }
        }
        Ok(())
    }
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for i in 0..self.len() {
            let layer = self.layer(i).expect("validated complete outer table");
            match layer.attention {
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) => {
                    cache.prepare_isolated_copy().visit_operands(visitor)
                }
                Some(MlxHybridAttentionState::Compressed(cache)) => cache
                    .prepare_isolated_copy()
                    .expect("validated immutable compressed source")
                    .visit_operands(visitor),
                _ => {}
            }
            for (_, value) in layer.fixed.iter() {
                if let Some(value) = value {
                    visitor(value.as_array());
                }
            }
        }
    }
    /// Complete source custody includes compressed capacity stores even when
    /// those stores are not destination operands. All fixed roles remain owned.
    pub(crate) fn visit_retained_arrays(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for i in 0..self.len() {
            let layer = self.layer(i).expect("validated complete source");
            match layer.attention {
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) => {
                    cache.prepare_isolated_copy().visit_operands(visitor)
                }
                Some(MlxHybridAttentionState::Compressed(cache)) => cache
                    .prepare_isolated_copy()
                    .expect("validated immutable compressed source")
                    .visit_retained_arrays(visitor),
                _ => {}
            }
            for (_, value) in layer.fixed.iter() {
                if let Some(value) = value {
                    visitor(value.as_array());
                }
            }
        }
    }
    /// Counts the actual outer/child Registered branches used by host_copy.
    /// Funded saved tables contribute no new registered source wrapper.
    pub(crate) fn copy_account_layout(
        &self,
    ) -> Result<eredu_runtime::working_memory::WorkspaceCopyAccountLayout, WorkingMemoryError> {
        // One actual fixed child per outer row, including empty tables.
        let tables = self
            .len()
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        eredu_runtime::working_memory::WorkspaceCopyAccountLayout::decoder_group(tables)
    }

    pub(crate) fn registered_source_tables(&self) -> Result<usize, WorkingMemoryError> {
        let mut count = usize::from(matches!(self.source, Source::Live(_)));
        for i in 0..self.len() {
            if matches!(
                self.layer(i)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?
                    .fixed,
                Fixed::Live(_)
            ) {
                count = count.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        Ok(count)
    }

    // Saved outer and child tables have authenticated private host holds, not
    // registry entries. Live inventory is supplied by the actual model state.
    pub(crate) fn visit_registered_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.validate()?;
        let mut error = None;
        self.visit_registered_child_metadata_borrowed(&mut |metadata| {
            if error.is_none() {
                error = visitor(metadata).err();
            }
        });
        error.map_or(Ok(()), Err)
    }
    /// The immutable prepared plan already validated this exact source topology.
    pub(crate) fn visit_registered_child_metadata_borrowed(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata),
    ) {
        if let Source::Live(s) = self.source {
            for layer in s.layers.slots() {
                visitor(layer.fixed.metadata());
            }
        }
    }
    fn copy_attention(
        &self,
        i: usize,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<Option<MlxHybridAttentionState>, Error> {
        self.copy_attention_with_metadata(i, stream, roots, None)
    }
    fn copy_attention_with_metadata(
        &self,
        i: usize,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Option<MlxHybridAttentionState>, Error> {
        fn cause(
            funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
            error: impl std::error::Error + Send + Sync + 'static,
        ) -> Error {
            match funding {
                Some(funding) => Error::Neural(funding.metadata_source(error)),
                None => other(error),
            }
        }
        match self
            .layer(i)
            .ok_or_else(|| cause(funding, WorkingMemoryError::IdentityMismatch))?
            .attention
        {
            None => Ok(None),
            Some(MlxHybridAttentionState::KeyValue(cache)) => {
                super::super::key_value::copy_resident_kv_layer_retained(cache, stream, roots)
                    .map(|cache| Some(MlxHybridAttentionState::KeyValue(cache)))
                    .map_err(|error| cause(funding, error))
            }
            Some(MlxHybridAttentionState::Compressed(cache)) => {
                let prepared = match funding {
                    Some(_) => cache
                        .prepare_isolated_copy_fixed()
                        .map_err(|error| cause(funding, error))?,
                    None => cache.prepare_isolated_copy()?,
                };
                prepared
                    .copy_retained(stream, roots)
                    .map(|cache| Some(MlxHybridAttentionState::Compressed(cache)))
                    .map_err(|error| cause(funding, error))
            }
        }
    }
    pub(crate) fn project_dense_workspace(
        &self,
        batch: NonZeroU32,
        context: &'a WorkspaceContext,
    ) -> Result<ProjectedHybridGroupedCopy, Error> {
        self.validate()?;
        if self.paged.is_some() {
            return self.project_paged_workspace(batch, context);
        }
        let mut projection = ExistingArrayProjection::new(context);
        let mut retained = projection
            .project_roots(|visitor| self.visit_retained_arrays(visitor))
            .map_err(other)?;
        context.begin_state_span(&retained).map_err(other)?;
        let factory = WorkspaceConcatStateFactory::new(batch, context).map_err(other)?;
        let state = DeviceState::create_workspace_with_shared_layout(
            self.shared_layout().clone(),
            context,
            |i, policy| {
                let layer = self.layer(i).expect("validated complete group source");
                if let Some(MlxHybridAttentionState::Compressed(cache)) = layer.attention {
                    let LayerCachePolicy::CompressedLatentRotary {
                        latent_dim,
                        rotary_dim,
                        ..
                    } = policy
                    else {
                        return Err(context
                            .metadata_error(format_args!("compressed source policy changed")));
                    };
                    let copied = cache
                        .prepare_isolated_copy_fixed()
                        .map_err(|cause| context.metadata_source(cause))?
                        .project_copied_workspace(
                            batch,
                            *latent_dim,
                            *rotary_dim,
                            &mut projection,
                        )?;
                    let copied = WorkspaceResidentLayerState::Compressed(copied);
                    use eredu_runtime::RuntimeLayerState;
                    for value in copied.retained_values() {
                        context.reserve_metadata_vec(&mut retained, 1)?;
                        retained.push(value.clone());
                    }
                    return Ok(copied);
                }
                super::workspace::project_ordinary_layer_with(
                    i,
                    policy,
                    layer.attention,
                    layer.fixed_offset,
                    layer.fixed.iter(),
                    &factory,
                    context,
                    |array| {
                        let copied = IsolatedArrayCopy::new(array).trace(&mut projection)?;
                        context.reserve_metadata_vec(&mut retained, 1)?;
                        retained.push(copied.clone());
                        Ok(copied)
                    },
                )
            },
        )
        .map_err(other)?;
        let copy = context.finish_report(&retained).map_err(other)?;
        let source_storage = projection.try_into_storage().map_err(other)?;
        Ok(ProjectedHybridGroupedCopy {
            state,
            source_storage,
            copy,
        })
    }
}
pub(crate) struct ProjectedHybridGroupedCopy {
    pub(crate) state: DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    pub(crate) source_storage: ProjectedNativeStorage,
    pub(crate) copy: WorkspaceTraceReport,
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;

use eredu_nn::workspace::WorkspaceMetadataAllocation;
