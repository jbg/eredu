//! Closed copies of actual Hybrid layers whose only payload is resident KV.
use super::*;
use crate::backend::runtime::cache::state::ResidentDecoderPreparationError;
use crate::backend::{
    error::Error,
    nn::workspace::{ExistingArrayProjection, ProjectedNativeStorage},
    runtime::residency::storage::StorageIdentity,
};
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTraceReport};
use eredu_runtime::{
    DenseHostSlotInitialization, HostSlotAttachmentError, HostSlotInitialization, HostSlotMetadata,
    SharedStateLayout,
    working_memory::{
        FundedDecoderSlots, FundedDenseHostSlots, InferencePromptCompletion, InferenceRetention,
        InitializedDecoderSlots, InitializedDenseDecoderSlots, RegisteredDecoderHostCopy,
        RegisteredDenseDecoderInitialization, WorkingMemoryError, WorkingMemoryPool,
        WorkspaceResidentLayerState,
    },
};
use std::{cell::RefCell, fmt, num::NonZeroU32};

fn other(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::Other(Box::new(error))
}
fn mismatch() -> Error {
    other(WorkingMemoryError::IdentityMismatch)
}

// Keep the established typed incomplete-bound gate for unsupported payloads.
// The selected mechanism above never strips fixed roles or retags a family.
fn unsupported() -> Error {
    other(WorkingMemoryError::UnknownBound)
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Live(&'a MlxHybridState),
    Saved(&'a SavedHybridKvCopy),
}

pub(crate) struct PreparedHybridKvCopy<'a> {
    source: Source<'a>,
}
impl<'a> PreparedHybridKvCopy<'a> {
    pub(crate) fn prepare(source: &'a MlxHybridState) -> Result<Self, Error> {
        Self::prepare_fixed(source).map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(crate) fn prepare_fixed(
        source: &'a MlxHybridState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        let plan = Self {
            source: Source::Live(source),
        };
        plan.validate_fixed()?;
        Ok(plan)
    }
    pub(in crate::backend::runtime::cache::state) fn len(&self) -> usize {
        match self.source {
            Source::Live(s) => s.layers.len(),
            Source::Saved(s) => s.layers.len(),
        }
    }
    fn layer(&self, index: usize) -> Option<&'a MlxHybridLayerState> {
        match self.source {
            Source::Live(s) => s.layers.slots().get(index),
            Source::Saved(s) => s.layers.get(index),
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
        if matches!(self.source, Source::Live(s) if s.manager.is_some())
            || self.len() != self.shared_layout().layout().len()
        {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        for index in 0..self.len() {
            let layer = self
                .layer(index)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if !matches!(
                self.shared_layout().layout().layer(index),
                Some(LayerCachePolicy::KeyValue { .. })
            ) || !layer.fixed.is_empty()
                || layer.fixed.payload_bytes() != Some(0)
                || !matches!(
                    layer.attention,
                    Some(MlxHybridAttentionState::KeyValue(
                        MlxKeyValueLayerState::Device(_)
                    ))
                )
            {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
        }
        self.slot_initialization_fixed()?;
        Ok(())
    }
    fn slot_initialization(
        &self,
    ) -> Result<HostSlotInitialization<'a, MlxHybridLayerState>, Error> {
        self.slot_initialization_fixed().map_err(other)
    }
    fn slot_initialization_fixed(
        &self,
    ) -> Result<
        HostSlotInitialization<'a, MlxHybridLayerState>,
        eredu_runtime::HostSlotInitializationError,
    > {
        match self.source {
            Source::Live(s) => s.layers.prepare_copy_slots(),
            Source::Saved(s) => s.layers.prepare_copy_slots(),
        }
    }
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for index in 0..self.len() {
            if let Some(MlxHybridLayerState {
                attention:
                    Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))),
                ..
            }) = self.layer(index)
            {
                cache.prepare_isolated_copy().visit_operands(visitor);
            }
        }
    }
    /// Actual empty child identities still participate in registry health and
    /// complete-source pins. The outer saved table keeps its separate host hold.
    pub(crate) fn visit_empty_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.validate()?;
        let mut error = None;
        self.visit_empty_child_metadata_borrowed(&mut |metadata| {
            if error.is_none() {
                error = visitor(metadata).err();
            }
        });
        error.map_or(Ok(()), Err)
    }
    /// Borrow only from the immutable, already validated selected plan.
    pub(crate) fn visit_empty_child_metadata_borrowed(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata),
    ) {
        for index in 0..self.len() {
            visitor(
                self.layer(index)
                    .expect("validated immutable child table")
                    .fixed
                    .metadata(),
            );
        }
    }

    /// Same source branch as host_copy; empty registered tables still count.
    pub(crate) fn registered_source_tables(&self) -> usize {
        usize::from(matches!(self.source, Source::Live(_)))
    }

    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, crate::backend::runtime::cache::state::ResidentDecoderPreparationError> {
        Ok(self
            .slot_initialization_fixed()?
            .initialization_peak_bytes())
    }

    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let outer = RegisteredDecoderHostCopy::<MlxHybridLayerState, StorageIdentity>::preparation_control_bytes(
            matches!(self.source, Source::Live(_)),
        )?;
        outer.checked_add(self.empty_children_preparation_bytes()?).ok_or(E::Overflow)
    }

    fn empty_children_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let frames = [
            std::mem::size_of::<std::ops::Range<usize>>(),
            std::mem::size_of::<Option<&eredu_core::HostPreparationAuthority>>(),
            std::mem::size_of::<Result<SavedHybridKvCopy, Error>>(),
            std::mem::size_of::<Result<PreparedDenseHybridKvState<'_>, Error>>(),
            std::mem::size_of::<Result<usize, E>>(),
            std::mem::size_of::<Option<&MlxHybridLayerState>>(),
        ].into_iter().try_fold(std::mem::size_of::<[usize; 6]>(), usize::checked_add)
            .ok_or(E::Overflow)?;
        (0..self.len()).try_fold(frames, |bytes, index| {
            let child = self.layer(index).ok_or(E::UnknownBound)?;
            bytes.checked_add(child.fixed.empty_copy_preparation_bytes().ok_or(E::UnknownBound)?)
                .ok_or(E::Overflow)
        })
    }

    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<RegisteredDecoderHostCopy<'a, MlxHybridLayerState, StorageIdentity>, Error> {
        self.host_copy_with_preparation(pool, None)
    }

    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<RegisteredDecoderHostCopy<'a, MlxHybridLayerState, StorageIdentity>, Error> {
        self.validate()?;
        match self.source {
            Source::Live(s) => RegisteredDecoderHostCopy::bind_with_preparation(
                pool,
                self.slot_initialization()?,
                StorageIdentity::HostMetadata(
                    s.layers.metadata().identity().registry_key().clone(),
                ),
                preparation,
            )
            .map_err(other),
            Source::Saved(s) => s.layers.prepare_copy().map_err(other),
        }
    }
    fn copy_layer(
        &self,
        index: usize,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<MlxHybridLayerState, Error> {
        let source = self.layer(index).ok_or_else(mismatch)?;
        let Some(MlxHybridAttentionState::KeyValue(attention)) = &source.attention else {
            return Err(unsupported());
        };
        // No child slot payload exists. Recreate only its independent identity
        // metadata under the admitted outer initializer; retain the full source
        // separately through native recovery. No role/value is discarded.
        let fixed = match host {
            Some(host) => source.fixed.copy_empty_prepared(host)?,
            None => source.fixed.copy_empty().ok_or_else(unsupported)?,
        };
        let attention =
            super::super::key_value::copy_resident_kv_layer_retained(attention, stream, roots)
                .map_err(other)?;
        Ok(MlxHybridLayerState {
            attention: Some(MlxHybridAttentionState::KeyValue(attention)),
            fixed,
            fixed_offset: source.fixed_offset,
        })
    }
    pub(crate) fn copy_retained(
        self,
        slots: InitializedDecoderSlots<MlxHybridLayerState>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedHybridKvCopy, Error> {
        self.copy_retained_with_preparation(slots, stream, roots, None)
    }
    pub(crate) fn copy_retained_with_preparation(
        self,
        mut slots: InitializedDecoderSlots<MlxHybridLayerState>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<SavedHybridKvCopy, Error> {
        self.validate()?;
        slots
            .validate_source(&self.slot_initialization()?)
            .map_err(other)?;
        for index in 0..self.len() {
            slots
                .push(self.copy_layer(index, stream, roots, host)?)
                .map_err(|_| mismatch())?;
        }
        let layers = slots.finish().map_err(|_| mismatch())?;
        Ok(SavedHybridKvCopy {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
        })
    }
    pub(crate) fn copy_dense_with_preparation(
        self,
        host: &eredu_core::HostPreparationAuthority,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxHybridState, Error> {
        use super::super::resident_copy::host_copy::copy_slots;
        self.validate_fixed()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let source = self
            .slot_initialization_fixed()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
            .for_dense_destination()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let layers = copy_slots(source, host, funding, |_, source| {
            let Some(MlxHybridAttentionState::KeyValue(attention)) = &source.attention else {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            };
            let child = source
                .fixed
                .prepare_slots()
                .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
                .for_dense_destination()
                .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
            let fixed = copy_slots(child, host, funding, |_, _| {
                Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))
            })?;
            let attention =
                super::super::key_value::copy_resident_kv_layer_retained(attention, stream, roots)
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
            Ok(MlxHybridLayerState {
                attention: Some(MlxHybridAttentionState::KeyValue(attention)),
                fixed: FixedStateSlots::from_published_slots(fixed),
                fixed_offset: source.fixed_offset,
            })
        })?;
        Ok(MlxHybridState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
            manager: None,
            inference_retention: InferenceRetention::new(),
        })
    }

    pub(crate) fn dense_initialization(
        &self,
    ) -> Result<DenseHostSlotInitialization<'a, MlxHybridLayerState, MlxHybridLayerState>, Error>
    {
        self.slot_initialization()?
            .for_dense_destination()
            .map_err(other)
    }
    pub(crate) fn dense_initialization_peak_bytes_fixed(
        &self,
    ) -> Result<u64, crate::backend::runtime::cache::state::ResidentDecoderPreparationError> {
        Ok(self
            .slot_initialization_fixed()?
            .for_dense_destination::<MlxHybridLayerState>()?
            .initialization_peak_bytes())
    }

    pub(crate) fn dense_host_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let nested =
            eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes().ok_or(E::UnknownBound)?;
        RegisteredDenseDecoderInitialization::<
            MlxHybridLayerState,
            MlxHybridLayerState,
            StorageIdentity,
        >::preparation_control_bytes(matches!(self.source, Source::Live(_)), nested)?
            .checked_add(self.empty_children_preparation_bytes()?).ok_or(E::Overflow)
    }

    pub(crate) fn dense_host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        RegisteredDenseDecoderInitialization<
            'a,
            MlxHybridLayerState,
            MlxHybridLayerState,
            StorageIdentity,
        >,
        Error,
    > {
        self.dense_host_copy_with_preparation(pool, None)
    }

    pub(crate) fn dense_host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<
        RegisteredDenseDecoderInitialization<
            'a,
            MlxHybridLayerState,
            MlxHybridLayerState,
            StorageIdentity,
        >,
        Error,
    > {
        self.host_copy_with_preparation(pool, preparation)?
            .for_dense_destination()
            .map_err(other)
    }
    pub(crate) fn copy_dense_retained(
        self,
        slots: InitializedDenseDecoderSlots<'a, MlxHybridLayerState, MlxHybridLayerState, StorageIdentity>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<PreparedDenseHybridKvState<'a>, Error> {
        self.copy_dense_retained_with_preparation(slots, stream, roots, None)
    }
    pub(crate) fn copy_dense_retained_with_preparation(
        self,
        mut slots: InitializedDenseDecoderSlots<
            'a,
            MlxHybridLayerState,
            MlxHybridLayerState,
            StorageIdentity,
        >,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        host: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<PreparedDenseHybridKvState<'a>, Error> {
        self.validate()?;
        slots
            .validate_source(&self.dense_initialization()?)
            .map_err(other)?;
        for index in 0..self.len() {
            slots
                .push(self.copy_layer(index, stream, roots, host)?)
                .map_err(|_| mismatch())?;
        }
        let layers = slots.finish().map_err(|_| mismatch())?;
        Ok(PreparedDenseHybridKvState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
        })
    }
    /// Full source/destination span. Binding residual credit after this trace is
    /// forbidden, exactly as for the ordinary dense KV projection.
    pub(crate) fn project_dense_workspace(
        &self,
        batch: NonZeroU32,
        context: &'a WorkspaceContext,
    ) -> Result<ProjectedHybridKvCopy, Error> {
        self.validate()?;
        let mut projection = ExistingArrayProjection::new(context);
        let mut retained = projection
            .project_roots(|visitor| self.visit_operands(visitor))
            .map_err(other)?;
        context.begin_state_span(&retained).map_err(other)?;
        let state = super::super::key_value::project_layers(
            self.shared_layout(),
            self.len(),
            batch,
            context,
            |index, policy| {
                let Some(MlxHybridAttentionState::KeyValue(cache)) =
                    self.layer(index).and_then(|s| s.attention.as_ref())
                else {
                    return Err(
                        context.metadata_error(format_args!("validated Hybrid KV source changed"))
                    );
                };
                cache.project_workspace_attention_with(policy, context, |array| {
                    let copied = crate::backend::array_copy::IsolatedArrayCopy::new(array)
                        .trace(&mut projection)?;
                    context.reserve_metadata_vec(&mut retained, 1)?;
                    retained.push(copied.clone());
                    Ok(copied)
                })
            },
        )
        .map_err(other)?;
        let copy = context.finish_report(&retained).map_err(other)?;
        let source_storage = projection.try_into_storage().map_err(other)?;
        Ok(ProjectedHybridKvCopy {
            state,
            source_storage,
            copy,
        })
    }
}

pub(crate) struct ProjectedHybridKvCopy {
    pub(crate) state: DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    pub(crate) source_storage: ProjectedNativeStorage,
    pub(crate) copy: WorkspaceTraceReport,
}

pub(crate) struct SavedHybridKvCopy {
    layers: FundedDecoderSlots<MlxHybridLayerState>,
    layout: SharedStateLayout,
    global_layer_start: usize,
}
impl SavedHybridKvCopy {
    pub(crate) fn prepare_copy(&self) -> Result<PreparedHybridKvCopy<'_>, Error> {
        self.prepare_copy_fixed()
            .map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(crate) fn prepare_copy_fixed(
        &self,
    ) -> Result<PreparedHybridKvCopy<'_>, ResidentDecoderPreparationError> {
        let plan = PreparedHybridKvCopy {
            source: Source::Saved(self),
        };
        plan.validate_fixed()?;
        Ok(plan)
    }
    pub(crate) fn shared_layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    pub(crate) fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }
    pub(crate) fn retained_slot_bytes(&self) -> u64 {
        self.layers.retained_bytes()
    }
    pub(crate) fn protected_slot_bytes(&self) -> u64 {
        self.layers.protected_bytes()
    }
}

pub(crate) struct PreparedDenseHybridKvState<'a> {
    layers: FundedDenseHostSlots<'a, MlxHybridLayerState, MlxHybridLayerState, StorageIdentity>,
    layout: SharedStateLayout,
    global_layer_start: usize,
}
impl PreparedDenseHybridKvState<'_> {
    pub(crate) fn slot_metadata(&self) -> &HostSlotMetadata {
        self.layers.metadata()
    }
    pub(crate) fn visit_operands<'s>(&'s self, visitor: &mut dyn FnMut(&'s Array)) {
        for index in 0..self.layers.len() {
            if let Some(MlxHybridLayerState {
                attention:
                    Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))),
                ..
            }) = self.layers.get(index)
            {
                cache.prepare_isolated_copy().visit_operands(visitor);
            }
        }
    }
}
impl<'a> PreparedDenseHybridKvState<'a> {
    pub(crate) fn publish_for_control(
        self,
    ) -> Result<(PublishedDenseHybridKvState, InferencePromptCompletion), HybridKvPublishError<'a>>
    {
        let Self {
            layers,
            layout,
            global_layer_start,
        } = self;
        let key =
            StorageIdentity::HostMetadata(layers.metadata().identity().registry_key().clone());
        match layers.publish(key) {
            Ok((layers, completion)) => Ok((
                PublishedDenseHybridKvState {
                    state: MlxHybridState {
                        layers,
                        layout,
                        global_layer_start,
                        manager: None,
                        inference_retention: InferenceRetention::new(),
                    },
                },
                completion,
            )),
            Err(error) => {
                let (layers, error) = error.into_parts();
                Err(HybridKvPublishError {
                    owner: Self {
                        layers,
                        layout,
                        global_layer_start,
                    },
                    error,
                })
            }
        }
    }
}
/// Only publication of the actual admitted table constructs this marker.
pub(crate) struct PublishedDenseHybridKvState {
    state: MlxHybridState,
}
impl PublishedDenseHybridKvState {
    pub(crate) fn visit_empty_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for layer in self.state.layers.slots() {
            if !layer.fixed.is_empty() || layer.fixed.payload_bytes() != Some(0) {
                return Err(unsupported());
            }
            visitor(layer.fixed.metadata())?;
        }
        Ok(())
    }

    pub(crate) fn into_state(self) -> MlxHybridState {
        self.state
    }
}

pub(crate) struct HybridKvPublishError<'a> {
    owner: PreparedDenseHybridKvState<'a>,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}
impl<'a> HybridKvPublishError<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedDenseHybridKvState<'a>,
        HostSlotAttachmentError<WorkingMemoryError>,
    ) {
        (self.owner, self.error)
    }
}
impl fmt::Debug for HybridKvPublishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl fmt::Display for HybridKvPublishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for HybridKvPublishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
