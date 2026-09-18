use super::*;

use eredu_runtime::working_memory::InferenceStateRetention;

fn validate_text_frontiers(
    layout: &eredu_runtime::StateLayout,
    positions: impl ExactSizeIterator<Item = i32>,
    expected: u64,
) -> Result<(), Error> {
    use eredu_runtime::working_memory::WorkingMemoryError;
    if positions.len() != layout.len() {
        return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
    }
    for (index, position) in positions.enumerate() {
        if matches!(
            layout.layer(index),
            Some(eredu_core::cache::LayerCachePolicy::NoState)
        ) {
            continue;
        }
        let actual = u64::try_from(position).map_err(|error| Error::Other(Box::new(error)))?;
        if actual != expected {
            return Err(Error::Other(Box::new(
                WorkingMemoryError::StateFrontierMismatch { expected, actual },
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod frontier_tests {
    use super::*;

    #[test]
    fn pooling_checkpoint_preserves_live_layout_owner_and_nonzero_state() {
        use eredu_core::{AttentionPolicy, LayerSchedule};
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let layout = eredu_runtime::StateLayout::new(
            LayerSchedule::new(
                1,
                vec![eredu_core::cache::LayerCachePolicy::key_only(
                    AttentionPolicy::sliding(7).unwrap(),
                    1,
                    2,
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let mut state = MlxPoolingAttentionStateFactory::device(layout).unwrap();
        state.as_mut()[0]
            .append_local(
                MlxTensor::from_array(Array::from_slice(&[1_f32, 2., 3., 4., 5., 6.], &[1, 3, 2])),
                &stream,
            )
            .unwrap();
        let checkpoint = state.deep_checkpoint().unwrap();
        assert!(state
            .shared_layout()
            .unwrap()
            .same_storage(checkpoint.shared_layout().unwrap()));
        assert_eq!(MlxStateMechanisms::offset(&checkpoint), 3);
        state.as_mut()[0]
            .append_local(
                MlxTensor::from_array(Array::from_slice(&[17_f32, 19.], &[1, 1, 2])),
                &stream,
            )
            .unwrap();
        assert_eq!(MlxStateMechanisms::offset(&state), 4);
        assert_eq!(MlxStateMechanisms::offset(&checkpoint), 3);
        state.restore_checkpoint(&checkpoint, &stream).unwrap();
        assert_eq!(MlxStateMechanisms::offset(&state), 3);
        assert!(state
            .shared_layout()
            .unwrap()
            .same_storage(checkpoint.shared_layout().unwrap()));
    }

    #[test]
    fn every_stateful_layer_must_match_even_when_the_first_layer_has_no_state() {
        use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
        let attention = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap();
        let layout = eredu_runtime::StateLayout::new(
            LayerSchedule::new(
                3,
                vec![LayerCachePolicy::NoState, attention.clone(), attention],
            )
            .unwrap(),
        )
        .unwrap();
        validate_text_frontiers(&layout, [0, 7, 7].into_iter(), 7).unwrap();
        let error = validate_text_frontiers(&layout, [0, 7, 6].into_iter(), 7).unwrap_err();
        let Error::Other(error) = error else {
            panic!("typed frontier rejection expected")
        };
        assert!(matches!(
            error.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>(),
            Some(
                eredu_runtime::working_memory::WorkingMemoryError::StateFrontierMismatch {
                    expected: 7,
                    actual: 6
                }
            )
        ));
        assert!(validate_text_frontiers(&layout, [0, 7].into_iter(), 7).is_err());
        assert!(validate_text_frontiers(&layout, [0, 7, -1].into_iter(), 7).is_err());
    }
}

/// The published native representation cannot populate this typed session.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PreparedDenseControlBindingError {
    #[error("published resident decoder state differs from the executable's native state type")]
    UnsupportedStateType,
}

#[derive(Clone, Copy)]
pub(crate) enum ResidentResetProfile {
    KeyValue,
    Hybrid,
}

pub(crate) trait MlxStateMechanisms:
    LayerRuntimeState<MlxNeuralBackend>
    + InferenceStateRetention
    + eredu_runtime::working_memory::ResidentResetProjection<MlxKeyValueState>
    + eredu_runtime::working_memory::ResidentResetProjection<MlxHybridState>
    + Sized
{
    fn resident_reset_profile() -> Option<ResidentResetProfile> {
        None
    }
    /// Fixed native fields for this actual representation; manager payloads,
    /// unknown custom state and outer allocation completeness remain separate.
    fn retained_owner_slot_counts(
        &self,
    ) -> Option<crate::backend::runtime::cache::state::NativeStateSlotCounts> {
        None
    }

    /// Moves a published dense KV owner into its exact native state type.
    /// Other representations reject instead of translating layers or inferring
    /// applicability from a model-family name. Rejection consumes the owner;
    /// the caller must have settled numerical work and retain external recovery.
    /// Only exact implementations whose conversion moves already published
    /// storage opt in. Custom ordinary conversions cannot certify this producer.
    const PREPARED_CONTROL_BINDING: bool = false;

    fn from_published_dense_control_state(
        state: crate::backend::runtime::cache::state::PublishedDenseResidentKvState,
    ) -> Result<Self, Error> {
        Self::from_published_dense_control_state_fixed(state)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn from_published_resident_control_state(
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
    ) -> Result<Self, Error> {
        Self::from_published_resident_control_state_fixed(state)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn from_published_dense_control_state_fixed(
        _state: crate::backend::runtime::cache::state::PublishedDenseResidentKvState,
    ) -> Result<Self, PreparedDenseControlBindingError> {
        Err(PreparedDenseControlBindingError::UnsupportedStateType)
    }

    fn from_published_resident_control_state_fixed(
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
    ) -> Result<Self, PreparedDenseControlBindingError> {
        match state {
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::KeyValue(
                state,
            ) => Self::from_published_dense_control_state_fixed(state),
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::HybridGrouped(
                _,
            )
            | crate::backend::runtime::cache::state::PublishedResidentDecoderState::Pooling(_) => {
                Err(PreparedDenseControlBindingError::UnsupportedStateType)
            }
        }
    }

    /// Borrows the complete supported native decoder representation. This does
    /// not run legacy estimate getters, copy slots or grant submission rights.
    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error>;

    /// Same exact native representation with fixed pregrant refusal causes.
    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    >;

    /// Source-qualified external paged storage uses the same registered copy
    /// worker with canonical source loans. Other representations keep the
    /// existing borrowed resident program. This grants no text execution.
    fn copy_original_paged_state(
        &self,
        _completed: Option<&crate::backend::runtime::cache::state::CompletedResidentSource>,
        _environment: &crate::backend::OriginalCopyEnvironment<'_>,
        _initialized: &safemlx::PrefillRootsRuntime,
        _mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        _funding: &eredu_nn::workspace::HostMetadataFunding,
        _host: &eredu_core::HostPreparationAuthority,
        _capacity: u64,
    ) -> Result<Option<crate::backend::runtime::cache::state::OriginalResidentState>, Error> {
        Ok(None)
    }

    /// Moves one completed original copy into its exact native representation.
    /// A mismatch returns the still-owned value for safe retirement; no erasure
    /// allocation, layer translation, or family selection occurs here.
    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        Err(state)
    }

    /// Checks every stateful layer's host frontier without reading native values.
    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error>;
    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_storage_fixed(storage)
            .map_err(Error::from)
    }

    /// Same storage producer before the ordinary backend error adapter.
    fn collect_retained_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError>;
    /// Visits actual fixed layer/role table owners. Inline slots and shared
    /// layout are inventoried separately from nested numerical backing. This
    /// neither constructs a new table nor proves a destination copy bound.
    fn visit_slot_metadata<E>(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), E>,
    ) -> Result<(), E>;

    /// Actual layout and fixed host tables are registered alongside idle model
    /// sources, separately from mutable decoder arrays. Tokens retain custody
    /// only; the inspected state keeps the corresponding tables alive.
    fn retained_host_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_host_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_host_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_host_storage_fixed(storage)
            .map_err(Error::from)
    }

    /// Shares metadata order with ordinary collection without formatting a cause.
    fn collect_retained_host_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
        match self.shared_layout() {
            Some(layout) => storage
                .include_metadata(eredu_runtime::SharedHostMetadata::Layout(layout.clone()))?,
            None if self.optional_layout().is_none() => {}
            None => storage.mark_incomplete(),
        }
        self.visit_slot_metadata(&mut |metadata| storage.include_slot_metadata(metadata.clone()))?;
        Ok(())
    }

    fn project_resident_workspace(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        eredu_runtime::DeviceState<
            eredu_nn::workspace::WorkspaceBackend,
            eredu_runtime::working_memory::WorkspaceResidentLayerState,
        >,
        Error,
    > {
        self.project_resident_workspace_with_storage(batch, context)
            .map(|projected| projected.state)
    }
    fn project_resident_workspace_with_storage(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error>;
    fn supports_isolated_snapshot(&self) -> bool {
        false
    }
    fn isolated_snapshot(&self, _stream: &Stream) -> Result<Self, Exception> {
        Err(Exception::custom(
            "complete isolated snapshot is unsupported for this state realization",
        ))
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn isolated_snapshot_auxiliary_growth(&self, _additional: u64) -> Option<u64> {
        Some(0)
    }
    fn original_isolated_snapshot_auxiliary_bytes(&self) -> Option<u64>;
    /// Implemented only by the concrete native owners using direct layer visits.
    /// No default iterator fallback is valid for the pre-grant path.
    fn visit_snapshot_arrays(&self, visitor: &mut dyn FnMut(&Array)) -> Option<()>;

    fn isolated_snapshot_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.isolated_snapshot_estimate_with(false)
    }
    fn original_isolated_snapshot_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.isolated_snapshot_estimate_with(true)
    }
    fn isolated_snapshot_estimate_with(
        &self,
        original: bool,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        use crate::backend::runtime::cache::state::snapshot_estimate;
        let mut retained = snapshot_estimate::state_metadata(
            self.layout(),
            std::mem::size_of::<Self>(),
            self.inference_retention().logical_metadata_bytes()?,
        )?;
        // Both routes consume the same logical policy. Original inspection
        // replaces native getters with borrowed descriptors only.
        let array_bytes = snapshot_estimate::array_bytes;
        if original {
            let mut total = Some(retained);
            self.visit_snapshot_arrays(&mut |array| {
                total = total.and_then(|bytes| {
                    let descriptor = array.try_descriptor().ok()?;
                    let facts = descriptor.facts();
                    bytes.checked_add(array_bytes(facts.logical_bytes(), facts.rank())?)
                });
            })?;
            retained = total?.checked_add(self.original_isolated_snapshot_auxiliary_bytes()?)?;
        } else {
            for array in self.retained_arrays() {
                retained =
                    retained.checked_add(array_bytes(array.nbytes(), array.shape().len())?)?;
            }
            retained = retained.checked_add(self.isolated_snapshot_auxiliary_bytes()?)?;
        }
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: retained,
            copy_bytes: retained,
        })
    }
    fn continuation_capacity_bound(&self, _additional: u64) -> Option<u64> {
        None
    }
    fn isolated_snapshot_growth(&self, additional: u64) -> Option<u64> {
        if !self.supports_isolated_snapshot() {
            return None;
        }
        // Ordinary text is a single sequence. Interpret only the declared
        // component geometry, including absent fixed tensors. MLX floating
        // storage is at most eight bytes; explicit integer components use four.
        // Charge the full future payload as an additional conservative allowance
        // (rather than subtracting currently retained data), including the same
        // materialization/descriptor allowance as an immutable copy.
        use eredu_core::cache::{StateTensorDimension as Dim, StateTensorDtype};
        let absolute = u64::try_from(self.offset()).ok()?.checked_add(additional)?;
        i32::try_from(absolute).ok()?;
        let prefix = absolute.max(self.continuation_capacity_bound(additional)?);
        i32::try_from(prefix).ok()?;
        let mut bytes = 0u64;
        for layer in 0..self.layout().len() {
            for component in self.layout().components(layer)? {
                let mut elements = 1u64;
                for dimension in component.shape() {
                    let extent = match dimension {
                        Dim::Batch | Dim::Scalar => 1,
                        Dim::Fixed(n) => u64::from(n.get()),
                        Dim::PrefixTokens => prefix,
                        Dim::PrefixTokensDiv(n) => prefix / u64::from(n.get()),
                        // The final remainder is not the maximum over a run.
                        Dim::PrefixTokensRem(n) => prefix.min(u64::from(n.get()) - 1),
                    };
                    elements = elements.checked_mul(extent)?;
                }
                let width = match component.dtype() {
                    StateTensorDtype::Floating => 8,
                    StateTensorDtype::Float32
                    | StateTensorDtype::Int32
                    | StateTensorDtype::Uint32 => 4,
                };
                bytes = bytes
                    .checked_add(elements.checked_mul(width)?.checked_mul(2)?)?
                    .checked_add(4096)?
                    .checked_add(
                        u64::try_from(component.shape().len())
                            .ok()?
                            .checked_mul(16)?,
                    )?;
            }
        }
        bytes.checked_add(self.isolated_snapshot_auxiliary_growth(additional)?)
    }
    fn offset(&self) -> i32;
    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error>;
    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn retained_arrays(&self) -> Vec<&Array>;
    fn deep_checkpoint(&self) -> Result<Self, Exception>;
    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception>;
    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception>;
    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;
    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    >;
    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception>;
}

impl eredu_runtime::working_memory::ResidentResetProjection<MlxKeyValueState> for MlxKeyValueState {
    fn resident_reset_ref(&self) -> Option<&MlxKeyValueState> {
        Some(self)
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxKeyValueState> {
        Some(self)
    }
}
impl eredu_runtime::working_memory::ResidentResetProjection<MlxKeyValueState> for MlxHybridState {
    fn resident_reset_ref(&self) -> Option<&MlxKeyValueState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxKeyValueState> {
        None
    }
}
impl eredu_runtime::working_memory::ResidentResetProjection<MlxKeyValueState>
    for MlxPoolingAttentionState
{
    fn resident_reset_ref(&self) -> Option<&MlxKeyValueState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxKeyValueState> {
        None
    }
}

impl eredu_runtime::working_memory::ResidentResetProjection<MlxHybridState> for MlxKeyValueState {
    fn resident_reset_ref(&self) -> Option<&MlxHybridState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxHybridState> {
        None
    }
}
impl eredu_runtime::working_memory::ResidentResetProjection<MlxHybridState> for MlxHybridState {
    fn resident_reset_ref(&self) -> Option<&MlxHybridState> {
        Some(self)
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxHybridState> {
        Some(self)
    }
}
impl eredu_runtime::working_memory::ResidentResetProjection<MlxHybridState>
    for MlxPoolingAttentionState
{
    fn resident_reset_ref(&self) -> Option<&MlxHybridState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxHybridState> {
        None
    }
}

pub(super) fn fork_mlx_prediction_target_state<S: MlxStateMechanisms>(
    state: &S,
    stream: &Stream,
) -> Result<S, Error> {
    state
        .fork_prediction_target_state(stream)
        .map_err(Into::into)
}

pub(super) fn selected_state_manager(
    selected: &SelectedStateRealization,
) -> Result<Option<CacheResidencyManager>, Error> {
    let needs_paging = selected
        .components()
        .iter()
        .any(|component| component.placement() == StateComponentPlacement::Paged);
    match (needs_paging, selected.policy()) {
        (_, CacheResidencyPolicy::Paged(options)) => CacheResidencyManager::new(options.clone())
            .map(Some)
            .map_err(|error| Error::Parallel(error.to_string())),
        (false, CacheResidencyPolicy::Device) => Ok(None),
        (true, CacheResidencyPolicy::Device) => Err(Error::Parallel(
            "selected paged state component has no paging policy".into(),
        )),
    }
}

impl MlxStateMechanisms for MlxKeyValueState {
    fn copy_original_paged_state(
        &self, completed: Option<&crate::backend::runtime::cache::state::CompletedResidentSource>,
        environment: &crate::backend::OriginalCopyEnvironment<'_>, initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        host: &eredu_core::HostPreparationAuthority, capacity: u64,
    ) -> Result<Option<crate::backend::runtime::cache::state::OriginalResidentState>, Error> {
        if !self.as_ref().iter().any(|layer| matches!(layer, crate::backend::runtime::cache::state::MlxKeyValueLayerState::Paged(_))) {
            return Ok(None);
        }
        let context = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(mechanisms, funding.clone())
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.copy_original_paged(completed, environment, initialized, mechanisms, &context, host, capacity)
            .map(|value| value.map(crate::backend::runtime::cache::state::OriginalResidentState::KeyValue))
    }

    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        match state {
            crate::backend::runtime::cache::state::OriginalResidentState::KeyValue(value) => Ok(value),
            other => Err(other),
        }
    }

    const PREPARED_CONTROL_BINDING: bool = true;
    fn resident_reset_profile() -> Option<ResidentResetProfile> {
        Some(ResidentResetProfile::KeyValue)
    }
    fn retained_owner_slot_counts(
        &self,
    ) -> Option<crate::backend::runtime::cache::state::NativeStateSlotCounts> {
        MlxKeyValueState::retained_owner_slot_counts(self)
    }

    fn from_published_dense_control_state_fixed(
        state: crate::backend::runtime::cache::state::PublishedDenseResidentKvState,
    ) -> Result<Self, PreparedDenseControlBindingError> {
        Ok(state.into_state())
    }

    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error> {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::key_value(self)
    }

    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::key_value_fixed(self)
    }

    fn visit_slot_metadata<E>(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), E>,
    ) -> Result<(), E> {
        visitor(self.layer_slot_metadata())
    }

    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error> {
        validate_text_frontiers(
            self.layout(),
            self.as_ref()
                .iter()
                .map(crate::backend::runtime::cache::kv::KeyValueCache::offset),
            expected,
        )
    }

    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
        MlxKeyValueState::collect_retained_storage(self, storage)
    }

    fn project_resident_workspace_with_storage(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        MlxKeyValueState::project_resident_workspace_with_storage(self, batch, context)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }
    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }

    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }
    fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        self.isolated_snapshot(stream)
    }
    fn offset(&self) -> i32 {
        MlxKeyValueState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxKeyValueState::save_prompt_cache(
            self,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxKeyValueState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxKeyValueState::retained_arrays(self)
    }

    fn visit_snapshot_arrays(&self, visitor: &mut dyn FnMut(&Array)) -> Option<()> {
        for layer in self.as_ref() {
            eredu_runtime::RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(
                layer,
                &mut |tensor| visitor(tensor.as_array()),
            );
        }
        Some(())
    }
    fn original_isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        MlxKeyValueState::original_isolated_snapshot_auxiliary_bytes(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxKeyValueState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxKeyValueState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .map(|layer| (eredu_nn::AttentionCache::offset(layer), Vec::new()))
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        Ok(Vec::new())
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        Ok(Vec::new())
    }
}

impl MlxStateMechanisms for MlxHybridState {
    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        match state {
            crate::backend::runtime::cache::state::OriginalResidentState::Hybrid(value) => Ok(value),
            other => Err(other),
        }
    }

    const PREPARED_CONTROL_BINDING: bool = true;
    fn resident_reset_profile() -> Option<ResidentResetProfile> {
        Some(ResidentResetProfile::Hybrid)
    }
    fn retained_owner_slot_counts(
        &self,
    ) -> Option<crate::backend::runtime::cache::state::NativeStateSlotCounts> {
        MlxHybridState::retained_owner_slot_counts(self)
    }

    fn from_published_resident_control_state_fixed(
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
    ) -> Result<Self, PreparedDenseControlBindingError> {
        match state {
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::HybridGrouped(
                state,
            ) => Ok(state.into_state()),
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::KeyValue(_)
            | crate::backend::runtime::cache::state::PublishedResidentDecoderState::Pooling(_) => {
                Err(PreparedDenseControlBindingError::UnsupportedStateType)
            }
        }
    }
    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error> {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::hybrid(self)
    }

    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::hybrid_fixed(self)
    }

    fn visit_slot_metadata<E>(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), E>,
    ) -> Result<(), E> {
        visitor(self.layer_slot_metadata())?;
        for metadata in self.fixed_slot_metadata() {
            visitor(metadata)?;
        }
        Ok(())
    }

    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error> {
        validate_text_frontiers(self.layout(), self.layer_positions(), expected)
    }

    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
        MlxHybridState::collect_retained_storage(self, storage)
    }

    fn project_resident_workspace_with_storage(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        MlxHybridState::project_resident_workspace_with_storage(self, batch, context)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        self.continuation_capacity_bound(additional)
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        self.isolated_snapshot_auxiliary_bytes()
    }
    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_auxiliary_growth(additional)
    }
    fn supports_isolated_snapshot(&self) -> bool {
        self.supports_isolated_snapshot()
    }
    fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        self.isolated_snapshot(stream)
    }
    fn offset(&self) -> i32 {
        MlxHybridState::offset(self)
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        MlxHybridState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let mut state = MlxHybridState::from_selected_with_global_layer_start(
            selected,
            Some(manager),
            expected.topology().cache_rank_identity(),
            identity.global_layer_start(),
        )?;
        state.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        MlxHybridState::save_prompt_cache(self, destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        MlxHybridState::residency_report(self)
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        MlxHybridState::retained_arrays(self)
    }

    fn visit_snapshot_arrays(&self, visitor: &mut dyn FnMut(&Array)) -> Option<()> {
        // This concrete native implementation visits its slot table directly.
        // It does not use RuntimeState's iterator-based default.
        eredu_runtime::RuntimeState::<MlxNeuralBackend>::visit_all_retained_values(
            self,
            &mut |tensor| visitor(tensor.as_array()),
        )
        .ok()
    }
    fn original_isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        MlxHybridState::original_isolated_snapshot_auxiliary_bytes(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        self.deep_clone_state()
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxHybridState::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        MlxHybridState::restore_checkpoint(self, checkpoint, stream)
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.semantic_snapshot()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        self.fixed_numeric_snapshot()
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_numeric_snapshot()
    }
}

impl MlxStateMechanisms for MlxPoolingAttentionState {
    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        match state {
            crate::backend::runtime::cache::state::OriginalResidentState::Pooling(value) => Ok(value),
            other => Err(other),
        }
    }

    const PREPARED_CONTROL_BINDING: bool = true;
    fn retained_owner_slot_counts(
        &self,
    ) -> Option<crate::backend::runtime::cache::state::NativeStateSlotCounts> {
        use crate::backend::runtime::cache::state::NativeStateSlotCounts;
        // A legacy DeviceState lacking actual shared tables has no closed host
        // owner count contract, even if its current tensor visit is complete.
        self.shared_layout()?;
        self.layer_slot_metadata()?;
        self.as_ref().iter().try_fold(
            NativeStateSlotCounts {
                layouts: 1,
                slot_tables: 1,
                ..Default::default()
            },
            |counts, layer| {
                let layer = layer.retained_owner_slot_counts();
                Some(NativeStateSlotCounts {
                    arrays: counts.arrays.checked_add(layer.arrays)?,
                    layouts: counts.layouts,
                    slot_tables: counts.slot_tables,
                    manager_roles: counts.manager_roles.checked_add(layer.manager_roles)?,
                })
            },
        )
    }

    fn from_published_resident_control_state_fixed(
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
    ) -> Result<Self, PreparedDenseControlBindingError> {
        use crate::backend::runtime::cache::state::PublishedResidentDecoderState;
        match state {
            PublishedResidentDecoderState::Pooling(state) => Ok(state.into_state()),
            PublishedResidentDecoderState::KeyValue(_)

            | PublishedResidentDecoderState::HybridGrouped(_) => {
                Err(PreparedDenseControlBindingError::UnsupportedStateType)
            }
        }
    }

    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error> {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::pooling(self)
    }

    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy::pooling_fixed(self)
    }

    fn visit_slot_metadata<E>(
        &self,
        visitor: &mut dyn FnMut(&eredu_runtime::HostSlotMetadata) -> Result<(), E>,
    ) -> Result<(), E> {
        if let Some(metadata) = self.layer_slot_metadata() {
            visitor(metadata)?;
        }
        Ok(())
    }

    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error> {
        validate_text_frontiers(
            self.layout(),
            self.as_ref().iter().map(|layer| layer.offset()),
            expected,
        )
    }

    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
        storage
            .include_retained_values::<crate::backend::runtime::residency::manager::ResidencyError>(
                |visitor| {
                    for layer in self.as_ref() {
                        eredu_runtime::RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(
                            layer, visitor,
                        );
                    }
                    Ok(true)
                },
            )?;
        for manager in self
            .as_ref()
            .iter()
            .filter_map(MlxPoolingAttentionCache::residency_manager)
        {
            manager.collect_retained_storage(storage)?;
        }
        Ok(())
    }

    fn project_resident_workspace_with_storage(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        MlxPoolingAttentionStateFactory::project_resident_workspace_with_storage(
            self, batch, context,
        )
        .map_err(Error::from)
    }

    fn supports_isolated_snapshot(&self) -> bool {
        MlxPoolingAttentionStateFactory::supports_isolated_snapshot(self)
    }
    fn isolated_snapshot(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxPoolingAttentionStateFactory::isolated_snapshot(self, stream)
    }
    fn isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        MlxPoolingAttentionStateFactory::isolated_snapshot_auxiliary_bytes(self)
    }
    fn isolated_snapshot_auxiliary_growth(&self, additional: u64) -> Option<u64> {
        MlxPoolingAttentionStateFactory::isolated_snapshot_auxiliary_growth(self, additional)
    }
    fn continuation_capacity_bound(&self, additional: u64) -> Option<u64> {
        MlxPoolingAttentionStateFactory::continuation_capacity_bound(self, additional)
    }
    fn offset(&self) -> i32 {
        self.as_ref().first().map_or(0, |layer| layer.offset())
    }

    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected)?;
        match manager {
            Some(manager) => MlxPoolingAttentionStateFactory::paged(
                selected.layout().clone(),
                manager,
                global_layer_start,
                0,
                rank,
            ),
            None => MlxPoolingAttentionStateFactory::device(selected.layout().clone()),
        }
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Self, PromptCacheManifest), Error> {
        let CacheResidencyPolicy::Paged(options) = selected.policy() else {
            return Err(Error::Parallel(
                "prompt-cache loading requires selected paged state".into(),
            ));
        };
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options.clone(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let prefix = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Error::Parallel("prompt-cache prefix exceeds i32".into()))?;
        let mut state = MlxPoolingAttentionStateFactory::paged(
            selected.layout().clone(),
            manager,
            identity.global_layer_start(),
            prefix,
            expected.topology().cache_rank_identity(),
        )?;
        let mut tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .into_iter()
            .map(|tensor| ((tensor.owner, tensor.role), tensor.array))
            .collect::<BTreeMap<_, _>>();
        for (layer, cache) in state.as_mut().iter_mut().enumerate() {
            let processed = prefix
                .checked_add(identity.layer_prefix_offsets()[layer])
                .ok_or_else(|| Error::Parallel("prompt-cache layer offset overflowed".into()))?;
            cache.restore_prompt_cache_state(
                identity.global_layer_start() + layer,
                &mut tensors,
                processed,
            )?;
        }
        if !tensors.is_empty() {
            return Err(Error::Parallel(
                "prompt cache contains unexpected state tensors".into(),
            ));
        }
        Ok((state, manifest))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let mut manager = None;
        for layer in self.as_mut() {
            layer.finalize()?;
            manager.get_or_insert_with(|| layer.residency_manager().cloned());
        }
        let fixed = self
            .as_ref()
            .iter()
            .enumerate()
            .flat_map(|(layer, cache)| cache.prompt_cache_state_arrays(layer))
            .collect::<Vec<_>>();
        manager
            .flatten()
            .ok_or_else(|| Error::Parallel("prompt-cache persistence requires paged state".into()))?
            .save_prompt_cache(destination, descriptor, prefix_token_ids, &fixed, options)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.as_ref()
            .iter()
            .find_map(MlxPoolingAttentionCache::residency_manager)
            .map(CacheResidencyManager::report)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn retained_arrays(&self) -> Vec<&Array> {
        self.as_ref()
            .iter()
            .flat_map(MlxPoolingAttentionCache::retained_arrays)
            .collect()
    }

    fn visit_snapshot_arrays(&self, visitor: &mut dyn FnMut(&Array)) -> Option<()> {
        for layer in self.as_ref() {
            eredu_runtime::RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(
                layer,
                &mut |tensor| visitor(tensor.as_array()),
            );
        }
        Some(())
    }
    fn original_isolated_snapshot_auxiliary_bytes(&self) -> Option<u64> {
        MlxPoolingAttentionStateFactory::original_isolated_snapshot_auxiliary_bytes(self)
    }

    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        // Pooling updates replace arrays. Retain exact views and the paged
        // sliding history that speculative verification could otherwise discard.
        let layout = self.shared_layout().cloned().ok_or_else(|| {
            Exception::custom("pooling checkpoint requires an owned state layout")
        })?;
        let mut checkpoint = Self::create_with_shared_layout(layout, |layer, _| {
            self.as_ref()[layer].checkpoint_clone_state()
        })?;
        checkpoint.inherit_inference_retention(self);
        Ok(checkpoint)
    }

    fn fork_prediction_target_state(&self, stream: &Stream) -> Result<Self, Exception> {
        MlxPoolingAttentionStateFactory::fork_prediction_target_state(self, stream)
    }

    fn restore_checkpoint(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        if self.layout() != checkpoint.layout() || self.as_ref().len() != checkpoint.as_ref().len()
        {
            return Err(Exception::custom(
                "pooling-attention checkpoint layout does not match canonical state",
            ));
        }
        self.inherit_inference_retention(checkpoint);
        self.inference_retention_mut()
            .restore_admission(checkpoint.inference_retention());
        for (current, previous) in self.as_mut().iter_mut().zip(checkpoint.as_ref()) {
            PoolingAttentionCache::restore(current, previous, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)> {
        self.as_ref()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let present = layer
                    .prompt_cache_state_arrays(index)
                    .into_iter()
                    .map(|state| state.role)
                    .collect::<std::collections::BTreeSet<_>>();
                let components = self
                    .layout()
                    .components(index)
                    .expect("pooling state layout contains each realized layer")
                    .iter()
                    .filter_map(|component| match component.role() {
                        eredu_core::cache::StateComponentRole::Fixed(role) => {
                            Some((role, present.contains(&role)))
                        }
                        _ => None,
                    })
                    .collect();
                (PoolingAttentionCache::offset(layer), components)
            })
            .collect()
    }

    #[cfg(test)]
    fn fixed_numeric_snapshot(
        &self,
    ) -> Result<
        Vec<(
            usize,
            eredu_core::cache::StateTensorRole,
            Vec<i32>,
            Vec<f32>,
        )>,
        Exception,
    > {
        let mut snapshot = Vec::new();
        for (layer, cache) in self.as_ref().iter().enumerate() {
            for state in cache.prompt_cache_state_arrays(layer) {
                let evaluated = state.array.evaluated()?.deep_clone()?;
                snapshot.push((
                    layer,
                    state.role,
                    state.array.shape().to_vec(),
                    evaluated.as_slice::<f32>().to_vec(),
                ));
            }
        }
        Ok(snapshot)
    }

    #[cfg(test)]
    fn retained_numeric_snapshot(&self) -> Result<RetainedNumericStateSnapshot, Exception> {
        self.retained_arrays()
            .into_iter()
            .map(|array| {
                let evaluated = array.evaluated()?.deep_clone()?;
                Ok((array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec()))
            })
            .collect()
    }
}

#[cfg(test)]
mod resident_copy_tests;

mod slot_bounds;
pub(crate) use slot_bounds::PreparedNativeStateSlotBounds;

/// Shared idle auxiliary-source order; neither adapter invents extra sources.
/// Classifies the actual partition auxiliaries of an idle executable. The
/// neutral communication authority contains a completion policy, terminal bit,
/// and fixed poison record; it owns no tensor, source payload, or callbacks.
/// A sampling group contributes the same retained native buffer as its parent
/// communicator table, deduplicated by the closed physical source identity.
pub(super) fn collect_partition_auxiliary_storage(
    storage:&mut crate::backend::runtime::residency::storage::RetainedStorage,
    group:Option<&crate::backend::runtime::distributed::Group>,
    authority:Option<&eredu_runtime::PartitionCommunicationAuthority>,
)->Result<(),crate::backend::runtime::residency::manager::ResidencyError>{
    if let Some(group)=group {
        if !authority.is_some_and(|authority|authority.completion_policy()==group.completion_policy()){
            storage.mark_incomplete();
        }
        group.collect_idle_retained_storage(storage)?;
    }
    Ok(())
}

pub(super) fn collect_snapshot_shared_sources(
    storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    paths: Option<&eredu_runtime::SharedLayeredObservationPaths>,
    input: Option<&eredu_runtime::SharedPreparedInputCacheIdentity>,
    complete: bool,
) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
    if let Some(paths) = paths {
        storage.include_metadata(eredu_runtime::SharedHostMetadata::ObservationPaths(
            paths.clone(),
        ))?;
    }
    if let Some(input) = input {
        storage.include_metadata(eredu_runtime::SharedHostMetadata::Input(input.clone()))?;
    }
    if !complete {
        storage.mark_incomplete();
    }
    Ok(())
}
