use super::*;

use crate::backend::runtime::cache::residency::PromptCacheMaterialization;
use eredu_core::cache::SharedPromptCacheManifest;
use eredu_runtime::cache::PromptCachePersistenceFunding;
use eredu_runtime::working_memory::InferenceStateRetention;

fn text_frontier(
    layout: &eredu_runtime::StateLayout,
    positions: impl ExactSizeIterator<Item = i32>,
    expected: Option<u64>,
) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
    use eredu_runtime::working_memory::WorkingMemoryError;
    if positions.len() != layout.len() {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    let mut frontier = expected;
    let mut stateful = false;
    for (index, position) in positions.enumerate() {
        if matches!(
            layout.layer(index),
            Some(eredu_core::cache::LayerCachePolicy::NoState)
        ) {
            continue;
        }
        let actual = u64::try_from(position).map_err(|_| WorkingMemoryError::Overflow)?;
        if let Some(expected) = frontier {
            if actual != expected {
                return Err(WorkingMemoryError::StateFrontierMismatch { expected, actual });
            }
        } else {
            frontier = Some(actual);
        }
        stateful = true;
    }
    Ok(if stateful { frontier } else { None })
}

fn validate_text_frontiers(
    layout: &eredu_runtime::StateLayout,
    positions: impl ExactSizeIterator<Item = i32>,
    expected: u64,
) -> Result<(), Error> {
    text_frontier(layout, positions, Some(expected))
        .map(|_| ())
        .map_err(|cause| Error::Other(Box::new(cause)))
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
                vec![
                    eredu_core::cache::LayerCachePolicy::key_only(
                        AttentionPolicy::sliding(7).unwrap(),
                        1,
                        2,
                    )
                    .unwrap(),
                ],
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
        assert!(
            state
                .shared_layout()
                .unwrap()
                .same_storage(checkpoint.shared_layout().unwrap())
        );
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
        assert!(
            state
                .shared_layout()
                .unwrap()
                .same_storage(checkpoint.shared_layout().unwrap())
        );
    }

    #[test]
    fn every_stateful_layer_must_match_even_when_the_first_layer_has_no_state() {
        use eredu_core::{AttentionPolicy, LayerSchedule, cache::LayerCachePolicy};
        let attention = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap();
        let layout = eredu_runtime::StateLayout::new(
            LayerSchedule::new(
                3,
                vec![LayerCachePolicy::NoState, attention.clone(), attention],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            text_frontier(&layout, [0, 7, 7].into_iter(), None).unwrap(),
            Some(7)
        );
        assert!(matches!(
            text_frontier(&layout, [0, 7, 6].into_iter(), None),
            Err(
                eredu_runtime::working_memory::WorkingMemoryError::StateFrontierMismatch {
                    expected: 7,
                    actual: 6,
                }
            )
        ));
        assert!(matches!(
            text_frontier(&layout, [0, 7, -1].into_iter(), None),
            Err(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
        ));
        let stateless = eredu_runtime::StateLayout::new(
            LayerSchedule::new(
                2,
                vec![LayerCachePolicy::NoState, LayerCachePolicy::NoState],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            text_frontier(&stateless, [0, 0].into_iter(), None).unwrap(),
            None
        );
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
    Pooling,
}

pub(crate) trait MlxStateMechanisms:
    LayerRuntimeState<MlxNeuralBackend>
    + InferenceStateRetention
    + eredu_runtime::working_memory::ResidentResetProjection<MlxKeyValueState>
    + eredu_runtime::working_memory::ResidentResetProjection<MlxHybridState>
    + eredu_runtime::working_memory::ResidentResetProjection<MlxPoolingAttentionState>
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
        _capacity: eredu_core::MemoryLimits,
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

    /// Checks actual stateful layers without allocating or reading native values.
    /// A stateless rank has no physical frontier and retains logical progress in
    /// its independently authenticated request.
    fn original_text_frontier(
        &self,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError>;

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
        crate::backend::runtime::cache::state::snapshot_estimate::continuation_growth(
            self.layout(),
            u64::try_from(self.offset()).ok()?,
            additional,
            self.continuation_capacity_bound(additional)?,
            self.isolated_snapshot_auxiliary_growth(additional)?,
        )
    }
    fn offset(&self) -> i32;
    fn realize(
        selected: &SelectedStateRealization,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
        global_layer_start: usize,
        stream: &Stream,
        transfer: Option<&crate::backend::runtime::cache::residency::PreparedCacheTransferStream>,
    ) -> Result<Self, Error>;
    fn load_prompt_cache(
        source: &Self,
        selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &PromptCacheMaterialization,
    ) -> Result<(Self, SharedPromptCacheManifest), Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, Error>;
    fn residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn retained_arrays(&self) -> Vec<&Array>;
    fn ordinary_checkpoint_program(
        &self,
        plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        Option<
            crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram,
        >,
        Error,
    > {
        let _ = (plan, context);
        Ok(None)
    }
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

impl eredu_runtime::working_memory::ResidentResetProjection<MlxPoolingAttentionState>
    for MlxKeyValueState
{
    fn resident_reset_ref(&self) -> Option<&MlxPoolingAttentionState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxPoolingAttentionState> {
        None
    }
}
impl eredu_runtime::working_memory::ResidentResetProjection<MlxPoolingAttentionState>
    for MlxHybridState
{
    fn resident_reset_ref(&self) -> Option<&MlxPoolingAttentionState> {
        None
    }
    fn resident_reset_mut(&mut self) -> Option<&mut MlxPoolingAttentionState> {
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
    stream: &Stream,
    transfer: Option<&crate::backend::runtime::cache::residency::PreparedCacheTransferStream>,
) -> Result<Option<CacheResidencyManager>, Error> {
    let needs_paging = selected
        .components()
        .iter()
        .any(|component| component.placement() == StateComponentPlacement::Paged);
    match (needs_paging, selected.policy()) {
        (_, CacheResidencyPolicy::Paged(options)) => {
            let manager = CacheResidencyManager::new(options.clone())
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            let ledger = crate::backend::managed_memory::try_ledger()?;
            let transfer = transfer.ok_or_else(|| {
                Error::Other(Box::new(
                crate::backend::runtime::cache::residency::CacheTransferStreamError::Unavailable,
            ))
            })?;
            manager
                .install_transfer_stream(transfer, &ledger, stream)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            Ok(Some(manager))
        }
        (false, CacheResidencyPolicy::Device) => Ok(None),
        (true, CacheResidencyPolicy::Device) => Err(Error::Parallel(
            "selected paged state component has no paging policy".into(),
        )),
    }
}

impl MlxStateMechanisms for MlxKeyValueState {
    fn copy_original_paged_state(
        &self,
        completed: Option<&crate::backend::runtime::cache::state::CompletedResidentSource>,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        host: &eredu_core::HostPreparationAuthority,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Option<crate::backend::runtime::cache::state::OriginalResidentState>, Error> {
        if !self.as_ref().iter().any(|layer| {
            matches!(
                layer,
                crate::backend::runtime::cache::state::MlxKeyValueLayerState::Paged(_)
            )
        }) {
            return Ok(None);
        }
        let context = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(
            mechanisms,
            funding.clone(),
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
        self.copy_original_paged(
            completed,
            environment,
            initialized,
            mechanisms,
            &context,
            host,
            &capacity,
        )
        .map(|value| {
            value.map(crate::backend::runtime::cache::state::OriginalResidentState::KeyValue)
        })
    }

    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        match state {
            crate::backend::runtime::cache::state::OriginalResidentState::KeyValue(value) => {
                Ok(value)
            }
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

    fn original_text_frontier(
        &self,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        self.optional_layout().map_or(Ok(None), |layout| {
            text_frontier(
                layout,
                self.as_ref()
                    .iter()
                    .map(crate::backend::runtime::cache::kv::KeyValueCache::offset),
                None,
            )
        })
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
        stream: &Stream,
        transfer: Option<&crate::backend::runtime::cache::residency::PreparedCacheTransferStream>,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected, stream, transfer)?;
        MlxKeyValueState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        source: &Self,
        _selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &PromptCacheMaterialization,
    ) -> Result<(Self, SharedPromptCacheManifest), Error> {
        let _ = (stream, materialization);
        source
            .load_prompt_cache_funded(directory, expected, identity, prefix_token_ids, funding)
            .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, Error> {
        self.save_prompt_cache_funded(destination, descriptor, prefix_token_ids, options, funding)
            .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
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

    fn ordinary_checkpoint_program(
        &self,
        plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        Option<
            crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram,
        >,
        Error,
    > {
        MlxKeyValueState::ordinary_checkpoint_program(self, plan, context).map(Some)
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
            crate::backend::runtime::cache::state::OriginalResidentState::Hybrid(value) => {
                Ok(value)
            }
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

    fn original_text_frontier(
        &self,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        self.optional_layout().map_or(Ok(None), |layout| {
            text_frontier(layout, self.layer_positions(), None)
        })
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
        stream: &Stream,
        transfer: Option<&crate::backend::runtime::cache::residency::PreparedCacheTransferStream>,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected, stream, transfer)?;
        MlxHybridState::from_selected_with_global_layer_start(
            selected,
            manager,
            rank,
            global_layer_start,
        )
        .map_err(Into::into)
    }

    fn load_prompt_cache(
        source: &Self,
        _selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &PromptCacheMaterialization,
    ) -> Result<(Self, SharedPromptCacheManifest), Error> {
        let _ = stream;
        source
            .load_prompt_cache_funded(
                directory,
                expected,
                identity,
                prefix_token_ids,
                stream,
                funding,
                materialization,
            )
            .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, Error> {
        self.save_prompt_cache_funded(destination, descriptor, prefix_token_ids, options, funding)
            .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
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

    fn ordinary_checkpoint_program(
        &self,
        plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        Option<
            crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram,
        >,
        Error,
    > {
        MlxHybridState::ordinary_checkpoint_program(self, plan, context).map(Some)
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
    fn resident_reset_profile() -> Option<ResidentResetProfile> {
        Some(ResidentResetProfile::Pooling)
    }
    fn from_original_resident_copy(
        state: crate::backend::runtime::cache::state::OriginalResidentState,
    ) -> Result<Self, crate::backend::runtime::cache::state::OriginalResidentState> {
        match state {
            crate::backend::runtime::cache::state::OriginalResidentState::Pooling(value) => {
                Ok(value)
            }
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

    fn original_text_frontier(
        &self,
    ) -> Result<Option<u64>, eredu_runtime::working_memory::WorkingMemoryError> {
        self.optional_layout().map_or(Ok(None), |layout| {
            text_frontier(
                layout,
                self.as_ref().iter().map(|layer| layer.offset()),
                None,
            )
        })
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
        stream: &Stream,
        transfer: Option<&crate::backend::runtime::cache::residency::PreparedCacheTransferStream>,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_allocation();
        let manager = selected_state_manager(selected, stream, transfer)?;
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
        source: &Self,
        _selected: &SelectedStateRealization,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        stream: &Stream,
        funding: &PromptCachePersistenceFunding,
        materialization: &PromptCacheMaterialization,
    ) -> Result<(Self, SharedPromptCacheManifest), Error> {
        let _ = stream;
        MlxPoolingAttentionStateFactory::load_prompt_cache_funded(
            source,
            directory,
            expected,
            identity,
            prefix_token_ids,
            stream,
            funding,
            materialization,
        )
        .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<SharedPromptCacheManifest, Error> {
        MlxPoolingAttentionStateFactory::save_prompt_cache_funded(
            self,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            funding,
        )
        .map_err(|cause| Error::Neural(funding.context().metadata_source(cause)))
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

    fn ordinary_checkpoint_program(
        &self,
        plan: &eredu_runtime::working_memory::InferenceSpanWorkspacePlan,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        Option<
            crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram,
        >,
        Error,
    > {
        let Some(layout) = self.shared_layout() else {
            return Ok(None);
        };
        let Some(metadata) = self.layer_slot_metadata() else {
            return Ok(None);
        };
        crate::backend::runtime::cache::state::ordinary_checkpoint::OrdinaryCheckpointProgram::prepare(layout, metadata, self.as_ref(), std::iter::empty(),
            self.inference_retention(), plan, context).map(Some)
    }
    fn deep_checkpoint(&self) -> Result<Self, Exception> {
        // Pooling updates replace arrays. Retain exact views and the paged
        // sliding history that speculative verification could otherwise discard.
        let layout = self.shared_layout().cloned().ok_or_else(|| {
            Exception::custom("pooling checkpoint requires an owned state layout")
        })?;
        if let Some(metadata) = self.layer_slot_metadata() {
            if let Some(loan) = crate::backend::runtime::cache::state::ordinary_checkpoint::begin(
                &layout,
                metadata,
                std::iter::empty(),
                self.inference_retention(),
            )? {
                let layers = loan.table(
                    self.as_ref(),
                    MlxPoolingAttentionCache::checkpoint_clone_state,
                )?;
                let retention = loan.retention(self.inference_retention())?;
                let mut checkpoint = Self::from_prepared_layers(layout, layers)
                    .map_err(|e| loan.retain(Exception::from_source(e)))?;
                *checkpoint.inference_retention_mut() = retention;
                return Ok(checkpoint);
            }
        }
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
    storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    group: Option<&crate::backend::runtime::distributed::Group>,
    authority: Option<&eredu_runtime::PartitionCommunicationAuthority>,
) -> Result<(), crate::backend::runtime::residency::manager::ResidencyError> {
    if let Some(group) = group {
        if !authority
            .is_some_and(|authority| authority.completion_policy() == group.completion_policy())
        {
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
