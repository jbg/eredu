use super::*;
use crate::backend::{managed_memory::NativeMemoryOwner, submission_recovery};
use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_runtime::working_memory::{InferenceRetention, WorkingMemoryError, WorkingMemoryPool};

pub(super) mod parameters;
pub(super) use parameters::MlxPredictionModule;
pub(crate) use parameters::{prepare_prediction_parameters, NativePredictionParameters, PredictionParameterStorage, MlxWorkspacePredictionParameterSource};
mod parameter_sources;
pub(crate) use parameter_sources::{
    CountedNativeParameterOwnerSource, NativeParameterOwnerSource, ParameterOwnerCounts,
    ParameterOwnerRole, ParameterOwnerRoleCounts, ParameterOwnerSourceError,
};
mod owned_cache;
mod snapshot;
mod startup;
mod original_copy;
pub(crate) mod workspace;
pub(crate) mod phase;
pub(crate) use phase::{MlxPredictionPhase, complete_original_prediction_state};
pub(crate) use original_copy::{OriginalPredictionCopyContext, OriginalPredictionTarget, OriginalEmbeddedCachePreparation, cache_metadata, cache_error, prepare_cache};
use super::super::speculative::SpeculativeExecutionStreams;
use owned_cache::OwnedPredictionCache;
pub(crate) use startup::{OriginalPredictionLane, OriginalPredictionStartupContext, PreparedLane, StartupCause};

#[cfg(test)]
mod memory_tests;

#[cfg(test)]
pub(super) type StatePresenceSnapshot = Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;

#[cfg(test)]
pub(super) type FixedNumericStateSnapshot = Vec<(
    usize,
    eredu_core::cache::StateTensorRole,
    Vec<i32>,
    Vec<f32>,
)>;

#[cfg(test)]
pub(super) type RetainedNumericStateSnapshot = Vec<(Vec<i32>, Vec<f32>)>;

#[cfg(test)]
pub(super) type CheckpointRestoreProbe = (
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    Vec<f32>,
);

pub(super) trait MlxParameterBankTelemetry {
    fn parameter_banks(
        &self,
    ) -> Result<std::collections::BTreeMap<eredu_runtime::RoutedBankId, crate::backend::runtime::residency::parameter_bank::IndexedBankSource>,Error> {
        Ok(Default::default())
    }
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    >;
}

impl MlxParameterBankTelemetry for eredu_runtime::DirectReplicatedTextExecution {
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<
        eredu_runtime::RoutedBankProviders<eredu_architectures::routed_text::PlannedResidentBank>,
    >
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl<E, G, R, I, T, U, V> MlxParameterBankTelemetry
    for eredu_runtime::PartitionedTextExecution<E, G, R, I, T, U, V>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

pub(super) type MlxAddressableBanks = eredu_runtime::RoutedBankProviders<
    eredu_architectures::routed_text::PlannedAddressableBank<
        MlxNeuralBackend,
        crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
    >,
>;
impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<MlxAddressableBanks>
{
    fn parameter_banks(
        &self,
    ) -> Result<std::collections::BTreeMap<eredu_runtime::RoutedBankId, crate::backend::runtime::residency::parameter_bank::IndexedBankSource>,Error> {
        self.provider()
            .banks()
            .iter()
            .map(|(id, provider)| provider.indexed_movement().indexed_bank_source().cloned()
                .map(|source|(*id,source)).ok_or(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)))
            .collect()
    }
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        let banks = self
            .provider()
            .banks()
            .iter()
            .map(|(id, provider)| {
                provider
                    .bank_report()
                    .map(|report| (*id, report))
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<_, _>>()?;
        Ok(Some(
            crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport::new(
                banks,
            ),
        ))
    }
}

/// Borrowed exact state roots during the existing sequence completion phase.
/// Implementations visit actual storage; they allocate no inventory or clones.
pub(crate) trait AutoregressiveStateRoots {
    fn visit_roots(&self, visitor: &mut dyn FnMut(&Array)) -> Result<(), Error>;
}
impl<S: MlxStateMechanisms> AutoregressiveStateRoots for S {
    fn visit_roots(&self, visitor: &mut dyn FnMut(&Array)) -> Result<(), Error> {
        eredu_runtime::RuntimeState::visit_all_retained_values(self, &mut |value: &MlxTensor| {
            visitor(value.as_array())
        })
        .map_err(Error::PrefillState)
    }
}
/// Closed typed ingress erased only after the selected architecture constructs it.
/// The concrete owner and its Box allocation retire before the original funding.
pub(crate) struct OriginalAutoregressiveMediaPrefill {
    source: Box<dyn std::any::Any>,
    visit: fn(&dyn std::any::Any, &mut dyn FnMut(&Array)),
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
impl OriginalAutoregressiveMediaPrefill {
    pub(super) fn new<T: 'static>(source:T,
        visit:fn(&dyn std::any::Any, &mut dyn FnMut(&Array)),
        metadata:&eredu_nn::workspace::WorkspaceContext) -> Result<Self,Error> {
        let funding=metadata.metadata_funding().ok_or(Error::PrefillScopeUnavailable)?;
        metadata.charge_metadata(std::mem::size_of::<(T,Self,Result<Self,Error>,
            fn(&dyn std::any::Any, &mut dyn FnMut(&Array)),
            &eredu_nn::workspace::WorkspaceContext)>())
            .map_err(|cause|Error::Neural(cause.into()))?;
        Ok(Self { source:Box::new(source), visit, _funding:funding })
    }
    pub(super) fn downcast_mut<T:'static>(&mut self)->Option<&mut T> { self.source.downcast_mut() }
    pub(crate) fn visit_retained_roots(&self, visitor:&mut dyn FnMut(&Array)) {
        (self.visit)(self.source.as_ref(), visitor)
    }
}
pub(crate) trait AutoregressiveSequenceCompletion {
    fn metadata_context(&self) -> eredu_nn::workspace::WorkspaceContext;
    fn role(&self) -> &eredu_runtime::working_memory::OriginalSpeculativeRole;
    fn metadata_funding(&self) -> eredu_nn::workspace::HostMetadataFunding;
    fn take_checkpoint(&mut self) -> Result<MlxPredictionTargetState, Error>;
    fn complete(
        &mut self,
        output: Option<&Array>,
        state: &dyn AutoregressiveStateRoots,
        stream: &Stream,
    ) -> Result<(), Error>;
}

pub(super) trait ErasedPredictionTargetState: std::any::Any {
    fn control_estimate(&self) -> Option<eredu_core::execution_control::SnapshotEstimate>;
    fn control_copy(
        &self,
        stream: &Stream,
    ) -> Result<Box<dyn ErasedPredictionTargetState>, Exception>;
    fn retain_memory_owner(&mut self, owner: &NativeMemoryOwner) -> Result<(), Error>;
    fn retention(&self) -> &InferenceRetention;
    fn inherit_retention(&mut self, source: &InferenceRetention);

    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception>;
    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
    fn offset(&self) -> i32;
    fn project_workspace(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error>;
    fn prepare_original_copy(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    >;
    fn visit_original_roots(&self, visitor: &mut dyn FnMut(&Array)) -> Result<(), Error>;
}

impl<S> ErasedPredictionTargetState for S
where
    S: MlxStateMechanisms + 'static,
{
    fn control_estimate(&self) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.isolated_snapshot_estimate()
    }
    fn control_copy(
        &self,
        stream: &Stream,
    ) -> Result<Box<dyn ErasedPredictionTargetState>, Exception> {
        super::super::speculative::state_snapshot::settle(self.retained_arrays())?;
        self.isolated_snapshot(stream)
            .map(|state| Box::new(state) as Box<dyn ErasedPredictionTargetState>)
    }
    fn retain_memory_owner(&mut self, owner: &NativeMemoryOwner) -> Result<(), Error> {
        self.inference_retention_mut()
            .retain_unquoted(&owner.unquoted_lease()?);
        super::super::speculative::state_snapshot::settle(self.retained_arrays())?;
        for array in self.retained_arrays() {
            owner.retain_array(array)?;
        }
        Ok(())
    }
    fn retention(&self) -> &InferenceRetention {
        self.inference_retention()
    }
    fn inherit_retention(&mut self, source: &InferenceRetention) {
        self.inference_retention_mut().extend_from(source);
    }
    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception> {
        self.deep_checkpoint()
            .map(|state| Box::new(state) as Box<dyn ErasedPredictionTargetState>)
    }

    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let checkpoint = checkpoint
            .as_any()
            .downcast_ref::<S>()
            .ok_or_else(|| Exception::custom("prediction target checkpoint state type changed"))?;
        self.restore_checkpoint(checkpoint, stream)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn offset(&self) -> i32 {
        MlxStateMechanisms::offset(self)
    }
    fn project_workspace(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        self.project_resident_workspace_with_storage(
            std::num::NonZeroU32::new(1).expect("one"),
            context,
        )
    }
    fn prepare_original_copy(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        self.prepare_resident_decoder_copy_fixed()
    }
    fn visit_original_roots(&self, visitor: &mut dyn FnMut(&Array)) -> Result<(), Error> {
        eredu_runtime::RuntimeState::visit_all_retained_values(self, &mut |value: &MlxTensor| {
            visitor(value.as_array())
        })
        .map_err(Error::PrefillState)
    }
}

/// Opaque MLX storage for one ordinary target lane.
///
/// Neutral prediction membership and transaction metadata live in
/// `EmbeddedPredictionCache`; this wrapper supplies only native clone,
/// restore, type transfer, and frontier inspection mechanisms.
pub(crate) struct MlxPredictionTargetState(
    Option<Box<dyn ErasedPredictionTargetState>>,
    // Erased payload and its Box shell retire before original constructor H.
    Option<eredu_core::HostPreparationAuthority>,
    Option<crate::backend::runtime::cache::state::CompletedResidentSource>,
);

impl MlxPredictionTargetState {
    pub(crate) fn control_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = self.0.as_ref()?.control_estimate()?;
        let authority =
            std::mem::size_of::<eredu_runtime::working_memory::WorkingMemoryUnquotedLease>() as u64;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(authority)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(authority)?;
        Some(estimate)
    }
    pub(crate) fn control_copy(
        &self,
        stream: &Stream,
        pool: &WorkingMemoryPool,
    ) -> Result<Self, Exception> {
        let owner = NativeMemoryOwner::acquire(pool).map_err(Exception::from_source)?;
        submission_recovery::detached_retained(owner.clone(), || {
            let mut copy = self
                .0
                .as_ref()
                .ok_or_else(|| Exception::custom("prediction target state is active"))?
                .control_copy(stream)?;
            copy.retain_memory_owner(&owner)?;
            Ok(Self(Some(copy), self.1.clone(), None))
        })
        .map_err(Exception::from_source)
    }

    /// Copies the actual independent lane through the shared original source
    /// projection, numerical admission, isolated workers and publication path.
    pub(crate) fn copy_original(
        &self,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        capacity: u64,
    ) -> Result<Self, Error> {
        self.with_original_copy_source(funding, |source, completed| {
            Self::copy_original_source_with_completion(
                source,
                completed,
                environment,
                initialized,
                mechanisms,
                funding,
                capacity,
            )
        })
    }

    /// Borrows the settled source selected by the existing copy preparation.
    /// The callback cannot export a source borrow or manufacture completion
    /// evidence; leaf copying still authenticates every actual native backing.
    pub(crate) fn with_original_copy_source<T>(
        &self,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        operation: impl for<'source> FnOnce(
            crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'source>,
            Option<&'source crate::backend::runtime::cache::state::CompletedResidentSource>,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        use crate::backend::runtime::cache::state::{
            CompletedResidentSource, PreparedResidentDecoderCopy,
        };
        let frames = [
            std::mem::size_of_val(&operation),
            std::mem::size_of::<PreparedResidentDecoderCopy<'_>>(),
            std::mem::size_of::<Option<&CompletedResidentSource>>(),
            std::mem::size_of::<Result<T, Error>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let source = self
            .0
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .prepare_original_copy()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        operation(source, self.2.as_ref())
    }
    pub(crate) fn copy_original_source(
        source: crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        capacity: u64,
    ) -> Result<Self, Error> {
        Self::copy_original_source_with_completion(
            source,
            None,
            environment,
            initialized,
            mechanisms,
            funding,
            capacity,
        )
    }
    fn copy_original_source_with_completion(
        source: crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        completed: Option<&crate::backend::runtime::cache::state::CompletedResidentSource>,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        capacity: u64,
    ) -> Result<Self, Error> {
        use crate::backend::runtime::cache::state::{
            OriginalResidentState, copy_completed_resident_state,
        };
        use eredu_core::HostPreparationAuthority;
        let bytes = HostPreparationAuthority::retention_bytes::<
            eredu_nn::workspace::HostMetadataFunding,
        >()
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let host = HostPreparationAuthority::retain(funding.clone());
        let state = copy_completed_resident_state(
            source,
            completed,
            environment,
            initialized,
            mechanisms,
            funding,
            &host,
            capacity,
        )?;
        let bytes = match &state {
            OriginalResidentState::KeyValue(state) => std::mem::size_of_val(state),
            OriginalResidentState::Hybrid(state) => std::mem::size_of_val(state),
            OriginalResidentState::Pooling(state) => std::mem::size_of_val(state),
        };
        let bytes = bytes
            .checked_add(std::mem::size_of::<Self>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Result<Self, Error>>()))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        Ok(match state {
            OriginalResidentState::KeyValue(state) => Self(Some(Box::new(state)), Some(host), None),
            OriginalResidentState::Hybrid(state) => Self(Some(Box::new(state)), Some(host), None),
            OriginalResidentState::Pooling(state) => Self(Some(Box::new(state)), Some(host), None),
        })
    }

    /// Called after successful native completion and exact Recovery retirement.
    /// The state keeps source facts through typed exchanges, while a copied
    /// checkpoint starts with its independently published registered backing.
    pub(crate) fn publish_completed_original_source(
        &mut self,
        budget: &safemlx::OriginalBufferBudget,
        role: &eredu_runtime::working_memory::OriginalSpeculativeRole,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        stream: &safemlx::Stream,
        additional: impl FnMut(&mut dyn FnMut(&Array)) -> Result<(), Error>,
    ) -> Result<(), Error> {
        use crate::backend::runtime::cache::state::{
            CompletedResidentSource, PreparedResidentDecoderCopy, ResidentDecoderPreparationError,
        };
        let frames = [
            std::mem::size_of::<PreparedResidentDecoderCopy<'_>>(),
            std::mem::size_of::<ResidentDecoderPreparationError>(),
            std::mem::size_of::<
                Result<PreparedResidentDecoderCopy<'_>, ResidentDecoderPreparationError>,
            >(),
            std::mem::size_of::<CompletedResidentSource>(),
            std::mem::size_of::<Option<CompletedResidentSource>>(),
            std::mem::size_of::<Result<(), Error>>(),
            std::mem::size_of_val(&additional),
            std::mem::size_of::<(
                &mut Self,
                &safemlx::OriginalBufferBudget,
                &eredu_runtime::working_memory::OriginalSpeculativeRole,
                &safemlx::Stream,
            )>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        // The selected validator produces a fixed cause. metadata_source then
        // reserves its actual erased destination before constructing that Box.
        let source = self
            .0
            .as_ref()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .prepare_original_copy()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        let completed = crate::backend::runtime::cache::state::CompletedResidentSource::capture(
            &source,
            self.2.as_ref(),
            budget,
            role,
            funding,
            additional,
        )?.with_completed_stream(stream)?;
        self.2 = Some(completed);
        Ok(())
    }

    /// Retains an already acquired owner before this state can escape its scope.
    pub(crate) fn retain_memory_owner(&mut self, owner: &NativeMemoryOwner) -> Result<(), Error> {
        self.0
            .as_mut()
            .ok_or_else(|| Exception::custom("prediction target state is active"))?
            .retain_memory_owner(owner)
    }

    /// Preserves existing backing owners before an atomic state replacement.
    pub(crate) fn inherit_retention(&mut self, source: &Self) -> Result<(), Exception> {
        let source = source
            .0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target checkpoint is active"))?;
        self.0
            .as_mut()
            .ok_or_else(|| Exception::custom("prediction target state is active"))?
            .inherit_retention(source.retention());
        Ok(())
    }

    pub(crate) fn new<S: MlxStateMechanisms + 'static>(state: S) -> Self {
        Self(Some(Box::new(state)), None, None)
    }

    pub(super) fn is<S: 'static>(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|state| state.as_ref().as_any().is::<S>())
    }

    /// Loans the existing erased allocation across both state exchanges. The
    /// same Box remains owned here on success, failure, recovery and unwind.
    pub(super) fn state_mut<S: 'static>(&mut self) -> Option<&mut S> {
        self.0.as_mut()?.as_any_mut().downcast_mut::<S>()
    }

    pub(super) fn state_and_original_source_mut<S: 'static>(&mut self) -> Option<(
        &mut S, &mut Option<crate::backend::runtime::cache::state::CompletedResidentSource>
    )> {
        Some((self.0.as_mut()?.as_any_mut().downcast_mut::<S>()?, &mut self.2))
    }

    pub(super) fn take_state<S: 'static>(&mut self) -> Result<S, Error> {
        Ok(*self
            .0
            .take()
            .ok_or_else(|| {
                Error::ArchitectureModel("prediction target state is already active".into())
            })?
            .into_any()
            .downcast::<S>()
            .expect("prediction target state type checked before transfer"))
    }

    pub(super) fn restore_state<S: MlxStateMechanisms + 'static>(&mut self, state: S) {
        self.0 = Some(Box::new(state));
    }

    pub(crate) fn deep_clone(&self) -> Result<Self, Exception> {
        self.0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?
            .deep_clone_box()
            .map(|state| Self(Some(state), self.1.clone(), None))
    }

    pub(crate) fn restore(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        let current = self
            .0
            .as_mut()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?;
        let checkpoint = checkpoint
            .0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target checkpoint is active"))?;
        current.restore_box(checkpoint.as_ref(), stream)
    }

    /// Borrows this independent lane, never the model's installed session state.
    pub(crate) fn project_workspace(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        self.0
            .as_ref()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?
            .project_workspace(context)
    }
    pub(crate) fn visit_original_roots(
        &self,
        visitor: &mut dyn FnMut(&Array),
    ) -> Result<(), Error> {
        self.0
            .as_ref()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?
            .visit_original_roots(visitor)
    }
    pub(crate) fn matches_media_binding(
        &self,
        execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
        binding: &eredu_runtime::working_memory::MediaSessionBinding,
    ) -> bool {
        let Some(state) = self.0.as_ref() else { return false; };
        self.generation_fixed().is_some_and(|frontier| binding.matches_retained_state(execution, state.retention(), frontier))
    }
    pub(crate) fn generation_fixed(&self) -> Option<u64> {
        u64::try_from(self.0.as_ref()?.offset()).ok()
    }
    pub(crate) fn generation(&self) -> Result<u64, Error> {
        let state = self.0.as_ref().ok_or_else(|| {
            Error::ArchitectureModel("prediction target state is already active".into())
        })?;
        u64::try_from(state.offset())
            .map_err(|_| Error::ArchitectureModel("target capture generation is negative".into()))
    }
}

pub(super) struct ExactPredictionCaptureObserver {
    paths: Vec<String>,
    pub(super) values: std::rc::Rc<std::cell::RefCell<Vec<Option<MlxTensor>>>>,
}

impl ExactPredictionCaptureObserver {
    pub(super) fn new(paths: Vec<String>) -> Result<Self, Error> {
        if paths.is_empty() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture declares no target paths".into(),
            ));
        }
        let unique = paths.iter().collect::<std::collections::BTreeSet<_>>();
        if unique.len() != paths.len() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture paths are not unique".into(),
            ));
        }
        let values = std::rc::Rc::new(std::cell::RefCell::new(vec![None; paths.len()]));
        Ok(Self { paths, values })
    }
}

impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>
    for ExactPredictionCaptureObserver
{
    fn requires_sequence_readout(&self) -> bool {
        self.paths
            .iter()
            .any(|path| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &MlxTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        if self.paths.iter().any(|expected| expected == path) {
            self.observe(path, &generate()?)?;
        }
        Ok(())
    }

    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        if let Some(index) = self.paths.iter().position(|expected| expected == path) {
            if self.values.borrow_mut()[index]
                .replace(value.clone())
                .is_some()
            {
                return Err(eredu_nn::Error::backend(format!(
                    "external-assistant target reached capture path {path} more than once"
                )));
            }
        }
        Ok(())
    }
}

pub(super) struct CompositePredictionTargetOperation<'a> {
    pub(super) operation: ExternalPredictionTargetOperation<'a, MlxTensor>,
}

impl<A>
    eredu_runtime::PredictionTargetOperation<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
    > for CompositePredictionTargetOperation<'_>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = MlxTensor;

    fn preserves_architecture_declarations(&self) -> bool { true }

    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _state: &mut MlxHybridState,
        parallel: Option<&<MlxNeuralBackend as NeuralBackend>::ParallelContext>,
        context: &Stream,
    ) -> Result<Self::Output, eredu_nn::Error> {
        if parallel.is_some() {
            return Err(eredu_nn::Error::backend(
                "external assistant target operations are unavailable under tensor parallelism",
            ));
        }
        architecture
            .inner_mut()
            .external_prediction_target_operation(self.operation, context)?
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "architecture does not implement the selected external target operation",
                )
            })
    }
}

pub(crate) struct MlxEmbeddedPredictionMaterializer;

pub(crate) type MaterializedEmbeddedPrediction =
    eredu_architectures::prediction_extension::MaterializedPredictionExtension<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    >;

pub(super) fn materialize_prepared_prediction_unit<M>(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
    layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    store: RetainedCheckpointSource,
    stream: &Stream,
    _weights_stream: &Stream,
) -> Result<MlxPredictionModule<M>, Error>
where
    M: Parameterized<MlxTensor>,
{
    use crate::backend::runtime::checkpoint::binding::build_mlx_exact_replicated_text_bindings;

    let residency = prepared.residency();
    let shared =
        prepared.role() == eredu_architectures::prediction_extension::PredictionModuleRole::Shared;
    let source_layout = prepared.source_layout().cloned();
    let (source, mut local, selected_tasks) = prepared.into_shared_parts();
    let mut store = store;
    let mut materialization = eredu_runtime::WeightMaterializationReport::default();
    for group in eredu_runtime::group_replicated_text_transform_tasks(&selected_tasks)
        .map_err(|error| Error::Quantization(error.to_string()))?
    {
        let tasks = group
            .tasks(&selected_tasks)
            .map_err(|error| Error::Quantization(error.to_string()))?;
        let (transformed, report) = quantize_exact_replicated_text_tasks(
            store,
            &source,
            &local,
            &[] as &[M],
            &[],
            source_layout.as_deref(),
            group.quantization(),
            &tasks,
            stream,
        )?;
        store = transformed;
        materialization.merge(report);
    }
    let task_refs = selected_tasks.iter().collect::<Vec<_>>();
    let bindings = build_mlx_exact_replicated_text_bindings(
        &local,
        store.as_ref(),
        &task_refs,
        &std::collections::BTreeSet::new(),
        layout.as_deref(),
    )?;
    let mut parameters = Vec::new();
    let mut declarations = Vec::new();
    super::session::prepared_parameters::collect_module(
        &local,
        &bindings,
        eredu_runtime::parameter_operations::PreparedParameterLocation::Prediction { module: 0 },
        store.as_ref(),
        &mut parameters,
        &mut declarations,
    )?;
    if parameters.len() != declarations.len() {
        return Err(Error::ArchitectureModel(
            "prediction parameters lack exact prepared binding owners".into(),
        ));
    }
    Ok(MlxPredictionModule {
        placeholders: parameters::placeholders(&mut local, stream)?,
        replacements: BTreeMap::new(),
        source: store,
        bindings,
        residency,
        shared,
        manager: Arc::new(std::sync::OnceLock::new()),
        id: None,
        original: None,
        stream: stream.clone(),
        inner: local,
        parameters,
        tasks: selected_tasks,
        layout,
        materialization,
    })
}

pub(crate) fn materialize_prediction_extension(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionExtension<
        MlxNeuralBackend,
    >,
    store: RetainedCheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedEmbeddedPrediction, Error> {
    let mut context = MlxPredictionMaterializationContext {
        store,
        stream,
        weights_stream,
        layouts: Vec::new(),
    };
    prepared.materialize::<MlxEmbeddedPredictionMaterializer>(&mut context)
}

pub(crate) struct MlxPredictionMaterializationContext<'a> {
    store: RetainedCheckpointSource,
    stream: &'a Stream,
    weights_stream: &'a Stream,
    // Ordinary loaded-source construction shares each actual placement layout.
    // Module aliases retain it after this temporary construction context retires.
    layouts: Vec<Arc<eredu_runtime::LocalModelLayout>>,
}

impl eredu_architectures::prediction_extension::PredictionExtensionMaterializer<MlxNeuralBackend>
    for MlxEmbeddedPredictionMaterializer
{
    type Error = Error;
    type Module<M> = MlxPredictionModule<M>;
    type PoolingState =
        OwnedPredictionCache<crate::backend::runtime::cache::state::MlxPoolingAttentionCache>;
    type SequentialState =
        OwnedPredictionCache<crate::backend::runtime::cache::kv::CompressedLatentCache>;
    type ModelState = MlxHybridState;
    type Context<'a> = MlxPredictionMaterializationContext<'a>;
    type SnapshotContext<'a> = SpeculativeExecutionStreams<'a>;

    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a MlxTensor>,
        _context: &Stream,
    ) -> Result<(), eredu_core::BackendFailure> {
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            values
                .into_iter()
                .map(|value| value.as_array())
                .chain(token_validations.iter()),
        )
        .and_then(|completion| completion.synchronize())
        .and_then(|()| validate_active_token_validations())
        .map_err(eredu_core::BackendFailure::from_error)
    }

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<MlxTensor>,
    {
        let layout = layout.map(|layout| {
            if let Some(retained) = context.layouts.iter().find(|retained| retained.as_ref() == layout) {
                return retained.clone();
            }
            let retained = Arc::new(layout.clone());
            context.layouts.push(retained.clone());
            retained
        });
        materialize_prepared_prediction_unit(
            prepared,
            layout,
            context.store.clone(),
            context.stream,
            context.weights_stream,
        )
    }

    fn invoke_module<U, O>(
        module: &mut Self::Module<U>,
        context: &Stream,
        operation: impl FnOnce(
            &mut U,
        )
            -> eredu_architectures::prediction_extension::PredictionInvocation<
            MlxTensor,
            O,
        >,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<MlxTensor>,
    {
        module
            .invoke(context, |inner| {
                let (outcome, roots) = operation(inner).into_parts();
                (outcome.map_err(Error::from), roots)
            })
            .map_err(|error| match error {
                Error::Neural(error) => error,
                error => eredu_nn::Error::backend_retained_source(error),
            })
    }

    fn invoke_module_with_roots<U, O>(
        module: &mut Self::Module<U>,
        context: &Stream,
        operation: impl FnOnce(&mut U, Option<&mut dyn eredu_architectures::prediction_extension::PreparedPredictionInvocationRoots<MlxTensor>>)
            -> eredu_architectures::prediction_extension::PredictionInvocation<MlxTensor,O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<MlxTensor>,
    {
        module.invoke_with_roots(context, operation)
    }

    fn invoke_module_with_shared_roots<U, V, O>(
        module: &mut Self::Module<U>,
        shared: Option<&mut Self::Module<V>>,
        context: &Stream,
        operation:impl FnOnce(&mut U,Option<&mut V>,Option<&mut dyn eredu_architectures::prediction_extension::PreparedPredictionInvocationRoots<MlxTensor>>)
            -> eredu_architectures::prediction_extension::PredictionInvocation<MlxTensor,O>,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<MlxTensor>,
        V: Parameterized<MlxTensor>,
    {
        if module.original.is_none() {
            return Self::invoke_module_with_shared(module, shared, context, |module, shared| {
                operation(module, shared, None)
            });
        }
        parameters::original::invoke_shared(module, shared, context, operation)
    }

    fn pooling_state(
        _context: &mut Self::Context<'_>,
        ordinal: usize,
        policy: eredu_core::cache::LayerCachePolicy,
    ) -> Result<Self::PoolingState, Self::Error> {
        Ok(OwnedPredictionCache::new(
            crate::backend::runtime::cache::state::MlxPoolingAttentionCache::resident_from_policy(
                ordinal, &policy,
            )?,
            Default::default(),
        ))
    }

    fn model_state(
        _context: &mut Self::Context<'_>,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self::ModelState, Self::Error> {
        Ok(MlxHybridState::device(layout)?)
    }

    fn sequential_state() -> Self::SequentialState {
        OwnedPredictionCache::new(
            crate::backend::runtime::cache::kv::CompressedLatentCache::new(),
            Default::default(),
        )
    }

    fn pooling_snapshot_estimate(
        state: &Self::PoolingState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = if let Ok(source)=state.inner().prepare_isolated_copy_fixed() {
            super::super::speculative::state_snapshot::estimate_visited::<Self::PoolingState>(
                |visitor|source.visit_operands(visitor),
            )?
        } else {
            // Existing ordinary paged snapshots retain their manager-copy path.
            // Original prepared prediction copies require resident leaf sources.
            super::super::speculative::state_snapshot::estimate_state::<Self::PoolingState>(
                state.inner().retained_arrays(),
            )?
        };
        let auxiliary = state
            .inner()
            .residency_manager()
            .map(|manager| manager.isolated_snapshot_bytes())
            .unwrap_or(Some(0))?;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(auxiliary)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(auxiliary)?;
        snapshot::with_cache_ownership(estimate, state)
    }

    fn pooling_snapshot(
        state: &Self::PoolingState,
        context: Self::SnapshotContext<'_>,
    ) -> Result<Option<Self::PoolingState>, eredu_core::BackendFailure> {
        use crate::backend::runtime::cache::state::MlxPoolingAttentionCache;
        snapshot::copy_cache(
            state,
            context,
            MlxPoolingAttentionCache::retained_arrays,
            MlxPoolingAttentionCache::isolated_snapshot,
        )
        .map(Some)
    }

    fn sequential_snapshot_estimate(
        state: &Self::SequentialState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if state.inner().is_paged() {
            return None;
        }
        let estimate = super::super::speculative::state_snapshot::estimate_state::<
            Self::SequentialState,
        >(state.inner().borrowed_retained_values().map(|value|value.as_array()))?;
        snapshot::with_cache_ownership(estimate, state)
    }

    fn sequential_snapshot(
        state: &Self::SequentialState,
        context: Self::SnapshotContext<'_>,
    ) -> Result<Option<Self::SequentialState>, eredu_core::BackendFailure> {
        if state.inner().is_paged() {
            return Ok(None);
        }
        use crate::backend::runtime::cache::kv::CompressedLatentCache;
        snapshot::copy_cache(
            state,
            context,
            CompressedLatentCache::retained_arrays,
            CompressedLatentCache::isolated_snapshot,
        )
        .map(Some)
    }

    fn model_snapshot_estimate(
        state: &Self::ModelState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = state.original_isolated_snapshot_estimate()?;
        let authority =
            std::mem::size_of::<eredu_runtime::working_memory::WorkingMemoryUnquotedLease>() as u64;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(authority)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(authority)?;
        Some(estimate)
    }

    fn model_snapshot(
        state: &Self::ModelState,
        context: Self::SnapshotContext<'_>,
    ) -> Result<Option<Self::ModelState>, eredu_core::BackendFailure> {
        super::super::prepared_speculative::copy_control_state(state, context)
            .map_err(eredu_core::BackendFailure::from_error)
    }
}

impl eredu_architectures::prediction_extension::PredictionModelState<MlxNeuralBackend>
    for MlxHybridState
{
    type LayerState = crate::backend::runtime::cache::state::MlxHybridLayerState;

    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.layers_mut()
    }
}

pub(super) struct SelectedPrediction<P> {
    pub(super) extension: P,
    pub(super) selected: eredu_runtime::SelectedSpeculativeRealization,
}

pub(super) struct NoSelectedPrediction;

pub(super) trait ReplicatedPredictionCapability<A, S, D>: Sized
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>>;

    /// The actual retained extension supplies the source to the paid destination.
    fn prepare_original_prediction(
        _model: &CompletedReplicatedText<A, S, D, Self>,
        _context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        None
    }

    fn present() -> bool;
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
    ) -> Result<(), Error>;

    fn count_parameter_owners(
        &self,
        _counts: &mut ParameterOwnerCounts,
        _guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        Err(ParameterOwnerSourceError::PredictionUnavailable)
    }

    fn publish_parameter_replacements(
        &mut self,
        _values: &BTreeMap<String, MlxTensor>,
        _active: bool,
    ) {
    }
    fn visit_parameter_slots(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) {
    }
    fn with_parameter_slots(
        &mut self,
        _module: usize,
        _operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        None
    }
}

impl<A, S, D> ReplicatedPredictionCapability<A, S, D> for NoSelectedPrediction
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Unit: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
{
    fn lend(
        _: &mut CompletedReplicatedText<A, S, D, Self>,
        _: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }

    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        <Self as ReplicatedPredictionCapability<A, S, D>>::collect_retained_storage(
            self,
            &mut storage,
        )?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn count_parameter_owners(
        &self,
        _counts: &mut ParameterOwnerCounts,
        _guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        Ok(())
    }

    fn present() -> bool {
        false
    }
}

pub(crate) trait ErasedExternalPredictionExecutable: 'static {
    fn prepare_external_prediction_target_cache(
        &mut self,
    ) -> Result<MlxPredictionTargetState, Error>;
    fn prefill_external_prediction_target(
        &mut self,
        input: input::ModelInput<'_>,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn prefill_external_prediction_spans(
        &mut self,
        input: crate::composition::mlx::MlxModelInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
            MlxTensor,
            Error,
        >,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_runtime::replicated_session::PrefillSourceProgress<Option<MlxTensor>>, Error>;
    fn verify_external_prediction_target_with_evidence(
        &mut self, tokens: &MlxTensor, request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState, context: SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_architectures::external_assistant::ExternalTargetResult<MlxTensor>, Error>;
    fn verify_external_prediction_target(
        &mut self,
        tokens: &MlxTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn apply_external_prediction_target_operation_with_source(
        &mut self,operation:ExternalPredictionTargetOperation<'_,MlxTensor>,
        context:SpeculativeExecutionStreams<'_>,
    )->Result<eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,Error>;
    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, MlxTensor>,
    ) -> Result<MlxTensor, Error>;
}

#[derive(Debug)]
pub(crate) struct CompletedMediaBindingError {
    cause: Option<eredu_runtime::replicated_session::MediaSemanticBindingError<Error>>,
    original: eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
}
impl CompletedMediaBindingError {
    pub(crate) fn boundary(
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        cause: eredu_runtime::working_memory::WorkingMemoryError,
    ) -> Self {
        Self {
            cause: None,
            original: original.reject_boundary(cause),
        }
    }
    pub(crate) fn mechanism(
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        cause: eredu_runtime::replicated_session::MediaSemanticBindingError<Error>,
    ) -> Self {
        Self {
            cause: Some(cause),
            original: original
                .reject_boundary(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound),
        }
    }
    pub(crate) fn semantic(
        original: eredu_runtime::working_memory::OriginalCompositeSemanticStorageError,
    ) -> Self {
        Self {
            cause: None,
            original,
        }
    }
}
impl std::fmt::Display for CompletedMediaBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(cause) = &self.cause {
            std::fmt::Display::fmt(cause, f)
        } else {
            std::fmt::Display::fmt(&self.original, f)
        }
    }
}
impl std::error::Error for CompletedMediaBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_ref()
            .map(|v| v as &dyn std::error::Error)
            .or(Some(&self.original))
    }
}

/// Backend-private erased operations for a paired architecture and mutable state.
/// Fixed, allocation-free inspection refusal before an original request begins.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OriginalTextFrontierError {
    #[error(transparent)]
    Boundary(eredu_runtime::replicated_session::RuntimeInspectionBoundary),
    #[error(transparent)]
    Mechanism(eredu_runtime::working_memory::WorkingMemoryError),
}

pub(crate) trait ErasedReplicatedTextExecutable {
    /// Actual constructor-selected independent bank sources, borrowed without
    /// building a table or rediscovering a provider from model/type information.
    fn indexed_bank_sources(&self)->Option<&std::collections::BTreeMap<eredu_runtime::RoutedBankId,crate::backend::runtime::residency::parameter_bank::IndexedBankSource>>{None}

    /// Exact selected unit mechanism, not a residency or family inference.
    fn uses_ordinary_unit_equations(&self) -> bool {
        false
    }

    /// Loans the exact prepared host/foreground source under the fixed session
    /// fence. No legacy source construction is allowed on this counted route.
    fn prepared_layerwise_workspace(
        &self,
        _allocation: crate::backend::nn::workspace::MetalAllocationFacts,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }

    /// Installs descriptive facts from this exact idle source into its private
    /// quote Context. Unavailable source strategies keep conservative facts.
    fn install_workspace_parameter_representations(
        &self,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        Ok(())
    }

    /// Descriptive declared static/prediction components under the fixed fence.
    /// Does not cover policy, manager/source, active lane, bank or recovery owners.
    fn count_parameter_owners(
        &self,
        _guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<ParameterOwnerCounts, ParameterOwnerSourceError> {
        Err(ParameterOwnerSourceError::ExecutableUnavailable)
    }

    /// Current initialized state identity without allocating an inspection error.
    fn original_request_media_binding(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::MediaSessionBinding,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
    }
    /// Read-only current source/session binding for ordinary media diagnostics.
    /// Never initializes a revision, synchronizes or grants a bind attempt.
    fn current_media_semantic_binding(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::MediaSessionBinding,
        eredu_runtime::replicated_session::MediaSemanticBindingError<Error>,
    > {
        Err(
            eredu_runtime::replicated_session::MediaSemanticBindingError::Boundary(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ),
        )
    }

    /// Original host source construction pays the existing session revision
    /// before compiling/binding B. Default performs no state/native work.
    fn prepare_original_media_semantic_binding(
        &self,
        _funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<(), eredu_runtime::replicated_session::OriginalMediaBindingError> {
        Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound.into())
    }

    #[cfg(test)]
    fn prepare_completed_media_binding_fixture(
        &self,
    ) -> Result<(), eredu_runtime::replicated_session::MediaSemanticBindingError<Error>> {
        Err(
            eredu_runtime::replicated_session::MediaSemanticBindingError::Boundary(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ),
        )
    }

    /// One-shot B caller supplies existing authority; this never initializes a
    /// lazy revision or uses the ordinary formatted error/target adapter.
    fn bind_completed_original_media_semantics(
        &self,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        _blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        _source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    ) -> Result<
        eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        CompletedMediaBindingError,
    > {
        Err(CompletedMediaBindingError::boundary(
            original,
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }

    /// Typed family binding installed before erasure; invoked only under the
    /// ordinary upload owner, before any tensor/source lowering occurs.
    fn bind_original_media_semantics(
        &self,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        _blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        _source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    ) -> Result<
        eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        eredu_core::BackendFailure,
    > {
        Err(eredu_core::BackendFailure::from_error(
            original
                .reject_boundary(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound),
        ))
    }

    /// Exact borrowed whole-KV source; readiness/admission remain separate.
    fn resident_reset_profile(&self) -> Option<ResidentResetProfile> {
        None
    }

    fn resident_reset_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetSource<'_, MlxKeyValueState>,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
    }

    /// Checked host-only installation. A rejected destination is returned by
    /// ownership; callers keep their prepared retirement slot outside this loan.
    fn install_resident_reset(
        &mut self,
        installation: eredu_runtime::working_memory::ResidentResetInstallation<MlxKeyValueState>,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetDisplaced<MlxKeyValueState>,
        (
            eredu_runtime::working_memory::WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<MlxKeyValueState>,
        ),
    > {
        Err((
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            installation,
        ))
    }

    /// Target component of the actual selected embedded executor, under the same
    /// fixed idle-state and origin checks as source inspection. This supplies no
    /// prediction-lane snapshot and never widens whole-text snapshot support.
    fn prepare_original_prediction_target_source(
        &self,
    ) -> Result<(
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ), StartupCause> {
        Err(Error::PrefillControl(WorkingMemoryError::UnknownBound).into())
    }

    /// Exact external target state source, independent of an Embedded lane.
    /// The selected assistant is authenticated separately before this loan.
    fn prepare_original_external_target_source(&self)->Result<(
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ),StartupCause> {
        Err(Error::PrefillControl(WorkingMemoryError::UnknownBound).into())
    }

    /// Cold source-only preparation under native control policy and quiescence
    /// checks. Returning a borrow grants no native work or full-copy bound.
    fn resident_hybrid_reset_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetSource<'_, MlxHybridState>,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
    }

    /// Checked host-only installation. A rejected destination is returned by
    /// ownership; callers keep their prepared retirement slot outside this loan.
    fn install_resident_hybrid_reset(
        &mut self,
        installation: eredu_runtime::working_memory::ResidentResetInstallation<MlxHybridState>,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetDisplaced<MlxHybridState>,
        (
            eredu_runtime::working_memory::WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<MlxHybridState>,
        ),
    > {
        Err((
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            installation,
        ))
    }

    /// Cold source-only preparation under native control policy and quiescence
    /// checks. Returning a borrow grants no native work or full-copy bound.
    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Same cold source selection without an allocated backend error adapter.
    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound.into())
    }

    /// Clones the actual committed shared input owner under the same cold
    /// checks. No raw cache identity is reconstructed or independently priced.
    fn resident_copy_input_identity(
        &self,
    ) -> Result<Option<eredu_runtime::SharedPreparedInputCacheIdentity>, Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Payload-free identity of this exact executable and parameter state.
    /// Native copy/continuation callers retain their existing source guards.
    fn resident_control_origin(
        &self,
    ) -> Result<eredu_runtime::replicated_session::ReplicatedTextControlOrigin, Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Same current executable/parameter-epoch loan with fixed boundary errors.
    /// Missing implementations remain unqualified; no state is copied.
    fn resident_control_origin_fixed(
        &self,
    ) -> Option<
        Result<
            eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
            eredu_runtime::replicated_session::PreparedControlBindingError,
        >,
    > {
        None
    }

    /// Cold exact-origin validation before any fresh destination construction.
    /// This does not admit a copy or issue execution/installation authority.
    fn validate_resident_control_origin(
        &self,
        _origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ) -> Result<(), Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Allocation-free origin loan for a saved source. This does not inspect
    /// the installed decoder or confer copy, execution or installation authority.
    fn validate_resident_control_origin_fixed(
        &self,
        _origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ) -> Option<Result<(), eredu_runtime::replicated_session::PreparedControlBindingError>> {
        None
    }

    /// Exact constructor controls for the actual typed returned state. Unknown
    /// custom conversions remain unqualified; this creates no source/native owner.
    fn prepared_resident_control_state_bytes(&self) -> Option<usize> {
        None
    }

    /// Binds the same published state under accepted host constructor custody.
    /// Only the sealed resume entry calls this and retains errors with that H.
    /// The returned owner keeps H through its erased allocation and any later
    /// displaced state. Fresh numerical authority remains the existing request.
    fn bind_original_resident_control_state(
        &self,
        _origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        _state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
        _prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        _host: &eredu_core::HostPreparationAuthority,
    ) -> Result<crate::composition::mlx::session::MlxNativeTextState, Error> {
        use crate::composition::mlx::session::{
            PreparedControlSlotError, prepared_control_slot_error,
        };
        Err(prepared_control_slot_error(
            PreparedControlSlotError::Unknown,
        ))
    }

    /// Exact typed placement sources, without copying retained charge directories.
    fn original_control_branch_sources(&self, _slot: &dyn std::any::Any)
        -> Result<[eredu_runtime::replicated_session::ControlBranchSource; 2], Error> {
        Err(crate::composition::mlx::session::prepared_control_slot_error(
            crate::composition::mlx::session::PreparedControlSlotError::Unknown))
    }

    /// Exchanges an exact prepared original slot through the shared control
    /// worker, without ordinary support estimates or diagnostic formatting.
    fn exchange_original_control_state(
        &mut self,
        _slot: &mut dyn std::any::Any,
        _metadata: &eredu_nn::workspace::HostMetadataFunding,
        _media: Option<&eredu_runtime::working_memory::MediaSessionBinding>,
        _branch: Option<(&eredu_runtime::working_memory::PendingTextBranchExchange,
            &[eredu_runtime::replicated_session::ControlBranchSource; 2])>,
    ) -> Result<eredu_runtime::replicated_session::ControlExchangeResult, Error> {
        use crate::composition::mlx::session::{
            PreparedControlSlotError, prepared_control_slot_error,
        };
        Err(prepared_control_slot_error(
            PreparedControlSlotError::Unknown,
        ))
    }

    /// Binds an already host-published dense state for later ordinary exchange.
    /// The concrete adapter rechecks policy, exact origin and selected geometry;
    /// it preserves the state's fresh retention and actual shared prompt owner.
    /// This performs no installation, native copy, completion or admission.
    /// Numerical work must already be settled/published, with caller recovery
    /// retained through this call. Inputs are consumed on typed rejection.
    fn bind_prepared_resident_control_state(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
        prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        match state {
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::KeyValue(
                state,
            ) => self.bind_prepared_dense_control_state(origin, state, prompt),
            crate::backend::runtime::cache::state::PublishedResidentDecoderState::HybridGrouped(
                _,
            )
            | crate::backend::runtime::cache::state::PublishedResidentDecoderState::Pooling(_) => {
                Err(Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                )))
            }
        }
    }

    fn bind_prepared_dense_control_state(
        &self,
        _origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        _state: crate::backend::runtime::cache::state::PublishedDenseResidentKvState,
        _prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::Other(Box::new(
            PreparedDenseControlBindingError::UnsupportedStateType,
        )))
    }

    /// Ordinary source attribution, bound while the architecture remains typed.
    fn prepared_control_attribution(
        &self,
        _input: input::ModelInput<'_>,
    ) -> Result<eredu_core::PreparedPromptAttribution, Error> {
        Err(Error::Other(Box::new(
            eredu_core::PreparedControlInputError::Unsupported,
        )))
    }

    /// Cold selected-ingress validation, before encoder or capture factories.
    fn validate_media_capture_input(
        &self,
        _input: input::ModelInput<'_>,
        _geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), Error> {
        Err(Error::Other(Box::new(
            eredu_core::PreparedControlInputError::InstrumentationUnavailable,
        )))
    }

    fn prepared_control_binding(&self) -> Result<PreparedControlBinding, Error> {
        Err(Error::Other(Box::new(
            eredu_core::PreparedControlInputError::Unsupported,
        )))
    }

    fn validate_prepared_control_binding(
        &self,
        _binding: &PreparedControlBinding,
    ) -> Result<(), Error> {
        Err(Error::Other(Box::new(
            eredu_core::PreparedControlInputError::Unsupported,
        )))
    }

    /// Checks all stateful layer frontiers at a quiescent host-only boundary.
    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error>;
    /// Borrows the existing neutral original frontier inspector without boxing,
    /// formatting, polling, or changing selected state.
    fn original_text_frontier(&self) -> Result<Option<u64>, OriginalTextFrontierError> {
        Err(OriginalTextFrontierError::Mechanism(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))
    }


    fn inference_execution_identity(
        &self,
    ) -> &eredu_runtime::working_memory::InferenceExecutionIdentity;

    /// Physical values in retained target modules, numerical helpers and policy overrides. Source
    /// and residency buffers, addressable banks, prediction modules and mutable
    /// state remain separately composed owners; this is not a whole-model bound.
    fn retained_target_module_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_target_module_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_target_module_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Retained prediction modules, overrides, prepared sources, residency
    /// buffers and immutable state prototypes. Active lanes, observers, outputs
    /// and request resources are separate owners. Shared target allocations
    /// deduplicate when this is merged with `retained_target_storage`.
    fn retained_prediction_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_prediction_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_prediction_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Retained target parameters, sources, weight/cache buffers, mutable state,
    /// and addressable banks. Prediction modules/prototypes, observers, escaped
    /// outputs and enclosing request resources are separate owners. This is
    /// physical inventory evidence, not a whole-request reservation.
    fn retained_target_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_target_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_target_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Parameters, sources and weight-residency buffers without decoder state.
    /// The complete state backing belongs to the request equation quote, so
    /// registration must not charge that same backing again as static storage.
    fn retained_target_nonstate_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_target_nonstate_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_target_nonstate_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Complete idle decoder storage, including native arrays, selected cache
    /// managers and host-backed state. This excludes target and prediction
    /// parameters; no empty-state claim may be inferred from tensor absence.
    fn retained_decoder_state_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_decoder_state_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_decoder_state_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Exact snapshot source census/fill through fixed inspection transports.
    /// Implementations share ordinary collection and retain unknown auxiliary owners.
    fn collect_snapshot_storage_fixed(
        &self,
        _storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::storage::SnapshotStorageInspectionError>
    {
        Err(crate::backend::runtime::residency::storage::SnapshotStorageInspectionError::Unavailable)
    }

    /// Clones only the authorities retained by idle decoder state. This never
    /// copies native state, acquires new authority or resets its admission.
    fn retained_inference_authority(
        &self,
    ) -> Result<eredu_runtime::working_memory::InferenceRetention, Error>;

    /// Actual shared ordinary observation source created under loading authority.
    /// Its private native binding may have been invalidated; this borrow is only
    /// for cold source publication/rebinding, never current execution permission.
    /// Partition-specific path/transport preparation remains separate.
    fn shared_observation_paths(&self) -> Option<&eredu_runtime::SharedLayeredObservationPaths> {
        None
    }

    /// Cold validation of the actual stored runtime token and exact expected
    /// shared source. Does not rebind or grant capture/submission authority.
    /// Strategies without prepared ordinary traversal retain a typed rejection.
    fn validate_prepared_observation_paths(
        &self,
        _expected: &eredu_runtime::SharedLayeredObservationPaths,
        _metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<(), Error> {
        Err(Error::PreparedObservation(
            eredu_runtime::PreparedSessionObservationError::Unavailable,
        ))
    }

    /// Exact local text-prefill control component, not complete media admission.
    /// Unknown state/parallel profiles remain unimplemented rather than guessed.
    fn original_state_slot_facts(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::cache::state::NativeStateSlotCounts>,
        eredu_runtime::replicated_session::RuntimeInspectionBoundary,
    > {
        Ok(None)
    }

    #[cfg(test)]
    fn inspection_adapters_for_test(&self, _: eredu_core::InferenceGeometry, _: &eredu_runtime::working_memory::WorkingMemoryPool) -> Result<(), Error> {
        Err(Error::PrefillScopeUnavailable)
    }
    #[cfg(test)]
    fn prefill_status_for_test(&self) -> Result<(bool, bool), Error> {
        Ok((false, false))
    }

    /// Augment actual equation recipes with the selected bounded policy's
    /// neural frontiers before choosing any native arena or pipeline capacity.
    fn bind_layerwise_neural_recipe(
        &self,
        _pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        _recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
        _funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<(), Error> {
        Err(Error::PrefillScopeUnavailable)
    }

    fn bind_speculative_neural_recipe(
        &self,
        _source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        _pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        _funding: &eredu_nn::workspace::HostMetadataFunding,
        _recipe: &mut crate::backend::nn::workspace::AutoregressiveEquationRecipe,
    ) -> Result<
        (
            u64,
            Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
        ),
        Error,
    > {
        Err(Error::PrefillScopeUnavailable)
    }
    fn prepare_speculative_neural_bank(
        &self,
        _source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        _recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        _role: eredu_runtime::working_memory::OriginalSpeculativeRole,
        _scope: &safemlx::SubmissionScope,
        _partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    {
        Err(Error::PrefillScopeUnavailable)
    }

    fn prepare_speculative_span_neural_bank(
        &self,
        _source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        _recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        _span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        _scope: &safemlx::SubmissionScope,
        _partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    {
        Err(Error::PrefillScopeUnavailable)
    }

    fn prefill_control_facts(
        &self,
        _pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        _geometry: eredu_core::InferenceGeometry,
        _graph_capacity: std::num::NonZeroU64,
        _native_root_capacity: Option<u64>,
        _retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        _native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        _funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Option<eredu_runtime::working_memory::TextPrefillScopeFacts>, Error> {
        Ok(None)
    }
    fn prepare_original_operation_banks(
        &self,
        _pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        _original: &eredu_runtime::working_memory::OriginalTextPrefillScopeSet,
        _step: &eredu_runtime::working_memory::InferenceTextStep,
        _registration: crate::backend::runtime::execution::generic::OriginalOperationRegistration,
        _controls: eredu_runtime::working_memory::OriginalTextControlGuard,
        _host_destinations: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
        _retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        _native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        _funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<
        Option<crate::backend::runtime::execution::generic::OriginalOperationBankOwner>,
        Error,
    > {
        Err(Error::PrefillScopeUnavailable)
    }
    fn native_storage_mechanism(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>,
        Error,
    > {
        Ok(None)
    }
    fn prefill_roots_runtime(&self) -> Result<safemlx::PrefillRootsRuntime, Error> {
        Err(Error::PrefillScopeUnavailable)
    }
    fn install_parallel_control(
        &self,control:Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection>,
    )->Result<(),Error>{
        if control.is_some(){Err(Error::PrefillScopeUnavailable)}else{Ok(())}
    }

    fn install_prefill_controls(
        &self,
        controls: Option<crate::backend::submission_recovery::prefill::PrefillControlProjection>,
    ) -> Result<(), Error> {
        if controls.is_some() {
            Err(Error::PrefillScopeUnavailable)
        } else {
            Ok(())
        }
    }

    /// Private original-admission native row proposal. The implementation uses
    /// the actual paired session inspector and retains no prepared authority.
    fn prepare_opening_rows(
        &self,
        _selection: eredu_runtime::layered::BoundCaptureSelection<'_>,
    ) -> Result<super::NativeOpeningRowsPlan, Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }
    fn install_opening_rows(&self, _rows: &super::NativeOpeningRowsOwner) -> Result<(), Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Explicit ordinary-idle retirement only; cold inspection never calls it.
    /// Executables without an installed native row slot have nothing to retire.
    fn retire_expired_opening_rows(&self) -> Result<(), Error> {
        Ok(())
    }

    #[cfg(test)]
    fn opening_rows_status_for_test(&self) -> Result<(bool, bool), Error> {
        Ok((false, false))
    }

    #[cfg(test)]
    fn busy_rows_retirement_for_test(&self) -> Result<(), Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
    }

    /// Other idle executable owners omitted by parameter and state traversal.
    /// Erased observers and communication owners with no physical inventory
    /// make this evidence explicitly incomplete rather than zero-sized.
    fn retained_idle_auxiliary_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_idle_auxiliary_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_idle_auxiliary_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    /// Exact selected bounded window, without submission or materialization.
    fn layerwise_workspace(
        &self,
        _allocation: crate::backend::nn::workspace::MetalAllocationFacts,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        Err(Error::Other(Box::new(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )))
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
        _batch: std::num::NonZeroU32,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        Err(Error::ArchitectureModel(
            "this executable has no resident text workspace projection".into(),
        ))
    }

    fn prepared_input_plans(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<
        Vec<eredu_architectures::media_plan::PreparedInputPartPlan>,
        eredu_core::CapabilityError,
    > {
        input
            .parts
            .iter()
            .map(|part| {
                eredu_architectures::media_plan::text_only_input_part(
                    self.effective_model_type(),
                    part,
                    &input::MlxInputInspector,
                )
            })
            .collect()
    }

    fn partition_observation_hooks(
        &self,
    ) -> Option<eredu_runtime::inspection::ObservationHookSupport> {
        None
    }

    fn partition_parameter_description(
        &self,
    ) -> Option<&std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>> {
        None
    }

    fn visit_loaded_parameters(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) -> bool {
        false
    }
    fn parameter_materialization_tasks(
        &self,
    ) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &[]
    }
    fn prepared_parameter_slots(
        &self,
    ) -> &[eredu_runtime::parameter_operations::PreparedParameterSlot] {
        &[]
    }
    fn with_parameter_slots(
        &mut self,
        _location: &eredu_runtime::parameter_operations::PreparedParameterLocation,
        _selected: &std::collections::BTreeSet<String>,
        _operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn publish_parameter_replacements(
        &mut self,
        _values: &std::collections::BTreeMap<String, MlxTensor>,
        _active: bool,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn invalidate_parameter_snapshots(&mut self) {}
    fn estimate_parameter_reset_state(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    fn prepare_parameter_reset_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "parameter state reset is unavailable".into(),
        ))
    }
    fn exchange_parameter_reset_state(
        &mut self,
        _slot: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "parameter state exchange is unavailable".into(),
        ))
    }
    fn prepare_autoregressive_cache(&mut self) -> Result<MlxPredictionTargetState, Error> {
        Err(Error::Speculative(
            "ordinary prediction state is unavailable".into(),
        ))
    }
    fn autoregressive_prefill(
        &mut self,
        _input: crate::backend::runtime::media::input::ModelInput<'_>,
        _cache: &mut MlxPredictionTargetState,
        _sample: bool,
        _cancellation: &eredu_core::GenerationCancellationToken,
        _stream: &Stream,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            eredu_runtime::speculative::autoregressive::AutoregressivePrefill<Array>,
        >,
        Error,
    > {
        Err(Error::Speculative(
            "selected independent prefill source is unavailable".into(),
        ))
    }
    fn autoregressive_forward(
        &mut self,
        _tokens: &Array,
        _cache: &mut MlxPredictionTargetState,
        _prefill: bool,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Speculative(
            "ordinary prediction state is unavailable".into(),
        ))
    }

    fn autoregressive_forward_with_completion(
        &mut self,
        _tokens: &Array,
        _cache: &mut MlxPredictionTargetState,
        _stream: &Stream,
        _completion: &mut dyn AutoregressiveSequenceCompletion,
    ) -> Result<Array, Error> {
        Err(Error::PrefillScopeUnavailable)
    }

    /// Compiles from the retained host source while the actual independent
    /// cache is installed. The resulting plan binds that cache's own revision.
    fn prepare_autoregressive_media_semantics(
        &mut self,
        _source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
        _cache: &mut MlxPredictionTargetState,
        _blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        _pool: &WorkingMemoryPool,
        _funding: &eredu_nn::workspace::HostMetadataFunding,
        _stream: &Stream,
    ) -> Result<eredu_architectures::media_plan::BoundPreparedMediaSemantics, Error> {
        Err(Error::PrefillScopeUnavailable)
    }
    fn prepare_autoregressive_media_prefill(
        &mut self, _packet: &input::OriginalMediaPacket, _semantics: &eredu_architectures::media_plan::BoundPreparedMediaSemantics, _cache: &mut MlxPredictionTargetState,
        _geometry: eredu_core::InferenceGeometry, _role: &eredu_runtime::working_memory::OriginalSpeculativeRole,
        _metadata: &eredu_nn::workspace::WorkspaceContext, _stream: &Stream,
    ) -> Result<OriginalAutoregressiveMediaPrefill,Error> { Err(Error::PrefillScopeUnavailable) }
    fn autoregressive_media_prefill_span_with_completion(
        &mut self, _source:&mut OriginalAutoregressiveMediaPrefill, _cache:&mut MlxPredictionTargetState,
        _span:&eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        _stream:&Stream, _completion:&mut dyn AutoregressiveSequenceCompletion,
    ) -> Result<Option<Array>,Error> { Err(Error::PrefillScopeUnavailable) }

    /// Actual bounded-driver prefill span with the same checkpoint/publication
    /// transaction as independent decode. Completion receives no output root
    /// for StateOnly; it must still complete every mutable state root.
    fn autoregressive_prefill_span_with_completion(
        &mut self,
        _tokens: &Array,
        _cache: &mut MlxPredictionTargetState,
        _span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        _stream: &Stream,
        _completion: &mut dyn AutoregressiveSequenceCompletion,
    ) -> Result<Option<Array>, Error> {
        Err(Error::PrefillScopeUnavailable)
    }

    fn native_control_support(&self) -> eredu_core::execution_control::ControlSupport<&'static str> {
        eredu_core::execution_control::ControlSupport::Unsupported {
            reason: "complete native state copying is unavailable for this executable",
        }
    }
    /// Exact live logical estimate without inventories, native clones or owned
    /// diagnostics. This does not certify physical original snapshot copying.
    fn estimate_original_native_control_state(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    fn estimate_native_control_state(
        &self,
        _saved: Option<&dyn std::any::Any>,
    ) -> Result<Option<eredu_core::execution_control::SnapshotEstimate>, Error> {
        Ok(None)
    }
    fn capture_native_control_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn estimate_native_control_growth(
        &self,
        _saved: &dyn std::any::Any,
        _additional: u64,
    ) -> Result<Option<u64>, Error> {
        Ok(None)
    }
    fn copy_native_control_state(
        &mut self,
        _saved: &dyn std::any::Any,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn validate_native_control_state(&self, _saved: &dyn std::any::Any) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn exchange_native_control_state(
        &mut self,
        _slot: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn effective_model_type(&self) -> &str;
    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate;
    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency;
    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot;
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception>;
    #[cfg(test)]
    fn retained_numeric_state_snapshot(
        &self,
    ) -> Option<Result<RetainedNumericStateSnapshot, Exception>> {
        None
    }

    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error>;
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport>;
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    >;
    fn has_partition_control(&self) -> bool {
        false
    }
    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        None
    }
    fn partition_public_output(&self) -> bool {
        true
    }
    fn with_embedded_prediction(
        &mut self,
        _continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }
    /// Fresh state only; no invocation or ordinary clone authority is implied.
    fn prepare_original_prediction(
        &self,
        _context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        None
    }
    fn has_embedded_prediction(&self) -> bool {
        false
    }
    fn speculative_activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        None
    }
    fn install_embedded_prediction_observers(
        &mut self,
        _observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        false
    }
    fn take_speculative_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        None
    }
    fn take_speculative_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        None
    }
    fn supports_external_capture_spans(&self) -> bool {
        false
    }
    /// Borrows only an originally qualified external target capability.
    fn original_external_prediction_mut(&mut self)
        ->Option<&mut (dyn ErasedExternalPredictionExecutable+'static)> { None }
    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut (dyn ErasedExternalPredictionExecutable + 'static)> {
        None
    }
    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity;
    fn reset_cache(&mut self) -> Result<(), Exception>;
    fn reset_cache_distributed(&mut self) -> Result<(), Error>;
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        let _ = input_identity;
        self.load_prompt_cache(directory, expected, prefix_token_ids)
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        self.prefill_result_with_observer(
            Ok(input),
            None,
            None,
            stream,
            &mut eredu_runtime::NoopObserver,
        )
    }
    #[cfg(test)]
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        self.decode_result_with_observer(Ok(tokens), stream, &mut eredu_runtime::NoopObserver)
    }
    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error>;
    #[cfg(test)]
    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        self.prefill_result_with_observer(Ok(input), mask, None, stream, observer)
    }
    fn prefill_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        self.prefill_cancellable_result_with_observer(
            input,
            mask,
            capture_geometry,
            &eredu_core::GenerationCancellationToken::new(),
            stream,
            observer,
        )?
        .ok_or_else(|| {
            Error::ArchitectureModel(
                "uncancellable caller participated in cancelled prefill".into(),
            )
        })
    }
    fn prefill_cancellable_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Option<Array>, Error>;
    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error>;
    /// Retains the actual funded resume Context for this adapter's continuation
    /// constructors. None preserves adapters without an explicit counted path.
    /// This is construction custody, not a prepared-input or native-work proof.
    fn retain_continuation_metadata(
        &self,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Option<eredu_nn::workspace::WorkspaceContext> {
        None
    }

    /// Explicit accepted-source metadata loan; ordinary implementations cannot
    /// silently claim counted native input construction.
    fn decode_result_with_observer_and_metadata(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Array, Error> {
        if metadata.is_some() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ));
        }

        self.decode_result_with_observer(tokens, stream, observer)
    }
}

pub(crate) fn prepared_composite_input(
    input: input::ModelInput<'_>,
) -> Result<eredu_runtime::PreparedModelInput<MlxTensor>, Error> {
    use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedInputPayload};

    input::validate(input)?;
    let parts = input
        .parts
        .iter()
        .map(|part| {
            let payload = match part.payload() {
                input::InputPayload::TokenIds(value) => {
                    PreparedInputPayload::TokenIds(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Tensor(value) => {
                    PreparedInputPayload::Tensor(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Embeddings(value) => {
                    PreparedInputPayload::Embeddings(MlxTensor::from_array(value.clone()))
                }
                _ => {
                    return Err(eredu_core::PreparedInputError::BackendTensorIdentity(
                        "MLX prepared input contains an unknown payload kind".into(),
                    ));
                }
            };
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                part.metadata()
                    .iter()
                    .map(|(key, value)| (*key, MlxTensor::from_array(value.clone()))),
                part.extents().iter().copied(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let inspector = input::MlxTensorInputInspector;
    eredu_runtime::PreparedModelInput::new(parts, |tensor| inspector.identity(tensor))
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    use crate::backend::{
        ExecutionContext,
        nn::tensor::{TokenValidationScope, validate_token_domain},
    };
    use eredu_architectures::prediction_extension::PredictionExtensionMaterializer;
    use safemlx::{Device, DeviceType};
    use std::error::Error as _;

    fn prove(device: DeviceType) {
        let execution = ExecutionContext::new(Device::new(device, 0));
        let stream = execution.stream();
        for token in [2, 4] {
            let scope = TokenValidationScope::begin().unwrap();
            let tokens = Array::from_slice(&[token], &[1]);
            let _validated = validate_token_domain(&tokens, 4, None, stream)
                .expect("deferred token checks must not fail during graph construction");
            // The assertion is independent of the retained tensor. Completion must
            // settle both without relying on an output's dependency graph.
            let retained = MlxTensor::from_array(
                Array::from_slice(&[2.0_f32, -3.0], &[2])
                    .multiply(Array::from_f32(3.0), stream)
                    .unwrap(),
            );
            let result =
                MlxEmbeddedPredictionMaterializer::complete_prediction_values([&retained], stream);
            assert_eq!(
                retained
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_as_slice::<f32>()
                    .unwrap(),
                &[6.0, -9.0]
            );
            if token == 2 {
                result.unwrap();
            } else {
                let failure = result.unwrap_err();
                let source = failure
                    .source()
                    .unwrap()
                    .downcast_ref::<Exception>()
                    .expect("native exception must remain the error source");
                assert!(source.to_string().contains("token ID is outside 0..4"));
            }
            drop(scope);
        }
    }

    #[test]
    fn prediction_completion_retains_dependencies_and_errors_cpu() {
        prove(DeviceType::Cpu);
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a local MLX Metal device"]
    fn prediction_completion_retains_dependencies_and_errors_metal() {
        prove(DeviceType::Gpu);
    }
}

/// Exact selected executable/parameter origin and installed branch revision.
/// None of these diagnostic identities grants work or memory authority.
pub(crate) struct PreparedControlBinding {
    pub(crate) origin: eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    pub(crate) revision: eredu_runtime::working_memory::InferenceStateRevision,
    pub(crate) frontier: u64,
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
