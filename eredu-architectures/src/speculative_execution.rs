//! Backend-generic speculative execution over architecture-owned prediction strategies.

use std::marker::PhantomData;
use crate::prediction_extension::equation::{
    execute_prediction_equation, PredictionEquation, PredictionEquationOutput,
};

use eredu_core::speculative::{
    AdmittedSpeculativeActivations, SpeculativeActivationCapture, SpeculativeActivationOrigin,
    SpeculativeActivationPhase, SpeculativeControlError,
};
use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill,
    SpeculativeTelemetry, Submission,
};
use eredu_runtime::inspection::SpeculativeActivationObserver;
use eredu_runtime::DraftStateTransaction;

mod capture;
mod captured_prefill;
mod observation;
mod prefill_source;
pub use captured_prefill::PrefillValues as EmbeddedPrefillValues;
mod prediction_phase;
pub use prediction_phase::{PredictionCompletionPoint, PredictionCompletionSources, OrdinaryPredictionPhase, PredictionPhaseRoots, PredictionPhaseState, PredictionPhaseEvidence, ReplicatedPredictionPhase};
pub use prefill_source::{
    AdmittedPredictionPrefill, CompositePredictionPrefill, PredictionPrefillPlan,
    PredictionPrefillSource, TextPredictionPrefill,
};
mod snapshot;
mod prepared_copy;
mod logit_block;
mod outer_observer;
pub use outer_observer::observe_embedded_tensor_ordinary;
pub use logit_block::{EmbeddedPredictionLogitBlock, EmbeddedPredictionTensor};
pub use prepared_copy::{PreparedEmbeddedCopy, PreparedEmbeddedCopyError, PreparedEmbeddedCopyProvider, PreparedEmbeddedEvidence, PreparedEmbeddedPayload, PreparedEmbeddedState};
pub use capture::{speculative_capture_scope, SpeculativeActivationExecution};

/// Observation path for the physical target capture consumed by embedded prediction.
pub const EMBEDDED_TARGET_CAPTURE_PATH: &str = "embedded_prediction.target_capture";
/// Observation path for a sequential prediction hidden output.
pub const EMBEDDED_PREDICTION_OUTPUT_PATH: &str = "embedded_prediction.output";
/// Observation path for logits consumed by speculative proposal sampling.
pub const EMBEDDED_PROPOSAL_LOGITS_PATH: &str = "embedded_prediction.proposal_logits";
/// Observation path for the full target verification logits tensor.
pub const EMBEDDED_VERIFICATION_LOGITS_PATH: &str = "embedded_prediction.verification_logits";

/// Production-carried observers for architecture-owned embedded prediction boundaries.
pub struct EmbeddedPredictionObservers<T, L, E> {
    tensors: Option<Box<dyn eredu_runtime::ActivationObserver<T, E>>>,
    logits: Box<dyn eredu_runtime::ActivationObserver<L, E>>,
    internal: Option<Box<dyn SpeculativeActivationObserver<T, E>>>,
}

impl<T, L, E> EmbeddedPredictionObservers<T, L, E> {
    /// Installs independent tensor and proposal-logit observers.
    pub fn new(
        tensors: impl eredu_runtime::ActivationObserver<T, E> + 'static,
        logits: impl eredu_runtime::ActivationObserver<L, E> + 'static,
    ) -> Self {
        Self {
            tensors: Some(Box::new(tensors)),
            logits: Box::new(logits),
            internal: None,
        }
    }

    /// Changes only the backend logit carrier at its final binding boundary.
    /// Tensor/internal observers move intact; the adapter owns and forwards the
    /// existing logit callback without invoking it during construction.
    pub fn map_logits<N>(
        self,
        adapt: impl FnOnce(
            Box<dyn eredu_runtime::ActivationObserver<L, E>>,
        ) -> Box<dyn eredu_runtime::ActivationObserver<N, E>>,
    ) -> EmbeddedPredictionObservers<T, N, E> {
        EmbeddedPredictionObservers {
            tensors: self.tensors, logits: adapt(self.logits), internal: self.internal,
        }
    }

    /// Adds explicitly admitted internal phase instrumentation. Existing outer
    /// observers do not enable this route or incur component tensor work.
    pub fn with_internal(
        mut self,
        observer: impl SpeculativeActivationObserver<T, E> + 'static,
    ) -> Self {
        self.internal = Some(Box::new(observer));
        self
    }

    /// Drains one admitted internal record after the executor scope has returned.
    pub fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.internal()?.take_activation_capture()
    }

    /// Recovers the original portable failure from native error propagation.
    pub fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.internal()?.take_activation_error()
    }

    fn internal(&mut self) -> Option<&mut dyn SpeculativeActivationObserver<T, E>> {
        match self.internal.as_mut() {
            Some(observer) => Some(&mut **observer),
            None => None,
        }
    }

    fn tensor<'a, M>(
        &mut self, path: &str, value: T,
        chunk: Option<&eredu_runtime::prefill::PrefillChunk>, context: M::Context<'a>,
    ) -> Result<T, E>
    where M: SpeculativeTensorMechanisms<Tensor = T, Error = E>,
    {
        let observer = match self.tensors.as_mut() {
            Some(observer) => Some(&mut **observer as &mut dyn eredu_runtime::ActivationObserver<T, E>),
            None => None,
        };
        M::observe_outer_tensor(value, path, chunk, observer, context)
    }

    fn supports_tensor_prefill_spans(&self) -> bool {
        self.tensors.as_ref().is_none_or(|observer| observer.supports_prefill_spans())
    }

    fn logits(&mut self, value: &L) -> Result<L, E>
    where
        L: Clone,
    {
        eredu_runtime::observe_and_intervene(
            self.logits.as_mut(),
            EMBEDDED_PROPOSAL_LOGITS_PATH,
            value,
        )
    }
}

impl<T, L, E> Default for EmbeddedPredictionObservers<T, L, E> {
    fn default() -> Self {
        Self {
            tensors: None,
            logits: Box::new(eredu_runtime::NoopObserver),
            internal: None,
        }
    }
}

/// Architecture-owned cache envelope for one embedded-prediction lane.
///
/// `T` and `L` are opaque backend-native storage values. Their membership, the
/// prepared-input binding, and the target capture frontier are neutral
/// prediction semantics and therefore remain outside any concrete backend.
pub struct EmbeddedPredictionCache<T, L> {
    target: Option<T>,
    prediction: L,
    prepared_input: Option<eredu_runtime::SpeculativeIdentity>,
    capture_generation: Option<u64>,
    prepared: Option<prepared_copy::CacheOwnership<T, L>>,
}

impl<T, L> EmbeddedPredictionCache<T, L> {
    /// Creates one lane from exact target and extension storage.
    pub const fn new(target: T, prediction: L) -> Self {
        Self {
            target: Some(target),
            prediction,
            prepared_input: None,
            capture_generation: None,
            prepared: None,
        }
    }

    /// Installs actual prepared target and prediction copies with their exact
    /// source providers. No ordinary state is adopted into this constructor.
    pub fn from_prepared(
        target: PreparedEmbeddedState<T>, prediction: PreparedEmbeddedState<L>,
    ) -> Result<Self, eredu_core::BackendFailure> {
        let bytes = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<Result<Self, eredu_core::BackendFailure>>())
            .ok_or_else(|| target.copy.reject(PreparedEmbeddedCopyError::Overflow))?;
        let metadata_host = target.copy.prepare_host(bytes)?;
        Ok(Self { target: Some(target.value), prediction: prediction.value,
            prepared_input: None, capture_generation: None,
            prepared: Some(prepared_copy::CacheOwnership {
                target: target.copy, prediction: prediction.copy,
                target_evidence: target.evidence, prediction_evidence: prediction.evidence,
                target_host: target.host, prediction_host: prediction.host, metadata_host,
            }) })
    }

    /// Installs the actual target completion/source witness before the next copy.
    /// Returns false for ordinary storage, which has no original provider.
    pub fn retain_target_evidence<E: 'static>(&mut self, evidence: E) -> Result<bool, eredu_core::BackendFailure> {
        let Some(prepared) = &mut self.prepared else { return Ok(false); };
        let evidence = prepared.target.retain_evidence(evidence)?;
        prepared.target_evidence = Some(evidence);
        Ok(true)
    }
    /// Installs prediction state evidence after its actual successful completion.
    pub fn retain_prediction_evidence<E: 'static>(&mut self, evidence: E) -> Result<bool, eredu_core::BackendFailure> {
        let Some(prepared) = &mut self.prepared else { return Ok(false); };
        let evidence = prepared.prediction.retain_evidence(evidence)?;
        prepared.prediction_evidence = Some(evidence);
        Ok(true)
    }

    /// Exact most recently completed target source, borrowed before it is copied
    /// into a particular output owner. Later state mutations cannot replace that copy.
    pub fn target_logit_evidence(&self) -> Option<&PreparedEmbeddedEvidence> {
        self.prepared.as_ref().and_then(|prepared| prepared.target_evidence.as_ref())
    }

    /// Borrows only the target witness slot while its state is in the session.
    pub fn target_phase_evidence(&mut self) -> PredictionPhaseEvidence<'_, T> {
        PredictionPhaseEvidence::new(self.prepared.as_mut().map(|prepared| {
            (&prepared.target, &mut prepared.target_evidence)
        }))
    }

    /// Borrows opaque target storage when it is not temporarily installed in a session.
    pub const fn target(&self) -> Option<&T> {
        self.target.as_ref()
    }

    /// Temporarily transfers opaque target storage into its singular session.
    pub fn take_target(&mut self) -> Option<T> {
        self.target.take()
    }

    /// Restores opaque target storage after singular-session execution.
    pub fn restore_target(&mut self, target: T) {
        self.target = Some(target);
    }

    /// Borrows architecture-typed prediction storage.
    pub const fn prediction(&self) -> &L {
        &self.prediction
    }

    /// Mutably borrows architecture-typed prediction storage.
    pub fn prediction_mut(&mut self) -> &mut L {
        &mut self.prediction
    }

    /// Lends this branch's state and exact evidence destination to one phase.
    pub fn prediction_phase_state(&mut self) -> PredictionPhaseState<'_, L> {
        PredictionPhaseState::new(&mut self.prediction, self.prepared.as_mut().map(|prepared| {
            (&prepared.prediction, &mut prepared.prediction_evidence)
        }))
    }

    /// Binds the lane to the exact prepared description and semantic content.
    pub fn bind_prepared_input(
        &mut self,
        identity: Option<&eredu_runtime::PreparedInputCacheIdentity>,
    ) -> Result<(), EmbeddedPredictionCacheError> {
        if let Some(prepared) = &mut self.prepared {
            let identity = identity.ok_or_else(|| EmbeddedPredictionCacheError::Prepared(
                prepared.target.reject(PreparedEmbeddedCopyError::MissingPreparedInput)))?;
            let prefix = "prepared-input/";
            let suffix = identity.prefix_content_fingerprint();
            if let Some(bound) = self.prepared_input.as_ref() {
                return if bound.as_str().strip_prefix(prefix) == Some(suffix) { Ok(()) }
                else { Err(EmbeddedPredictionCacheError::Prepared(prepared.target.reject(PreparedEmbeddedCopyError::SourceMismatch))) };
            }
            let capacity = prefix.len().checked_add(suffix.len()).ok_or_else(||
                EmbeddedPredictionCacheError::Prepared(prepared.target.reject(PreparedEmbeddedCopyError::Overflow)))?;
            let parts = [capacity, std::mem::size_of::<String>(),
                std::mem::size_of::<Option<eredu_runtime::SpeculativeIdentity>>(),
                std::mem::size_of::<std::collections::TryReserveError>(),
                std::mem::size_of::<Result<(), std::collections::TryReserveError>>()];
            let bytes = parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| EmbeddedPredictionCacheError::Prepared(prepared.target.reject(PreparedEmbeddedCopyError::Overflow)))?;
            let host = prepared.target.prepare_host(bytes).map_err(EmbeddedPredictionCacheError::Prepared)?;
            let mut text = String::new();
            text.try_reserve_exact(capacity).map_err(|cause|
                EmbeddedPredictionCacheError::Prepared(prepared.target.allocation_failure(cause)))?;
            text.push_str(prefix); text.push_str(suffix);
            let identity = eredu_runtime::SpeculativeIdentity::new(text).map_err(|_| {
                EmbeddedPredictionCacheError::Prepared(prepared.target.reject(PreparedEmbeddedCopyError::SourceMismatch))
            })?;
            match self.prepared_input.as_ref() {
                Some(bound) if bound != &identity => return Err(EmbeddedPredictionCacheError::Prepared(
                    prepared.target.reject(PreparedEmbeddedCopyError::SourceMismatch))),
                Some(_) => return Ok(()),
                None => { self.prepared_input = Some(identity); prepared.metadata_host = host; return Ok(()); }
            }
        }
        let identity = identity.ok_or(EmbeddedPredictionCacheError::MissingPreparedInput)?;
        let identity = eredu_runtime::SpeculativeIdentity::new(format!(
            "prepared-input/{}",
            identity.prefix_content_fingerprint()
        ))
        .map_err(|error| EmbeddedPredictionCacheError::Identity(error.to_string()))?;
        match self.prepared_input.as_ref() {
            Some(bound) if bound != &identity => {
                Err(EmbeddedPredictionCacheError::DifferentPreparedInput)
            }
            Some(_) => Ok(()),
            None => {
                self.prepared_input = Some(identity);
                Ok(())
            }
        }
    }

    /// Forms the exact selected lane identity at the current target frontier.
    pub fn lane_identity<E>(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        generation: impl FnOnce(&T) -> Result<u64, E>,
    ) -> Result<eredu_runtime::SpeculativeLaneIdentity, EmbeddedPredictionCacheAccessError<E>> {
        let prepared =
            self.prepared_input
                .clone()
                .ok_or(EmbeddedPredictionCacheAccessError::Cache(
                    EmbeddedPredictionCacheError::CaptureBeforePreparedInput,
                ))?;
        let generation = match self.target.as_ref() {
            Some(target) => {
                generation(target).map_err(EmbeddedPredictionCacheAccessError::Native)?
            }
            None => self
                .capture_generation
                .ok_or(EmbeddedPredictionCacheAccessError::Cache(
                    EmbeddedPredictionCacheError::MissingCaptureGeneration,
                ))?,
        };
        Ok(selected.lane_identity(prepared, generation))
    }

    /// Borrows the exact selected/input owners at this target frontier.
    pub fn lane_identity_ref<'a, E>(
        &'a self, selected: &'a eredu_runtime::SelectedSpeculativeRealization,
        generation: impl FnOnce(&T) -> Result<u64, E>,
    ) -> Result<eredu_runtime::SpeculativeLaneIdentityRef<'a>, EmbeddedPredictionCacheAccessError<E>> {
        let prepared = self.prepared_input.as_ref().ok_or(EmbeddedPredictionCacheError::CaptureBeforePreparedInput)?;
        let generation = match &self.target {
            Some(target) => generation(target).map_err(EmbeddedPredictionCacheAccessError::Native)?,
            None => self.capture_generation.ok_or(EmbeddedPredictionCacheError::MissingCaptureGeneration)?,
        };
        Ok(selected.lane_identity_ref(prepared, generation))
    }

    /// Retains a successfully published target frontier for prediction-only forks.
    pub fn retain_capture_generation<E>(
        &mut self,
        generation: impl FnOnce(&T) -> Result<u64, E>,
    ) -> Result<(), EmbeddedPredictionCacheAccessError<E>> {
        let target = self
            .target
            .as_ref()
            .ok_or(EmbeddedPredictionCacheAccessError::Cache(
                EmbeddedPredictionCacheError::TargetStateActive,
            ))?;
        self.capture_generation =
            Some(generation(target).map_err(EmbeddedPredictionCacheAccessError::Native)?);
        Ok(())
    }
}

impl<T, L: Clone> EmbeddedPredictionCache<T, L> {
    /// Creates an exact checkpoint using the backend's opaque target-storage clone mechanism.
    pub fn checkpoint<E>(
        &self,
        clone_target: impl FnOnce(&T) -> Result<T, E>,
    ) -> Result<Self, EmbeddedPredictionCacheAccessError<E>> {
        if let Some(prepared) = &self.prepared {
            let target = self.target.as_ref().map(|state| prepared.target.copy_state(state, prepared.target_evidence.as_ref()))
                .transpose().map_err(EmbeddedPredictionCacheAccessError::Prepared)?;
            let prediction = prepared.prediction.copy_state(&self.prediction, prepared.prediction_evidence.as_ref())
                .map_err(EmbeddedPredictionCacheAccessError::Prepared)?;
            let (prepared_input, metadata_host) = prepared_copy::identity::<_, Self>(&self.prepared_input, &prepared.target)
                .map_err(EmbeddedPredictionCacheAccessError::Prepared)?;
            let (target, target_evidence, target_host) = match target {
                Some(value) => (Some(value.value), value.evidence, value.host),
                None => (None, prepared.target_evidence.clone(), prepared.target_host.clone()),
            };
            return Ok(Self { target, prediction: prediction.value, prepared_input,
                capture_generation: self.capture_generation,
                prepared: Some(prepared_copy::CacheOwnership {
                    target: prepared.target.clone(), prediction: prepared.prediction.clone(),
                    target_evidence, prediction_evidence: prediction.evidence,
                    target_host, prediction_host: prediction.host, metadata_host,
                }) });
        }
        let target = self
            .target
            .as_ref()
            .map(clone_target)
            .transpose()
            .map_err(EmbeddedPredictionCacheAccessError::Native)?;
        Ok(Self {
            target,
            prediction: self.prediction.clone(),
            prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation,
            prepared: None,
        })
    }

    /// Forks prediction-local state without transferring ordinary target storage.
    pub fn prediction_fork(&self) -> Result<EmbeddedPredictionDraftCache<L>, eredu_core::BackendFailure> {
        if let Some(prepared) = &self.prepared {
            let prediction = prepared.prediction.copy_state(&self.prediction, prepared.prediction_evidence.as_ref())?;
            let (prepared_input, metadata_host) = prepared_copy::identity::<_, EmbeddedPredictionDraftCache<L>>(
                &self.prepared_input, &prepared.prediction)?;
            return Ok(EmbeddedPredictionDraftCache {
                prediction: prediction.value, prepared_input, capture_generation: self.capture_generation,
                prepared: Some(prepared_copy::PredictionOwnership { copy: prepared.prediction.clone(),
                    evidence: prediction.evidence, payload_host: prediction.host, metadata_host }),
            });
        }
        Ok(EmbeddedPredictionDraftCache {
            prediction: self.prediction.clone(), prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation, prepared: None,
        })
    }

    /// Commits a successful prediction-local transaction.
    pub fn commit_prediction(&mut self, draft: &EmbeddedPredictionDraftCache<L>) -> Result<(), eredu_core::BackendFailure> {
        match (&mut self.prepared, &draft.prepared) {
            (Some(current), Some(source)) if current.prediction.same_source(&source.copy) => {
                let prediction = source.copy.copy_state(&draft.prediction, source.evidence.as_ref())?;
                let (identity, metadata_host) = prepared_copy::identity::<_, Self>(&draft.prepared_input, &source.copy)?;
                // Both fallible destinations complete before replacing canonical state.
                self.prediction = prediction.value;
                self.prepared_input = identity;
                self.capture_generation = draft.capture_generation;
                current.prediction_evidence = prediction.evidence;
                current.prediction_host = prediction.host;
                current.metadata_host = metadata_host;
                Ok(())
            }
            (Some(current), _) => Err(current.prediction.reject(PreparedEmbeddedCopyError::SourceMismatch)),
            (_, Some(source)) => Err(source.copy.reject(PreparedEmbeddedCopyError::SourceMismatch)),
            (None, None) => {
                self.prediction.clone_from(&draft.prediction);
                self.prepared_input.clone_from(&draft.prepared_input);
                self.capture_generation = draft.capture_generation;
                Ok(())
            }
        }
    }

    /// Restores a completed provisional checkpoint by moving its existing paid
    /// destinations. Recovery cannot require another copy after a failed action.
    pub fn rollback_prediction(&mut self, checkpoint: EmbeddedPredictionDraftCache<L>) -> Result<(), eredu_core::BackendFailure> {
        match (&self.prepared, &checkpoint.prepared) {
            (Some(current), Some(source)) if current.prediction.same_source(&source.copy) => {},
            (Some(current), _) => return Err(current.prediction.reject(PreparedEmbeddedCopyError::SourceMismatch)),
            (_, Some(source)) => return Err(source.copy.reject(PreparedEmbeddedCopyError::SourceMismatch)),
            (None, None) => {},
        }
        self.prediction = checkpoint.prediction;
        self.prepared_input = checkpoint.prepared_input;
        self.capture_generation = checkpoint.capture_generation;
        if let (Some(current), Some(source)) = (&mut self.prepared, checkpoint.prepared) {
            current.prediction_evidence = source.evidence;
            current.prediction_host = source.payload_host;
            current.metadata_host = source.metadata_host;
        }
        Ok(())
    }

    /// Restores all neutral membership around an opaque target-state restore.
    pub fn restore<E>(
        &mut self,
        checkpoint: &Self,
        restore_target: impl FnOnce(&mut T, &T) -> Result<(), E>,
    ) -> Result<(), EmbeddedPredictionCacheAccessError<E>> {
        match (&self.prepared, &checkpoint.prepared) {
            (Some(current), Some(source)) if current.target.same_source(&source.target)
                && current.prediction.same_source(&source.prediction) => {
                if self.target.is_some() != checkpoint.target.is_some() {
                    return Err(EmbeddedPredictionCacheAccessError::Prepared(current.target.reject(PreparedEmbeddedCopyError::SourceMismatch)));
                }
                let replacement = checkpoint.checkpoint(|_| -> Result<T, E> { unreachable!("prepared copy owns its target worker") })?;
                *self = replacement;
                return Ok(());
            }
            (Some(current), _) => return Err(EmbeddedPredictionCacheAccessError::Prepared(current.target.reject(PreparedEmbeddedCopyError::SourceMismatch))),
            (_, Some(source)) => return Err(EmbeddedPredictionCacheAccessError::Prepared(source.target.reject(PreparedEmbeddedCopyError::SourceMismatch))),
            (None, None) => {}
        }
        match (&mut self.target, &checkpoint.target) {
            (Some(current), Some(previous)) => restore_target(current, previous)
                .map_err(EmbeddedPredictionCacheAccessError::Native)?,
            (None, None) => {}
            _ => {
                return Err(EmbeddedPredictionCacheAccessError::Cache(
                    EmbeddedPredictionCacheError::TargetPresenceChanged,
                ))
            }
        }
        self.prediction.clone_from(&checkpoint.prediction);
        self.prepared_input.clone_from(&checkpoint.prepared_input);
        self.capture_generation = checkpoint.capture_generation;
        Ok(())
    }
}

/// Prediction-only state forked from an architecture-owned lane envelope.
pub struct EmbeddedPredictionDraftCache<L> {
    prediction: L,
    prepared_input: Option<eredu_runtime::SpeculativeIdentity>,
    capture_generation: Option<u64>,
    prepared: Option<prepared_copy::PredictionOwnership<L>>,
}

impl<L> EmbeddedPredictionDraftCache<L> {
    /// Installs this branch's actual prediction completion/source witness.
    pub fn retain_evidence<E: 'static>(&mut self, evidence: E) -> Result<bool, eredu_core::BackendFailure> {
        let Some(prepared) = &mut self.prepared else { return Ok(false); };
        let evidence = prepared.copy.retain_evidence(evidence)?;
        prepared.evidence = Some(evidence);
        Ok(true)
    }
    /// Borrow the actual completed invocation source for this branch's output.
    pub fn logit_evidence(&self) -> Option<&PreparedEmbeddedEvidence> {
        self.prepared.as_ref().and_then(|prepared| prepared.evidence.as_ref())
    }

    /// Borrows prediction-local storage.
    pub const fn prediction(&self) -> &L {
        &self.prediction
    }

    /// Mutably borrows prediction-local storage.
    pub fn prediction_mut(&mut self) -> &mut L {
        &mut self.prediction
    }

    /// Lends this fork's state and its own completed-source witness slot.
    pub fn prediction_phase_state(&mut self) -> PredictionPhaseState<'_, L> {
        PredictionPhaseState::new(&mut self.prediction, self.prepared.as_mut().map(|prepared| {
            (&prepared.copy, &mut prepared.evidence)
        }))
    }

    /// Borrows this prediction branch's exact selected/input identity.
    pub fn lane_identity_ref<'a>(&'a self, selected: &'a eredu_runtime::SelectedSpeculativeRealization)
        -> Result<eredu_runtime::SpeculativeLaneIdentityRef<'a>, EmbeddedPredictionCacheError> {
        let prepared = self.prepared_input.as_ref().ok_or(EmbeddedPredictionCacheError::CaptureBeforePreparedInput)?;
        let generation = self.capture_generation.ok_or(EmbeddedPredictionCacheError::MissingCaptureGeneration)?;
        Ok(selected.lane_identity_ref(prepared, generation))
    }

    /// Reconstructs the selected lane identity retained by this prediction-only fork.
    pub fn lane_identity(
        &self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
    ) -> Result<eredu_runtime::SpeculativeLaneIdentity, EmbeddedPredictionCacheError> {
        let prepared = self
            .prepared_input
            .clone()
            .ok_or(EmbeddedPredictionCacheError::CaptureBeforePreparedInput)?;
        let generation = self
            .capture_generation
            .ok_or(EmbeddedPredictionCacheError::MissingCaptureGeneration)?;
        Ok(selected.lane_identity(prepared, generation))
    }
}

impl<L: Clone> EmbeddedPredictionDraftCache<L> {
    /// Copies current state with its retained provider, preserving ordinary behavior.
    pub fn try_copy(&self) -> Result<Self, eredu_core::BackendFailure> {
        if let Some(prepared) = &self.prepared {
            let prediction = prepared.copy.copy_state(&self.prediction, prepared.evidence.as_ref())?;
            let (prepared_input, metadata_host) = prepared_copy::identity::<_, Self>(&self.prepared_input, &prepared.copy)?;
            return Ok(Self { prediction: prediction.value, prepared_input,
                capture_generation: self.capture_generation,
                prepared: Some(prepared_copy::PredictionOwnership { copy: prepared.copy.clone(),
                    evidence: prediction.evidence, payload_host: prediction.host, metadata_host }) });
        }
        Ok(Self { prediction: self.prediction.clone(), prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation, prepared: None })
    }
}

/// Neutral embedded-prediction cache contract failure.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddedPredictionCacheError {
    /// Input omitted its exact prepared/content identity.
    #[error("embedded speculative input is missing its prepared-input cache identity")]
    MissingPreparedInput,
    /// One lane was reused with different prepared/content identity.
    #[error("embedded speculative cache belongs to a different prepared input")]
    DifferentPreparedInput,
    /// A target capture was requested before input identity binding.
    #[error("embedded target capture precedes prepared-input binding")]
    CaptureBeforePreparedInput,
    /// A prediction-only fork has no retained target frontier.
    #[error("prediction cache has no bound target generation")]
    MissingCaptureGeneration,
    /// Target storage is temporarily installed in its singular session.
    #[error("prediction target cache is already active")]
    TargetStateActive,
    /// Checkpoint and live target storage membership differ.
    #[error("prediction target checkpoint state presence changed")]
    TargetPresenceChanged,
    /// Prepared-input identity construction failed.
    #[error("invalid embedded prepared-input identity: {0}")]
    Identity(String),
    /// Exact source-funded construction or copy refusal.
    #[error(transparent)]
    Prepared(eredu_core::BackendFailure),
}

/// Cache failure preserving a backend-native opaque storage error.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddedPredictionCacheAccessError<E> {
    /// Neutral envelope contract failure.
    #[error(transparent)]
    Cache(#[from] EmbeddedPredictionCacheError),
    /// Opaque native storage mechanism failure.
    #[error("embedded prediction native state operation failed: {0}")]
    Native(E),
    /// Source-funded preparation/copy failure retains its exact typed cause.
    #[error(transparent)]
    Prepared(eredu_core::BackendFailure),
}

/// Fixed protocol diagnostics for the shared embedded prediction driver.
#[derive(Clone,Copy,Debug,thiserror::Error)]
pub enum EmbeddedPredictionContractError {
    /// The erased value did not match the executor's declared concrete type.
    #[error("embedded executor carried a mismatched erased {value}")]
    ErasedValue { /// Declared boundary value.
        value: &'static str },
    /// A commit exceeds the actual provisional block.
    #[error("cannot commit {verified} embedded-prediction inputs from a block of {available}")]
    Commit { /// Requested prefix.
        verified:usize, /// Actual provisional length.
        available:usize },
    /// A completed output disagrees with the declared sequence geometry.
    #[error("embedded prediction output lengths disagree: logits={logits}, capture={capture}, tokens={tokens}, expected={expected:?}")]
    Output { /// Readout rows.
        logits:usize, /// Captured rows.
        capture:usize, /// Input positions.
        tokens:usize, /// Requested rows, when fixed.
        expected:Option<usize> },
    /// Fused output is shorter than the selected proposal count.
    #[error("fused embedded prediction block has {available} rows, but {requested} were requested")]
    FusedCapacity { /// Selected proposal count.
        requested:usize, /// Actual rows.
        available:usize },
}

/// Tensor, token, transfer, and completion mechanisms needed by embedded prediction.
///
/// Implementations contain no architecture identity, prediction depth, capture path, or replay
/// policy. The selected execution context determines where each operation runs.
pub trait SpeculativeTensorMechanisms: 'static {
    /// Native retained tensor value.
    type Tensor: Clone;
    /// Native logits value consumed by the selected sampling mechanism.
    type Logits: Clone;
    /// Selected target/draft execution assignment.
    type Context<'a>: Copy;
    /// Exact completion retaining submitted verification resources.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Native mechanism failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Moves an already retained source error into the public neutral boundary.
    /// Ordinary errors remain in their native domain for the existing adapter.
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error> {Err(error)}

    /// Coordinates host scheduler facts without executing a model equation.
    fn coordinate_speculative_step<'a>(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Ok(local)
    }

    /// Agrees host readiness through retained native session transport.
    fn agree_text_preparation<'a>(
        _stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        _context: Self::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        use eredu_core::run_preparation::{
            TextPreparationOutcome as O, TextPreparationStatus as S,
        };
        Ok(match status {
            S::Ready => O::Ready,
            S::Cancelled => O::Cancelled,
            S::Failed => O::Rejected { rank: 0 },
        })
    }

    /// Translates a neutral internal-observation rejection without native work.
    fn observation_error(message: &'static str) -> Self::Error;

    /// Validates outer callback ownership before even querying its optional
    /// prefill capabilities. This check invokes no caller callback.
    fn validate_outer_tensor_observer<'a>(
        _has_caller: bool, _context: Self::Context<'a>,
    ) -> Result<(), Self::Error> { Ok(()) }

    /// Observes an owned outer tensor through the existing callback worker.
    /// An absent callback is structural: original backends may move the value
    /// after validating their context, while ordinary execution keeps its clone.
    fn observe_outer_tensor<'a>(
        value: Self::Tensor, path: &str, chunk: Option<&eredu_runtime::prefill::PrefillChunk>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Self::Tensor, Self::Error>>,
        _context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        observe_embedded_tensor_ordinary(value, path, chunk, observer)
    }

    /// Complete bound for a durable copy of one captured seed tensor.
    fn control_tensor_estimate(
        _value: &Self::Tensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies a seed tensor after reservation and settles native work before
    /// returning. The portable error retains the original native cause.
    fn control_tensor_snapshot<'a>(
        _value: &Self::Tensor,
        _context: Self::Context<'a>,
    ) -> Result<Option<Self::Tensor>, SpeculativeControlError> {
        Ok(None)
    }

    /// Durable immutable seed ownership. Ordinary snapshots keep their existing
    /// independent tensor copy; original packets may share their completed value.
    fn control_tensor_packet_estimate(value:&EmbeddedPredictionTensor<Self::Tensor>)
        ->Option<eredu_core::execution_control::SnapshotEstimate>{Self::control_tensor_estimate(value)}
    /// Copies or shares the seed through its selected immutable ownership policy.
    fn control_tensor_packet_snapshot<'a>(value:&EmbeddedPredictionTensor<Self::Tensor>,context:Self::Context<'a>)
        ->Result<Option<EmbeddedPredictionTensor<Self::Tensor>>,SpeculativeControlError>{
        Self::control_tensor_snapshot(value,context).map(|value|value.map(EmbeddedPredictionTensor::ordinary))
    }

    /// Binds an admitted portable collector to native capture mechanisms. Scopes
    /// and family semantics have already been resolved by architecture admission.
    fn activation_observer<'a>(
        plan: &AdmittedSpeculativeActivations,
        _request: eredu_core::SpeculativeRequestId,
        _context: Self::Context<'a>,
    ) -> Result<
        Option<Box<dyn SpeculativeActivationObserver<Self::Tensor, Self::Error>>>,
        SpeculativeControlError,
    > {
        if plan.is_empty() {
            Ok(None)
        } else {
            Err(SpeculativeControlError::Unsupported(
                "selected mechanisms have no speculative activation collector",
            ))
        }
    }

    /// Constructs the stable empty-input failure in the backend error domain.
    fn empty_prediction_input() -> Self::Error;

    /// Constructs the stable fused-block exhaustion failure in the backend error domain.
    fn fused_prediction_exhausted() -> Self::Error;

    /// Constructs the stable invalid-commit failure in the backend error domain.
    fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error;

    /// Constructs the stable output-geometry failure in the backend error domain.
    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Self::Error;

    /// Constructs the stable proposal-capacity failure in the backend error domain.
    fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error;

    /// Returns the sequence width of a retained tensor.
    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error>;

    /// Moves an already selected public score tensor into sampling ownership.
    /// No native indexing, allocation or execution may occur in this conversion.
    fn selected_prefill_logits(value: Self::Tensor) -> Result<Self::Logits, Self::Error>;

    /// Selects one logits row from a sequence tensor.
    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Selects from one retained completed output proof; the ordinary backend
    /// keeps its original row worker. Evidence grants no new model invocation.
    fn logits_row_with_source<'a>(
        value: &Self::Tensor, row: usize, _source: Option<&PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> { Self::logits_row(value, row, context) }

    /// Selects who owns the final prefill score-row operation. A source-bound
    /// implementation may retain selected positions and use its existing paid
    /// numerical consumer after the target phase completes.
    fn prefill_score_layout<'a>(
        _context: Self::Context<'a>,
    ) -> eredu_runtime::replicated_session::PrefillScoreLayout {
        eredu_runtime::replicated_session::PrefillScoreLayout::FinalScores
    }

    /// Receives the declared score layout with its exact completed source.
    /// SelectedPositions requires the implementation to select and complete the
    /// final causal row under that source's own execution authority.
    fn selected_prefill_logits_with_source<'a>(
        value: Self::Tensor, _source: Option<&PreparedEmbeddedEvidence>,
        _context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> { Self::selected_prefill_logits(value) }

    /// Retains one immutable block; original backends pay the shared fixed
    /// destination before construction and validate the supplied output source.
    fn retain_logit_block<'a>(
        value: Self::Tensor, _source: Option<PreparedEmbeddedEvidence>,
        _context: Self::Context<'a>,
    ) -> Result<EmbeddedPredictionLogitBlock<Self::Tensor>, Self::Error> {
        Ok(EmbeddedPredictionLogitBlock::ordinary(value))
    }

    /// A completed capture uses the same immutable owner as fused logits.
    fn retain_tensor_packet<'a>(value:Self::Tensor,source:Option<PreparedEmbeddedEvidence>,context:Self::Context<'a>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{Self::retain_logit_block(value,source,context)}
    /// The returned row retains this exact output's source independently of cache mutation.
    fn tensor_row_with_source<'a>(value:&Self::Tensor,row:usize,_source:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'a>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        Self::tensor_row(value,row,context).map(EmbeddedPredictionTensor::ordinary)
    }
    /// A prefix retains the particular output proof supplied by its owner.
    fn tensor_prefix_with_source<'a>(value:&Self::Tensor,end:usize,_source:Option<&PreparedEmbeddedEvidence>,context:Self::Context<'a>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        Self::tensor_prefix(value,end,context).map(EmbeddedPredictionTensor::ordinary)
    }
    /// Views of existing immutable packets also retain their earlier operation custody.
    fn tensor_row_packet<'a>(value:&EmbeddedPredictionTensor<Self::Tensor>,row:usize,context:Self::Context<'a>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        Self::tensor_row_with_source(value,row,value.evidence(),context)
    }
    /// A prefix of an owned packet preserves its predecessor custody.
    fn tensor_prefix_packet<'a>(value:&EmbeddedPredictionTensor<Self::Tensor>,end:usize,context:Self::Context<'a>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        Self::tensor_prefix_with_source(value,end,value.evidence(),context)
    }
    /// Ordinary realization remains the caller's existing tensor worker. An
    /// original realization executes only its fixed, source-bound numerical cut.
    fn tensor_concatenate_packet<'a>(left:&EmbeddedPredictionTensor<Self::Tensor>,right:&EmbeddedPredictionTensor<Self::Tensor>,
        _context:Self::Context<'a>,ordinary:impl FnOnce(&Self::Tensor,&Self::Tensor)->Result<Self::Tensor,Self::Error>)
        ->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        ordinary(left,right).map(EmbeddedPredictionTensor::ordinary)
    }

    /// Selects one sequence row while retaining its sequence dimension.
    fn tensor_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Selects a prefix while retaining its sequence dimension.
    fn tensor_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Selects a half-open token range while retaining its sequence dimension.
    fn token_range<'a>(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Selects a token prefix while retaining its sequence dimension.
    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Constructs an exact token tensor on the selected target placement.
    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Source-carrying token input. Ordinary mechanisms retain their existing
    /// producer; original mechanisms return the actual completed tensor owner.
    fn target_tokens_packet<'a>(tokens: &[u32], context: Self::Context<'a>)
        -> Result<EmbeddedPredictionTensor<Self::Tensor>, Self::Error> {
        Self::target_tokens(tokens, context).map(EmbeddedPredictionTensor::ordinary)
    }
    /// Borrows the exact admitted prefill token packet. Ordinary sources keep
    /// their existing handle clone; original sources must supply their owner.
    fn prefill_token_packet<'a>(value:&Self::Tensor,prepared:Option<&EmbeddedPredictionTensor<Self::Tensor>>,
        _context:Self::Context<'a>)->Result<EmbeddedPredictionTensor<Self::Tensor>,Self::Error>{
        Ok(prepared.cloned().unwrap_or_else(||EmbeddedPredictionTensor::ordinary(value.clone())))
    }
    /// Static token view retaining its particular immutable source packet.
    fn token_range_packet<'a>(value: &EmbeddedPredictionTensor<Self::Tensor>, start: usize, end: usize,
        context: Self::Context<'a>) -> Result<EmbeddedPredictionTensor<Self::Tensor>, Self::Error> {
        Self::token_range(value, start, end, context).map(EmbeddedPredictionTensor::ordinary)
    }
    /// Lends this actual input's evidence only for the callback's model call.
    /// The callback cannot return a borrow of its shortened execution context.
    fn with_tensor_source<R>(
        source: Option<&PreparedEmbeddedEvidence>, context: Self::Context<'_>,
        run: impl for<'scope> FnOnce(Self::Context<'scope>) -> Result<R, Self::Error>,
    ) -> Result<R, Self::Error> {
        match source {Some(source)=>Self::with_tensor_sources(&[source],context,run),
            None=>Self::with_tensor_sources(&[],context,run)}
    }
    /// Extends an existing lexical input loan without replacing its sources.
    fn with_tensor_sources<R>(
        _sources:&[&PreparedEmbeddedEvidence],context:Self::Context<'_>,
        run:impl for<'scope> FnOnce(Self::Context<'scope>)->Result<R,Self::Error>,
    )->Result<R,Self::Error>{run(context)}

    /// Selects one row from a fused proposal block.
    fn fused_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Lazily selects only the requested fused row from its own completed source.
    fn fused_logits_row_with_source<'a>(
        value: &Self::Tensor, row: usize, _source: Option<&PreparedEmbeddedEvidence>,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> { Self::fused_logits_row(value, row, context) }

    /// An original target phase may already have completed its actual roots;
    /// the backend validates and retains that output's own source evidence.
    fn submit_verification_completion_with_source<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>, inputs: &Self::Tensor,
        _source: Option<&PreparedEmbeddedEvidence>, context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        Self::submit_verification_completion(output, inputs, context)
    }

    /// Submits exact verification completion while retaining required resources.
    fn submit_verification_completion<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error>;
}

/// Target output and architecture-owned capture used by an embedded prediction strategy.
#[derive(Debug, Clone)]
pub struct EmbeddedPredictionOutput<T> {
    /// Target logits for every evaluated input position.
    pub logits: T,
    /// Architecture-selected target capture for every evaluated position.
    pub capture: T,
    /// Exact evaluated token ids.
    pub tokens: EmbeddedPredictionTensor<T>,
}

impl<T> EmbeddedPredictionOutput<T> {
    /// Creates one exact target output.
    pub const fn new(logits: T, capture: T, tokens: T) -> Self {
        Self {
            logits,
            capture,
            tokens: EmbeddedPredictionTensor::ordinary(tokens),
        }
    }
    /// Keeps the exact immutable input packet rather than cloning its native value.
    pub const fn with_tokens(logits: T, capture: T, tokens: EmbeddedPredictionTensor<T>) -> Self {
        Self { logits, capture, tokens }
    }

    /// Borrows target logits.
    pub const fn logits(&self) -> &T {
        &self.logits
    }

    /// Borrows the architecture-selected target capture.
    pub const fn capture(&self) -> &T {
        &self.capture
    }

    /// Borrows exact evaluated token ids.
    pub fn tokens(&self) -> &T {
        &self.tokens
    }
}

/// Architecture-owned embedded prediction behavior over backend mechanisms.
///
/// Implementations are typed to one prepared prediction extension. The neutral executor below
/// owns proposal ordering, cache forking, verification replay, and commit policy.
pub trait EmbeddedPredictionStrategy<M: SpeculativeTensorMechanisms + 'static> {
    /// Prepared target input.
    type Input;
    /// Complete ordinary-target lane cache.
    type TargetCache;
    /// Separately typed embedded-prediction cache.
    type PredictionCache;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;

    /// Borrow the just-completed target proof for a particular returned output.
    fn target_logit_evidence<'a>(&self, _cache: &'a Self::TargetCache)
        -> Option<&'a PreparedEmbeddedEvidence> { None }
    /// Exact completed prediction output, separate from target output evidence.
    fn prediction_tensor_evidence<'a>(&self,_cache:&'a Self::PredictionCache)
        ->Option<&'a PreparedEmbeddedEvidence>{None}


    /// Complete durable target-lane estimate, including canonical prediction state.
    fn control_target_estimate(
        &self,
        _cache: &Self::TargetCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Independent bound for the separately retained prediction seed replica.
    fn control_prediction_estimate(
        &self,
        _cache: &Self::PredictionCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies the full canonical lane after reservation.
    fn control_target_snapshot<'a>(
        &self,
        _cache: &Self::TargetCache,
        _context: M::Context<'a>,
    ) -> Result<Option<Self::TargetCache>, SpeculativeControlError> {
        Ok(None)
    }
    /// Copies the independently retained seed replica after reservation.
    fn control_prediction_snapshot<'a>(
        &self,
        _cache: &Self::PredictionCache,
        _context: M::Context<'a>,
    ) -> Result<Option<Self::PredictionCache>, SpeculativeControlError> {
        Ok(None)
    }

    /// Maximum number of proposal tokens in one verification transaction.
    fn proposal_capacity(&self) -> usize;

    /// Complete internal hook coverage of the selected target and predictor.
    /// An outer tensor/logit observer alone does not establish this capability.
    fn supports_internal_observations(&self) -> bool {
        false
    }

    /// Native invocation adapters may require scheduler provenance independently
    /// of optional internal observation. Ordinary strategies retain no origin.
    fn requires_activation_origin(&self) -> bool { false }
    fn set_activation_origin(&mut self, _origin: Option<SpeculativeActivationOrigin>) {}

    /// Prepares the concrete two-cause error destination before an operation
    /// can mutate target state. A failed restoration must retain both causes.
    fn prepare_rollback_failure<'context: 'context>(context: M::Context<'context>)
        -> Result<impl FnOnce(eredu_core::speculative::SpeculativeRollbackFailure<M::Error, M::Error>) -> M::Error, M::Error>;

    /// Creates a fallible exact checkpoint of ordinary target and prediction state.
    fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, M::Error>;

    /// Enables optional component telemetry.
    fn set_telemetry_enabled(&mut self, _enabled: bool) {}

    /// Whether optional component telemetry is available.
    fn supports_telemetry(&self) -> bool {
        false
    }

    /// Drains completed component telemetry.
    fn take_telemetry(&mut self) -> Result<Self::Telemetry, M::Error>;

    /// Drains telemetry retained by a target verification output.
    fn take_verification_telemetry(
        &mut self,
        _output: &mut EmbeddedPredictionOutput<M::Tensor>,
    ) -> Result<Self::Telemetry, M::Error>;

    /// Runs ordinary-target prefill and returns its exact selected capture.
    fn prefill_target<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
    ) -> Result<EmbeddedPredictionOutput<M::Tensor>, M::Error>;

    /// Shared registration enters the same cancellable initial prefill. Legacy
    /// strategies retain their one complete call; materialized strategies use
    /// the selected captured-span driver for default and explicit chunk policies.
    fn prefill_cancellable<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        observers: &mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
        _cancellation: &eredu_core::GenerationCancellationToken,
        context: M::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            captured_prefill::PrefillValues<M::Tensor, M::Logits>,
        >,
        M::Error,
    > {
        captured_prefill::legacy(self, input, cache, observers, context)
            .map(eredu_core::SpeculativePrefillOutcome::Complete)
    }

    /// Runs ordinary-target verification and returns its exact selected capture.
    fn verify_target<'a>(
        &mut self,
        tokens: &EmbeddedPredictionTensor<M::Tensor>,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<M::Tensor>, M::Error>;

    /// Seeds prediction-local state from a successful ordinary-target transaction.
    fn seed_prediction_cache<'a>(
        &mut self,
        output: &EmbeddedPredictionOutput<M::Tensor>,
        tokens: &M::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
    ) -> Result<(), M::Error>;

    /// Forks prediction-local state from the authoritative target lane.
    fn prediction_cache(&self, cache: &Self::TargetCache)
        -> Result<Self::PredictionCache, M::Error>;

    /// Copies an existing prediction fork before independent advancement.
    /// Retained implementations use the same source-bound copy provider as checkpoints.
    fn copy_prediction_cache(&self, cache: &Self::PredictionCache)
        -> Result<Self::PredictionCache, M::Error>;

    /// Commits prediction-local state into the authoritative target lane.
    fn commit_prediction_cache(
        &self,
        cache: &mut Self::TargetCache,
        prediction: &Self::PredictionCache,
    ) -> Result<(), M::Error>;

    /// Restores an exact ordinary-target checkpoint.
    fn restore_target_checkpoint<'a>(
        cache: &mut Self::TargetCache,
        checkpoint: &Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<(), M::Error>;

    /// Runs one sequential prediction depth.
    fn sequential_logits<'a>(
        &mut self,
        capture: &M::Tensor,
        last_token: u32,
        depth: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
    ) -> Result<(M::Logits, M::Tensor), M::Error>;

    /// Optionally runs one fused proposal block from the exact target capture.
    fn fused_logits<'a>(
        &mut self,
        _capture: &M::Tensor,
        _last_token: u32,
        _capacity: usize,
        _cache: &mut Self::PredictionCache,
        _context: M::Context<'a>,
        _observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
    ) -> Result<Option<EmbeddedPredictionLogitBlock<M::Tensor>>, M::Error> {
        Ok(None)
    }

    /// Applies an architecture-declared token-conditioned fused adjustment.
    fn adjust_fused_logits<'a>(
        &mut self,
        logits: M::Logits,
        _last_token: u32,
        _context: M::Context<'a>,
    ) -> Result<M::Logits, M::Error> {
        Ok(logits)
    }

    /// Advances prediction-local state for newly retained verified inputs.
    fn advance_prediction_cache<'a>(
        &mut self,
        captures: &M::Tensor,
        tokens: &M::Tensor,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
    ) -> Result<(), M::Error>;
}

/// Backend-owned input lowering for one statically typed replicated target.
///
/// Plain text and composite admission have different borrowed input forms.  This mechanism owns
/// only that lowering step; the architecture strategy retains prefill/decode, capture validation,
/// extension state, proposal ordering, replay, and commit policy.
pub trait ReplicatedPredictionInput<A, B, S, E>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
{
    /// Owned request input accepted by the public backend.
    type Input;

    /// Exact source plan for the shared captured-span driver.
    type Prefill: PredictionPrefillPlan<A, B, S>;

    /// Keeps the original native input owner alive throughout the callback.
    /// Preparation failures are delivered to the callback so the shared runtime
    /// can agree them before any participant enters a target span.
    fn with_prefill_source<R>(
        &mut self,
        input: Self::Input,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl FnOnce(Result<Self::Prefill, E>) -> Result<R, E>,
    ) -> Result<R, E>;

    fn with_prefill_source_with_metadata<R>(
        &mut self,
        input: Self::Input,
        prepared: eredu_runtime::input::PreparedModelInputOwner<B::Tensor>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _metadata: &eredu_nn::workspace::WorkspaceContext,
        operation: impl FnOnce(Result<Self::Prefill, E>) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<eredu_nn::Error>,
    {
        let result = operation(Err(eredu_nn::Error::from(
            eredu_nn::workspace::WorkspaceMetadataError::Unqualified,
        )
        .into()));
        drop((prepared, input));
        result
    }

    /// Reads the retained scheduling policy without lowering or copying input.
    fn requested_chunks(input: &Self::Input) -> Option<std::num::NonZeroU64>;

    /// Lowers an owned prefill request and lends the exact architecture input plus token tensor.
    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl for<'a> FnOnce(
            A::Input<'a>,
            B::Tensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, E>,
    ) -> Result<R, E>;

    /// Lowers verified tokens to the architecture's exact decode input.
    fn with_decode<R>(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, E>,
    ) -> Result<R, E>;
}

/// Backend-native mechanics needed by the typed replicated prediction strategy.
///
/// Implementations contain no architecture identity, capture path, prediction depth, proposal
/// equation, replay rule, or cache-membership policy.
pub trait ReplicatedPredictionNative<A, B, S, M>
where
    B: eredu_nn::NeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    /// Owned backend input accepted by the final erased executor.
    type Input;
    /// Optional backend component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Fixed scheduler-facing type bundle used after architecture pairing.
    type ExecutorTypes: EmbeddedExecutorTypes<
        Input = Self::Input,
        Logits = M::Logits,
        Completion = M::Completion,
        Telemetry = Self::Telemetry,
        Error = M::Error,
    >;
    /// Binds the fixed erased context to the typed tensor mechanisms.
    fn executor_context<'a>(
        context: <Self::ExecutorTypes as EmbeddedExecutorTypes>::Context<'a>,
    ) -> M::Context<'a>;
    /// Returns the ordinary target tensor context selected by composition.
    fn target_context<'a>(context: M::Context<'a>) -> &'a <B::Tensor as eredu_nn::Tensor>::Context;
    fn with_prefill_source<I, P, R>(
        lowerer: &mut I,
        input: Self::Input,
        context: M::Context<'_>,
        operation: impl for<'source> FnOnce(Result<P, M::Error>, M::Context<'source>) -> Result<R, M::Error>,
    ) -> Result<R, M::Error>
    where
        I: ReplicatedPredictionInput<A, B, S, M::Error, Input = Self::Input, Prefill = P>,
    {
        lowerer.with_prefill_source(input, Self::target_context(context), |source|operation(source,context))
    }

    fn prepare_prefill_chunk<P>(
        source: &P,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        context: M::Context<'_>,
    ) -> Result<P::Chunk, A::Error>
    where
        P: PredictionPrefillSource<A, B, S>,
    {
        source.prepare_chunk(chunk, Self::target_context(context))
    }

    /// Preserves snapshot resource authority when projecting the full execution
    /// context for an independently retained prediction-state copy.
    fn prediction_snapshot_context<'a>(
        context: M::Context<'a>,
    ) -> <Self as crate::prediction_extension::PredictionExtensionMaterializer<B>>::SnapshotContext<'a>
    where
        Self: crate::prediction_extension::PredictionExtensionMaterializer<B>,
        B: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend;
    /// Whether the execution context requires the source-funded cache producer.
    fn uses_prepared_cache(_context:M::Context<'_>)->bool {false}
    /// Pays concrete shared cache/erasure controls before their construction.
    fn prepared_cache_metadata(_bytes:Option<usize>,_context:M::Context<'_>)->Result<eredu_core::HostPreparationAuthority,eredu_core::BackendFailure> {
        Ok(eredu_core::HostPreparationAuthority::unmanaged())
    }
    /// Pays and retains the concrete typed preparation failure without formatting.
    fn prepared_cache_error<E:std::error::Error+Send+Sync+'static>(cause:E,_context:M::Context<'_>)->eredu_core::BackendFailure {
        eredu_core::BackendFailure::from_error(cause)
    }
    /// Copies the actual canonical target and consumes the paid prediction lane.
    /// It runs only inside the shared mutating target-preparation transaction.
    fn prepared_cache<P>(source:&S,extension:&P,selected:&eredu_runtime::SelectedSpeculativeRealization,context:M::Context<'_>)
        ->Result<EmbeddedPredictionCache<S,P::LaneState>,eredu_core::BackendFailure>
    where Self:Sized+crate::prediction_extension::PredictionExtensionMaterializer<B>,
        S:'static,
        B:eredu_nn::BlockwiseAttentionBackend+eredu_nn::DistributedNeuralBackend+eredu_nn::GroupedNeuralBackend+eredu_nn::HyperNeuralBackend,
        P:crate::prediction_extension::MaterializedPredictionExecutor<A,B,Self>,
    {
        let _=(source,extension,selected,context);
        Err(eredu_core::BackendFailure::from_error(PreparedEmbeddedCopyError::SourceMismatch))
    }
    /// Creates an exact native target-state checkpoint.
    fn checkpoint(state: &S) -> Result<S, M::Error>;
    /// Complete isolated snapshot bound for the target's native state profile.
    fn control_state_estimate(
        _state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies complete native state and settles copying before returning.
    /// The full execution context preserves backend resource authority while
    /// the ordinary tensor context alone may identify only a queue or device.
    fn control_state_snapshot<'a>(
        _state: &S,
        _context: M::Context<'a>,
    ) -> Result<Option<S>, SpeculativeControlError> {
        Ok(None)
    }
    /// Restores an exact native target-state checkpoint.
    fn restore(
        state: &mut S,
        checkpoint: &S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), M::Error>;
    /// Returns the current target-state frontier.
    fn generation(state: &S) -> Result<u64, M::Error>;
    /// Constructs a one-token tensor for an architecture prediction operation.
    fn token(
        token: u32,
        context: M::Context<'_>,
    ) -> Result<B::Tensor, M::Error>;
    /// Returns the physical shape used to close the selected capture contract.
    fn shape(tensor: &B::Tensor) -> &[i32];
    /// Completes the actual architecture-selected state/output values using the
    /// full execution source. Ordinary implementations retain their old worker.
    fn complete_prediction_state<P>(
        extension: &P,
        state: &mut P::LaneState,
        outputs: &[&B::Tensor],
        point: PredictionCompletionPoint,
        sources: PredictionCompletionSources<'_>,
        context: M::Context<'_>,
    ) -> Result<(), M::Error>
    where
        Self: Sized + crate::prediction_extension::PredictionExtensionMaterializer<B>,
        B: eredu_nn::BlockwiseAttentionBackend + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend + eredu_nn::HyperNeuralBackend,
        P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, Self>,
    {
        let _ = (point, sources);
        extension.complete_state(state, outputs, Self::target_context(context))
            .map_err(Self::session_failure)
    }
    /// Runs an operation inside the backend's deferred-validation transaction.
    fn validate<T>(operation: impl FnOnce() -> Result<T, M::Error>) -> Result<T, M::Error>;
    /// Uses an already prepared invocation validation collector when present;
    /// ordinary execution keeps its existing deferred-validation transaction.
    fn validate_with_context<T>(
        _context: M::Context<'_>,
        operation: impl FnOnce() -> Result<T, M::Error>,
    ) -> Result<T, M::Error> {
        Self::validate(operation)
    }
    /// The captured-span driver may contain separate native target and seed
    /// invocations. Ordinary execution retains its enclosing collector.
    fn validate_captured_span<T>(
        _context: M::Context<'_>,
        operation: impl FnOnce() -> Result<T, M::Error>,
    ) -> Result<T, M::Error> { Self::validate(operation) }
    /// Maps a neutral-session failure into the backend error domain.
    fn session_error(error: impl std::fmt::Display) -> M::Error;
    /// Preserves an owned native/session cause across the prediction error domain.
    fn session_failure(error: eredu_core::BackendFailure) -> M::Error;
    /// Preserves a typed session/policy cause using the actual source context.
    /// Native implementations pay their existing retained error transport;
    /// allocation refusal remains a typed error, not formatted fallback text.
    fn session_cause_with_context<E:std::error::Error+Send+Sync+'static>(
        error:E,_context:M::Context<'_>,
    )->M::Error {Self::session_error(error)}

    /// Prepares one concrete session-error conversion before the fallible
    /// operation begins. Consuming this callback retains the original cause;
    /// it must not require another metadata grant after that operation fails.
    /// Additional caller transport controls are reserved with that destination;
    /// `None` denotes overflow. Ordinary conversion keeps its existing policy.
    // The explicit lifetime bound gives the GAT loan and its concrete native
    // implementation the same early-bound lifetime in the opaque return type.
    fn prepare_session_cause<'context: 'context, E: std::error::Error + Send + Sync + 'static>(
        context: M::Context<'context>,
        _additional_controls: Option<usize>,
    ) -> Result<impl FnOnce(E) -> M::Error, M::Error> {
        Ok(move |error| Self::session_cause_with_context(error, context))
    }

    /// Constructs a rejected operation diagnostic from borrowed arguments.
    fn session_arguments_with_context(arguments: std::fmt::Arguments<'_>, _context:M::Context<'_>) -> M::Error {
        Self::session_error(arguments)
    }

    /// Crosses the shared neural driver boundary, preserving typed custody.
    fn neural_cause_with_context<E:std::error::Error+Send+Sync+'static>(error:E, _context:M::Context<'_>) -> eredu_nn::Error {
        eredu_nn::Error::backend_source(error)
    }

    /// The observer bridge retains the actual failure separately from its marker.
    fn neural_observer_error(error:&M::Error, _context:M::Context<'_>) -> eredu_nn::Error {
        eredu_nn::Error::backend(error.to_string())
    }

    /// Drains native component telemetry.
    fn take_telemetry() -> Result<Self::Telemetry, M::Error>;
}

/// Architecture-owned embedded strategy over one typed replicated session and paired extension.
pub struct ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M, H = OrdinaryPredictionPhase>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    >,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    phase: H,
    session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
    extension: &'a mut P,
    selected: &'a eredu_runtime::SelectedSpeculativeRealization,
    input: I,
    cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    _native: PhantomData<fn() -> (S, N, M)>,
}

impl<'a, A, B, S, SM, D, P, I, N, M, H>
    ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M, H>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    H: ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>,
{
    /// Completes the statically checked pairing before any executor erasure.
    pub fn new(
        session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        extension: &'a mut P,
        selected: &'a eredu_runtime::SelectedSpeculativeRealization,
        input: I,
        cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Self
    where H: Default,
    {
        Self::new_with_phase(session, extension, selected, input, cache_context, H::default())
    }

    /// Supplies typed backend invocation mechanics after architecture pairing.
    pub fn new_with_phase(
        session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        extension: &'a mut P,
        selected: &'a eredu_runtime::SelectedSpeculativeRealization,
        input: I,
        cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
        phase: H,
    ) -> Self {
        Self {
            phase,
            session,
            extension,
            selected,
            input,
            cache_context,
            _native: PhantomData,
        }
    }

    /// Keeps one auxiliary invocation in the ordinary session transaction. The
    /// unobserved path retains its existing execution and validation behavior.
    fn prediction_phase<R>(
        &mut self,
        lane: PredictionPhaseState<'_, P::LaneState>,
        pass: eredu_runtime::ExpertPass,
        context: M::Context<'_>,
        equation: &PredictionEquation<&B::Tensor>,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: impl for<'execution> FnOnce(
            &mut P,
            &mut ReplicatedPredictionInvoker<'_, 'execution, A, B, S, SM, D, N, M>,
            &mut P::LaneState,
            Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
            M::Context<'execution>,
        ) -> Result<R, M::Error>,
        complete: impl for<'execution> FnOnce(&P, &mut P::LaneState, &R, M::Context<'execution>) -> Result<(), M::Error>,
    ) -> Result<R, M::Error>
    where
        R: PredictionPhaseRoots<B::Tensor, M::Logits>,
        N: ReplicatedPredictionNative<A, B, S, M>,
    {
        self.phase.run(
            self.session, self.extension, lane, pass, equation,
            observer.is_some().then_some(PredictionCompletionPoint::ObservedEquation),
            None, context, observer,
            |session, extension, lane, observer, context| {
        let tensor_context = N::target_context(context);
        N::validate_with_context(context, || {
            let Some(observer) = observer else {
                return execute(
                    extension,
                    &mut ReplicatedPredictionInvoker {
                        session,
                        context: tensor_context,
                        execution_context: context,
                        _native: PhantomData,
                    },
                    lane,
                    None,
                    context,
                );
            };
            session.with_prediction_observation(
                pass,
                tensor_context,
                observer,
                &mut (extension, lane),
                |session, (extension, lane), observer| {
                    execute(
                        extension,
                        &mut ReplicatedPredictionInvoker {
                            session,
                            context: tensor_context,
                        execution_context: context,
                            _native: PhantomData,
                        },
                        lane,
                        Some(observer),
                        context,
                    )
                },
                |_, (extension, lane), output| {
                    complete(extension, lane, output, context)
                },
                |error| N::session_cause_with_context(error, context),
            )
        })
            },
        )
    }

    /// Context-bearing original construction preserves the existing shared
    /// target preparation agreement and retains typed failures directly.
    pub fn new_cache_with_context(&mut self,context:M::Context<'_>)
        ->Result<EmbeddedPredictionCache<S,P::LaneState>,eredu_core::BackendFailure>
    where N:ReplicatedPredictionNative<A,B,S,M>,S:'static {
        if !N::uses_prepared_cache(context) {
            return self.new_cache().map_err(eredu_core::BackendFailure::from_error);
        }
        let parts=[
            std::mem::size_of::<EmbeddedPredictionCache<S,P::LaneState>>(),
            std::mem::size_of::<Result<EmbeddedPredictionCache<S,P::LaneState>,eredu_core::BackendFailure>>(),
            std::mem::size_of::<eredu_runtime::replicated_session::ReplicatedTextSessionError<A::Error,SM::PolicyError,SM::Error>>(),
            std::mem::size_of::<eredu_runtime::replicated_session::PredictionTargetPreparationError>(),
            std::mem::size_of::<(&P,&eredu_runtime::SelectedSpeculativeRealization,M::Context<'_>)>(),
        ];
        let controls=parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add);
        let _host=N::prepared_cache_metadata(controls,context)?;
        let extension=&*self.extension;
        let selected=self.selected;
        self.session.prepare_prediction_target_state_with(
            N::target_context(context),
            |_,source,_|N::prepared_cache(source,extension,selected,context),
            |cache|cache.target().expect("prepared constructor retains target state"),
            |cause|N::prepared_cache_error(cause,context),
            |cause|N::prepared_cache_error(cause,context),
        )
    }

    /// Realizes one typed target/prediction lane before final executor erasure.
    pub fn new_cache(&mut self) -> Result<EmbeddedPredictionCache<S, P::LaneState>, M::Error>
    where
        N: ReplicatedPredictionNative<A, B, S, M>,
    {
        let state = self
            .session
            .prepare_prediction_target_state(self.cache_context)
            .map_err(N::session_error)?;
        Ok(EmbeddedPredictionCache::new(
            state,
            self.extension.new_state(),
        ))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct PredictionContractFailure(&'static str);

#[derive(Debug, thiserror::Error)]
#[error("prediction target state exchange failed: {exchange}; local ownership recovery failed: {recovery}")]
struct TargetStateRecoveryFailure<E:std::error::Error+'static,R:std::error::Error+'static> {
    #[source]
    exchange:E,
    recovery:R,
}

struct ReplicatedPredictionInvoker<'a, 'context, A, B, S, SM, D, N, M>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
    >,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    execution_context: M::Context<'context>,
    _native: PhantomData<fn() -> (N, M)>,
}

impl<A, B, S, SM, D, N, M> crate::prediction_extension::PredictionOperationInvoker<A, B, S>
    for ReplicatedPredictionInvoker<'_, '_, A, B, S, SM, D, N, M>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    N: ReplicatedPredictionNative<A, B, S, M>,
{
    type Error = M::Error;

    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Self::Error>
    where
        O: eredu_runtime::PredictionTargetOperation<A, B, S>,
    {
        self.session
            .apply_prediction_target_operation(operation, self.context)
            .map_err(|error| N::session_cause_with_context(error,self.execution_context))
    }

    fn invalid(message: String) -> Self::Error {
        N::session_error(message)
    }
    fn invalid_arguments(&self, arguments:std::fmt::Arguments<'_>)->Self::Error {
        N::session_arguments_with_context(arguments,self.execution_context)
    }
}

impl<A, B, S, SM, D, P, I, N, M, H> EmbeddedPredictionStrategy<M>
    for ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M, H>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B> + 'static,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    A::Error: std::error::Error + Send + Sync + 'static,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>
        + ReplicatedPredictionNative<A, B, S, M>,
    I: ReplicatedPredictionInput<A, B, S, M::Error, Input = N::Input>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    H: ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>,
{
    type Input = I::Input;
    type TargetCache = EmbeddedPredictionCache<S, P::LaneState>;
    type PredictionCache = EmbeddedPredictionDraftCache<P::LaneState>;
    fn target_logit_evidence<'a>(&self, cache: &'a Self::TargetCache)
        -> Option<&'a PreparedEmbeddedEvidence> { cache.target_logit_evidence() }
    fn prediction_tensor_evidence<'a>(&self,cache:&'a Self::PredictionCache)
        ->Option<&'a PreparedEmbeddedEvidence>{cache.logit_evidence()}


    type Telemetry = N::Telemetry;
    fn prepare_rollback_failure<'context: 'context>(context: M::Context<'context>)
        -> Result<impl FnOnce(eredu_core::speculative::SpeculativeRollbackFailure<M::Error, M::Error>) -> M::Error, M::Error> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<(M::Error, &mut Self::TargetCache, &Self::TargetCache, M::Context<'context>)>(),
            size_of::<Result<(), M::Error>>(),
            size_of::<M::Error>(),
            size_of::<eredu_core::speculative::SpeculativeRollbackFailure<M::Error, M::Error>>(),
            size_of::<Option<usize>>(),
        ];
        N::prepare_session_cause(context,
            controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add))
    }


    fn requires_activation_origin(&self) -> bool {
        self.phase.requires_activation_origin()
    }
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.phase.set_activation_origin(origin);
    }

    fn control_target_estimate(
        &self,
        cache: &Self::TargetCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            snapshot::host::<Self::TargetCache>(cache.prepared_input.as_ref()),
            N::control_state_estimate(cache.target.as_ref()?),
            self.extension.snapshot_estimate(&cache.prediction),
        ])
    }

    fn control_prediction_estimate(
        &self,
        cache: &Self::PredictionCache,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            snapshot::host::<Self::PredictionCache>(cache.prepared_input.as_ref()),
            self.extension.snapshot_estimate(&cache.prediction),
        ])
    }

    fn control_target_snapshot<'a>(
        &self,
        cache: &Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<Option<Self::TargetCache>, SpeculativeControlError> {
        if cache.target.is_none() {
            return Ok(None);
        }
        if cache.prepared.is_some() {
            return cache.checkpoint(|_| -> Result<S, std::convert::Infallible> {
                unreachable!("prepared snapshot uses its retained copy provider")
            }).map(Some).map_err(|cause| match cause {
                EmbeddedPredictionCacheAccessError::Prepared(cause) => SpeculativeControlError::Backend(cause),
                EmbeddedPredictionCacheAccessError::Native(never) => match never {},
                EmbeddedPredictionCacheAccessError::Cache(_) => unreachable!("prepared checkpoint does not validate a new binding"),
            });
        }
        let Some(target) = cache.target.as_ref() else {
            return Ok(None);
        };
        let Some(target) = N::control_state_snapshot(target, context)? else {
            return Ok(None);
        };
        let context = N::prediction_snapshot_context(context);
        let Some(prediction) = self
            .extension
            .snapshot(&cache.prediction, context)
            .map_err(SpeculativeControlError::Backend)?
        else {
            return Ok(None);
        };
        Ok(Some(EmbeddedPredictionCache {
            target: Some(target),
            prediction,
            prepared_input: cache.prepared_input.clone(),
            capture_generation: cache.capture_generation,
            prepared: None,
        }))
    }

    fn control_prediction_snapshot<'a>(
        &self,
        cache: &Self::PredictionCache,
        context: M::Context<'a>,
    ) -> Result<Option<Self::PredictionCache>, SpeculativeControlError> {
        if cache.prepared.is_some() {
            return cache.try_copy().map(Some).map_err(SpeculativeControlError::Backend);
        }
        let Some(prediction) = self
            .extension
            .snapshot(&cache.prediction, N::prediction_snapshot_context(context))
            .map_err(SpeculativeControlError::Backend)?
        else {
            return Ok(None);
        };
        Ok(Some(EmbeddedPredictionDraftCache {
            prediction,
            prepared_input: cache.prepared_input.clone(),
            capture_generation: cache.capture_generation,
            prepared: None,
        }))
    }

    fn supports_internal_observations(&self) -> bool {
        self.extension.activation_execution(self.selected).is_some()
    }

    fn proposal_capacity(&self) -> usize {
        self.selected
            .requirements()
            .strategy()
            .proposal_capacity()
            .get()
    }

    fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, M::Error> {
        cache.checkpoint(N::checkpoint).map_err(|error| match error {
            EmbeddedPredictionCacheAccessError::Prepared(cause) => N::session_failure(cause),
            other => N::session_error(other),
        })
    }

    fn take_telemetry(&mut self) -> Result<Self::Telemetry, M::Error> {
        N::take_telemetry()
    }

    fn take_verification_telemetry(
        &mut self,
        _output: &mut EmbeddedPredictionOutput<B::Tensor>,
    ) -> Result<Self::Telemetry, M::Error> {
        N::take_telemetry()
    }

    fn prefill_cancellable<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        observers: &mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: M::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            captured_prefill::PrefillValues<M::Tensor, M::Logits>,
        >,
        M::Error,
    > {
        self.prefill_captured_spans(input, cache, observers, cancellation, context)
    }

    fn prefill_target<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<EmbeddedPredictionOutput<B::Tensor>, M::Error> {
        let tensor_context = N::target_context(context);
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            phase: phase_adapter,
            ..
        } = self;
        lowerer.with_prefill(input, tensor_context, |prepared, tokens, identity| {
            let sequence = M::sequence_len(&tokens)?;
            cache
                .bind_prepared_input(identity)
                .map_err(|error| match error {
                    EmbeddedPredictionCacheError::Prepared(cause) => N::session_failure(cause),
                    other => N::session_cause_with_context(other,context),
                })?;
            let session_error = N::prepare_session_cause(context, Some(0))?;
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_cause_with_context(EmbeddedPredictionCacheError::TargetStateActive,context))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_cause_with_context(error,context));
            }
            let result = observation::neural_with_error(observer, SpeculativeActivationPhase::TargetPrefill,
                sequence, |cause|N::session_cause_with_context(cause,context), |error|N::neural_observer_error(error,context), |observer| phase_adapter.run_target(
                    session, cache.target_phase_evidence(), Some(&tokens),
                    SpeculativeActivationPhase::TargetPrefill, eredu_core::OutputDemand::Sequence, None,
                    context, observer,
                    |session, observer, phase_context| N::validate_with_context(phase_context, || {
                    match observer {
                        Some(observer) => session.prefill_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.prefill_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture)| (logits, capture))
                    .map_err(session_error)
                })));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => match session.recover_prediction_target_state_after_failure(&mut lane) {
                    Ok(()) => Err(N::session_cause_with_context(error, context)),
                    Err(recovery) => Err(N::session_cause_with_context(
                        TargetStateRecoveryFailure { exchange:error, recovery }, context)),
                },
            };
            cache.restore_target(lane);
            let output = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity_ref(selected, N::generation)
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            let output = EmbeddedPredictionOutput::new(output.0, output.1, tokens);
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            cache
                .retain_capture_generation(N::generation)
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            Ok(output)
        })
    }

    fn verify_target<'a>(
        &mut self,
        tokens: &EmbeddedPredictionTensor<B::Tensor>,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<B::Tensor>, M::Error> {
        M::with_tensor_source(tokens.evidence(), context, |context| {
        let sequence = M::sequence_len(tokens)?;
        let tensor_context = N::target_context(context);
        let retained = tokens.clone();
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            phase: phase_adapter,
            ..
        } = self;
        lowerer.with_decode(tokens, tensor_context, |prepared| {
            let session_error = N::prepare_session_cause(context, Some(0))?;
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_cause_with_context(EmbeddedPredictionCacheError::TargetStateActive,context))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_cause_with_context(error,context));
            }
            let result = observation::neural_with_error(observer, phase, sequence, |cause|N::session_cause_with_context(cause,context),
            |error|N::neural_observer_error(error,context),
                |observer| phase_adapter.run_target(
                    session, cache.target_phase_evidence(), Some(&**tokens),
                    phase, eredu_core::OutputDemand::Sequence, None,
                    context, observer,
                    |session, observer, phase_context| N::validate_with_context(phase_context, || {
                    match observer {
                        Some(observer) => session.decode_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.decode_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture)| EmbeddedPredictionOutput::with_tokens(logits, capture, retained))
                    .map_err(session_error)
                })));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => match session.recover_prediction_target_state_after_failure(&mut lane) {
                    Ok(()) => Err(N::session_cause_with_context(error, context)),
                    Err(recovery) => Err(N::session_cause_with_context(
                        TargetStateRecoveryFailure { exchange:error, recovery }, context)),
                },
            };
            cache.restore_target(lane);
            let output = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity_ref(selected, N::generation)
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            cache
                .retain_capture_generation(N::generation)
                .map_err(|cause|N::session_cause_with_context(cause,context))?;
            Ok(output)
        })
        })
    }

    fn seed_prediction_cache<'a>(
        &mut self,
        output: &EmbeddedPredictionOutput<B::Tensor>,
        tokens: &B::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(), M::Error> {
        let sequence = M::sequence_len(tokens)?;
        let lane_identity = cache
            .lane_identity_ref(self.selected, N::generation)
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        self.extension
            .validate_capture(self.selected, &lane_identity, N::shape(output.capture()))
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        let prediction_sequence = self.extension.prefill_sequence_len(sequence);
        if prediction_sequence == 0 {
            return Ok(());
        }
        let target_source=cache.target_logit_evidence().cloned();
        let hidden = M::tensor_prefix_with_source(output.capture(),sequence.saturating_sub(1),target_source.as_ref(),context)?;
        let tokens=M::prefill_token_packet(tokens,Some(&output.tokens),context)?;
        let next = M::token_range_packet(&tokens,1,sequence,context)?;
        let checkpoint = cache.prediction_fork().map_err(N::session_failure)?;
        let tensor_context = N::target_context(context);
        let equation = PredictionEquation::Prefill { target_capture: output.capture(), hidden: &*hidden, tokens: &*next };
        let result = M::with_tensor_source(target_source.as_ref(),context,|context|
            M::with_tensor_source(hidden.evidence(),context,|context|
            M::with_tensor_source(next.evidence(),context,|context|observation::neural_with_error(
            observer,
            SpeculativeActivationPhase::PredictionPrefill,
            prediction_sequence,
            |cause|N::session_cause_with_context(cause,context),
            |error|N::neural_observer_error(error,context),
            |observer| {
                self.prediction_phase(
                    cache.prediction_phase_state(),
                    eredu_runtime::ExpertPass::Prefill,
                    context,
                    &equation,
                    observer,
                    |extension, invoker, lane, observer, phase_context| {
                        execute_prediction_equation::<A, B, S, N, _, _>(
                            equation, extension, invoker, lane, observer,
                            |token| N::token(token, phase_context),
                        ).map(|output| {
                            let PredictionEquationOutput::StateOnly = output else { unreachable!("prefill equation result") };
                        })
                    },
                    |extension, lane, _, phase_context| N::complete_prediction_state(
                        extension, lane, &[], PredictionCompletionPoint::ObservedEquation,
                        PredictionCompletionSources::default(), phase_context,
                    ),
                )
            },
        ))));
        if result.is_err() {
            cache.rollback_prediction(checkpoint).map_err(N::session_failure)?;
        }
        result
    }

    fn prediction_cache(&self, cache: &Self::TargetCache) -> Result<Self::PredictionCache, M::Error> {
        cache.prediction_fork().map_err(N::session_failure)
    }

    fn copy_prediction_cache(&self, cache: &Self::PredictionCache) -> Result<Self::PredictionCache, M::Error> {
        cache.try_copy().map_err(N::session_failure)
    }

    fn commit_prediction_cache(
        &self,
        cache: &mut Self::TargetCache,
        prediction: &Self::PredictionCache,
    ) -> Result<(), M::Error> {
        cache.commit_prediction(prediction).map_err(N::session_failure)
    }

    fn restore_target_checkpoint<'a>(
        cache: &mut Self::TargetCache,
        checkpoint: &Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<(), M::Error> {
        cache
            .restore(checkpoint, |state, previous| {
                N::restore(state, previous, N::target_context(context))
            })
            .map_err(|error| match error {
                EmbeddedPredictionCacheAccessError::Prepared(cause) => N::session_failure(cause),
                other => N::session_cause_with_context(other,context),
            })
    }

    fn sequential_logits<'a>(
        &mut self,
        capture: &B::Tensor,
        last_token: u32,
        depth: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(M::Logits, B::Tensor), M::Error> {
        let tensor_context = N::target_context(context);
        let equation = PredictionEquation::Sequential { hidden: capture, token: last_token, depth };
        observation::neural_with_error(
            observer,
            SpeculativeActivationPhase::Proposal { depth },
            1,
            |cause|N::session_cause_with_context(cause,context),
            |error|N::neural_observer_error(error,context),
            |observer| {
                self.prediction_phase(
                    cache.prediction_phase_state(),
                    eredu_runtime::ExpertPass::Decode,
                    context,
                    &equation,
                    observer,
                    |extension, invoker, lane, observer, phase_context| {
                        let observed = observer.is_some();
                        let output = execute_prediction_equation::<A, B, S, N, _, _>(
                            equation, extension, invoker, lane, observer,
                            |token| N::token(token, phase_context),
                        )?;
                        let PredictionEquationOutput::Sequential { logits, hidden } = output else { unreachable!("sequential equation result") };
                        let row = M::logits_row(&logits, 0, phase_context)?;
                        Ok((row, hidden, observed.then_some(logits)))
                    },
                    |extension, lane, (_, hidden, logits), phase_context| {
                        let logits = logits.as_ref().expect("observed output retains its logits");
                        N::complete_prediction_state(extension, lane, &[logits, hidden], PredictionCompletionPoint::ObservedEquation, PredictionCompletionSources::default(), phase_context)
                    },
                )
                .map(|(row, hidden, _)| (row, hidden))
            },
        )
    }

    fn fused_logits<'a>(
        &mut self,
        capture: &B::Tensor,
        last_token: u32,
        capacity: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<Option<EmbeddedPredictionLogitBlock<B::Tensor>>, M::Error> {
        let lane = cache
            .lane_identity_ref(self.selected)
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(capture))
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        if self.selected.requirements().strategy().class()
            == eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential
        {
            // Sequential proposals still validate the retained lane and capture
            // above; no fused invocation exists to instrument for this strategy.
            return Ok(None);
        }
        let tensor_context = N::target_context(context);
        let equation = PredictionEquation::Fused { anchor: last_token, capacity };
        if observer.is_some() {
            // Retain all temporary proposal cache members through the same
            // completion/commit protocol as sequential prediction phases. The
            // persistent accepted-context lane is never modified by proposals.
            let mut proposal = cache.try_copy().map_err(N::session_failure)?;
            let output = observation::neural_with_error(
                observer,
                SpeculativeActivationPhase::FusedProposal,
                capacity,
                |cause|N::session_cause_with_context(cause,context),
            |error|N::neural_observer_error(error,context),
                |observer| {
                    self.prediction_phase(
                        proposal.prediction_phase_state(),
                        eredu_runtime::ExpertPass::Decode,
                        context,
                        &equation,
                        observer,
                        |extension, invoker, lane, observer, phase_context| {
                            let output = execute_prediction_equation::<A, B, S, N, _, _>(
                                equation, extension, invoker, lane, observer,
                                |token| N::token(token, phase_context),
                            )?;
                            let PredictionEquationOutput::Fused(output) = output else { unreachable!("fused equation result") };
                            Ok(output)
                        },
                        |extension, lane, logits, phase_context| match logits {
                            Some(logits) => {
                                N::complete_prediction_state(extension, lane, &[logits], PredictionCompletionPoint::ObservedEquation, PredictionCompletionSources::default(), phase_context)
                            }
                            None => N::complete_prediction_state(extension, lane, &[], PredictionCompletionPoint::ObservedEquation, PredictionCompletionSources::default(), phase_context),
                        },
                    )
                },
            )?;
            return output.map(|value| M::retain_logit_block(
                value, proposal.logit_evidence().cloned(), context,
            )).transpose();
        }
        let output = self.prediction_phase(
            cache.prediction_phase_state(),
            eredu_runtime::ExpertPass::Decode,
            context,
            &equation,
            None,
            |extension, invoker, lane, observer, phase_context| {
                let output = execute_prediction_equation::<A, B, S, N, _, _>(
                    equation, extension, invoker, lane, observer,
                    |token| N::token(token, phase_context),
                )?;
                let PredictionEquationOutput::Fused(output) = output else { unreachable!("fused equation result") };
                Ok(output)
            },
            |_, _, _, _| unreachable!("unobserved phase has no observation completion"),
        )?;
        output.map(|value| M::retain_logit_block(
            value, cache.logit_evidence().cloned(), context,
        )).transpose()
    }

    fn advance_prediction_cache<'a>(
        &mut self,
        captures: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<(), M::Error> {
        let lane = cache
            .lane_identity_ref(self.selected)
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(captures))
            .map_err(|cause|N::session_cause_with_context(cause,context))?;
        let tensor_context = N::target_context(context);
        let equation = PredictionEquation::Replay { captures, tokens };
        observation::neural_with_error(
            observer,
            SpeculativeActivationPhase::PredictionReplay,
            M::sequence_len(tokens)?,
            |cause|N::session_cause_with_context(cause,context),
            |error|N::neural_observer_error(error,context),
            |observer| {
                self.prediction_phase(
                    cache.prediction_phase_state(),
                    eredu_runtime::ExpertPass::Decode,
                    context,
                    &equation,
                    observer,
                    |extension, invoker, lane, observer, phase_context| {
                        execute_prediction_equation::<A, B, S, N, _, _>(
                            equation, extension, invoker, lane, observer,
                            |token| N::token(token, phase_context),
                        ).map(|output| {
                            let PredictionEquationOutput::StateOnly = output else { unreachable!("replay equation result") };
                        })
                    },
                    |extension, lane, _, phase_context| N::complete_prediction_state(
                        extension, lane, &[], PredictionCompletionPoint::ObservedEquation,
                        PredictionCompletionSources::default(), phase_context,
                    ),
                )
            },
        )
    }
}

impl<A, B, S, SM, D, P, I, N, M, H> EmbeddedExecutorCacheFactory<N::ExecutorTypes>
    for EmbeddedPredictionExecutor<
        '_,
        ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M, H>,
        M,
    >
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    SM: eredu_runtime::ReplicatedTextSessionMechanisms<A, B, State = S>,
    S: eredu_runtime::RuntimeState<B> + 'static,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        B,
        S,
        SM::ResidentPolicy,
        SM::BoundedPolicy,
    >,
    SM::PolicyError: std::error::Error + Send + Sync + 'static,
    SM::Error: std::error::Error + Send + Sync + 'static,
    P: crate::prediction_extension::MaterializedPredictionExecutor<A, B, N>,
    N: crate::prediction_extension::PredictionExtensionMaterializer<B>
        + ReplicatedPredictionNative<A, B, S, M>,
    I: ReplicatedPredictionInput<A, B, S, M::Error, Input = N::Input>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
    H: ReplicatedPredictionPhase<A, B, S, SM, D, P, N, M>,
{
    fn new_cache(&mut self) -> Result<Self::Cache, Self::Error> {
        self.strategy_mut().new_cache()
    }
    fn new_cache_with_context<'a>(&mut self,context:Self::Context<'a>)->Result<Self::Cache,eredu_core::BackendFailure>
    where Self:'a {
        self.strategy_mut().new_cache_with_context(context)
    }
    fn cache_metadata<'a>(bytes:Option<usize>,context:Self::Context<'a>)->Result<eredu_core::HostPreparationAuthority,eredu_core::BackendFailure>
    where Self:'a {
        N::prepared_cache_metadata(bytes,context)
    }

    fn bind_context<'a>(
        context: <N::ExecutorTypes as EmbeddedExecutorTypes>::Context<'a>,
    ) -> Self::Context<'a> {
        N::executor_context(context)
    }
}

/// Seed state matching the authoritative ordinary-target cache.
pub struct EmbeddedPredictionTargetState<T, C> {
    capture: EmbeddedPredictionTensor<T>,
    prediction_cache: C,
}

/// Private proposal state forked from one exact target seed.
#[derive(Clone)]
pub struct EmbeddedPredictionDraftState<T, C> {
    capture: EmbeddedPredictionTensor<T>,
    prediction_cache: DraftStateTransaction<C>,
    depth: usize,
    fused_logits: Option<EmbeddedPredictionLogitBlock<T>>,
    fused_cursor: usize,
    proposal_capacity: usize,
}

/// Retained verification output and its exact input tokens.
pub struct EmbeddedPredictionVerification<T> {
    output: EmbeddedPredictionOutput<T>,
    inputs: EmbeddedPredictionTensor<T>,
    logits_source: Option<PreparedEmbeddedEvidence>,
}

/// Fixed backend-facing types shared by every erased embedded-prediction executor.
///
/// Architecture construction remains statically typed through the target and its paired
/// extension.  This bundle fixes only the values which the backend scheduler must handle after
/// that pairing has completed; target state, prediction state, checkpoints, and verification
/// payloads are erased by [`DynEmbeddedExecutor`] itself.
pub trait EmbeddedExecutorTypes: 'static {
    /// Backend-owned model input.
    type Input;
    /// Opaque logits consumed by backend sampling.
    type Logits;
    /// Selected execution assignment.
    type Context<'a>: Copy
    where
        Self: 'a;
    /// Exact native completion.
    type Completion: BoundedCompletion<Error = Self::Error>;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Structured backend failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Projects a stable request ID through the existing batch assignment.
    /// No allocation or new admission is permitted during this projection.
    fn request_context<'a>(
        _request: eredu_core::generation::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<Self::Context<'a>, Self::Error>
    where Self: 'a,
    {
        Ok(context)
    }
    /// Actual host construction for the shared driver after typed pairing.
    /// Original backends debit the same request source before every birth.
    fn driver_buffer<V>(capacity: usize, _context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeBuffer<V>, Self::Error> {
        Ok(eredu_core::SpeculativeBuffer::with_capacity(capacity))
    }
    /// Exact query for the same host buffer producer.
    fn driver_buffer_bytes<V>(capacity: usize) -> Option<usize> {
        eredu_core::SpeculativeBuffer::<V>::retained_control_bytes(capacity)
    }
    /// Non-vector payload and erasure controls; an unknown original bound refuses.
    fn driver_host_metadata(_bytes: Option<usize>, _context: Self::Context<'_>)
        -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        Ok(eredu_core::HostPreparationAuthority::unmanaged())
    }
    /// Same source-owned logical identity for initial and restored driver state.
    fn driver_identity(_context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeRequestIdentity, Self::Error> {
        Ok(eredu_core::SpeculativeRequestIdentity::new())
    }
    /// Copy the actual fixed canonical sequence through its existing provider.
    fn copy_sequence(source: eredu_core::SpeculativeSequenceRef<'_>, _context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Self::Error>> {
        source.copy_ordinary().map_err(eredu_core::SpeculativeDriverError::Preparation)
    }
    /// Logical snapshot cost for the same copied sequence provider.
    fn sequence_copy_bytes(source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        match source {
            eredu_core::SpeculativeSequence::Ordinary(_) => source.snapshot_storage_bytes(),
            eredu_core::SpeculativeSequence::Retained(_) => None,
        }
    }
    /// Keep owned scheduler facts in the same transport. Ordinary facts continue
    /// through the paired executor's existing coordination callback.
    fn coordinate_retained_buffer(
        _local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>,
        _context: Self::Context<'_>,
    ) -> Result<eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Err(eredu_core::HostMetadataFundingError::Unavailable.into())
    }
    /// Fund one authenticated restored future before mutable copies begin.
    /// The original occurrence source remains outside all snapshot branches.
    fn prepare_control_continuation(
        _committed: usize, _status: eredu_core::generation::SpeculativeRequestStatus,
        _context: Self::Context<'_>,
    ) -> Result<(), SpeculativeControlError> { Ok(()) }

    /// Moves an already retained source error into the public neutral boundary.
    /// Ordinary errors remain in their native domain for the existing adapter.
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error> {Err(error)}

    /// Constructs the stable failure for an internally inconsistent erased value.
    fn erased_type_mismatch(value: &'static str) -> Self::Error;
}

/// Architecture-owned erased target cache for a completed embedded executor.
pub struct DynEmbeddedCache(Box<dyn std::any::Any>, eredu_core::HostPreparationAuthority);
/// Architecture-owned erased proposal seed state.
pub struct DynEmbeddedTargetState(Box<dyn std::any::Any>, eredu_core::HostPreparationAuthority);
/// Architecture-owned erased, discardable prediction branch.
pub struct DynEmbeddedDraftState(Box<dyn std::any::Any>, eredu_core::HostPreparationAuthority);
/// Architecture-owned erased target-cache checkpoint.
pub struct DynEmbeddedCheckpoint(Box<dyn std::any::Any>, eredu_core::HostPreparationAuthority);
/// Architecture-owned erased verification payload.
pub struct DynEmbeddedVerification(Box<dyn std::any::Any>, eredu_core::HostPreparationAuthority);

/// An executor which can realize a fresh lane cache before scheduling.
pub trait EmbeddedExecutorCacheFactory<T: EmbeddedExecutorTypes>: SpeculativeExecutor {
    /// Realizes one independent authoritative target/prediction lane.
    fn new_cache(&mut self) -> Result<Self::Cache, Self::Error>;

    /// Context-bearing prepared constructor, retaining its typed source failure.
    fn new_cache_with_context<'a>(&mut self,_context:Self::Context<'a>)->Result<Self::Cache,eredu_core::BackendFailure>
    where Self:'a {
        self.new_cache().map_err(eredu_core::BackendFailure::from_error)
    }
    /// Concrete erasure shell and fixed controls paid before Box construction.
    fn cache_metadata<'a>(_bytes:Option<usize>,_context:Self::Context<'a>)->Result<eredu_core::HostPreparationAuthority,eredu_core::BackendFailure>
    where Self:'a {Ok(eredu_core::HostPreparationAuthority::unmanaged())}

    /// Binds the backend's fixed scheduler context to this exact typed executor.
    fn bind_context<'a>(context: T::Context<'a>) -> Self::Context<'a>;
}

/// Object-safe ABI for one already paired architecture-owned embedded executor.
///
/// The trait is deliberately not a backend integration surface.  Its blanket implementation
/// below erases a completed [`SpeculativeExecutor`] only after architecture composition has
/// paired the exact target, extension, state, and capture contract.
pub trait ErasedEmbeddedExecutor<T: EmbeddedExecutorTypes> {
    /// Preserves preparation agreement through executable erasure.
    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>;
    /// Preserves preparation agreement through executable erasure.
    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>;
    /// Preserves mandatory loaded validation through executable erasure.
    fn validate_activation_readmission(&self, plan: &AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        plan.validate(discovery.ok_or(SpeculativeControlError::Unsupported(
            "loaded execution has no internal activation discovery",
        ))?).map_err(Into::into)
    }
    /// Installs validated prospective internal edits through the typed collector.
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError>;
    /// Exact complete cache/seed cost from the typed executor.
    fn control_snapshot_estimate(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate>;
    /// Copies complete typed state without erasing a native failure's source.
    fn control_snapshot<'a>(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<(DynEmbeddedCheckpoint, DynEmbeddedTargetState)>, SpeculativeControlError>;
    /// Restores all typed state atomically after copying succeeds.
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut DynEmbeddedCache,
        saved: &DynEmbeddedCheckpoint,
        state: &DynEmbeddedTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<DynEmbeddedTargetState>, SpeculativeControlError>;
    /// Installs one request's admitted collector for the borrowed executor scope.
    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError>;
    /// Whether the installed internal observer requires scheduler provenance.
    fn requires_activation_origin(&self) -> bool;
    /// Forwards the shared scheduler's bounded operation scope.
    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>);
    /// Moves one already charged internal record through erasure.
    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture>;
    /// Drains the original portable capture failure.
    fn take_activation_error(&mut self)
        -> Option<eredu_core::speculative::SpeculativeControlError>;
    /// Maximum proposals supported by the paired realization.
    fn max_proposals(&self) -> usize;
    /// Enables optional component telemetry.
    fn set_telemetry_enabled(&mut self, enabled: bool);
    /// Whether component telemetry is available.
    fn supports_telemetry(&self) -> bool;
    /// Drains component telemetry.
    fn take_telemetry(&mut self) -> Result<T::Telemetry, T::Error>;
    /// Drains telemetry retained in one verification payload.
    fn take_verification_telemetry(
        &mut self,
        output: &mut DynEmbeddedVerification,
    ) -> Result<T::Telemetry, T::Error>;
    /// Whether an exact cloned branch may be promoted.
    fn supports_exact_optimistic_promotion(&self) -> bool;
    /// Realizes one lane cache.
    fn new_cache(&mut self) -> Result<DynEmbeddedCache, T::Error>;
    /// Uses the exact context for prepared payload and erasure destinations.
    fn new_cache_with_context<'a>(&mut self,context:T::Context<'a>)->Result<DynEmbeddedCache,eredu_core::BackendFailure>
    where Self:'a;
    /// Prefills one lane.
    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynEmbeddedTargetState, T::Logits>, T::Error>;
    /// Forwards cancellation without bypassing the concrete executor contract.
    fn prefill_cancellable<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: T::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            SpeculativePrefill<DynEmbeddedTargetState, T::Logits>,
        >,
        T::Error,
    >;
    /// Fallibly copies the actual pending branch for optimistic advancement.
    fn copy_draft_state<'a>(
        &self,
        state: &DynEmbeddedDraftState,
        context: T::Context<'a>,
    ) -> Result<DynEmbeddedDraftState, T::Error>
    where Self: 'a;
    /// Forks a proposal branch.
    fn begin_proposal<'a>(
        &mut self,
        state: &DynEmbeddedTargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: T::Context<'a>,
    ) -> Result<DynEmbeddedDraftState, T::Error>;
    /// Advances a proposal branch and produces logits.
    fn proposal_logits<'a>(
        &mut self,
        state: &mut DynEmbeddedDraftState,
        last_token: u32,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error>;
    /// Captures an exact target-cache checkpoint.
    fn checkpoint(&self, cache: &DynEmbeddedCache) -> Result<DynEmbeddedCheckpoint, T::Error>;
    /// The same checkpoint with paid erasure and request identity.
    fn checkpoint_with_context<'a>(&self,cache:&DynEmbeddedCache,context:T::Context<'a>)
        ->Result<DynEmbeddedCheckpoint,T::Error>;
    /// Restores an exact checkpoint.
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut DynEmbeddedCache,
        checkpoint: &DynEmbeddedCheckpoint,
        context: T::Context<'a>,
    ) -> Result<(), T::Error>;
    /// Submits target verification.
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<Submission<DynEmbeddedVerification, T::Completion>, T::Error>;
    /// Selects one retained verification-logits row.
    fn verification_logits<'a>(
        &self,
        output: &DynEmbeddedVerification,
        index: usize,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error>;
    /// Commits an exact verified prefix.
    #[allow(clippy::too_many_arguments)]
    fn commit_verification<'a>(
        &mut self,
        output: DynEmbeddedVerification,
        draft_state: DynEmbeddedDraftState,
        cache: &mut DynEmbeddedCache,
        checkpoint: &DynEmbeddedCheckpoint,
        verified_inputs: usize,
        context: T::Context<'a>,
    ) -> Result<SpeculativeCommit<DynEmbeddedTargetState>, T::Error>;
}

fn erased_host<V, T: EmbeddedExecutorTypes>(context:T::Context<'_>)
    ->Result<eredu_core::HostPreparationAuthority,T::Error> {
    use std::mem::{size_of,size_of_val};
    let parts=[size_of::<V>(),size_of::<Box<V>>(),size_of::<Box<dyn std::any::Any>>(),
        size_of::<eredu_core::HostPreparationAuthority>(),
        size_of::<(Box<dyn std::any::Any>,eredu_core::HostPreparationAuthority)>(),
        size_of::<Result<V,T::Error>>(),
        size_of::<Result<eredu_core::HostPreparationAuthority,T::Error>>(),
        size_of::<T::Context<'_>>(),size_of::<Option<usize>>()];
    T::driver_host_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add),context)
}

fn erased_ref<'a, T: 'static, E: EmbeddedExecutorTypes>(
    value: &'a Box<dyn std::any::Any>,
    name: &'static str,
) -> Result<&'a T, E::Error> {
    value
        .downcast_ref::<T>()
        .ok_or_else(|| E::erased_type_mismatch(name))
}

fn erased_mut<'a, T: 'static, E: EmbeddedExecutorTypes>(
    value: &'a mut Box<dyn std::any::Any>,
    name: &'static str,
) -> Result<&'a mut T, E::Error> {
    value
        .downcast_mut::<T>()
        .ok_or_else(|| E::erased_type_mismatch(name))
}

impl<T, E> ErasedEmbeddedExecutor<T> for E
where
    T: EmbeddedExecutorTypes,
    E: EmbeddedExecutorCacheFactory<
        T,
        Input = T::Input,
        Logits = T::Logits,
        Completion = T::Completion,
        Telemetry = T::Telemetry,
        Error = T::Error,
    >,
    E::Cache: 'static,
    E::TargetState: 'static,
    E::DraftState: 'static,
    E::CacheCheckpoint: 'static,
    E::Verification: 'static,
{
    fn validate_activation_readmission(&self, plan: &AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        SpeculativeExecutor::validate_activation_readmission(self, plan, discovery)
    }
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        SpeculativeExecutor::readmit_activation_interventions(self, plan)
    }
    fn control_snapshot_estimate(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            SpeculativeExecutor::control_snapshot_estimate(
                self,
                erased_ref::<E::Cache, T>(&cache.0, "snapshot cache").ok()?,
                erased_ref::<E::TargetState, T>(&state.0, "snapshot seed").ok()?,
            ),
            snapshot::host::<DynEmbeddedCheckpoint>(None),
            snapshot::host::<DynEmbeddedTargetState>(None),
        ])
    }
    fn control_snapshot<'a>(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<(DynEmbeddedCheckpoint, DynEmbeddedTargetState)>, SpeculativeControlError>
    {
        let cache = erased_ref::<E::Cache, T>(&cache.0, "snapshot cache")
            .map_err(SpeculativeControlError::backend)?;
        let state = erased_ref::<E::TargetState, T>(&state.0, "snapshot seed")
            .map_err(SpeculativeControlError::backend)?;
        let checkpoint_host=erased_host::<E::CacheCheckpoint,T>(context)
            .map_err(|cause|SpeculativeControlError::backend_with_retained(cause,T::take_retained_failure))?;
        let seed_host=erased_host::<E::TargetState,T>(context)
            .map_err(|cause|SpeculativeControlError::backend_with_retained(cause,T::take_retained_failure))?;
        SpeculativeExecutor::control_snapshot(
            self,
            cache,
            state,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|pair| {
            pair.map(|(cache, state)| {
                (
                    DynEmbeddedCheckpoint(Box::new(cache),checkpoint_host),
                    DynEmbeddedTargetState(Box::new(state),seed_host),
                )
            })
        })
    }
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut DynEmbeddedCache,
        saved: &DynEmbeddedCheckpoint,
        state: &DynEmbeddedTargetState,
        context: T::Context<'a>,
    ) -> Result<Option<DynEmbeddedTargetState>, SpeculativeControlError> {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "snapshot cache")
            .map_err(SpeculativeControlError::backend)?;
        let saved = erased_ref::<E::CacheCheckpoint, T>(&saved.0, "snapshot checkpoint")
            .map_err(SpeculativeControlError::backend)?;
        let state = erased_ref::<E::TargetState, T>(&state.0, "snapshot seed")
            .map_err(SpeculativeControlError::backend)?;
        let host=erased_host::<E::TargetState,T>(context)
            .map_err(|cause|SpeculativeControlError::backend_with_retained(cause,T::take_retained_failure))?;
        SpeculativeExecutor::restore_control_snapshot(
            self,
            cache,
            saved,
            state,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|state| state.map(|state| DynEmbeddedTargetState(Box::new(state),host)))
    }
    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        SpeculativeExecutor::coordinate_speculative_step(
            self,
            local,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        SpeculativeExecutor::agree_text_preparation(
            self,
            stage,
            status,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::configure_activation_capture(self, plan, request, context)
    }

    fn requires_activation_origin(&self) -> bool {
        SpeculativeExecutor::requires_activation_origin(self)
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        SpeculativeExecutor::set_activation_origin(self, origin);
    }

    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        SpeculativeExecutor::take_activation_capture(self)
    }

    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        SpeculativeExecutor::take_activation_error(self)
    }

    fn max_proposals(&self) -> usize {
        SpeculativeExecutor::max_proposals(self)
    }

    fn set_telemetry_enabled(&mut self, enabled: bool) {
        SpeculativeExecutor::set_telemetry_enabled(self, enabled);
    }

    fn supports_telemetry(&self) -> bool {
        SpeculativeExecutor::supports_telemetry(self)
    }

    fn take_telemetry(&mut self) -> Result<T::Telemetry, T::Error> {
        SpeculativeExecutor::take_telemetry(self)
    }

    fn take_verification_telemetry(
        &mut self,
        output: &mut DynEmbeddedVerification,
    ) -> Result<T::Telemetry, T::Error> {
        let output = erased_mut::<E::Verification, T>(&mut output.0, "verification telemetry")?;
        SpeculativeExecutor::take_verification_telemetry(self, output)
    }

    fn supports_exact_optimistic_promotion(&self) -> bool {
        SpeculativeExecutor::supports_exact_optimistic_promotion(self)
    }

    fn new_cache(&mut self) -> Result<DynEmbeddedCache, T::Error> {
        EmbeddedExecutorCacheFactory::<T>::new_cache(self)
            .map(|cache| DynEmbeddedCache(Box::new(cache),eredu_core::HostPreparationAuthority::unmanaged()))
    }

    fn new_cache_with_context<'a>(&mut self,context:T::Context<'a>)->Result<DynEmbeddedCache,eredu_core::BackendFailure>
    where Self:'a {
        let context=<E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        let parts=[std::mem::size_of::<E::Cache>(),std::mem::size_of::<DynEmbeddedCache>(),
            std::mem::size_of::<Box<E::Cache>>(),std::mem::size_of::<Box<dyn std::any::Any>>(),
            std::mem::size_of::<Result<DynEmbeddedCache,eredu_core::BackendFailure>>()];
        let bytes=parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add);
        let host=<E as EmbeddedExecutorCacheFactory<T>>::cache_metadata(bytes,context)?;
        let cache=<E as EmbeddedExecutorCacheFactory<T>>::new_cache_with_context(self,context)?;
        Ok(DynEmbeddedCache(Box::new(cache),host))
    }

    fn copy_draft_state<'a>(
        &self,
        state: &DynEmbeddedDraftState,
        context: T::Context<'a>,
    ) -> Result<DynEmbeddedDraftState, T::Error>
    where Self: 'a,
    {
        let state = erased_ref::<E::DraftState, T>(&state.0, "draft state")?;
        let host=erased_host::<E::DraftState,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::copy_draft_state(self, state, context)
            .map(|state| DynEmbeddedDraftState(Box::new(state),host))
    }

    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynEmbeddedTargetState, T::Logits>, T::Error> {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let host=erased_host::<E::TargetState,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::prefill(self, input, cache, context).map(|prefill| {
            let (logits, state, target_tokens) = prefill.into_parts();
            SpeculativePrefill::new(
                logits,
                DynEmbeddedTargetState(Box::new(state),host),
                target_tokens,
            )
        })
    }

    fn prefill_cancellable<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: T::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            SpeculativePrefill<DynEmbeddedTargetState, T::Logits>,
        >,
        T::Error,
    > {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let host=erased_host::<E::TargetState,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::prefill_cancellable(self, input, cache, cancellation, context).map(
            |outcome| {
                outcome.map(|prefill| {
                    let (logits, state, target_tokens) = prefill.into_parts();
                    SpeculativePrefill::new(
                        logits,
                        DynEmbeddedTargetState(Box::new(state),host),
                        target_tokens,
                    )
                })
            },
        )
    }

    fn begin_proposal<'a>(
        &mut self,
        state: &DynEmbeddedTargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: T::Context<'a>,
    ) -> Result<DynEmbeddedDraftState, T::Error> {
        let state = erased_ref::<E::TargetState, T>(&state.0, "target state")?;
        let host=erased_host::<E::DraftState,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::begin_proposal(self, state, last_token, proposal_capacity, context)
            .map(|state| DynEmbeddedDraftState(Box::new(state),host))
    }

    fn proposal_logits<'a>(
        &mut self,
        state: &mut DynEmbeddedDraftState,
        last_token: u32,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error> {
        let state = state
            .0
            .downcast_mut::<E::DraftState>()
            .ok_or_else(|| T::erased_type_mismatch("draft state"))?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::proposal_logits(self, state, last_token, context)
    }

    fn checkpoint(&self, cache: &DynEmbeddedCache) -> Result<DynEmbeddedCheckpoint, T::Error> {
        if !cache.1.is_unmanaged() {
            return Err(T::erased_type_mismatch("prepared checkpoint requires its execution context"));
        }
        let cache = erased_ref::<E::Cache, T>(&cache.0, "target cache")?;
        SpeculativeExecutor::checkpoint(self, cache)
            .map(|checkpoint| DynEmbeddedCheckpoint(Box::new(checkpoint),eredu_core::HostPreparationAuthority::unmanaged()))
    }
    fn checkpoint_with_context<'a>(&self,cache:&DynEmbeddedCache,context:T::Context<'a>)
        ->Result<DynEmbeddedCheckpoint,T::Error> {
        let cache=erased_ref::<E::Cache,T>(&cache.0,"target cache")?;
        let host=erased_host::<E::CacheCheckpoint,T>(context)?;
        let context=<E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::checkpoint_with_context(self,cache,context)
            .map(|checkpoint|DynEmbeddedCheckpoint(Box::new(checkpoint),host))
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut DynEmbeddedCache,
        checkpoint: &DynEmbeddedCheckpoint,
        context: T::Context<'a>,
    ) -> Result<(), T::Error> {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let checkpoint = erased_ref::<E::CacheCheckpoint, T>(&checkpoint.0, "target checkpoint")?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::restore_checkpoint(self, cache, checkpoint, context)
    }

    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<Submission<DynEmbeddedVerification, T::Completion>, T::Error> {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let host=erased_host::<E::Verification,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::submit_verification(self, input_tokens, cache, context).map(
            |submission| Submission {
                output: DynEmbeddedVerification(Box::new(submission.output),host),
                completion: submission.completion,
            },
        )
    }

    fn verification_logits<'a>(
        &self,
        output: &DynEmbeddedVerification,
        index: usize,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error> {
        let output = erased_ref::<E::Verification, T>(&output.0, "verification")?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::verification_logits(self, output, index, context)
    }

    fn commit_verification<'a>(
        &mut self,
        output: DynEmbeddedVerification,
        draft_state: DynEmbeddedDraftState,
        cache: &mut DynEmbeddedCache,
        checkpoint: &DynEmbeddedCheckpoint,
        verified_inputs: usize,
        context: T::Context<'a>,
    ) -> Result<SpeculativeCommit<DynEmbeddedTargetState>, T::Error> {
        let output = output
            .0
            .downcast::<E::Verification>()
            .map_err(|_| T::erased_type_mismatch("verification"))?;
        let draft_state = draft_state
            .0
            .downcast::<E::DraftState>()
            .map_err(|_| T::erased_type_mismatch("draft state"))?;
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let checkpoint = erased_ref::<E::CacheCheckpoint, T>(&checkpoint.0, "target checkpoint")?;
        let host=erased_host::<E::TargetState,T>(context)?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::commit_verification(
            self,
            *output,
            *draft_state,
            cache,
            checkpoint,
            verified_inputs,
            context,
        )
        .map(|commit| {
            let (state, replayed) = commit.into_parts();
            SpeculativeCommit::new(DynEmbeddedTargetState(Box::new(state),host), replayed)
        })
    }
}

/// Concrete `SpeculativeExecutor` view over an architecture-erased paired executor.
pub struct DynEmbeddedExecutor<'a, T: EmbeddedExecutorTypes> {
    inner: &'a mut dyn ErasedEmbeddedExecutor<T>,
}

impl<'a, T: EmbeddedExecutorTypes> DynEmbeddedExecutor<'a, T> {
    /// Lends one already paired executor to backend scheduling.
    pub fn new(inner: &'a mut dyn ErasedEmbeddedExecutor<T>) -> Self {
        Self { inner }
    }

    /// Realizes one independent lane cache from the paired executor.
    pub fn new_cache(&mut self) -> Result<DynEmbeddedCache, T::Error> {
        self.inner.new_cache()
    }
    /// Constructs the lane through the same typed factory with actual funding.
    pub fn new_cache_with_context<'c>(&mut self,context:T::Context<'c>)->Result<DynEmbeddedCache,eredu_core::BackendFailure>
    where Self:'c {
        self.inner.new_cache_with_context(context)
    }
}

impl<T: EmbeddedExecutorTypes> SpeculativeExecutor for DynEmbeddedExecutor<'_, T> {
    type Input = T::Input;
    type Cache = DynEmbeddedCache;
    type TargetState = DynEmbeddedTargetState;
    type DraftState = DynEmbeddedDraftState;

    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where Self: 'a,
    {
        self.inner.copy_draft_state(state, context)
    }

    type CacheCheckpoint = DynEmbeddedCheckpoint;
    type Verification = DynEmbeddedVerification;
    type Logits = T::Logits;
    type Context<'a> = T::Context<'a>;
    type Completion = T::Completion;
    type Telemetry = T::Telemetry;
    type Error = T::Error;
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error> {
        T::take_retained_failure(error)
    }

    fn driver_buffer<V>(&self, capacity: usize, context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeBuffer<V>, Self::Error> {
        T::driver_buffer(capacity, context)
    }
    fn driver_buffer_bytes<V>(&self, capacity: usize) -> Option<usize> {
        T::driver_buffer_bytes::<V>(capacity)
    }
    fn driver_host_metadata(&self, bytes: Option<usize>, context: Self::Context<'_>)
        -> Result<eredu_core::HostPreparationAuthority, Self::Error> {
        T::driver_host_metadata(bytes, context)
    }
    fn request_context<'a>(&self, request: eredu_core::generation::SpeculativeRequestId,
        context: Self::Context<'a>) -> Result<Self::Context<'a>, Self::Error>
    where Self: 'a,
    {
        T::request_context(request, context)
    }
    fn driver_identity(&self, context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeRequestIdentity, Self::Error> {
        T::driver_identity(context)
    }
    fn copy_sequence(&self, source: eredu_core::SpeculativeSequenceRef<'_>, context: Self::Context<'_>)
        -> Result<eredu_core::SpeculativeSequence, eredu_core::SpeculativeDriverError<Self::Error>> {
        T::copy_sequence(source, context)
    }
    fn sequence_copy_bytes(&self, source: &eredu_core::SpeculativeSequence) -> Option<u64> {
        T::sequence_copy_bytes(source)
    }
    fn coordinate_speculative_buffer<'a>(&mut self,
        local: eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, context: Self::Context<'a>,
    ) -> Result<eredu_core::SpeculativeBuffer<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        match local.try_into_ordinary() {
            Ok(local) => self.inner.coordinate_speculative_step(local, context).map(Into::into),
            Err(local) => T::coordinate_retained_buffer(local, context),
        }
    }
    fn prepare_control_continuation<'a>(&mut self, committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus, context: Self::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        T::prepare_control_continuation(committed, status, context)
    }

    fn validate_activation_readmission(&self, plan: &AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        self.inner.validate_activation_readmission(plan, discovery)
    }
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.inner.readmit_activation_interventions(plan)
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.inner.control_snapshot_estimate(cache, state)
    }
    fn control_snapshot<'a>(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError> {
        self.inner.control_snapshot(cache, state, context)
    }
    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError> {
        self.inner
            .restore_control_snapshot(cache, saved, state, context)
    }

    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: T::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        self.inner.coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: T::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        self.inner.agree_text_preparation(stage, status, context)
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: T::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        self.inner
            .configure_activation_capture(plan, request, context)
    }

    fn requires_activation_origin(&self) -> bool {
        self.inner.requires_activation_origin()
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.inner.set_activation_origin(origin);
    }

    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.inner.take_activation_capture()
    }

    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.inner.take_activation_error()
    }

    fn max_proposals(&self) -> usize {
        self.inner.max_proposals()
    }
    fn set_telemetry_enabled(&mut self, enabled: bool) {
        self.inner.set_telemetry_enabled(enabled);
    }
    fn supports_telemetry(&self) -> bool {
        self.inner.supports_telemetry()
    }
    fn take_telemetry(&mut self) -> Result<Self::Telemetry, Self::Error> {
        self.inner.take_telemetry()
    }
    fn take_verification_telemetry(
        &mut self,
        output: &mut Self::Verification,
    ) -> Result<Self::Telemetry, Self::Error> {
        self.inner.take_verification_telemetry(output)
    }
    fn supports_exact_optimistic_promotion(&self) -> bool {
        self.inner.supports_exact_optimistic_promotion()
    }
    fn prefill<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
        self.inner.prefill(input, cache, context)
    }
    fn prefill_cancellable<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<SpeculativePrefill<Self::TargetState, Self::Logits>>,
        Self::Error,
    > {
        self.inner
            .prefill_cancellable(input, cache, cancellation, context)
    }
    fn begin_proposal<'a>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error> {
        self.inner
            .begin_proposal(state, last_token, proposal_capacity, context)
    }
    fn proposal_logits<'a>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        self.inner.proposal_logits(state, last_token, context)
    }
    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        self.inner.checkpoint(cache)
    }
    fn checkpoint_with_context<'c>(&self,cache:&Self::Cache,context:Self::Context<'c>)
        ->Result<Self::CacheCheckpoint,Self::Error>{
        self.inner.checkpoint_with_context(cache,context)
    }
    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        self.inner.restore_checkpoint(cache, checkpoint, context)
    }
    fn submit_verification<'a>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: Self::Context<'a>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
        self.inner.submit_verification(input_tokens, cache, context)
    }
    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        self.inner.verification_logits(output, index, context)
    }
    fn commit_verification<'a>(
        &mut self,
        output: Self::Verification,
        draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: Self::Context<'a>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        self.inner.commit_verification(
            output,
            draft_state,
            cache,
            checkpoint,
            verified_inputs,
            context,
        )
    }
}

/// Neutral embedded prediction executor consumed by the shared runtime scheduler.
pub struct EmbeddedPredictionExecutor<'a, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    strategy: &'a mut S,
    observers: EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    prior_internal: Option<Option<Box<dyn SpeculativeActivationObserver<M::Tensor, M::Error>>>>,
    execution_started: bool,
    _mechanisms: PhantomData<fn() -> M>,
}

/// Transactional target checkpoint, optionally carrying durable internal
/// authority. The architecture driver composes it with complete native state.
#[derive(Debug)]
pub struct EmbeddedPredictionCheckpoint<C> {
    cache: C,
    activations: Option<eredu_runtime::capture::SpeculativeActivationCheckpoint>,
}

impl<'a, S, M> EmbeddedPredictionExecutor<'a, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    /// Pairs one typed prepared strategy with its already selected mechanisms.
    pub fn new(strategy: &'a mut S) -> Self {
        Self {
            strategy,
            observers: EmbeddedPredictionObservers::default(),
            prior_internal: None,
            execution_started: false,
            _mechanisms: PhantomData,
        }
    }

    /// Pairs a typed strategy with production observers before execution begins.
    pub fn with_observers(
        strategy: &'a mut S,
        observers: EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    ) -> Self {
        Self {
            strategy,
            observers,
            prior_internal: None,
            execution_started: false,
            _mechanisms: PhantomData,
        }
    }

    /// Returns the production observer set after execution so a session can retain it.
    pub fn into_observers(mut self) -> EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error> {
        if let Some(prior) = self.prior_internal.take() {
            self.observers.internal = prior;
        }
        self.observers
    }

    /// Mutably borrows the statically paired strategy for cache realization.
    pub fn strategy_mut(&mut self) -> &mut S {
        self.strategy
    }

    // Enter only after failure. Keeping the restore result, error pair and
    // conversion call here prevents them enlarging every active cold quote.
    #[inline(never)]
    fn restore_target_failure<F>(
        operation: M::Error,
        cache: &mut S::TargetCache,
        checkpoint: &S::TargetCache,
        context: M::Context<'_>,
        retain: F,
    ) -> M::Error
    where
        F: FnOnce(eredu_core::speculative::SpeculativeRollbackFailure<M::Error, M::Error>) -> M::Error,
    {
        match S::restore_target_checkpoint(cache, checkpoint, context) {
            Ok(()) => operation,
            Err(rollback) => retain(eredu_core::speculative::SpeculativeRollbackFailure { operation, rollback }),
        }
    }

    fn state_at<'context>(
        output: &EmbeddedPredictionOutput<M::Tensor>,
        row: usize,
        prediction_cache: S::PredictionCache,
        source: Option<&PreparedEmbeddedEvidence>,
        context: M::Context<'context>,
    ) -> Result<EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>, M::Error>
    where
        M: 'context,
    {
        Ok(EmbeddedPredictionTargetState {
            capture: M::tensor_row_with_source(output.capture(), row, source, context)?,
            prediction_cache,
        })
    }

    fn validate_output(
        output: &EmbeddedPredictionOutput<M::Tensor>,
        expected: Option<usize>,
    ) -> Result<usize, M::Error> {
        let logits = M::sequence_len(output.logits())?;
        let capture = M::sequence_len(output.capture())?;
        let tokens = M::sequence_len(output.tokens())?;
        if logits != capture || logits != tokens || expected.is_some_and(|value| value != logits) {
            return Err(M::invalid_prediction_output(
                logits, capture, tokens, expected,
            ));
        }
        Ok(logits)
    }
}

impl<S, M> SpeculativeExecutor for EmbeddedPredictionExecutor<'_, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    type Input = S::Input;
    type Cache = S::TargetCache;
    type TargetState = EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>;
    type DraftState = EmbeddedPredictionDraftState<M::Tensor, S::PredictionCache>;

    fn copy_draft_state<'a>(
        &self,
        state: &Self::DraftState,
        _context: Self::Context<'a>,
    ) -> Result<Self::DraftState, Self::Error>
    where Self: 'a,
    {
        let prediction_cache = state.prediction_cache
            .try_copy(|cache| self.strategy.copy_prediction_cache(cache))?;
        Ok(EmbeddedPredictionDraftState {
            capture: state.capture.clone(),
            prediction_cache,
            depth: state.depth,
            fused_logits: state.fused_logits.clone(),
            fused_cursor: state.fused_cursor,
            proposal_capacity: state.proposal_capacity,
        })
    }

    type CacheCheckpoint = EmbeddedPredictionCheckpoint<S::TargetCache>;
    type Verification = EmbeddedPredictionVerification<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = S::Telemetry;
    type Error = M::Error;
    fn take_retained_failure(error:Self::Error)->Result<eredu_core::BackendFailure,Self::Error> {
        M::take_retained_failure(error)
    }

    fn control_snapshot_estimate(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            snapshot::host::<Self::TargetState>(None),
            self.strategy.control_target_estimate(cache),
            self.strategy
                .control_prediction_estimate(&state.prediction_cache),
            M::control_tensor_packet_estimate(&state.capture),
            self.observers
                .internal
                .as_ref()
                .map_or(Some(0), |observer| observer.activation_checkpoint_bytes())
                .and_then(|bytes| {
                    bytes.checked_add(std::mem::size_of::<Self::CacheCheckpoint>() as u64)
                })
                .map(|bytes| eredu_core::execution_control::SnapshotEstimate {
                    retained_bytes: bytes,
                    copy_bytes: bytes,
                }),
        ])
    }

    fn control_snapshot<'a>(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError> {
        if self.control_snapshot_estimate(cache, state).is_none() {
            return Ok(None);
        }
        let activations = self
            .observers
            .internal
            .as_ref()
            .map(|observer| observer.activation_checkpoint())
            .transpose()?;
        let Some((cache, state)) = snapshot::copy::<S, M>(self.strategy, cache, state, context)?
        else {
            return Ok(None);
        };
        Ok(Some((
            EmbeddedPredictionCheckpoint { cache, activations },
            state,
        )))
    }

    fn restore_control_snapshot<'a>(
        &mut self,
        cache: &mut Self::Cache,
        saved: &Self::CacheCheckpoint,
        state: &Self::TargetState,
        context: Self::Context<'a>,
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError> {
        let prepared = match (self.observers.internal.as_mut(), saved.activations.as_ref()) {
            (Some(observer), Some(saved)) => Some(observer.prepare_activation_restore(saved)?),
            (None, None) => None,
            _ => {
                return Err(SpeculativeControlError::Invalid(
                    "internal snapshot authority changed",
                ))
            }
        };
        let Some((replacement, seed)) =
            snapshot::copy::<S, M>(self.strategy, &saved.cache, state, context)?
        else {
            return Ok(None);
        };
        *cache = replacement;
        if let Some(prepared) = prepared {
            prepared.commit();
        }
        Ok(Some(seed))
    }

    fn validate_activation_readmission(&self, plan: &AdmittedSpeculativeActivations,
        discovery: Option<&eredu_core::speculative::SpeculativeActivationDiscovery>,
    ) -> Result<(), SpeculativeControlError> {
        match self.observers.internal.as_ref() {
            Some(observer) => observer.validate_activation_readmission(plan, discovery),
            None => plan.validate(discovery.ok_or(SpeculativeControlError::Unsupported(
            "loaded execution has no internal activation discovery",
        ))?).map_err(Into::into),
        }
    }

    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.observers
            .internal()
            .ok_or(SpeculativeControlError::Unsupported(
                "internal re-admission requires capture authority from run creation",
            ))?
            .readmit_activation_interventions(plan)
    }

    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: M::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        M::coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: M::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        M::agree_text_preparation(stage, status, context)
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: M::Context<'a>,
    ) -> Result<(), SpeculativeControlError> {
        if self.execution_started || self.prior_internal.is_some() {
            return Err(SpeculativeControlError::Unsupported(
                "activation authority must be installed once, before speculative execution",
            ));
        }
        if !plan.is_empty() && !self.strategy.supports_internal_observations() {
            return Err(SpeculativeControlError::Unsupported(
                "selected speculative strategy has no complete internal activation path",
            ));
        }
        let observer = M::activation_observer(&plan, request, context)?;
        self.prior_internal = Some(std::mem::replace(&mut self.observers.internal, observer));
        Ok(())
    }

    fn requires_activation_origin(&self) -> bool {
        self.strategy.requires_activation_origin() || self.observers.internal.is_some()
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
        self.strategy.set_activation_origin(origin);
        if let Some(observer) = self.observers.internal() {
            observer.set_activation_origin(origin);
        }
    }

    fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.observers.take_activation_capture()
    }

    fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.observers.take_activation_error()
    }

    fn max_proposals(&self) -> usize {
        self.strategy.proposal_capacity()
    }

    fn set_telemetry_enabled(&mut self, enabled: bool) {
        self.strategy.set_telemetry_enabled(enabled);
    }

    fn supports_telemetry(&self) -> bool {
        self.strategy.supports_telemetry()
    }

    fn take_telemetry(&mut self) -> Result<Self::Telemetry, Self::Error> {
        self.strategy.take_telemetry()
    }

    fn take_verification_telemetry(
        &mut self,
        output: &mut Self::Verification,
    ) -> Result<Self::Telemetry, Self::Error> {
        self.strategy
            .take_verification_telemetry(&mut output.output)
    }

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error> {
        match self.prefill_cancellable(
            input,
            cache,
            &eredu_core::GenerationCancellationToken::default(),
            context,
        )? {
            eredu_core::SpeculativePrefillOutcome::Complete(value) => Ok(value),
            eredu_core::SpeculativePrefillOutcome::Cancelled { .. } => Err(M::observation_error(
                "uncancellable prefill observed peer cancellation",
            )),
        }
    }

    fn prefill_cancellable<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: Self::Context<'context>,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<SpeculativePrefill<Self::TargetState, Self::Logits>>,
        Self::Error,
    > {
        self.execution_started = true;
        if self.observers.internal.is_some() && !self.strategy.supports_internal_observations() {
            return Err(M::observation_error(
                "selected speculative strategy has no complete internal activation path",
            ));
        }
        let rollback_failure = S::prepare_rollback_failure(context)?;
        let checkpoint = S::checkpoint_target(cache)?;
        let result = self.strategy.prefill_cancellable(
            input,
            cache,
            &mut self.observers,
            cancellation,
            context,
        );
        match result {
            Ok(eredu_core::SpeculativePrefillOutcome::Complete(value)) => Ok(
                eredu_core::SpeculativePrefillOutcome::Complete(SpeculativePrefill::new(
                    value.logits,
                    EmbeddedPredictionTargetState {
                        capture: value.capture,
                        prediction_cache: self.strategy.prediction_cache(cache)?,
                    },
                    value.evaluated,
                )),
            ),
            Ok(eredu_core::SpeculativePrefillOutcome::Cancelled { evaluated_tokens }) => {
                // Shared registration owns agreed cancellation rollback.
                Ok(eredu_core::SpeculativePrefillOutcome::Cancelled { evaluated_tokens })
            }
            Err(operation) => Err(Self::restore_target_failure(operation, cache, &checkpoint, context, rollback_failure)),
        }
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: M::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        let mut prediction_cache = DraftStateTransaction::try_fork(
            &state.prediction_cache,
            |cache| self.strategy.copy_prediction_cache(cache),
        )?;
        let fused_logits = M::with_tensor_source(state.capture.evidence(),context,|context|self.strategy.fused_logits(
            &state.capture,last_token,proposal_capacity,prediction_cache.draft_mut(),context,self.observers.internal(),
        ))?;
        if let Some(logits) = &fused_logits {
            let available = M::sequence_len(logits)?;
            if available < proposal_capacity {
                return Err(M::invalid_fused_capacity(proposal_capacity, available));
            }
        }
        Ok(EmbeddedPredictionDraftState {
            capture: state.capture.clone(),
            prediction_cache,
            depth: 0,
            fused_logits,
            fused_cursor: 0,
            proposal_capacity,
        })
    }

    fn proposal_logits(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: M::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        if state.depth.max(state.fused_cursor) >= state.proposal_capacity {
            return Err(M::fused_prediction_exhausted());
        }
        if let Some(logits) = &state.fused_logits {
            let available = M::sequence_len(logits)?;
            if state.fused_cursor >= available {
                return Err(M::fused_prediction_exhausted());
            }
            let row = state.fused_cursor;
            state.fused_cursor += 1;
            let logits = M::fused_logits_row_with_source(logits, row, logits.evidence(), context)?;
            let logits = self
                .strategy
                .adjust_fused_logits(logits, last_token, context)?;
            return self.observers.logits(&logits);
        }
        let (logits, capture) = M::with_tensor_source(state.capture.evidence(),context,|context|self.strategy.sequential_logits(
            &state.capture,last_token,state.depth,state.prediction_cache.draft_mut(),context,self.observers.internal(),
        ))?;
        let capture = self.observers.tensor::<M>(EMBEDDED_PREDICTION_OUTPUT_PATH,capture,None,context)?;
        state.capture = M::retain_tensor_packet(capture,
            self.strategy.prediction_tensor_evidence(state.prediction_cache.draft()).cloned(),context)?;
        state.depth += 1;
        self.observers.logits(&logits)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        S::checkpoint_target(cache).map(|cache| EmbeddedPredictionCheckpoint {
            cache,
            activations: None,
        })
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        S::restore_target_checkpoint(cache, &checkpoint.cache, context)
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: M::Context<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
        let inputs = M::target_tokens_packet(input_tokens, context)?;
        let mut output = self.strategy.verify_target(
            &inputs,
            cache,
            context,
            self.observers.internal(),
            SpeculativeActivationPhase::Verification,
        )?;
        output.logits = self
            .observers
            .tensor::<M>(EMBEDDED_VERIFICATION_LOGITS_PATH, output.logits, None, context)?;
        output.capture = self
            .observers
            .tensor::<M>(EMBEDDED_TARGET_CAPTURE_PATH, output.capture, None, context)?;
        Self::validate_output(&output, Some(input_tokens.len()))?;
        let completion = M::submit_verification_completion_with_source(
            &output, &inputs, self.strategy.target_logit_evidence(cache), context,
        )?;
        Ok(Submission {
            output: EmbeddedPredictionVerification {
                output, inputs, logits_source: self.strategy.target_logit_evidence(cache).cloned(),
            },
            completion,
        })
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::logits_row_with_source(output.output.logits(), index, output.logits_source.as_ref(), context)
    }

    fn commit_verification(
        &mut self,
        output: Self::Verification,
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: M::Context<'_>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error> {
        let rollback_failure = S::prepare_rollback_failure(context)?;
        let result = (|| {
            let input_len = M::sequence_len(&output.inputs)?;
            Self::validate_output(&output.output, Some(input_len))?;
            if verified_inputs == 0 || verified_inputs > input_len {
                return Err(M::invalid_prediction_commit(verified_inputs, input_len));
            }
            if verified_inputs > 1 {
                let captures = M::tensor_prefix_with_source(output.output.capture(),verified_inputs-1,output.logits_source.as_ref(),context)?;
                let tokens = M::token_range_packet(&output.inputs,1,verified_inputs,context)?;
                M::with_tensor_source(captures.evidence(),context,|context|
                    M::with_tensor_source(tokens.evidence(),context,|context|self.strategy.advance_prediction_cache(
                        &captures,&tokens,draft_state.prediction_cache.draft_mut(),context,self.observers.internal(),
                    )))?;
            }
            let (committed, committed_source, replayed_tokens) = if verified_inputs == input_len {
                (output.output, output.logits_source, 0)
            } else {
                S::restore_target_checkpoint(cache, &checkpoint.cache, context)?;
                let retained = M::token_range_packet(&output.inputs, 0, verified_inputs, context)?;
                let replayed=self.strategy.verify_target(&retained,cache,context,
                    self.observers.internal(),SpeculativeActivationPhase::TargetReplay)?;
                let replayed_source=self.strategy.target_logit_evidence(cache).cloned();
                (replayed,replayed_source,verified_inputs)
            };
            self.strategy
                .commit_prediction_cache(cache, draft_state.prediction_cache.draft())?;
            let state = Self::state_at(
                &committed,
                verified_inputs - 1,
                self.strategy.prediction_cache(cache)?,
                committed_source.as_ref(),
                context,
            )?;
            Ok(SpeculativeCommit::new(state, replayed_tokens))
        })();
        match result {
            Ok(commit) => Ok(commit),
            Err(operation) => Err(Self::restore_target_failure(operation, cache, &checkpoint.cache, context, rollback_failure)),
        }
    }
}

#[cfg(test)]
mod tests;
