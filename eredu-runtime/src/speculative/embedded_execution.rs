//! Generic embedded-prediction execution over a portable strategy.

use std::marker::PhantomData;

use eredu_core::{
    speculative::{
        AdmittedSpeculativeActivations, SpeculativeActivationCapture, SpeculativeActivationOrigin,
        SpeculativeActivationPhase, SpeculativeControlError,
    },
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill, Submission,
};

use crate::{inspection::SpeculativeActivationObserver, DraftStateTransaction};

use super::{
    EmbeddedPredictionObservers, EmbeddedPredictionOutput, SpeculativeTensorMechanisms,
    EMBEDDED_PREDICTION_OUTPUT_PATH, EMBEDDED_TARGET_CAPTURE_PATH,
    EMBEDDED_VERIFICATION_LOGITS_PATH,
};

mod snapshot;

/// Portable cache envelope for one embedded-prediction lane.
///
/// `T` and `L` are opaque backend-native storage values. Their membership, the
/// prepared-input binding, and the target capture frontier are neutral
/// prediction semantics and therefore remain outside any concrete backend.
pub struct EmbeddedPredictionCache<T, L> {
    target: Option<T>,
    prediction: L,
    prepared_input: Option<crate::SpeculativeIdentity>,
    capture_generation: Option<u64>,
}

impl<T, L> EmbeddedPredictionCache<T, L> {
    /// Creates one lane from exact target and extension storage.
    pub const fn new(target: T, prediction: L) -> Self {
        Self {
            target: Some(target),
            prediction,
            prepared_input: None,
            capture_generation: None,
        }
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

    /// Accounts for the envelope and both native snapshot owners.
    pub fn snapshot_estimate(
        &self,
        target: Option<eredu_core::execution_control::SnapshotEstimate>,
        prediction: Option<eredu_core::execution_control::SnapshotEstimate>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            snapshot::host::<Self>(self.prepared_input.as_ref()),
            target,
            prediction,
        ])
    }

    /// Pairs completed native snapshot copies with the exact retained identity.
    pub fn with_snapshot_storage(&self, target: T, prediction: L) -> Self {
        Self {
            target: Some(target),
            prediction,
            prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation,
        }
    }

    /// Binds the lane to the exact prepared description and semantic content.
    pub fn bind_prepared_input(
        &mut self,
        identity: Option<&crate::PreparedInputCacheIdentity>,
    ) -> Result<(), EmbeddedPredictionCacheError> {
        let identity = identity.ok_or(EmbeddedPredictionCacheError::MissingPreparedInput)?;
        let identity = crate::SpeculativeIdentity::new(format!(
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
        selected: &crate::SelectedSpeculativeRealization,
        generation: impl FnOnce(&T) -> Result<u64, E>,
    ) -> Result<crate::SpeculativeLaneIdentity, EmbeddedPredictionCacheAccessError<E>> {
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
        })
    }

    /// Forks prediction-local state without transferring ordinary target storage.
    pub fn prediction_fork(&self) -> EmbeddedPredictionDraftCache<L> {
        EmbeddedPredictionDraftCache {
            prediction: self.prediction.clone(),
            prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation,
        }
    }

    /// Commits a successful prediction-local transaction.
    pub fn commit_prediction(&mut self, draft: &EmbeddedPredictionDraftCache<L>) {
        self.prediction.clone_from(&draft.prediction);
        self.prepared_input.clone_from(&draft.prepared_input);
        self.capture_generation = draft.capture_generation;
    }

    /// Restores all neutral membership around an opaque target-state restore.
    pub fn restore<E>(
        &mut self,
        checkpoint: &Self,
        restore_target: impl FnOnce(&mut T, &T) -> Result<(), E>,
    ) -> Result<(), EmbeddedPredictionCacheAccessError<E>> {
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
#[derive(Clone)]
pub struct EmbeddedPredictionDraftCache<L> {
    prediction: L,
    prepared_input: Option<crate::SpeculativeIdentity>,
    capture_generation: Option<u64>,
}

impl<L> EmbeddedPredictionDraftCache<L> {
    /// Borrows prediction-local storage.
    pub const fn prediction(&self) -> &L {
        &self.prediction
    }

    /// Mutably borrows prediction-local storage.
    pub fn prediction_mut(&mut self) -> &mut L {
        &mut self.prediction
    }

    /// Accounts for the envelope and the native prediction snapshot owner.
    pub fn snapshot_estimate(
        &self,
        prediction: Option<eredu_core::execution_control::SnapshotEstimate>,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        snapshot::sum([
            snapshot::host::<Self>(self.prepared_input.as_ref()),
            prediction,
        ])
    }

    /// Pairs completed native prediction storage with the exact retained identity.
    pub fn with_snapshot_storage(&self, prediction: L) -> Self {
        Self {
            prediction,
            prepared_input: self.prepared_input.clone(),
            capture_generation: self.capture_generation,
        }
    }

    /// Reconstructs the selected lane identity retained by this prediction-only fork.
    pub fn lane_identity(
        &self,
        selected: &crate::SelectedSpeculativeRealization,
    ) -> Result<crate::SpeculativeLaneIdentity, EmbeddedPredictionCacheError> {
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
    type PredictionCache: Clone;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }

    /// Reads the target budget shared with embedded prediction owners without
    /// polling, settling, or touching request state.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

    /// Reads installed target/prediction state and shared parameter ownership.
    fn continuation_memory_observation(
        &self,
        _cache: &Self::TargetCache,
        _additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        Ok(None)
    }

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

    /// Creates a fallible exact checkpoint of ordinary target and prediction state.
    fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, M::Error>;

    /// Streams target and prediction prefill, retaining only the final target chunk.
    /// The returned count covers the complete input, independently of that chunk.
    fn prefill_with_prediction<'a>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observers: &mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    ) -> Result<(EmbeddedPredictionOutput<M::Tensor>, usize), M::Error>;

    /// Runs ordinary-target verification and returns its exact selected capture.
    fn verify_target<'a>(
        &mut self,
        tokens: &M::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<M::Tensor, M::Error>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<M::Tensor>, M::Error>;

    /// Forks prediction-local state from the authoritative target lane.
    fn prediction_cache(&self, cache: &Self::TargetCache) -> Self::PredictionCache;

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
    ) -> Result<Option<M::Tensor>, M::Error> {
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

/// Configures the neutral executor's cache and scheduler context for a selected strategy.
pub trait EmbeddedPredictionCacheFactory<T, M>: EmbeddedPredictionStrategy<M>
where
    T: eredu_core::speculative::GenerationExecutorTypes,
    M: SpeculativeTensorMechanisms,
{
    /// Realizes one fresh lane cache for this strategy.
    fn new_cache<'a>(&mut self, context: M::Context<'a>) -> Result<Self::TargetCache, M::Error>
    where
        Self: 'a;

    /// Translates the erased scheduler context into the strategy's native context.
    fn bind_context<'a>(context: T::Context<'a>) -> M::Context<'a>
    where
        Self: 'a;
}

impl<T, S, M> eredu_core::speculative::GenerationExecutorCacheFactory<T>
    for EmbeddedPredictionExecutor<'_, S, M>
where
    T: eredu_core::speculative::GenerationExecutorTypes,
    S: EmbeddedPredictionCacheFactory<T, M>,
    M: SpeculativeTensorMechanisms,
{
    fn new_cache<'a>(
        &mut self,
        _: &T::Binding,
        context: Self::Context<'a>,
    ) -> Result<Self::Cache, Self::Error>
    where
        Self: 'a,
    {
        self.strategy.new_cache(context)
    }

    fn bind_context<'a>(context: T::Context<'a>) -> Self::Context<'a>
    where
        Self: 'a,
    {
        S::bind_context(context)
    }
}

/// Seed state matching the authoritative ordinary-target cache.
pub struct EmbeddedPredictionTargetState<T, C> {
    capture: T,
    prediction_cache: C,
}

/// Private proposal state forked from one exact target seed.
#[derive(Clone)]
pub struct EmbeddedPredictionDraftState<T, C: Clone> {
    capture: T,
    // The committed target frontier remains available after proposal captures
    // diverge. Sequential cache commitment replays only target-aligned inputs.
    anchor_capture: T,
    prediction_cache: DraftStateTransaction<C>,
    depth: usize,
    fused_logits: Option<T>,
    fused_cursor: usize,
    proposal_capacity: usize,
}

/// Neutral embedded prediction executor consumed by the shared runtime scheduler.
pub struct EmbeddedPredictionExecutor<'a, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    strategy: S,
    observers: &'a mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    prior_internal: Option<Option<Box<dyn SpeculativeActivationObserver<M::Tensor, M::Error>>>>,
    execution_started: bool,
    _mechanisms: PhantomData<fn() -> M>,
}

/// Transactional target checkpoint, optionally carrying durable internal
/// authority. The architecture driver composes it with complete native state.
#[derive(Debug)]
pub struct EmbeddedPredictionCheckpoint<C> {
    cache: C,
    activations: Option<crate::capture::SpeculativeActivationCheckpoint>,
}

impl<'a, S, M> EmbeddedPredictionExecutor<'a, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    /// Binds a typed strategy and its retained observers for the transaction.
    pub fn bind<T>(
        strategy: S,
        observers: &'a mut EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error>,
    ) -> Box<dyn eredu_core::speculative::erasure::ErasedGenerationExecutor<T, S::Input> + 'a>
    where
        S: 'a,
        T: eredu_core::speculative::GenerationExecutorTypes,
        Self: eredu_core::speculative::erasure::ErasedGenerationExecutor<T, S::Input>,
    {
        Box::new(Self {
            strategy,
            observers,
            prior_internal: None,
            execution_started: false,
            _mechanisms: PhantomData,
        })
    }

    fn state_at<'context>(
        output: &EmbeddedPredictionOutput<M::Tensor>,
        row: usize,
        prediction_cache: S::PredictionCache,
        context: M::Context<'context>,
    ) -> Result<EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>, M::Error>
    where
        M: 'context,
    {
        Ok(EmbeddedPredictionTargetState {
            capture: M::tensor_row(output.capture(), row, context)?,
            prediction_cache,
        })
    }

    fn validate_output(
        output: &EmbeddedPredictionOutput<M::Tensor>,
        expected: Option<usize>,
    ) -> Result<usize, M::Error> {
        validate_prediction_output::<M>(output, expected)
    }
}

impl<S, M> Drop for EmbeddedPredictionExecutor<'_, S, M>
where
    S: EmbeddedPredictionStrategy<M>,
    M: SpeculativeTensorMechanisms + 'static,
{
    fn drop(&mut self) {
        if let Some(prior) = self.prior_internal.take() {
            self.observers.replace_internal(prior);
        }
    }
}

/// Checks that a target result has matching logits, capture, and token extents.
pub fn validate_prediction_output<M: SpeculativeTensorMechanisms>(
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

impl<S, M> SpeculativeExecutor for EmbeddedPredictionExecutor<'_, S, M>
where
    M: SpeculativeTensorMechanisms + 'static,
    S: EmbeddedPredictionStrategy<M>,
{
    fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        eredu_core::decoding::DecodingCapabilities::causal_drafting(self.max_proposals())
            .with_completion_support(
                M::Completion::supports_cancellation(
                    eredu_core::CompletionCancellationMode::NativeCancel,
                ),
                M::Completion::supports_cancellation(
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                ),
            )
    }
    fn logits_snapshot_estimate(
        &self,
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::control_logits_estimate(logits)
    }
    type Input = S::Input;
    type Cache = S::TargetCache;
    type TargetState = EmbeddedPredictionTargetState<M::Tensor, S::PredictionCache>;
    type DraftState = EmbeddedPredictionDraftState<M::Tensor, S::PredictionCache>;
    type CacheCheckpoint = EmbeddedPredictionCheckpoint<S::TargetCache>;
    type Verification =
        crate::speculative::CausalVerification<EmbeddedPredictionOutput<M::Tensor>, M::Tensor>;
    type Logits = M::Logits;
    type Context<'a>
        = M::Context<'a>
    where
        Self: 'a;
    type Completion = M::Completion;
    type Telemetry = ();
    type Error = M::Error;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.strategy.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        self.strategy.parameter_conversion_retention()
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::Cache,
        state: &Self::TargetState,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        let Some(mut observation) = self
            .strategy
            .continuation_memory_observation(cache, additional)?
        else {
            return Ok(None);
        };
        if let Some(embedded) = &mut observation.embedded {
            // A proposal retains its immutable target anchor alongside the
            // mutable prediction capture. Charge both even for native shared views.
            embedded.retained_feature_bytes = M::control_tensor_estimate(&state.capture)
                .and_then(|e| e.retained_bytes.checked_mul(2));
        }
        observation.seed_bytes = self
            .strategy
            .control_prediction_estimate(&state.prediction_cache)
            .map(|e| e.retained_bytes);
        Ok(Some(observation))
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
            M::control_tensor_estimate(&state.capture),
            self.observers
                .internal_ref()
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
    ) -> Result<Option<(Self::CacheCheckpoint, Self::TargetState)>, SpeculativeControlError>
    where
        Self: 'a,
    {
        if self.control_snapshot_estimate(cache, state).is_none() {
            return Ok(None);
        }
        let activations = self
            .observers
            .internal_ref()
            .map(|observer| observer.activation_checkpoint())
            .transpose()?;
        let Some((cache, state)) = snapshot::copy::<S, M>(&self.strategy, cache, state, context)?
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
    ) -> Result<Option<Self::TargetState>, SpeculativeControlError>
    where
        Self: 'a,
    {
        let prepared = match (self.observers.internal(), saved.activations.as_ref()) {
            (Some(observer), Some(saved)) => Some(observer.prepare_activation_restore(saved)?),
            (None, None) => None,
            _ => {
                return Err(SpeculativeControlError::Invalid(
                    "internal snapshot authority changed",
                ))
            }
        };
        let Some((replacement, seed)) =
            snapshot::copy::<S, M>(&self.strategy, &saved.cache, state, context)?
        else {
            return Ok(None);
        };
        *cache = replacement;
        if let Some(prepared) = prepared {
            prepared.commit();
        }
        Ok(Some(seed))
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

    fn coordinate_decoding_control<'a>(
        &mut self,
        local: eredu_core::decoding::DecodingCoordination,
        context: M::Context<'a>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure>
    where
        Self: 'a,
    {
        M::coordinate_decoding_control(local, context)
    }

    fn coordinate_speculative_step<'a>(
        &mut self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: M::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure>
    where
        Self: 'a,
    {
        M::coordinate_speculative_step(local, context)
    }

    fn agree_text_preparation<'a>(
        &mut self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: M::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    where
        Self: 'a,
    {
        M::agree_text_preparation(stage, status, context)
    }

    fn configure_activation_capture<'a>(
        &mut self,
        plan: AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: M::Context<'a>,
    ) -> Result<(), SpeculativeControlError>
    where
        Self: 'a,
    {
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
        let observer =
            crate::capture::bind_speculative_observer::<M::Capture>(&plan, request, context)?;
        self.prior_internal = Some(self.observers.replace_internal(observer));
        Ok(())
    }

    fn requires_activation_origin(&self) -> bool {
        self.observers.internal_ref().is_some()
    }

    fn set_activation_origin(&mut self, origin: Option<SpeculativeActivationOrigin>) {
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

    fn prefill<'context>(
        &mut self,
        input: Self::Input,
        cache: &mut Self::Cache,
        context: Self::Context<'context>,
    ) -> Result<SpeculativePrefill<Self::TargetState, Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
        self.execution_started = true;
        if self.observers.internal_ref().is_some()
            && !self.strategy.supports_internal_observations()
        {
            return Err(M::observation_error(
                "selected speculative strategy has no complete internal activation path",
            ));
        }
        let checkpoint = S::checkpoint_target(cache)?;
        let result = (|| {
            let (output, total_tokens) =
                self.strategy
                    .prefill_with_prediction(input, cache, context, self.observers)?;
            let sequence = Self::validate_output(&output, None)?;
            if sequence == 0 {
                return Err(M::empty_prediction_input());
            }
            let row = sequence - 1;
            let logits = M::logits_row(output.logits(), row, context)?;
            let state =
                Self::state_at(&output, row, self.strategy.prediction_cache(cache), context)?;
            Ok(SpeculativePrefill::new(logits, state, total_tokens))
        })();
        match result {
            Ok(prefill) => Ok(prefill),
            Err(error) => {
                S::restore_target_checkpoint(cache, &checkpoint, context)?;
                Err(error)
            }
        }
    }

    fn begin_proposal<'context>(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: M::Context<'context>,
    ) -> Result<Self::DraftState, Self::Error>
    where
        Self: 'context,
    {
        let mut prediction_cache = DraftStateTransaction::fork(&state.prediction_cache);
        let fused_logits = if proposal_capacity == 0 {
            None
        } else {
            self.strategy.fused_logits(
                &state.capture,
                last_token,
                proposal_capacity,
                prediction_cache.draft_mut(),
                context,
                self.observers.internal(),
            )?
        };
        if let Some(logits) = &fused_logits {
            let available = M::sequence_len(logits)?;
            if available < proposal_capacity {
                return Err(M::invalid_fused_capacity(proposal_capacity, available));
            }
        }
        Ok(EmbeddedPredictionDraftState {
            capture: state.capture.clone(),
            anchor_capture: state.capture.clone(),
            prediction_cache,
            depth: 0,
            fused_logits,
            fused_cursor: 0,
            proposal_capacity,
        })
    }

    fn proposal_step<'context>(
        &mut self,
        state: &mut Self::DraftState,
        last_token: u32,
        context: M::Context<'context>,
    ) -> Result<eredu_core::speculative::SpeculativeProposalStep<Self::Logits>, Self::Error>
    where
        Self: 'context,
    {
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
            let logits = M::fused_logits_row(logits, row, context)?;
            let logits = self
                .strategy
                .adjust_fused_logits(logits, last_token, context)?;
            return self
                .observers
                .logits(&logits)
                .map(eredu_core::speculative::SpeculativeProposalStep::CausalLogits);
        }
        let (logits, capture) = self.strategy.sequential_logits(
            &state.capture,
            last_token,
            state.depth,
            state.prediction_cache.draft_mut(),
            context,
            self.observers.internal(),
        )?;
        state.capture = self
            .observers
            .tensor(EMBEDDED_PREDICTION_OUTPUT_PATH, &capture)?;
        state.depth += 1;
        self.observers
            .logits(&logits)
            .map(eredu_core::speculative::SpeculativeProposalStep::CausalLogits)
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
    ) -> Result<(), Self::Error>
    where
        Self: 'a,
    {
        S::restore_target_checkpoint(cache, &checkpoint.cache, context)
    }

    fn submit_verification<'context>(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: M::Context<'context>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error>
    where
        Self: 'context,
    {
        let inputs = M::target_tokens(input_tokens, context)?;
        let mut output = self.strategy.verify_target(
            &inputs,
            cache,
            context,
            self.observers.internal(),
            SpeculativeActivationPhase::Verification,
        )?;
        output.logits = self
            .observers
            .tensor(EMBEDDED_VERIFICATION_LOGITS_PATH, &output.logits)?;
        output.capture = self
            .observers
            .tensor(EMBEDDED_TARGET_CAPTURE_PATH, &output.capture)?;
        Self::validate_output(&output, Some(input_tokens.len()))?;
        let completion = M::submit_verification_completion(&output, &inputs, context)?;
        Ok(Submission {
            output: crate::speculative::CausalVerification::new(output, inputs),
            completion,
        })
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>
    where
        Self: 'a,
    {
        M::logits_row(output.output().logits(), index, context)
    }

    fn commit_verification<'context>(
        &mut self,
        output: Self::Verification,
        mut draft_state: Self::DraftState,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        verified_inputs: usize,
        context: M::Context<'context>,
    ) -> Result<SpeculativeCommit<Self::TargetState>, Self::Error>
    where
        Self: 'context,
    {
        let (verified, inputs) = output.into_parts();
        let result = (|| {
            let input_len = M::sequence_len(&inputs)?;
            Self::validate_output(&verified, Some(input_len))?;
            if verified_inputs == 0 || verified_inputs > input_len {
                return Err(M::invalid_prediction_commit(verified_inputs, input_len));
            }
            if draft_state.fused_logits.is_none() {
                // Proposal depths may have consumed rejected tokens and predicted
                // captures. Neither belongs in the accepted-context cache, even
                // when all sampled tokens happen to be accepted. Restore the
                // pre-proposal prediction lane and replay the committed anchor
                // from the target's actual capture through every prediction unit.
                let accepted = draft_state.prediction_cache.checkpoint().clone();
                *draft_state.prediction_cache.draft_mut() = accepted;
                let anchor = M::token_prefix(&inputs, 1, context)?;
                self.strategy.advance_prediction_cache(
                    &draft_state.anchor_capture,
                    &anchor,
                    draft_state.prediction_cache.draft_mut(),
                    context,
                    self.observers.internal(),
                )?;
            }
            if verified_inputs > 1 {
                let captures = M::tensor_prefix(verified.capture(), verified_inputs - 1, context)?;
                let tokens = M::token_range(&inputs, 1, verified_inputs, context)?;
                self.strategy.advance_prediction_cache(
                    &captures,
                    &tokens,
                    draft_state.prediction_cache.draft_mut(),
                    context,
                    self.observers.internal(),
                )?;
            }
            let (committed, replayed_tokens) = if verified_inputs == input_len {
                (verified, 0)
            } else {
                S::restore_target_checkpoint(cache, &checkpoint.cache, context)?;
                let retained = M::token_prefix(&inputs, verified_inputs, context)?;
                (
                    self.strategy.verify_target(
                        &retained,
                        cache,
                        context,
                        self.observers.internal(),
                        SpeculativeActivationPhase::TargetReplay,
                    )?,
                    verified_inputs,
                )
            };
            self.strategy
                .commit_prediction_cache(cache, draft_state.prediction_cache.draft())?;
            let state = Self::state_at(
                &committed,
                verified_inputs - 1,
                self.strategy.prediction_cache(cache),
                context,
            )?;
            Ok(SpeculativeCommit::new(state, replayed_tokens))
        })();
        match result {
            Ok(commit) => Ok(commit),
            Err(error) => {
                S::restore_target_checkpoint(cache, &checkpoint.cache, context)?;
                Err(error)
            }
        }
    }
}
