//! Backend-generic speculative execution over architecture-owned prediction strategies.

use std::marker::PhantomData;

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
mod observation;
mod snapshot;
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
    tensors: Box<dyn eredu_runtime::ActivationObserver<T, E>>,
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
            tensors: Box::new(tensors),
            logits: Box::new(logits),
            internal: None,
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

    fn tensor(&mut self, path: &str, value: &T) -> Result<T, E>
    where
        T: Clone,
    {
        eredu_runtime::observe_and_intervene(self.tensors.as_mut(), path, value)
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
        Self::new(eredu_runtime::NoopObserver, eredu_runtime::NoopObserver)
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

    /// Binds the lane to the exact prepared description and semantic content.
    pub fn bind_prepared_input(
        &mut self,
        identity: Option<&eredu_runtime::PreparedInputCacheIdentity>,
    ) -> Result<(), EmbeddedPredictionCacheError> {
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
    prepared_input: Option<eredu_runtime::SpeculativeIdentity>,
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

    /// Selects one logits row from a sequence tensor.
    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

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

    /// Selects one row from a fused proposal block.
    fn fused_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error>;

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
    pub tokens: T,
}

impl<T> EmbeddedPredictionOutput<T> {
    /// Creates one exact target output.
    pub const fn new(logits: T, capture: T, tokens: T) -> Self {
        Self {
            logits,
            capture,
            tokens,
        }
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
    pub const fn tokens(&self) -> &T {
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
    type PredictionCache: Clone;
    /// Optional component telemetry.
    type Telemetry: SpeculativeTelemetry;

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

    /// Runs ordinary-target verification and returns its exact selected capture.
    fn verify_target<'a>(
        &mut self,
        tokens: &M::Tensor,
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
    /// Creates an exact native target-state checkpoint.
    fn checkpoint(state: &S) -> Result<S, M::Error>;
    /// Current native state and horizon capacity allowances; no synchronization.
    fn state_memory_bounds(_state: &S, _additional: u64) -> (Option<u64>, Option<u64>) {
        (None, None)
    }

    /// Complete isolated snapshot bound for the target's native state profile.
    fn control_state_estimate(
        _state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Copies complete native state and settles copying before returning.
    fn control_state_snapshot(
        _state: &S,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
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
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, M::Error>;
    /// Returns the physical shape used to close the selected capture contract.
    fn shape(tensor: &B::Tensor) -> &[i32];
    /// Runs an operation inside the backend's deferred-validation transaction.
    fn validate<T>(operation: impl FnOnce() -> Result<T, M::Error>) -> Result<T, M::Error>;
    /// Maps a neutral-session failure into the backend error domain.
    fn session_error(error: impl std::fmt::Display) -> M::Error;
    /// Preserves an owned native/session cause across the prediction error domain.
    fn session_failure(error: eredu_core::BackendFailure) -> M::Error;
    /// Drains native component telemetry.
    fn take_telemetry() -> Result<Self::Telemetry, M::Error>;
}

/// Architecture-owned embedded strategy over one typed replicated session and paired extension.
pub struct ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M>
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
    session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
    extension: &'a mut P,
    selected: &'a eredu_runtime::SelectedSpeculativeRealization,
    input: I,
    cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    _native: PhantomData<fn() -> (S, N, M)>,
}

impl<'a, A, B, S, SM, D, P, I, N, M>
    ReplicatedMaterializedPredictionStrategy<'a, A, B, S, SM, D, P, I, N, M>
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
{
    /// Completes the statically checked pairing before any executor erasure.
    pub fn new(
        session: &'a mut eredu_runtime::ReplicatedTextSession<A, B, SM, D>,
        extension: &'a mut P,
        selected: &'a eredu_runtime::SelectedSpeculativeRealization,
        input: I,
        cache_context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Self {
        Self {
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
        lane: &mut P::LaneState,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        execute: impl FnOnce(
            &mut P,
            &mut ReplicatedPredictionInvoker<'_, A, B, S, SM, D, N, M>,
            &mut P::LaneState,
            Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, A::Error>>,
        ) -> Result<R, M::Error>,
        complete: impl FnOnce(&P, &mut P::LaneState, &R) -> Result<(), eredu_core::BackendFailure>,
    ) -> Result<R, M::Error>
    where
        N: ReplicatedPredictionNative<A, B, S, M>,
    {
        N::validate(|| {
            let Some(observer) = observer else {
                return execute(
                    self.extension,
                    &mut ReplicatedPredictionInvoker {
                        session: self.session,
                        context,
                        _native: PhantomData,
                    },
                    lane,
                    None,
                );
            };
            self.session.with_prediction_observation(
                pass,
                context,
                observer,
                &mut (&mut *self.extension, lane),
                |session, (extension, lane), observer| {
                    execute(
                        extension,
                        &mut ReplicatedPredictionInvoker {
                            session,
                            context,
                            _native: PhantomData,
                        },
                        lane,
                        Some(observer),
                    )
                },
                |_, (extension, lane), output| {
                    complete(extension, lane, output).map_err(N::session_failure)
                },
                |error| N::session_failure(eredu_core::BackendFailure::from_error(error)),
            )
        })
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

struct ReplicatedPredictionInvoker<'a, A, B, S, SM, D, N, M>
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
    _native: PhantomData<fn() -> (N, M)>,
}

impl<A, B, S, SM, D, N, M> crate::prediction_extension::PredictionOperationInvoker<A, B, S>
    for ReplicatedPredictionInvoker<'_, A, B, S, SM, D, N, M>
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
            .map_err(|error| N::session_failure(eredu_core::BackendFailure::from_error(error)))
    }

    fn invalid(message: String) -> Self::Error {
        N::session_error(message)
    }
}

impl<A, B, S, SM, D, P, I, N, M> EmbeddedPredictionStrategy<M>
    for ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M>
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
{
    type Input = I::Input;
    type TargetCache = EmbeddedPredictionCache<S, P::LaneState>;
    type PredictionCache = EmbeddedPredictionDraftCache<P::LaneState>;
    type Telemetry = N::Telemetry;

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.session.trim_parameter_conversions().map(|target| {
            eredu_core::residency::ExecutionConversionRetentionTrimReport {
                target,
                external_drafter: None,
            }
        })
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        self.session
            .parameter_conversion_retention()
            .map(|target| {
                Some(eredu_core::residency::ExecutionConversionRetentionReport {
                    target,
                    external_drafter: None,
                })
            })
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn continuation_memory_observation(
        &self,
        cache: &Self::TargetCache,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        let Some(target) = cache.target.as_ref() else {
            return Ok(None);
        };
        let Some(parameters) = self
            .session
            .parameter_memory_observation()
            .map_err(eredu_core::BackendFailure::from_error)?
        else {
            return Ok(None);
        };
        let Some(prediction) = self
            .extension
            .memory_observation(&cache.prediction, additional)
        else {
            return Ok(None);
        };
        let (current_state_bytes, peak_state_bytes) = N::state_memory_bounds(target, additional);
        Ok(Some(
            eredu_core::speculative::SpeculativeContinuationObservation {
                target: eredu_core::speculative::SpeculativeModelMemoryObservation {
                    parameter_conversions: parameters.parameter_conversions.clone(),
                    current_positions: N::generation(target)
                        .map_err(eredu_core::BackendFailure::from_error)?,
                    current_state_bytes,
                    peak_state_bytes,
                    parameters: parameters.parameters,
                    available: parameters.available,
                    allocator_cache_limit: parameters.allocator_cache_limit,
                },
                draft: None,
                embedded: Some(eredu_core::speculative::EmbeddedContinuationObservation {
                    prediction,
                    retained_feature_bytes: None,
                }),
                parameter_conversions: parameters.parameter_conversions,
                seed_bytes: None,
            },
        ))
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
        let Some(target) = cache.target.as_ref() else {
            return Ok(None);
        };
        let context = N::target_context(context);
        let Some(target) = N::control_state_snapshot(target, context)? else {
            return Ok(None);
        };
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
        }))
    }

    fn control_prediction_snapshot<'a>(
        &self,
        cache: &Self::PredictionCache,
        context: M::Context<'a>,
    ) -> Result<Option<Self::PredictionCache>, SpeculativeControlError> {
        let Some(prediction) = self
            .extension
            .snapshot(&cache.prediction, N::target_context(context))
            .map_err(SpeculativeControlError::Backend)?
        else {
            return Ok(None);
        };
        Ok(Some(EmbeddedPredictionDraftCache {
            prediction,
            prepared_input: cache.prepared_input.clone(),
            capture_generation: cache.capture_generation,
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
        cache.checkpoint(N::checkpoint).map_err(N::session_error)
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
            ..
        } = self;
        lowerer.with_prefill(input, tensor_context, |prepared, tokens, identity| {
            let sequence = M::sequence_len(&tokens)?;
            cache
                .bind_prepared_input(identity)
                .map_err(N::session_error)?;
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_error(EmbeddedPredictionCacheError::TargetStateActive))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_error(error));
            }
            let result = observation::neural(observer, SpeculativeActivationPhase::TargetPrefill,
                sequence, N::session_error, |observer| N::validate(|| {
                    match observer {
                        Some(observer) => session.prefill_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.prefill_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture)| EmbeddedPredictionOutput::new(logits, capture, tokens))
                    .map_err(N::session_error)
                }));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => session
                    .recover_prediction_target_state_after_failure(&mut lane)
                    .map_err(|recovery| {
                        N::session_error(format!(
                            "prediction target state exchange failed: {error}; local ownership recovery failed: {recovery}"
                        ))
                    })
                    .and(Err(N::session_error(error))),
            };
            cache.restore_target(lane);
            let output = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity(selected, N::generation)
                .map_err(N::session_error)?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(N::session_error)?;
            cache
                .retain_capture_generation(N::generation)
                .map_err(N::session_error)?;
            Ok(output)
        })
    }

    fn verify_target<'a>(
        &mut self,
        tokens: &B::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
        observer: Option<&mut dyn SpeculativeActivationObserver<B::Tensor, M::Error>>,
        phase: SpeculativeActivationPhase,
    ) -> Result<EmbeddedPredictionOutput<B::Tensor>, M::Error> {
        let sequence = M::sequence_len(tokens)?;
        let tensor_context = N::target_context(context);
        let retained = tokens.clone();
        let Self {
            session,
            extension,
            selected,
            input: lowerer,
            ..
        } = self;
        lowerer.with_decode(tokens, tensor_context, |prepared| {
            let mut lane = cache
                .take_target()
                .ok_or_else(|| N::session_error(EmbeddedPredictionCacheError::TargetStateActive))?;
            if let Err(error) =
                session.exchange_prediction_target_state(&mut lane, tensor_context)
            {
                cache.restore_target(lane);
                return Err(N::session_error(error));
            }
            let result = observation::neural(observer, phase, sequence, N::session_error,
                |observer| N::validate(|| {
                    match observer {
                        Some(observer) => session.decode_input_prediction_target_observed(prepared, tensor_context, observer),
                        None => session.decode_input_prediction_target(prepared, tensor_context),
                    }
                    .map(|(logits, capture)| EmbeddedPredictionOutput::new(logits, capture, retained))
                    .map_err(N::session_error)
                }));
            let restored = match session
                .exchange_prediction_target_state(&mut lane, tensor_context)
            {
                Ok(()) => Ok(()),
                Err(error) => session
                    .recover_prediction_target_state_after_failure(&mut lane)
                    .map_err(|recovery| {
                        N::session_error(format!(
                            "prediction target state exchange failed: {error}; local ownership recovery failed: {recovery}"
                        ))
                    })
                    .and(Err(N::session_error(error))),
            };
            cache.restore_target(lane);
            let output = match (result, restored) {
                (Err(error), _) => return Err(error),
                (Ok(output), Ok(())) => output,
                (Ok(_), Err(error)) => return Err(error),
            };
            let lane = cache
                .lane_identity(selected, N::generation)
                .map_err(N::session_error)?;
            extension
                .validate_capture(selected, &lane, N::shape(output.capture()))
                .map_err(N::session_error)?;
            cache
                .retain_capture_generation(N::generation)
                .map_err(N::session_error)?;
            Ok(output)
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
            .lane_identity(self.selected, N::generation)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane_identity, N::shape(output.capture()))
            .map_err(N::session_error)?;
        let prediction_sequence = self.extension.prefill_sequence_len(sequence);
        if prediction_sequence == 0 {
            return Ok(());
        }
        let hidden = M::tensor_prefix(output.capture(), sequence.saturating_sub(1), context)?;
        let next = M::token_range(tokens, 1, sequence, context)?;
        let checkpoint = cache.prediction_fork();
        let tensor_context = N::target_context(context);
        let result = observation::neural(
            observer,
            SpeculativeActivationPhase::PredictionPrefill,
            prediction_sequence,
            N::session_error,
            |observer| {
                self.prediction_phase(
                    cache.prediction_mut(),
                    eredu_runtime::ExpertPass::Prefill,
                    tensor_context,
                    observer,
                    |extension, invoker, lane, observer| {
                        extension.prefill_observed::<S, _>(
                            invoker,
                            output.capture(),
                            &hidden,
                            &next,
                            lane,
                            observer,
                        )
                    },
                    |extension, lane, _| extension.complete_state(lane, &[], tensor_context),
                )
            },
        );
        if result.is_err() {
            cache.commit_prediction(&checkpoint);
        }
        result
    }

    fn prediction_cache(&self, cache: &Self::TargetCache) -> Self::PredictionCache {
        cache.prediction_fork()
    }

    fn commit_prediction_cache(
        &self,
        cache: &mut Self::TargetCache,
        prediction: &Self::PredictionCache,
    ) -> Result<(), M::Error> {
        cache.commit_prediction(prediction);
        Ok(())
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
            .map_err(N::session_error)
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
        let token = N::token(last_token, tensor_context)?;
        observation::neural(
            observer,
            SpeculativeActivationPhase::Proposal { depth },
            1,
            N::session_error,
            |observer| {
                self.prediction_phase(
                    cache.prediction_mut(),
                    eredu_runtime::ExpertPass::Decode,
                    tensor_context,
                    observer,
                    |extension, invoker, lane, observer| {
                        let observed = observer.is_some();
                        let (logits, hidden) = extension.logits_observed::<S, _>(
                            invoker, capture, &token, depth, lane, observer,
                        )?;
                        let row = M::logits_row(&logits, 0, context)?;
                        Ok((row, hidden, observed.then_some(logits)))
                    },
                    |extension, lane, (_, hidden, logits)| {
                        let logits = logits.as_ref().expect("observed output retains its logits");
                        extension.complete_state(lane, &[logits, hidden], tensor_context)
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
    ) -> Result<Option<B::Tensor>, M::Error> {
        let lane = cache
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(capture))
            .map_err(N::session_error)?;
        if self.selected.requirements().strategy().class()
            == eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential
        {
            // Sequential proposals still validate the retained lane and capture
            // above; no fused invocation exists to instrument for this strategy.
            return Ok(None);
        }
        let tensor_context = N::target_context(context);
        let token = N::token(last_token, tensor_context)?;
        if observer.is_some() {
            // Retain all temporary proposal cache members through the same
            // completion/commit protocol as sequential prediction phases. The
            // persistent accepted-context lane is never modified by proposals.
            let mut proposal = cache.prediction().clone();
            return observation::neural(
                observer,
                SpeculativeActivationPhase::FusedProposal,
                capacity,
                N::session_error,
                |observer| {
                    self.prediction_phase(
                        &mut proposal,
                        eredu_runtime::ExpertPass::Decode,
                        tensor_context,
                        observer,
                        |extension, invoker, lane, observer| {
                            extension.fused_logits_observed::<S, _>(
                                invoker, &token, capacity, lane, observer,
                            )
                        },
                        |extension, lane, logits| match logits {
                            Some(logits) => {
                                extension.complete_state(lane, &[logits], tensor_context)
                            }
                            None => extension.complete_state(lane, &[], tensor_context),
                        },
                    )
                },
            );
        }
        N::validate(|| {
            self.extension.fused_logits::<S, _>(
                &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                    session: self.session,
                    context: tensor_context,
                    _native: PhantomData,
                },
                &token,
                capacity,
                cache.prediction(),
            )
        })
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
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(captures))
            .map_err(N::session_error)?;
        let tensor_context = N::target_context(context);
        observation::neural(
            observer,
            SpeculativeActivationPhase::PredictionReplay,
            M::sequence_len(tokens)?,
            N::session_error,
            |observer| {
                self.prediction_phase(
                    cache.prediction_mut(),
                    eredu_runtime::ExpertPass::Decode,
                    tensor_context,
                    observer,
                    |extension, invoker, lane, observer| {
                        extension
                            .advance_observed::<S, _>(invoker, captures, tokens, lane, observer)
                    },
                    |extension, lane, _| extension.complete_state(lane, &[], tensor_context),
                )
            },
        )
    }
}

impl<A, B, S, SM, D, P, I, N, M> EmbeddedExecutorCacheFactory<N::ExecutorTypes>
    for EmbeddedPredictionExecutor<
        '_,
        ReplicatedMaterializedPredictionStrategy<'_, A, B, S, SM, D, P, I, N, M>,
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
{
    fn new_cache(&mut self) -> Result<Self::Cache, Self::Error> {
        self.strategy_mut().new_cache()
    }

    fn bind_context<'a>(
        context: <N::ExecutorTypes as EmbeddedExecutorTypes>::Context<'a>,
    ) -> Self::Context<'a> {
        N::executor_context(context)
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
    prediction_cache: DraftStateTransaction<C>,
    depth: usize,
    fused_logits: Option<T>,
    fused_cursor: usize,
    proposal_capacity: usize,
}

/// Retained verification output and its exact input tokens.
pub struct EmbeddedPredictionVerification<T> {
    output: EmbeddedPredictionOutput<T>,
    inputs: T,
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

    /// Constructs the stable failure for an internally inconsistent erased value.
    fn erased_type_mismatch(value: &'static str) -> Self::Error;
}

trait CloneAny: std::any::Any {
    fn clone_any(&self) -> Box<dyn CloneAny>;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

impl<T: Clone + 'static> CloneAny for T {
    fn clone_any(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// Architecture-owned erased target cache for a completed embedded executor.
pub struct DynEmbeddedCache(Box<dyn std::any::Any>);
/// Architecture-owned erased proposal seed state.
pub struct DynEmbeddedTargetState(Box<dyn std::any::Any>);
/// Architecture-owned erased, discardable prediction branch.
pub struct DynEmbeddedDraftState(Box<dyn CloneAny>);
/// Architecture-owned erased target-cache checkpoint.
pub struct DynEmbeddedCheckpoint(Box<dyn std::any::Any>);
/// Architecture-owned erased verification payload.
pub struct DynEmbeddedVerification(Box<dyn std::any::Any>);

impl Clone for DynEmbeddedDraftState {
    fn clone(&self) -> Self {
        Self(self.0.as_ref().clone_any())
    }
}

/// An executor which can realize a fresh lane cache before scheduling.
pub trait EmbeddedExecutorCacheFactory<T: EmbeddedExecutorTypes>: SpeculativeExecutor {
    /// Realizes one independent authoritative target/prediction lane.
    fn new_cache(&mut self) -> Result<Self::Cache, Self::Error>;

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
    /// Installs validated prospective internal edits through the typed collector.
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError>;
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    >;

    /// Reads shared retention budgets without touching a lane or native work.
    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    >;

    /// Reads one settled lane without polling, copying, or evaluating native work.
    fn continuation_memory_observation(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    >;
    /// Complete durable snapshot bound for the erased canonical lane.
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
    /// Prefills one lane.
    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynEmbeddedTargetState, T::Logits>, T::Error>;
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
    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        SpeculativeExecutor::readmit_activation_interventions(self, plan)
    }
    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        SpeculativeExecutor::trim_parameter_conversions(self)
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        SpeculativeExecutor::parameter_conversion_retention(self)
    }

    fn continuation_memory_observation(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
        additional: u64,
    ) -> Result<
        Option<eredu_core::speculative::SpeculativeContinuationObservation>,
        eredu_core::BackendFailure,
    > {
        SpeculativeExecutor::continuation_memory_observation(
            self,
            cache.0.downcast_ref().expect("paired embedded cache"),
            state
                .0
                .downcast_ref()
                .expect("paired embedded target state"),
            additional,
        )
    }

    fn control_snapshot_estimate(
        &self,
        cache: &DynEmbeddedCache,
        state: &DynEmbeddedTargetState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        SpeculativeExecutor::control_snapshot_estimate(
            self,
            erased_ref::<E::Cache, T>(&cache.0, "snapshot cache").ok()?,
            erased_ref::<E::TargetState, T>(&state.0, "snapshot seed").ok()?,
        )
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
        SpeculativeExecutor::control_snapshot(
            self,
            cache,
            state,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|pair| {
            pair.map(|(cache, state)| {
                (
                    DynEmbeddedCheckpoint(Box::new(cache)),
                    DynEmbeddedTargetState(Box::new(state)),
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
        SpeculativeExecutor::restore_control_snapshot(
            self,
            cache,
            saved,
            state,
            <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context),
        )
        .map(|state| state.map(|state| DynEmbeddedTargetState(Box::new(state))))
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
            .map(|cache| DynEmbeddedCache(Box::new(cache)))
    }

    fn prefill<'a>(
        &mut self,
        input: T::Input,
        cache: &mut DynEmbeddedCache,
        context: T::Context<'a>,
    ) -> Result<SpeculativePrefill<DynEmbeddedTargetState, T::Logits>, T::Error> {
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::prefill(self, input, cache, context).map(|prefill| {
            let (logits, state, target_tokens) = prefill.into_parts();
            SpeculativePrefill::new(
                logits,
                DynEmbeddedTargetState(Box::new(state)),
                target_tokens,
            )
        })
    }

    fn begin_proposal<'a>(
        &mut self,
        state: &DynEmbeddedTargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: T::Context<'a>,
    ) -> Result<DynEmbeddedDraftState, T::Error> {
        let state = erased_ref::<E::TargetState, T>(&state.0, "target state")?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::begin_proposal(self, state, last_token, proposal_capacity, context)
            .map(|state| DynEmbeddedDraftState(Box::new(state)))
    }

    fn proposal_logits<'a>(
        &mut self,
        state: &mut DynEmbeddedDraftState,
        last_token: u32,
        context: T::Context<'a>,
    ) -> Result<T::Logits, T::Error> {
        let state = state
            .0
            .as_any_mut()
            .downcast_mut::<E::DraftState>()
            .ok_or_else(|| T::erased_type_mismatch("draft state"))?;
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::proposal_logits(self, state, last_token, context)
    }

    fn checkpoint(&self, cache: &DynEmbeddedCache) -> Result<DynEmbeddedCheckpoint, T::Error> {
        let cache = erased_ref::<E::Cache, T>(&cache.0, "target cache")?;
        SpeculativeExecutor::checkpoint(self, cache)
            .map(|checkpoint| DynEmbeddedCheckpoint(Box::new(checkpoint)))
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
        let context = <E as EmbeddedExecutorCacheFactory<T>>::bind_context(context);
        SpeculativeExecutor::submit_verification(self, input_tokens, cache, context).map(
            |submission| Submission {
                output: DynEmbeddedVerification(Box::new(submission.output)),
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
            .into_any()
            .downcast::<E::DraftState>()
            .map_err(|_| T::erased_type_mismatch("draft state"))?;
        let cache = erased_mut::<E::Cache, T>(&mut cache.0, "target cache")?;
        let checkpoint = erased_ref::<E::CacheCheckpoint, T>(&checkpoint.0, "target checkpoint")?;
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
            SpeculativeCommit::new(DynEmbeddedTargetState(Box::new(state)), replayed)
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
}

impl<T: EmbeddedExecutorTypes> SpeculativeExecutor for DynEmbeddedExecutor<'_, T> {
    type Input = T::Input;
    type Cache = DynEmbeddedCache;
    type TargetState = DynEmbeddedTargetState;
    type DraftState = DynEmbeddedDraftState;
    type CacheCheckpoint = DynEmbeddedCheckpoint;
    type Verification = DynEmbeddedVerification;
    type Logits = T::Logits;
    type Context<'a> = T::Context<'a>;
    type Completion = T::Completion;
    type Telemetry = T::Telemetry;
    type Error = T::Error;

    fn readmit_activation_interventions(
        &mut self,
        plan: AdmittedSpeculativeActivations,
    ) -> Result<(), SpeculativeControlError> {
        self.inner.readmit_activation_interventions(plan)
    }

    /// Releases optional conversions; caller must establish a settled boundary.
    fn trim_parameter_conversions(
        &mut self,
    ) -> Result<
        eredu_core::residency::ExecutionConversionRetentionTrimReport,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.inner.trim_parameter_conversions()
    }

    fn parameter_conversion_retention(
        &self,
    ) -> Result<
        Option<eredu_core::residency::ExecutionConversionRetentionReport>,
        eredu_core::BackendFailure,
    > {
        self.inner.parameter_conversion_retention()
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
        self.inner
            .continuation_memory_observation(cache, state, additional)
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
    type CacheCheckpoint = EmbeddedPredictionCheckpoint<S::TargetCache>;
    type Verification = EmbeddedPredictionVerification<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = S::Telemetry;
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
            embedded.retained_feature_bytes =
                M::control_tensor_estimate(&state.capture).map(|e| e.retained_bytes);
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
        self.observers.internal.is_some()
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
        self.execution_started = true;
        if self.observers.internal.is_some() && !self.strategy.supports_internal_observations() {
            return Err(M::observation_error(
                "selected speculative strategy has no complete internal activation path",
            ));
        }
        let checkpoint = S::checkpoint_target(cache)?;
        let result = (|| {
            let mut output =
                self.strategy
                    .prefill_target(input, cache, context, self.observers.internal())?;
            output.capture = self
                .observers
                .tensor(EMBEDDED_TARGET_CAPTURE_PATH, &output.capture)?;
            let sequence = Self::validate_output(&output, None)?;
            if sequence == 0 {
                return Err(M::empty_prediction_input());
            }
            let tokens = output.tokens().clone();
            self.strategy.seed_prediction_cache(
                &output,
                &tokens,
                cache,
                context,
                self.observers.internal(),
            )?;
            let row = sequence - 1;
            let logits = M::logits_row(output.logits(), row, context)?;
            let state =
                Self::state_at(&output, row, self.strategy.prediction_cache(cache), context)?;
            Ok(SpeculativePrefill::new(logits, state, sequence))
        })();
        match result {
            Ok(prefill) => Ok(prefill),
            Err(error) => {
                S::restore_target_checkpoint(cache, &checkpoint, context)?;
                Err(error)
            }
        }
    }

    fn begin_proposal(
        &mut self,
        state: &Self::TargetState,
        last_token: u32,
        proposal_capacity: usize,
        context: M::Context<'_>,
    ) -> Result<Self::DraftState, Self::Error> {
        let mut prediction_cache = DraftStateTransaction::fork(&state.prediction_cache);
        let fused_logits = self.strategy.fused_logits(
            &state.capture,
            last_token,
            proposal_capacity,
            prediction_cache.draft_mut(),
            context,
            self.observers.internal(),
        )?;
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
            let logits = M::fused_logits_row(logits, row, context)?;
            let logits = self
                .strategy
                .adjust_fused_logits(logits, last_token, context)?;
            return self.observers.logits(&logits);
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
            output: EmbeddedPredictionVerification { output, inputs },
            completion,
        })
    }

    fn verification_logits<'a>(
        &self,
        output: &Self::Verification,
        index: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        M::logits_row(output.output.logits(), index, context)
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
        let result = (|| {
            let input_len = M::sequence_len(&output.inputs)?;
            Self::validate_output(&output.output, Some(input_len))?;
            if verified_inputs == 0 || verified_inputs > input_len {
                return Err(M::invalid_prediction_commit(verified_inputs, input_len));
            }
            if verified_inputs > 1 {
                let captures =
                    M::tensor_prefix(output.output.capture(), verified_inputs - 1, context)?;
                let tokens = M::token_range(&output.inputs, 1, verified_inputs, context)?;
                self.strategy.advance_prediction_cache(
                    &captures,
                    &tokens,
                    draft_state.prediction_cache.draft_mut(),
                    context,
                    self.observers.internal(),
                )?;
            }
            let (committed, replayed_tokens) = if verified_inputs == input_len {
                (output.output, 0)
            } else {
                S::restore_target_checkpoint(cache, &checkpoint.cache, context)?;
                let retained = M::token_prefix(&output.inputs, verified_inputs, context)?;
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

#[cfg(test)]
mod tests;
