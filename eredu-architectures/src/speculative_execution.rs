//! Backend-generic speculative execution over architecture-owned prediction strategies.

use std::marker::PhantomData;

use eredu_core::{
    BoundedCompletion, SpeculativeCommit, SpeculativeExecutor, SpeculativePrefill,
    SpeculativeTelemetry, Submission,
};
use eredu_runtime::DraftStateTransaction;

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

    /// Maximum number of proposal tokens in one verification transaction.
    fn proposal_capacity(&self) -> usize;

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
    ) -> Result<EmbeddedPredictionOutput<M::Tensor>, M::Error>;

    /// Runs ordinary-target verification and returns its exact selected capture.
    fn verify_target<'a>(
        &mut self,
        tokens: &M::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
    ) -> Result<EmbeddedPredictionOutput<M::Tensor>, M::Error>;

    /// Seeds prediction-local state from a successful ordinary-target transaction.
    fn seed_prediction_cache<'a>(
        &mut self,
        output: &EmbeddedPredictionOutput<M::Tensor>,
        tokens: &M::Tensor,
        cache: &mut Self::TargetCache,
        context: M::Context<'a>,
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
    ) -> Result<(M::Logits, M::Tensor), M::Error>;

    /// Optionally runs one fused proposal block from the exact target capture.
    fn fused_logits<'a>(
        &mut self,
        _capture: &M::Tensor,
        _last_token: u32,
        _capacity: usize,
        _cache: &mut Self::PredictionCache,
        _context: M::Context<'a>,
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
    A::Error: std::fmt::Display,
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
    A::Error: std::fmt::Display,
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
    A::Error: std::fmt::Display,
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
    A::Error: std::fmt::Display,
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
            .map_err(N::session_error)
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
    A::Error: std::fmt::Display,
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
            let result = N::validate(|| {
                session
                    .prefill_input_prediction_target(prepared, tensor_context)
                    .map(|(logits, capture)| {
                        EmbeddedPredictionOutput::new(logits, capture, tokens)
                    })
                    .map_err(N::session_error)
            });
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
    ) -> Result<EmbeddedPredictionOutput<B::Tensor>, M::Error> {
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
            let result = N::validate(|| {
                session
                    .decode_input_prediction_target(prepared, tensor_context)
                    .map(|(logits, capture)| {
                        EmbeddedPredictionOutput::new(logits, capture, retained)
                    })
                    .map_err(N::session_error)
            });
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
    ) -> Result<(), M::Error> {
        let sequence = M::sequence_len(tokens)?;
        let lane_identity = cache
            .lane_identity(self.selected, N::generation)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane_identity, N::shape(output.capture()))
            .map_err(N::session_error)?;
        if sequence <= 1 && !self.extension.prefill_single_token() {
            return Ok(());
        }
        let hidden = M::tensor_prefix(output.capture(), sequence.saturating_sub(1), context)?;
        let next = M::token_range(tokens, 1, sequence, context)?;
        let checkpoint = cache.prediction_fork();
        let tensor_context = N::target_context(context);
        let result = N::validate(|| {
            self.extension.prefill::<S, _>(
                &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                    session: self.session,
                    context: tensor_context,
                    _native: PhantomData,
                },
                output.capture(),
                &hidden,
                &next,
                cache.prediction_mut(),
            )
        });
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
    ) -> Result<(M::Logits, B::Tensor), M::Error> {
        let tensor_context = N::target_context(context);
        let token = N::token(last_token, tensor_context)?;
        N::validate(|| {
            self.extension
                .logits::<S, _>(
                    &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                        session: self.session,
                        context: tensor_context,
                        _native: PhantomData,
                    },
                    capture,
                    &token,
                    depth,
                    cache.prediction_mut(),
                )
                .and_then(|(logits, hidden)| {
                    M::logits_row(&logits, 0, context).map(|logits| (logits, hidden))
                })
        })
    }

    fn fused_logits<'a>(
        &mut self,
        capture: &B::Tensor,
        last_token: u32,
        capacity: usize,
        cache: &mut Self::PredictionCache,
        context: M::Context<'a>,
    ) -> Result<Option<B::Tensor>, M::Error> {
        let lane = cache
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(capture))
            .map_err(N::session_error)?;
        let tensor_context = N::target_context(context);
        let token = N::token(last_token, tensor_context)?;
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
    ) -> Result<(), M::Error> {
        let lane = cache
            .lane_identity(self.selected)
            .map_err(N::session_error)?;
        self.extension
            .validate_capture(self.selected, &lane, N::shape(captures))
            .map_err(N::session_error)?;
        let tensor_context = N::target_context(context);
        N::validate(|| {
            self.extension.advance::<S, _>(
                &mut ReplicatedPredictionInvoker::<A, B, S, SM, D, N, M> {
                    session: self.session,
                    context: tensor_context,
                    _native: PhantomData,
                },
                captures,
                tokens,
                cache.prediction_mut(),
            )
        })
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
    SM::PolicyError: std::fmt::Display,
    SM::Error: std::fmt::Display,
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
    _mechanisms: PhantomData<fn() -> M>,
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
            _mechanisms: PhantomData,
        }
    }

    /// Returns the production observer set after execution so a session can retain it.
    pub fn into_observers(self) -> EmbeddedPredictionObservers<M::Tensor, M::Logits, M::Error> {
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
    type CacheCheckpoint = S::TargetCache;
    type Verification = EmbeddedPredictionVerification<M::Tensor>;
    type Logits = M::Logits;
    type Context<'a> = M::Context<'a>;
    type Completion = M::Completion;
    type Telemetry = S::Telemetry;
    type Error = M::Error;

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
        let checkpoint = S::checkpoint_target(cache)?;
        let result = (|| {
            let mut output = self.strategy.prefill_target(input, cache, context)?;
            output.capture = self
                .observers
                .tensor(EMBEDDED_TARGET_CAPTURE_PATH, &output.capture)?;
            let sequence = Self::validate_output(&output, None)?;
            if sequence == 0 {
                return Err(M::empty_prediction_input());
            }
            let tokens = output.tokens().clone();
            self.strategy
                .seed_prediction_cache(&output, &tokens, cache, context)?;
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
        )?;
        state.capture = self
            .observers
            .tensor(EMBEDDED_PREDICTION_OUTPUT_PATH, &capture)?;
        state.depth += 1;
        self.observers.logits(&logits)
    }

    fn checkpoint(&self, cache: &Self::Cache) -> Result<Self::CacheCheckpoint, Self::Error> {
        S::checkpoint_target(cache)
    }

    fn restore_checkpoint<'a>(
        &mut self,
        cache: &mut Self::Cache,
        checkpoint: &Self::CacheCheckpoint,
        context: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        S::restore_target_checkpoint(cache, checkpoint, context)
    }

    fn submit_verification(
        &mut self,
        input_tokens: &[u32],
        cache: &mut Self::Cache,
        context: M::Context<'_>,
    ) -> Result<Submission<Self::Verification, Self::Completion>, Self::Error> {
        let inputs = M::target_tokens(input_tokens, context)?;
        let mut output = self.strategy.verify_target(&inputs, cache, context)?;
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
                )?;
            }
            let (committed, replayed_tokens) = if verified_inputs == input_len {
                (output.output, 0)
            } else {
                S::restore_target_checkpoint(cache, checkpoint, context)?;
                let retained = M::token_prefix(&output.inputs, verified_inputs, context)?;
                (
                    self.strategy.verify_target(&retained, cache, context)?,
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
                S::restore_target_checkpoint(cache, checkpoint, context)?;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt,
        sync::{Arc, Mutex},
    };

    use eredu_core::{Completion, SpeculativeExecutor};

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Tensor(Vec<i32>);

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct Cache {
        target: Vec<i32>,
        prediction: Vec<i32>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Failure {
        None,
        Advance,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestError(String);

    impl fmt::Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.0)
        }
    }

    impl std::error::Error for TestError {}

    #[derive(Debug)]
    struct Done {
        retained: Arc<[Tensor]>,
    }

    impl Completion for Done {
        type Error = TestError;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl eredu_core::BoundedCompletion for Done {
        fn wait_bounded(
            self,
            _policy: eredu_core::BoundedCompletionWait,
        ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
            Ok(eredu_core::BoundedCompletionOutcome::Completed)
        }
    }

    struct Mechanisms;

    impl SpeculativeTensorMechanisms for Mechanisms {
        type Tensor = Tensor;
        type Logits = i32;
        type Context<'a> = ();
        type Completion = Done;
        type Error = TestError;

        fn empty_prediction_input() -> Self::Error {
            TestError("empty prediction input".into())
        }

        fn fused_prediction_exhausted() -> Self::Error {
            TestError("prediction block exhausted".into())
        }

        fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error {
            TestError(format!("invalid commit {verified}/{available}"))
        }

        fn invalid_prediction_output(
            logits: usize,
            capture: usize,
            tokens: usize,
            expected: Option<usize>,
        ) -> Self::Error {
            TestError(format!(
                "invalid output {logits}/{capture}/{tokens}/{expected:?}"
            ))
        }

        fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error {
            TestError(format!("invalid fused capacity {requested}/{available}"))
        }

        fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
            Ok(value.0.len())
        }

        fn logits_row<'a>(
            value: &Self::Tensor,
            row: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            value
                .0
                .get(row)
                .copied()
                .ok_or_else(|| TestError("logits row is missing".into()))
        }

        fn tensor_row<'a>(
            value: &Self::Tensor,
            row: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Tensor, Self::Error> {
            Self::logits_row(value, row, ()).map(|value| Tensor(vec![value]))
        }

        fn tensor_prefix<'a>(
            value: &Self::Tensor,
            end: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor(value.0[..end].to_vec()))
        }

        fn token_range<'a>(
            value: &Self::Tensor,
            start: usize,
            end: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor(value.0[start..end].to_vec()))
        }

        fn token_prefix<'a>(
            value: &Self::Tensor,
            end: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Tensor, Self::Error> {
            Self::tensor_prefix(value, end, ())
        }

        fn target_tokens<'a>(
            tokens: &[u32],
            _: Self::Context<'a>,
        ) -> Result<Self::Tensor, Self::Error> {
            Ok(Tensor(tokens.iter().map(|&token| token as i32).collect()))
        }

        fn fused_logits_row<'a>(
            value: &Self::Tensor,
            row: usize,
            _: Self::Context<'a>,
        ) -> Result<Self::Logits, Self::Error> {
            Self::logits_row(value, row, ())
        }

        fn submit_verification_completion<'a>(
            output: &EmbeddedPredictionOutput<Self::Tensor>,
            inputs: &Self::Tensor,
            _: Self::Context<'a>,
        ) -> Result<Self::Completion, Self::Error> {
            Ok(Done {
                retained: Arc::from([
                    output.logits.clone(),
                    output.capture.clone(),
                    output.tokens.clone(),
                    inputs.clone(),
                ]),
            })
        }
    }

    #[derive(Clone, Debug)]
    struct Strategy {
        fused_rows: Option<usize>,
        corrupt_capture: bool,
        failure: Failure,
    }

    impl Strategy {
        fn output(tokens: Tensor, corrupt_capture: bool) -> EmbeddedPredictionOutput<Tensor> {
            let logits = Tensor(tokens.0.iter().map(|token| token + 100).collect());
            let mut capture = Tensor(tokens.0.iter().map(|token| token + 200).collect());
            if corrupt_capture {
                capture.0.pop();
            }
            EmbeddedPredictionOutput::new(logits, capture, tokens)
        }
    }

    impl EmbeddedPredictionStrategy<Mechanisms> for Strategy {
        type Input = Vec<u32>;
        type TargetCache = Cache;
        type PredictionCache = Vec<i32>;
        type Telemetry = ();

        fn proposal_capacity(&self) -> usize {
            3
        }

        fn checkpoint_target(cache: &Self::TargetCache) -> Result<Self::TargetCache, TestError> {
            Ok(cache.clone())
        }

        fn take_telemetry(&mut self) -> Result<Self::Telemetry, TestError> {
            Ok(())
        }

        fn take_verification_telemetry(
            &mut self,
            _: &mut EmbeddedPredictionOutput<Tensor>,
        ) -> Result<Self::Telemetry, TestError> {
            Ok(())
        }

        fn prefill_target<'a>(
            &mut self,
            input: Self::Input,
            cache: &mut Self::TargetCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<EmbeddedPredictionOutput<Tensor>, TestError> {
            let tokens = Tensor(input.into_iter().map(|token| token as i32).collect());
            cache.target.extend_from_slice(&tokens.0);
            Ok(Self::output(tokens, self.corrupt_capture))
        }

        fn verify_target<'a>(
            &mut self,
            tokens: &Tensor,
            cache: &mut Self::TargetCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<EmbeddedPredictionOutput<Tensor>, TestError> {
            cache.target.extend_from_slice(&tokens.0);
            Ok(Self::output(tokens.clone(), self.corrupt_capture))
        }

        fn seed_prediction_cache<'a>(
            &mut self,
            _: &EmbeddedPredictionOutput<Tensor>,
            tokens: &Tensor,
            cache: &mut Self::TargetCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<(), TestError> {
            cache.prediction.clone_from(&tokens.0);
            Ok(())
        }

        fn prediction_cache(&self, cache: &Self::TargetCache) -> Self::PredictionCache {
            cache.prediction.clone()
        }

        fn commit_prediction_cache(
            &self,
            cache: &mut Self::TargetCache,
            prediction: &Self::PredictionCache,
        ) -> Result<(), TestError> {
            cache.prediction.clone_from(prediction);
            Ok(())
        }

        fn restore_target_checkpoint<'a>(
            cache: &mut Self::TargetCache,
            checkpoint: &Self::TargetCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<(), TestError> {
            cache.clone_from(checkpoint);
            Ok(())
        }

        fn sequential_logits<'a>(
            &mut self,
            capture: &Tensor,
            last_token: u32,
            depth: usize,
            cache: &mut Self::PredictionCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<(i32, Tensor), TestError> {
            cache.push(last_token as i32);
            Ok((last_token as i32 + depth as i32, capture.clone()))
        }

        fn fused_logits<'a>(
            &mut self,
            _: &Tensor,
            _: u32,
            _: usize,
            _: &mut Self::PredictionCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<Option<Tensor>, TestError> {
            Ok(self
                .fused_rows
                .map(|rows| Tensor((0..rows as i32).collect())))
        }

        fn advance_prediction_cache<'a>(
            &mut self,
            _: &Tensor,
            tokens: &Tensor,
            cache: &mut Self::PredictionCache,
            _: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
        ) -> Result<(), TestError> {
            if self.failure == Failure::Advance {
                return Err(TestError("injected prediction advance failure".into()));
            }
            cache.extend_from_slice(&tokens.0);
            Ok(())
        }
    }

    fn cache() -> Cache {
        Cache {
            target: vec![9],
            prediction: vec![9],
        }
    }

    #[test]
    fn embedded_cache_envelope_owns_prediction_fork_commit_and_target_membership() {
        let mut cache = EmbeddedPredictionCache::new(7_i32, vec![1_i32]);
        let checkpoint = cache
            .checkpoint(|target| Ok::<_, TestError>(*target))
            .unwrap();
        let mut draft = cache.prediction_fork();
        draft.prediction_mut().push(2);
        cache.commit_prediction(&draft);
        assert_eq!(cache.prediction(), &[1, 2]);

        let active = cache.take_target().unwrap();
        let error = cache
            .restore(&checkpoint, |current, previous| {
                *current = *previous;
                Ok::<_, TestError>(())
            })
            .unwrap_err();
        assert!(matches!(
            error,
            EmbeddedPredictionCacheAccessError::Cache(
                EmbeddedPredictionCacheError::TargetPresenceChanged
            )
        ));
        assert_eq!(cache.prediction(), &[1, 2]);
        cache.restore_target(active);
        cache
            .restore(&checkpoint, |current, previous| {
                *current = *previous;
                Ok::<_, TestError>(())
            })
            .unwrap();
        assert_eq!(cache.target(), Some(&7));
        assert_eq!(cache.prediction(), &[1]);
    }

    #[test]
    fn sequential_partial_commit_replays_target_and_commits_prediction_state() {
        let mut strategy = Strategy {
            fused_rows: None,
            corrupt_capture: false,
            failure: Failure::None,
        };
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
        let mut cache = cache();
        let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
        let (_, state, _) = prefill.into_parts();
        let mut draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
        assert_eq!(executor.proposal_logits(&mut draft, 2, ()).unwrap(), 2);
        assert_eq!(executor.proposal_logits(&mut draft, 3, ()).unwrap(), 4);
        assert_eq!(
            executor.proposal_logits(&mut draft, 4, ()).unwrap_err().0,
            "prediction block exhausted"
        );
        let checkpoint = executor.checkpoint(&cache).unwrap();
        let submission = executor
            .submit_verification(&[2, 3, 4], &mut cache, ())
            .unwrap();
        assert_eq!(submission.completion.retained.len(), 4);
        let commit = executor
            .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
            .unwrap();
        let (_, replayed_tokens) = commit.into_parts();
        assert_eq!(replayed_tokens, 2);
        assert_eq!(cache.target, vec![9, 1, 2, 2, 3]);
        assert_eq!(cache.prediction, vec![1, 2, 2, 3, 3]);
    }

    #[test]
    fn prefill_geometry_failure_restores_target_and_prediction_cache() {
        let mut strategy = Strategy {
            fused_rows: None,
            corrupt_capture: true,
            failure: Failure::None,
        };
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
        let mut cache = cache();
        let checkpoint = cache.clone();
        let error = executor.prefill(vec![1, 2], &mut cache, ()).err().unwrap();
        assert!(error.0.starts_with("invalid output"));
        assert_eq!(cache, checkpoint);
    }

    #[test]
    fn fused_capacity_is_rejected_before_any_proposal_row() {
        let mut strategy = Strategy {
            fused_rows: Some(1),
            corrupt_capture: false,
            failure: Failure::None,
        };
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
        let mut cache = cache();
        let prefill = executor.prefill(vec![1], &mut cache, ()).unwrap();
        let (_, state, _) = prefill.into_parts();
        let error = executor.begin_proposal(&state, 1, 2, ()).err().unwrap();
        assert_eq!(error.0, "invalid fused capacity 2/1");
    }

    #[test]
    fn commit_failure_restores_exact_preverification_checkpoint() {
        let mut strategy = Strategy {
            fused_rows: None,
            corrupt_capture: false,
            failure: Failure::Advance,
        };
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
        let mut cache = cache();
        let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
        let (_, state, _) = prefill.into_parts();
        let draft = executor.begin_proposal(&state, 2, 2, ()).unwrap();
        let checkpoint = cache.clone();
        let submission = executor
            .submit_verification(&[2, 3, 4], &mut cache, ())
            .unwrap();
        let error = executor
            .commit_verification(submission.output, draft, &mut cache, &checkpoint, 2, ())
            .err()
            .unwrap();
        assert_eq!(error.0, "injected prediction advance failure");
        assert_eq!(cache, checkpoint);
    }

    #[test]
    fn production_observers_reach_causal_embedded_boundaries_and_can_intervene() {
        struct TensorTrace(Arc<Mutex<Vec<String>>>);

        impl eredu_runtime::ActivationObserver<Tensor, TestError> for TensorTrace {
            fn observe(&mut self, path: &str, _: &Tensor) -> Result<(), TestError> {
                self.0.lock().unwrap().push(path.into());
                Ok(())
            }

            fn intervene(
                &mut self,
                path: &str,
                value: &Tensor,
            ) -> Result<Option<Tensor>, TestError> {
                Ok((path == EMBEDDED_PREDICTION_OUTPUT_PATH).then(|| {
                    let mut replacement = value.clone();
                    replacement.0.fill(77);
                    replacement
                }))
            }
        }

        struct LogitsTrace(Arc<Mutex<Vec<String>>>);

        impl eredu_runtime::ActivationObserver<i32, TestError> for LogitsTrace {
            fn observe(&mut self, path: &str, _: &i32) -> Result<(), TestError> {
                self.0.lock().unwrap().push(path.into());
                Ok(())
            }

            fn intervene(&mut self, _: &str, value: &i32) -> Result<Option<i32>, TestError> {
                Ok(Some(value + 100))
            }
        }

        let paths = Arc::new(Mutex::new(Vec::new()));
        let observers = EmbeddedPredictionObservers::new(
            TensorTrace(Arc::clone(&paths)),
            LogitsTrace(Arc::clone(&paths)),
        );
        let mut strategy = Strategy {
            fused_rows: None,
            corrupt_capture: false,
            failure: Failure::None,
        };
        let mut executor =
            EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(&mut strategy, observers);
        let mut cache = cache();
        let prefill = executor.prefill(vec![1, 2], &mut cache, ()).unwrap();
        let (_, state, _) = prefill.into_parts();
        let mut draft = executor.begin_proposal(&state, 2, 1, ()).unwrap();
        assert_eq!(executor.proposal_logits(&mut draft, 2, ()).unwrap(), 102);
        let _ = executor
            .submit_verification(&[2, 3], &mut cache, ())
            .unwrap();
        assert_eq!(
            *paths.lock().unwrap(),
            [
                EMBEDDED_TARGET_CAPTURE_PATH,
                EMBEDDED_PREDICTION_OUTPUT_PATH,
                EMBEDDED_PROPOSAL_LOGITS_PATH,
                EMBEDDED_VERIFICATION_LOGITS_PATH,
                EMBEDDED_TARGET_CAPTURE_PATH,
            ]
        );
    }
}
