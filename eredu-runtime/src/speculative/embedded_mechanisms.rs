//! Native tensor, state and input mechanisms for portable embedded prediction.

use crate::capture::SpeculativeCaptureMechanisms as _;

use eredu_core::{
    speculative::{GenerationExecutorTypes, SpeculativeControlError},
    BoundedCompletion, SpeculativeTelemetry,
};

/// Tensor, token, transfer, and completion mechanisms needed by embedded prediction.
///
/// Implementations contain no architecture identity, prediction depth, capture path, or replay
/// policy. The selected execution context determines where each operation runs.
pub trait SpeculativeTensorMechanisms: 'static {
    /// Complete immutable score-row backing allowance for settled handoffs.
    fn control_logits_estimate(
        _logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    /// Source-free mode agreement through the retained request transport.
    fn coordinate_decoding_control<'a>(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        Self::Capture::coordinate_decoding_control(&context, local)
    }
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
        context: Self::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        Self::Capture::coordinate_speculative_step(&context, local)
    }

    /// Agrees host readiness through retained native session transport.
    fn agree_text_preparation<'a>(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        Self::Capture::agree_text_preparation(&context, stage, status)
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

    /// Native transforms and retained transport for the portable speculative observer.
    type Capture: for<'a> crate::capture::SpeculativeCaptureMechanisms<
            Context<'a> = Self::Context<'a>,
            Provider: crate::capture::CaptureBackendProvider<Tensor = Self::Tensor>,
            Error = Self::Error,
        > + 'static;

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

/// Backend-owned input lowering for one statically typed replicated target.
///
/// Plain text and composite admission have different borrowed input forms.  This mechanism owns
/// only that lowering step; the architecture strategy retains prefill/decode, capture validation,
/// extension state, proposal ordering, replay, and commit policy.
pub trait ReplicatedPredictionInput<A, B, S, E>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    A: crate::LayeredArchitecture<B, S>,
{
    /// Owned request input accepted by the public backend.
    type Input;

    /// Lends successive causal chunks, preserving the complete request identity.
    /// Each callback must finish before the next chunk's native input is created.
    /// Its returned forward context supplies architecture-owned continuation
    /// products; lowering completes those retained tensors before reusing them.
    fn with_prefill_chunks(
        &mut self,
        input: Self::Input,
        maximum_chunk_tokens: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        operation: impl for<'a> FnMut(
            A::Input<'a>,
            B::Tensor,
            Option<&crate::PreparedInputCacheIdentity>,
        ) -> Result<A::ForwardContext, E>,
    ) -> Result<(), E>;

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
pub trait ReplicatedPredictionNative<B, S, M>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    M: SpeculativeTensorMechanisms<Tensor = B::Tensor>,
{
    /// Optional backend component telemetry.
    type Telemetry: SpeculativeTelemetry;
    /// Fixed scheduler-facing type bundle used after architecture pairing.
    type ExecutorTypes: for<'a> GenerationExecutorTypes<
        Context<'a> = M::Context<'a>,
        Logits = M::Logits,
        Completion = M::Completion,
        Telemetry = Self::Telemetry,
        Error = M::Error,
    >;
    /// Returns the ordinary target tensor context selected by composition.
    fn target_context<'a>(context: M::Context<'a>) -> &'a <B::Tensor as eredu_nn::Tensor>::Context;
    /// Portable state lifecycle paired with the selected target.
    type TargetState: super::PredictionTargetLifecycle<S, <B::Tensor as eredu_nn::Tensor>::Context>;
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

use eredu_core::speculative::embedded::EmbeddedPredictionOperators;
use eredu_nn::Tensor as _;

/// Portable erasure profile derived from the exact embedded tensor operations.
pub struct PortableEmbeddedExecutor<M>(std::marker::PhantomData<fn() -> M>);

impl<M: EmbeddedPredictionOperators> GenerationExecutorTypes for PortableEmbeddedExecutor<M> {
    type Logits = M::Logits;
    type Context<'a> = crate::capture::SpeculativeContextFor<'a, M::Capture>;
    type Completion = M::Completion;
    type Telemetry = M::Telemetry;
    type Error = M::Error;

    fn erased_type_mismatch(value: &'static str) -> Self::Error {
        M::invalid(format!(
            "embedded executor carried a mismatched erased {value}"
        ))
    }
}

impl<M> SpeculativeTensorMechanisms for M
where
    M: EmbeddedPredictionOperators,
    M::Capture: for<'a> crate::capture::SpeculativeCaptureMechanisms<
        Context<'a> = crate::capture::SpeculativeContextFor<'a, M::Capture>,
        Provider: crate::capture::CaptureBackendProvider<Tensor = M::Tensor>,
        Error = M::Error,
    >,
{
    type Tensor = M::Tensor;
    type Logits = M::Logits;
    type Context<'a> = crate::capture::SpeculativeContextFor<'a, M::Capture>;
    type Completion = M::Completion;
    type Capture = M::Capture;
    type Error = M::Error;

    fn control_logits_estimate(
        logits: &M::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::logits_estimate(logits)
    }

    fn observation_error(message: &'static str) -> M::Error {
        M::invalid(message)
    }

    fn control_tensor_estimate(
        value: &M::Tensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        M::tensor_estimate(value)
    }

    fn control_tensor_snapshot<'a>(
        value: &M::Tensor,
        context: Self::Context<'a>,
    ) -> Result<Option<M::Tensor>, SpeculativeControlError> {
        M::copy_tensor(value, context.target())
            .map(Some)
            .map_err(SpeculativeControlError::backend)
    }

    fn empty_prediction_input() -> M::Error {
        M::invalid("embedded prediction input must contain at least one token")
    }

    fn fused_prediction_exhausted() -> M::Error {
        M::invalid("fused embedded prediction proposal block is exhausted")
    }

    fn invalid_prediction_commit(verified: usize, available: usize) -> M::Error {
        M::invalid(format!(
            "cannot commit {verified} embedded-prediction inputs from a block of {available}"
        ))
    }

    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> M::Error {
        M::invalid(format!(
            "embedded prediction output lengths disagree: logits={logits}, capture={capture}, tokens={tokens}, expected={expected:?}"
        ))
    }

    fn invalid_fused_capacity(requested: usize, available: usize) -> M::Error {
        M::invalid(format!(
            "fused embedded prediction block has {available} rows, but {requested} were requested"
        ))
    }

    fn sequence_len(value: &M::Tensor) -> Result<usize, M::Error> {
        M::sequence_len(value)
    }

    fn logits_row<'a>(
        value: &M::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Logits, M::Error> {
        M::logits_row(value, row, context.target())
    }

    fn tensor_row<'a>(
        value: &M::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Tensor, M::Error> {
        M::tensor_row(value, row, context.target())
    }

    fn tensor_prefix<'a>(
        value: &M::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Tensor, M::Error> {
        M::tensor_prefix(value, end, context.target())
    }

    fn token_range<'a>(
        value: &M::Tensor,
        start: usize,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Tensor, M::Error> {
        M::token_range(value, start, end, context.target())
    }

    fn token_prefix<'a>(
        value: &M::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Tensor, M::Error> {
        M::token_range(value, 0, end, context.target())
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<M::Tensor, M::Error> {
        M::target_tokens(tokens, context.crosses_devices().then(|| context.target()))
    }

    fn fused_logits_row<'a>(
        value: &M::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<M::Logits, M::Error> {
        M::logits_row(value, row, context.draft())
    }

    fn submit_verification_completion<'a>(
        output: &EmbeddedPredictionOutput<M::Tensor>,
        inputs: &M::Tensor,
        _: Self::Context<'a>,
    ) -> Result<M::Completion, M::Error> {
        M::submit_verification(&[output.logits(), output.capture(), output.tokens(), inputs])
    }
}

impl<B, S, M, N> ReplicatedPredictionNative<B, S, M> for N
where
    B: eredu_nn::NeuralBackend<Tensor: Clone>,
    M: EmbeddedPredictionOperators<Tensor = B::Tensor>
        + for<'a> SpeculativeTensorMechanisms<
            Tensor = B::Tensor,
            Context<'a> = crate::capture::SpeculativeContextFor<
                'a,
                <M as EmbeddedPredictionOperators>::Capture,
            >,
            Logits = <M as EmbeddedPredictionOperators>::Logits,
            Error = <M as EmbeddedPredictionOperators>::Error,
            Completion = <M as EmbeddedPredictionOperators>::Completion,
        >,
    N: eredu_core::checkpoint_mechanisms::prediction::NativePredictionMaterialization<B>,
    N::ModelCacheStorage: eredu_core::cache::mechanisms::CacheTensorStorage<
        Queue = <B::Tensor as eredu_nn::Tensor>::Context,
    >,
    S: crate::state::PortableTextState<B, N::ModelCacheStorage>,
{
    type Telemetry = M::Telemetry;
    type ExecutorTypes = PortableEmbeddedExecutor<M>;
    type TargetState = super::PortablePredictionTargetState<B, N::ModelCacheStorage, N::Error>;

    fn target_context<'a>(
        context: <M as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> &'a <B::Tensor as eredu_nn::Tensor>::Context {
        context.target()
    }

    fn token(
        token: u32,
        queue: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, <M as SpeculativeTensorMechanisms>::Error> {
        M::token(token, queue)
    }

    fn shape(tensor: &B::Tensor) -> &[i32] {
        tensor.shape()
    }

    fn validate<T>(
        operation: impl FnOnce() -> Result<T, <M as SpeculativeTensorMechanisms>::Error>,
    ) -> Result<T, <M as SpeculativeTensorMechanisms>::Error> {
        M::validate(operation)
    }

    fn session_error(error: impl std::fmt::Display) -> <M as SpeculativeTensorMechanisms>::Error {
        M::invalid(error.to_string())
    }

    fn session_failure(
        error: eredu_core::BackendFailure,
    ) -> <M as SpeculativeTensorMechanisms>::Error {
        M::failure(error)
    }

    fn take_telemetry() -> Result<Self::Telemetry, <M as SpeculativeTensorMechanisms>::Error> {
        M::take_telemetry()
    }
}
