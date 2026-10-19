//! Whole-session MLX speculative generation capability.

use eredu_architectures::speculative_execution::{
    EmbeddedPredictionOutput, ReplicatedPredictionInput, ReplicatedPredictionNative,
    SpeculativeTensorMechanisms,
};
use eredu_core::speculative::{DynGenerationExecutor, GenerationExecutorTypes};
use eredu_core::{
    generation::{GenerationCancellationToken, GenerationRequestConfig, SemanticEvent},
    GenerationSequence, ModelRuntime, PreparedSpeculativeLane, SpeculativeCallbackPublisher,
    SpeculativeCapability, SpeculativeDraft, SpeculativeExecutor, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeOutputRuntime, SpeculativeSampling,
    SpeculativeSemanticConstraint, SpeculativeSemanticState, SpeculativeTokenFilterController,
};
use eredu_runtime::{ConstrainedSampler, SpeculativeSampler};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use super::{
    session::MlxTextSampler,
    speculative::{
        scheduler::{component_timing_enabled, MlxSpeculativeRuntime},
        MlxAssistantPreparationVisitor, MlxDrafter, MlxSpeculativeSampling,
        SpeculativeExecutionStreams,
    },
    MlxBackend, MlxModelInput,
};
use crate::backend::error::Error;
use crate::backend::nn::{shared::MlxNeuralBackend, tensor::TokenValidationScope};
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::backend::runtime::media::input;
use crate::MlxTensor;

pub(crate) struct MlxEmbeddedPredictionMechanisms;

pub(crate) struct MlxGenerationExecutorTypes;

pub(crate) struct MlxSharedGenerationExecutorTypes;
impl GenerationExecutorTypes for MlxSharedGenerationExecutorTypes {
    type Input = eredu_runtime::speculative::shared_backbone::SharedBackboneInput<
        MlxModelInput,
        eredu_runtime::speculative::shared_backbone::SharedBackboneDraftOptions,
        crate::backend::random::RandomState,
    >;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
    type Error = Exception;
    fn erased_type_mismatch(value: &'static str) -> Exception {
        Exception::custom(format!(
            "shared generation carried a mismatched erased {value}"
        ))
    }
}
impl eredu_core::speculative::GenerationExecutorCacheFactory<MlxSharedGenerationExecutorTypes>
    for eredu_runtime::speculative::shared_backbone::SharedBackboneExecutor<
        '_,
        super::speculative::autoregressive::MlxAutoregressiveMechanisms,
    >
{
    fn new_cache<'a>(&mut self, context: Self::Context<'a>) -> Result<Self::Cache, Exception>
    where
        Self: 'a,
    {
        eredu_runtime::speculative::shared_backbone::SharedBackboneExecutor::new_cache(
            self, context,
        )
    }
    fn bind_context<'a>(
        context: <MlxSharedGenerationExecutorTypes as GenerationExecutorTypes>::Context<'a>,
    ) -> Self::Context<'a>
    where
        Self: 'a,
    {
        context
    }
}

pub(crate) trait MlxEmbeddedExecutorContinuation {
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynGenerationExecutor<'_, MlxGenerationExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>;
}

impl GenerationExecutorTypes for MlxGenerationExecutorTypes {
    type Input = MlxModelInput;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
    type Error = Exception;

    fn erased_type_mismatch(value: &'static str) -> Self::Error {
        Exception::custom(format!(
            "embedded executor carried a mismatched erased {value}"
        ))
    }
}

pub(crate) struct MlxTextPredictionInput;

impl<A, S> ReplicatedPredictionInput<A, MlxNeuralBackend, S, Exception> for MlxTextPredictionInput
where
    S: eredu_runtime::RuntimeState<MlxNeuralBackend>,
    A: eredu_runtime::ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    type Input = MlxModelInput;

    fn with_prefill_chunks(
        &mut self,
        input: Self::Input,
        maximum_chunk_tokens: usize,
        context: &Stream,
        mut operation: impl for<'a> FnMut(
            A::Input<'a>,
            MlxTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<A::ForwardContext, Exception>,
    ) -> Result<(), Exception> {
        if maximum_chunk_tokens == 0 {
            return Err(Exception::custom(
                "prediction prefill chunk size must be positive",
            ));
        }
        input.with_borrowed(|input| {
            let tokens = input::text_token_ids(input, context)
                .map(MlxTensor::from_array)
                .map_err(Exception::from_source)?;
            let sequence = tokens.as_array().shape()[1] as usize;
            if sequence > maximum_chunk_tokens && !A::supports_chunked_prefill() {
                return Err(Exception::custom(
                    "selected architecture cannot split prediction prefill input",
                ));
            }
            for start in (0..sequence).step_by(maximum_chunk_tokens) {
                let end = start.saturating_add(maximum_chunk_tokens).min(sequence);
                let chunk = if start == 0 && end == sequence {
                    tokens.clone()
                } else {
                    MlxTensor::from_array(
                        tokens
                            .as_array()
                            .try_index_device((.., start as i32..end as i32), context)?,
                    )
                };
                operation(
                    A::text_input(&chunk, None),
                    chunk.clone(),
                    input.cache_identity(),
                )?;
            }
            Ok(())
        })
    }

    fn with_decode<R>(
        &mut self,
        tokens: &MlxTensor,
        _context: &Stream,
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        operation(A::text_input(tokens, None))
    }
}

impl<A, S> ReplicatedPredictionNative<A, MlxNeuralBackend, S, MlxEmbeddedPredictionMechanisms>
    for super::replicated_text::MlxEmbeddedPredictionMaterializer
where
    S: super::replicated_text::MlxStateMechanisms,
    A: eredu_runtime::LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    type Input = MlxModelInput;
    type Telemetry = super::speculative::scheduler::SpeculativeComponentTimings;
    type ExecutorTypes = MlxGenerationExecutorTypes;

    fn executor_context<'a>(
        context: <Self::ExecutorTypes as GenerationExecutorTypes>::Context<'a>,
    ) -> <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Context<'a> {
        context
    }

    fn target_context<'a>(
        context: <MlxEmbeddedPredictionMechanisms as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> &'a Stream {
        context.target()
    }

    fn checkpoint(state: &S) -> Result<S, Exception> {
        state.deep_checkpoint()
    }

    fn state_memory_bounds(state: &S, additional: u64) -> (Option<u64>, Option<u64>) {
        let retained = state.isolated_snapshot_estimate().map(|e| e.retained_bytes);
        (
            retained.and_then(|n| n.checked_add(state.isolated_snapshot_growth(0)?)),
            retained.and_then(|n| n.checked_add(state.isolated_snapshot_growth(additional)?)),
        )
    }

    fn control_state_estimate(
        state: &S,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.isolated_snapshot_estimate()
    }

    fn control_state_snapshot(
        state: &S,
        stream: &Stream,
    ) -> Result<Option<S>, eredu_core::speculative::SpeculativeControlError> {
        if !state.supports_isolated_snapshot() {
            return Ok(None);
        }
        let copy = (|| {
            super::speculative::state_snapshot::settle(state.retained_arrays())?;
            let copy = state.isolated_snapshot(stream)?;
            super::speculative::state_snapshot::settle(copy.retained_arrays())?;
            Ok::<_, Exception>(copy)
        })()
        .map_err(eredu_core::speculative::SpeculativeControlError::backend)?;
        Ok(Some(copy))
    }

    fn restore(state: &mut S, checkpoint: &S, context: &Stream) -> Result<(), Exception> {
        state.restore_checkpoint(checkpoint, context)
    }

    fn generation(state: &S) -> Result<u64, Exception> {
        u64::try_from(state.offset())
            .map_err(|_| Exception::custom("target capture generation is negative"))
    }

    fn token(token: u32, _context: &Stream) -> Result<MlxTensor, Exception> {
        let token = i32::try_from(token)
            .map_err(|_| Exception::custom("prediction token exceeds int32 storage"))?;
        Ok(MlxTensor::from_array(Array::from_slice(&[token], &[1, 1])))
    }

    fn shape(tensor: &MlxTensor) -> &[i32] {
        tensor.as_array().shape()
    }

    fn validate<T>(operation: impl FnOnce() -> Result<T, Exception>) -> Result<T, Exception> {
        let scope = TokenValidationScope::begin()?;
        let output = operation()?;
        let validations = scope.finish();
        if !validations.is_empty() {
            async_eval_with_event(validations.arrays())?.synchronize()?;
            validations.validate_completed()?;
        }
        Ok(output)
    }

    fn session_error(error: impl std::fmt::Display) -> Exception {
        Exception::custom(error.to_string())
    }

    fn session_failure(error: eredu_core::BackendFailure) -> Exception {
        Exception::from_source(error)
    }

    fn take_telemetry() -> Result<Self::Telemetry, Exception> {
        Ok(Self::Telemetry::default())
    }
}

impl SpeculativeTensorMechanisms for MlxEmbeddedPredictionMechanisms {
    fn control_logits_estimate(
        logits: &Self::Logits,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::speculative::state_snapshot::estimate_array(logits)
    }
    type Tensor = MlxTensor;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Error = Exception;

    fn observation_error(message: &'static str) -> Self::Error {
        Exception::custom(message)
    }

    fn control_tensor_estimate(
        value: &MlxTensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::speculative::state_snapshot::estimate_array(value.as_array())
    }

    fn control_tensor_snapshot<'a>(
        value: &MlxTensor,
        context: Self::Context<'a>,
    ) -> Result<Option<MlxTensor>, eredu_core::speculative::SpeculativeControlError> {
        super::speculative::state_snapshot::copy_array(value.as_array(), context.target())
            .map(|array| Some(MlxTensor::from_array(array)))
            .map_err(eredu_core::speculative::SpeculativeControlError::backend)
    }

    fn coordinate_decoding_control<'a>(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        context.coordinate_decoding_control(local)
    }

    fn coordinate_speculative_step<'a>(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'a>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        context.coordinate_speculative_step(local)
    }

    fn agree_text_preparation<'a>(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'a>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        context.agree_text_preparation(stage, status)
    }

    fn activation_observer<'a>(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        context: Self::Context<'a>,
    ) -> Result<
        Option<
            Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Exception>>,
        >,
        eredu_core::speculative::SpeculativeControlError,
    > {
        match context.capture_binding() {
            Some(binding) => binding.observer(plan, request, context.target()),
            None => super::session::bounded_capture::speculative_capture(
                plan,
                request,
                context.target(),
            ),
        }
    }

    fn empty_prediction_input() -> Self::Error {
        Exception::custom("embedded prediction input must contain at least one token")
    }

    fn fused_prediction_exhausted() -> Self::Error {
        Exception::custom("fused embedded prediction proposal block is exhausted")
    }

    fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error {
        Exception::custom(format!(
            "cannot commit {verified} embedded-prediction inputs from a block of {available}"
        ))
    }

    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Self::Error {
        Exception::custom(format!(
            "embedded prediction output lengths disagree: logits={logits}, capture={capture}, tokens={tokens}, expected={expected:?}"
        ))
    }

    fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error {
        Exception::custom(format!(
            "fused embedded prediction block has {available} rows, but {requested} were requested"
        ))
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        usize::try_from(value.as_array().dim(1))
            .map_err(|_| Exception::custom("prediction sequence length exceeds usize"))
    }

    fn logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| Exception::custom("prediction row exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., row, ..), context.target())
    }

    fn tensor_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| Exception::custom("prediction row exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., row..row + 1, ..), context.target())
            .map(MlxTensor::from_array)
    }

    fn tensor_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let end =
            i32::try_from(end).map_err(|_| Exception::custom("prediction prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end, ..), context.target())
            .map(MlxTensor::from_array)
    }

    fn token_range<'a>(
        value: &Self::Tensor,
        start: usize,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let start = i32::try_from(start)
            .map_err(|_| Exception::custom("prediction token start exceeds i32"))?;
        let end = i32::try_from(end)
            .map_err(|_| Exception::custom("prediction token end exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., start..end), context.target())
            .map(MlxTensor::from_array)
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let end = i32::try_from(end)
            .map_err(|_| Exception::custom("prediction token prefix exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., ..end), context.target())
            .map(MlxTensor::from_array)
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        context: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let width = i32::try_from(tokens.len())
            .map_err(|_| Exception::custom("prediction token count exceeds i32"))?;
        let mut value = Array::from_slice(tokens, &[1, width]);
        if context.crosses_devices() {
            value = value.copy(context.target())?;
        }
        Ok(MlxTensor::from_array(value))
    }

    fn fused_logits_row<'a>(
        value: &Self::Tensor,
        row: usize,
        context: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        let row =
            i32::try_from(row).map_err(|_| Exception::custom("prediction row exceeds i32"))?;
        value
            .as_array()
            .try_index_device((.., row, ..), context.draft())
    }

    fn submit_verification_completion<'a>(
        output: &EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _context: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        super::speculative::MlxSpeculativeCompletion::submit([
            output.logits().as_array(),
            output.capture().as_array(),
            output.tokens().as_array(),
            inputs.as_array(),
        ])
    }
}

impl<'world> SpeculativeGenerationBackend for MlxBackend<'world> {
    type Drafter = MlxDrafter;

    fn speculative_capability(runtime: &ModelRuntime<Self>) -> SpeculativeCapability {
        runtime.session().speculative_capability()
    }

    fn speculative_activation_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        runtime.session().speculative_activation_discovery()
    }

    fn validate_speculative_activations(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        plan.validate(&Self::speculative_activation_discovery(runtime)?)?;
        eredu_runtime::intervention::preflight(
            plan.captures(),
            plan.interventions(),
            &super::session::intervention::NativeInterventionEstimator,
        )
    }

    fn validate_speculative_capture(
        runtime: &ModelRuntime<Self>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        super::speculative::validate_control_capture(plan)?;
        <Self as eredu_core::TextGenerationBackend>::validate_text_capture(runtime, plan)
    }

    fn speculative_intervention_discovery(
        runtime: &ModelRuntime<Self>,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        let mut discovery =
            <Self as eredu_core::TextGenerationBackend>::intervention_discovery(runtime)?;
        discovery
            .points
            .retain(|point| point.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
        Ok(discovery)
    }
    fn validate_speculative_interventions(
        runtime: &ModelRuntime<Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        Self::validate_speculative_capture(runtime, capture)?;
        let discovery = Self::speculative_intervention_discovery(runtime)?;
        eredu_runtime::intervention::validate_session(
            capture,
            plan,
            &discovery,
            &super::session::intervention::NativeInterventionEstimator,
        )
    }

    fn with_speculative_execution<C, V>(
        runtime: &mut ModelRuntime<Self>,
        request: SpeculativeGenerationBatchRequest<'_, Self, Self::Drafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        MlxSpeculativeSession::new(runtime).with_execution(request, visitor)
    }
}

struct MlxSpeculativeSession<'runtime, 'world> {
    runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>,
}

struct MlxSpeculativeLaneRuntime<'a, C, I = MlxModelInput> {
    input: I,
    config: GenerationRequestConfig,
    prng_key: Option<Array>,
    refinement_seed: u64,
    sampler: MlxPreparedSampler<C>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
    control_handle: Option<eredu_core::execution_control::GenerationControlHandle>,
}

impl<'a, C, I> MlxSpeculativeLaneRuntime<'a, C, I> {
    fn map_input<J>(
        self,
        prepare: impl FnOnce(I, u64) -> Result<J, Exception>,
    ) -> Result<MlxSpeculativeLaneRuntime<'a, C, J>, Exception> {
        Ok(MlxSpeculativeLaneRuntime {
            input: prepare(self.input, self.refinement_seed)?,
            config: self.config,
            prng_key: self.prng_key,
            refinement_seed: self.refinement_seed,
            sampler: self.sampler,
            semantic: self.semantic,
            cancellation: self.cancellation,
            on_event: self.on_event,
            control_handle: self.control_handle,
        })
    }
}

type MlxPreparedSampler<C> = ConstrainedSampler<MlxTextSampler, C>;

fn validate_lane_proposal_capacity(
    config: &GenerationRequestConfig,
    proposal_capacity: usize,
) -> Result<(), Error> {
    if proposal_capacity == 0 && config.proposal_capacity() > 0 {
        return Err(eredu_core::generation::GenerationError::NoBackendDraftCapacity.into());
    }
    if config.proposal_capacity() > proposal_capacity {
        return Err(Error::Speculative(format!(
            "lane requests {} draft tokens, but the selected speculative realization admits at most {proposal_capacity}",
            config.proposal_capacity()
        )));
    }
    Ok(())
}

struct MlxExternalBatchRunner<'target, 'lane, C, V> {
    target: &'target mut (dyn super::replicated_text::ErasedReplicatedTextExecutable + 'target),
    lanes: Vec<MlxSpeculativeLaneRuntime<'lane, C>>,
    streams: SpeculativeExecutionStreams<'target>,
    visitor: V,
    capture: eredu_architectures::composite_execution::ExternalPredictionCaptureRequest,
    selected: eredu_runtime::SelectedSpeculativeRealization,
}

struct MlxExternalBatchVisitor<'target, 'lane, C, V> {
    runner: MlxExternalBatchRunner<'target, 'lane, C, V>,
}

impl<'target, 'lane, C, V>
    eredu_architectures::MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor>
    for MlxExternalBatchVisitor<'target, 'lane, C, V>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    type Output = Result<SpeculativeGenerationBatchOutput, Error>;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        assistant: &mut super::speculative::MlxExternalAssistant<A>,
    ) -> Self::Output {
        let runner = self.runner;
        let target = runner.target.external_prediction_mut().ok_or_else(|| {
            Error::ArchitectureModel(
                "selected target has no external-assistant prediction capability".into(),
            )
        })?;
        let caches = (0..runner.lanes.len())
            .map(|_| {
                target
                    .prepare_external_prediction_target_cache()
                    .map(|cache| {
                        super::speculative::external::MlxExternalPredictionCache::new(
                            cache,
                            runner.selected.clone(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>();
        let caches = runner.streams.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Sampling,
            caches,
            |error| Error::Exception(Exception::from_source(error)),
        )?;
        A::visit_executor::<
            crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
            _,
        >(
            target,
            assistant,
            runner.capture,
            MlxExternalExecutorRunner {
                lanes: runner.lanes,
                caches,
                streams: runner.streams,
                visitor: runner.visitor,
            },
        )
    }
}

struct MlxExternalExecutorRunner<'lane, 'streams, C, V> {
    lanes: Vec<MlxSpeculativeLaneRuntime<'lane, C>>,
    caches: Vec<super::speculative::external::MlxExternalPredictionCache>,
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: V,
}

impl<'lane, 'streams, A, C, V>
    eredu_architectures::ExternalAssistantExecutorVisitor<
        A,
        crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
    > for MlxExternalExecutorRunner<'lane, 'streams, C, V>
where
    A: eredu_architectures::ExternalAssistantArchitecture,
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    type Output = Result<SpeculativeGenerationBatchOutput, Error>;

    fn execute<'run, E>(self, backend: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: SpeculativeExecutor<
                Input = MlxModelInput,
                Cache = super::speculative::external::MlxExternalPredictionCache,
                Logits = Array,
                Context<'run> = SpeculativeExecutionStreams<'run>,
                Completion = super::speculative::MlxSpeculativeCompletion,
                Telemetry = crate::composition::mlx::speculative::scheduler::SpeculativeComponentTimings,
                Error = Exception,
            > + 'run,
    {
        run_speculative_batch(
            backend,
            self.lanes,
            self.caches,
            Ok,
            self.streams,
            self.visitor,
        )
        .map_err(|error| Error::Exception(Exception::from_source(error)))
    }
}

fn run_speculative_batch<'run, 'lane, 'streams, B, C, S>(
    backend: &'run mut B,
    lanes: Vec<MlxSpeculativeLaneRuntime<'lane, C, B::Input>>,
    caches: Vec<B::Cache>,
    wrap_sampler: impl Fn(MlxPreparedSampler<C>) -> Result<S, Exception>,
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: impl SpeculativeGenerationVisitor,
) -> Result<SpeculativeGenerationBatchOutput, Exception>
where
    'lane: 'run,
    'streams: 'run,
    B: MlxSpeculativeRuntime<'run>,
    C: SpeculativeTokenFilterController + 'run,
    S: SpeculativeSampler<MlxSamplingBackend> + Clone + 'run,
{
    let topology = streams.topology();
    let component_timings_collected = component_timing_enabled() && backend.supports_telemetry();
    let prepared = (|| {
        if caches.len() != lanes.len() {
            return Err(Exception::custom(format!(
                "speculative cache has {} lanes but the request has {} lanes",
                caches.len(),
                lanes.len()
            )));
        }
        let mut prepared = Vec::with_capacity(lanes.len());
        for (lane, cache) in lanes.into_iter().zip(caches) {
            let MlxSpeculativeLaneRuntime {
                input,
                config,
                prng_key,
                refinement_seed: _,
                sampler,
                semantic,
                cancellation,
                on_event,
                control_handle,
            } = lane;
            let sampling = MlxSpeculativeSampling::new(wrap_sampler(sampler)?);
            let randomness =
                <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
                    prng_key,
                    config.temperature,
                    streams,
                )?
                .with_verification_policy(backend.verification_policy());
            let sequence =
                GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied());
            prepared.push(PreparedSpeculativeLane::new(
                eredu_core::speculative::GenerationCache::Owned(cache),
                input,
                config,
                SpeculativeOutputRuntime::new(
                    sampling,
                    sequence,
                    SpeculativeSemanticConstraint::semantic(semantic),
                    SpeculativeCallbackPublisher::semantic(on_event),
                    cancellation,
                )
                .with_control_handle(control_handle),
                randomness,
            ));
        }
        Ok(prepared)
    })();
    let prepared = streams.finish_preparation(
        eredu_core::run_preparation::TextPreparationStage::Delivery,
        prepared,
        Exception::from_source,
    )?;
    visitor
        .run(
            backend,
            prepared,
            topology,
            streams.is_split(),
            component_timings_collected,
            streams,
        )
        .map_err(Exception::from_source)
}

impl<'runtime, 'world> MlxSpeculativeSession<'runtime, 'world> {
    fn new(runtime: &'runtime mut ModelRuntime<MlxBackend<'world>>) -> Self {
        Self { runtime }
    }

    fn prepare_mlx_speculative_sampling<C>(
        generation: eredu_core::TextGenerationConfig,
        constraint: C,
    ) -> Result<(Option<Array>, MlxPreparedSampler<C>), Error>
    where
        C: SpeculativeTokenFilterController,
    {
        let resolved = generation.sampling();
        let sampler = MlxTextSampler::from_config(&generation)
            .map_err(|error| Error::Exception(Exception::from_source(error)))?;
        let prng_key = (resolved.temperature != 0.0)
            .then(|| safemlx::random::key(generation.seed()))
            .transpose()?;
        Ok((prng_key, ConstrainedSampler::new(sampler, constraint)))
    }

    fn prepare_speculative_batch_lanes<'a, C>(
        lanes: Vec<SpeculativeGenerationLane<'a, MlxBackend<'world>, C>>,
        proposal_capacity: usize,
    ) -> Result<Vec<MlxSpeculativeLaneRuntime<'a, C>>, Error>
    where
        C: SpeculativeTokenFilterController,
    {
        for lane in &lanes {
            validate_lane_proposal_capacity(lane.config(), proposal_capacity)?;
        }
        let mut prepared_lanes = Vec::with_capacity(lanes.len());
        for mut lane in lanes {
            let prompt = lane.take_prompt();
            let generation = lane.take_generation();
            let config = lane.take_config();
            let constraint = lane.take_constraint();
            let semantic = lane.take_semantic();
            let cancellation = lane.take_cancellation();
            let on_event = lane.take_on_event();
            let control_handle = lane.take_control_handle();
            let refinement_seed = generation.seed();
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, constraint)?;
            prepared_lanes.push(MlxSpeculativeLaneRuntime {
                input: prompt,
                config,
                prng_key,
                refinement_seed,
                sampler,
                semantic,
                cancellation,
                on_event,
                control_handle,
            });
        }
        Ok(prepared_lanes)
    }

    fn with_execution<C, V>(
        &mut self,
        mut request: SpeculativeGenerationBatchRequest<'_, MlxBackend<'world>, MlxDrafter, C>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let drafting = request.take_drafting();
        if !matches!(drafting, SpeculativeDraft::Target)
            && <MlxBackend<'_> as eredu_core::TextGenerationBackend>::generation_score_kind(
                self.runtime,
            ) != eredu_core::refinement::PredictionScoreKind::CausalConditional
        {
            return Err(
                eredu_core::generation::GenerationError::RequiresCausalConditionalScores.into(),
            );
        }
        let lanes = request.take_lanes();
        match drafting {
            SpeculativeDraft::Target => self.generate_batch_with_target(lanes, visitor),
            SpeculativeDraft::SharedBackbone => {
                self.generate_speculative_batch_with_shared_backbone(lanes, visitor)
            }
            SpeculativeDraft::External(drafter) => {
                self.generate_speculative_batch_with_external_draft(drafter, lanes, visitor)
            }
            SpeculativeDraft::Embedded => {
                self.generate_speculative_batch_with_embedded_draft(lanes, visitor)
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported speculative draft source".to_string(),
            )),
        }
    }

    fn generate_batch_with_target<C, V>(
        &mut self,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        if <MlxBackend<'_> as eredu_core::TextGenerationBackend>::decoding_capabilities(
            self.runtime,
        )
        .speculation
        {
            return self.generate_speculative_batch_with_shared_backbone(lanes, visitor);
        }
        use super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_runtime::speculative::autoregressive::{
            AutoregressiveInput, CausalExecutor, SharedTargetRefinement,
            SharedTargetRefinementState,
        };
        self.runtime.session().ensure_no_submission_in_flight()?;
        let stream = self.runtime.backend().stream().clone();
        let partition_binding = self.runtime.session().speculative_partition_binding();
        let prepared = Self::prepare_speculative_batch_lanes(lanes, 0)?;
        self.runtime.session_mut().with_model_operation(|model| {
            let streams = SpeculativeExecutionStreams::single(&stream)
                .with_capture_binding(partition_binding.as_ref());
            let substream = model
                .erased()
                .capability_estimate()
                .ordinary_refinement()
                .and_then(|policy| policy.randomness_substream());
            let mut executor =
                CausalExecutor::<MlxAutoregressiveMechanisms, SharedTargetRefinement>::new(
                    model.erased_mut(),
                    SharedTargetRefinement,
                );
            let prepared = prepared
                .into_iter()
                .map(|lane| {
                    lane.map_input(|input, seed| {
                        let root = safemlx::random::key(seed)?;
                        let key = match substream {
                            Some(index) => crate::backend::random::split_key_at(
                                &root,
                                index,
                                streams.target(),
                            )?,
                            None => root,
                        };
                        super::speculative::state_snapshot::settle(vec![&key])?;
                        Ok(AutoregressiveInput::new(
                            input,
                            SharedTargetRefinementState::new(
                                crate::backend::random::RandomState::from_key(key),
                            ),
                        ))
                    })
                })
                .collect::<Result<Vec<_>, Exception>>()?;
            let caches = (0..prepared.len())
                .map(|_| executor.new_cache(streams))
                .collect::<Result<Vec<_>, _>>()?;
            run_speculative_batch(&mut executor, prepared, caches, Ok, streams, visitor)
                .map_err(|error| Error::Exception(Exception::from_source(error)))
        })
    }

    fn generate_speculative_batch_with_shared_backbone<C, V>(
        &mut self,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        use super::speculative::autoregressive::MlxAutoregressiveMechanisms;
        use eredu_core::decoding::{DecodingMode, SpeculationOptions};
        use eredu_runtime::speculative::shared_backbone::{
            SharedBackboneExecutor, SharedBackboneInput,
        };
        self.runtime.session().ensure_no_submission_in_flight()?;
        let policy = self
            .runtime
            .session()
            .capability_estimate()
            .map_err(|error| Error::Exception(Exception::from_source(error)))?
            .shared_backbone_draft()
            .ok_or_else(|| {
                Error::Speculative(
                    "selected execution has no shared-backbone draft declaration".into(),
                )
            })?;
        let stream = self.runtime.backend().stream().clone();
        let mut prepared = Vec::with_capacity(lanes.len());
        for mut lane in lanes {
            let input = lane.take_prompt();
            let generation = lane.take_generation();
            let config = lane.take_config();
            validate_lane_proposal_capacity(&config, policy.proposal_capacity().get())?;
            if generation
                .decoding_mode()
                .is_some_and(|mode| mode != config.decoding_mode)
            {
                return Err(eredu_core::generation::GenerationError::DecodingMode(
                    eredu_core::decoding::DecodingModeError::Unsupported(
                        "lane sampling and execution modes disagree",
                    ),
                )
                .into());
            }
            let options = match config.decoding_mode {
                DecodingMode::SelfSpeculation(options) => options,
                DecodingMode::Autoregressive | DecodingMode::Diffusion { .. } => {
                    SpeculationOptions::default().with_proposal_capacity(policy.proposal_capacity())
                }
            };
            let options = policy
                .prepare(options, config.temperature)
                .map_err(|error| Error::Exception(Exception::from_source(error)))?;
            let refinement_seed = generation.seed();
            let root = safemlx::random::key(refinement_seed)?;
            let proposal_key = crate::backend::random::split_key_at(&root, 1, &stream)?;
            let refinement_key = crate::backend::random::split_key_at(&root, 2, &stream)?;
            super::speculative::state_snapshot::settle(vec![&proposal_key, &refinement_key])
                .map_err(Error::Exception)?;
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, lane.take_constraint())?;
            prepared.push(MlxSpeculativeLaneRuntime {
                input: SharedBackboneInput::new(
                    input,
                    options,
                    crate::backend::random::RandomState::from_key(proposal_key),
                    crate::backend::random::RandomState::from_key(refinement_key),
                ),
                config,
                prng_key,
                refinement_seed,
                sampler,
                semantic: lane.take_semantic(),
                cancellation: lane.take_cancellation(),
                on_event: lane.take_on_event(),
                control_handle: lane.take_control_handle(),
            });
        }
        let partition_binding = self.runtime.session().speculative_partition_binding();
        self.runtime.session_mut().with_model_operation(|model| {
            let streams = SpeculativeExecutionStreams::single(&stream)
                .with_capture_binding(partition_binding.as_ref());
            let mut typed = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                model.erased_mut(),
                policy.proposal_capacity(),
            );
            let mut executor =
                DynGenerationExecutor::<MlxSharedGenerationExecutorTypes>::new(&mut typed);
            let caches = (0..prepared.len())
                .map(|_| executor.new_cache(streams))
                .collect::<Result<Vec<_>, _>>()?;
            run_speculative_batch(&mut executor, prepared, caches, Ok, streams, visitor)
                .map_err(|error| Error::Exception(Exception::from_source(error)))
        })
    }

    fn generate_speculative_batch_with_external_draft<C, V>(
        &mut self,
        drafter: &mut MlxDrafter,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        let proposal_capacity = drafter
            .selected()
            .requirements()
            .strategy()
            .proposal_capacity()
            .get();
        let target_stream = self.runtime.backend().stream().clone();
        let draft_stream = drafter.stream().clone();
        let partition_binding = self.runtime.session().speculative_partition_binding();
        let preparation = SpeculativeExecutionStreams::single(&target_stream)
            .with_capture_binding(partition_binding.as_ref());
        preparation.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Request,
            self.runtime
                .session()
                .ensure_no_submission_in_flight()
                .and_then(|()| drafter.validate_target(self.runtime.session())),
            |error| Error::Exception(Exception::from_source(error)),
        )?;
        let streams = preparation.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Request,
            SpeculativeExecutionStreams::bind(&target_stream, &draft_stream, drafter.topology())
                .map(|streams| streams.with_capture_binding(partition_binding.as_ref())),
            Exception::from_source,
        )?;
        let agree_ownership = |local| {
            streams.finish_preparation(
                eredu_core::run_preparation::TextPreparationStage::Request,
                local,
                |error| Error::Exception(Exception::from_source(error)),
            )
        };
        if drafter.is_autoregressive() {
            return self.runtime.session_mut().with_prepared_model_operation(move |local| {
            agree_ownership(local)?;
            let prepared_lanes = streams.finish_preparation(
                eredu_core::run_preparation::TextPreparationStage::Sampling,
                Self::prepare_speculative_batch_lanes(lanes, proposal_capacity),
                |error| Error::Exception(Exception::from_source(error)),
            )?;
            let draft = drafter.prepare_operation(agree_ownership)?;
            Ok((draft, prepared_lanes))
            }, |model, (draft, prepared_lanes)| {
            draft.with_autoregressive(|draft| {
                let mut executor =
                    eredu_runtime::speculative::autoregressive::AutoregressiveExecutor::<
                        super::speculative::autoregressive::MlxAutoregressiveMechanisms,
                        eredu_runtime::speculative::autoregressive::SharedTargetRefinement,
                    >::new(
                        model.erased_mut(),
                        draft.executable_mut().erased_mut(),
                        std::num::NonZeroUsize::new(proposal_capacity)
                            .expect("selected nonzero capacity"),
                        eredu_runtime::speculative::autoregressive::SharedTargetRefinement,
                    );
                let prepared_lanes = prepared_lanes.into_iter().map(|lane| lane.map_input(|input, seed| {
                    let root = safemlx::random::key(seed)?;
                    let key = crate::backend::random::split_key_at(&root, 2, streams.target())?;
                    super::speculative::state_snapshot::settle(vec![&key])?;
                    Ok(eredu_runtime::speculative::autoregressive::AutoregressiveInput::new(input,
                        eredu_runtime::speculative::autoregressive::SharedTargetRefinementState::new(crate::backend::random::RandomState::from_key(key))))
                })).collect::<Result<Vec<_>, Exception>>();
                let prepared_lanes = streams.finish_preparation(
                    eredu_core::run_preparation::TextPreparationStage::Sampling,
                    prepared_lanes, Exception::from_source,
                )?;
                let caches = (0..prepared_lanes.len())
                    .map(|_| executor.new_cache(streams))
                    .collect::<Result<Vec<_>, _>>();
                let caches = streams.finish_preparation(
                    eredu_core::run_preparation::TextPreparationStage::Sampling,
                    caches, Exception::from_source,
                )?;
                run_speculative_batch(
                    &mut executor,
                    prepared_lanes,
                    caches,
                    Ok,
                    streams,
                    visitor,
                )
                .map_err(|error| Error::Exception(Exception::from_source(error)))
            })
        });
        }
        let capture = drafter.capture().clone();
        let selected = drafter.selected().clone();
        self.runtime.session_mut().with_prepared_model_operation(
            move |local| {
                agree_ownership(local)?;
                let prepared_lanes = streams.finish_preparation(
                    eredu_core::run_preparation::TextPreparationStage::Sampling,
                    Self::prepare_speculative_batch_lanes(lanes, proposal_capacity),
                    |error| Error::Exception(Exception::from_source(error)),
                )?;
                let draft = drafter.prepare_operation(agree_ownership)?;
                Ok((draft, prepared_lanes))
            },
            |model, (draft, prepared_lanes)| {
                draft.visit(MlxExternalBatchVisitor {
                    runner: MlxExternalBatchRunner {
                        target: model.erased_mut(),
                        lanes: prepared_lanes,
                        streams,
                        visitor,
                        capture,
                        selected,
                    },
                })
            },
        )
    }

    fn generate_speculative_batch_with_embedded_draft<C, V>(
        &mut self,
        lanes: Vec<SpeculativeGenerationLane<'_, MlxBackend<'world>, C>>,
        visitor: V,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>
    where
        C: SpeculativeTokenFilterController,
        V: SpeculativeGenerationVisitor,
    {
        self.runtime.session().ensure_no_submission_in_flight()?;
        let stream = self.runtime.backend().stream().clone();
        let capture = self.runtime.session().speculative_partition_binding();
        let streams =
            SpeculativeExecutionStreams::single(&stream).with_capture_binding(capture.as_ref());
        let mut continuation = MlxEmbeddedBatchContinuation {
            lanes,
            streams,
            visitor: Some(visitor),
        };
        self.runtime.session_mut().with_model_operation(|model| {
            model
                .erased_mut()
                .with_embedded_prediction(&mut continuation)
                .ok_or_else(|| {
                    Error::Speculative(
                        "neutral target has no installed prediction-extension contract".into(),
                    )
                })?
        })
    }
}

struct MlxEmbeddedBatchContinuation<'lane, 'world, 'streams, C, V>
where
    C: SpeculativeTokenFilterController,
{
    lanes: Vec<SpeculativeGenerationLane<'lane, MlxBackend<'world>, C>>,
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: Option<V>,
}

impl<C, V> MlxEmbeddedExecutorContinuation for MlxEmbeddedBatchContinuation<'_, '_, '_, C, V>
where
    C: SpeculativeTokenFilterController,
    V: SpeculativeGenerationVisitor,
{
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynGenerationExecutor<'_, MlxGenerationExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error> {
        let proposal_capacity = selected.requirements().strategy().proposal_capacity().get();
        let lanes = std::mem::take(&mut self.lanes);
        // Cache construction may enter architecture-owned preparation
        // collectives. Every rank must first accept its sampler/lane policy.
        let prepared_lanes = self.streams.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Sampling,
            MlxSpeculativeSession::prepare_speculative_batch_lanes(lanes, proposal_capacity),
            |error| Error::Exception(Exception::from_source(error)),
        )?;
        let caches = (0..prepared_lanes.len())
            .map(|_| executor.new_cache(self.streams))
            .collect::<Result<Vec<_>, _>>();
        let caches = self.streams.finish_preparation(
            eredu_core::run_preparation::TextPreparationStage::Sampling,
            caches,
            Exception::from_source,
        )?;
        run_speculative_batch(
            executor,
            prepared_lanes,
            caches,
            Ok,
            self.streams,
            self.visitor
                .take()
                .expect("embedded executor continuation is invoked once"),
        )
        .map_err(|error| Error::Exception(Exception::from_source(error)))
    }
}

#[cfg(test)]
mod mechanism_tests;
