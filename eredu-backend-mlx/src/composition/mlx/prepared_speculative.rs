//! Whole-session MLX speculative generation capability.

use eredu_architectures::speculative_execution::{
    DynEmbeddedExecutor, EmbeddedExecutorTypes, EmbeddedPredictionOutput,
    ReplicatedPredictionInput, ReplicatedPredictionNative, SpeculativeTensorMechanisms,
};
use eredu_core::{
    generation::{GenerationCancellationToken, SemanticEvent, SpeculativeConfig},
    GenerationSequence, ModelRuntime, PreparedSpeculativeLane, SpeculativeCallbackPublisher,
    SpeculativeCapability, SpeculativeDraft, SpeculativeExecutor, SpeculativeGenerationBackend,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationBatchRequest, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeOutputRuntime, SpeculativeSampling,
    SpeculativeSemanticConstraint, SpeculativeSemanticState, SpeculativeTokenFilterController,
};
use eredu_runtime::{ConstrainedSampler, GenerationSampler, SpeculativeSampler};
use safemlx::{
    error::Exception, ops::indexing::TryIndexOp, transforms::async_eval_with_event, Array, Stream,
};

use super::{
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

pub(crate) struct MlxEmbeddedExecutorTypes;

pub(crate) trait MlxEmbeddedExecutorContinuation {
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut DynEmbeddedExecutor<'_, MlxEmbeddedExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error>;
}

impl EmbeddedExecutorTypes for MlxEmbeddedExecutorTypes {
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

    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        context: &Stream,
        operation: impl for<'a> FnOnce(
            A::Input<'a>,
            MlxTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        input.with_borrowed(|input| {
            let tokens = input::text_token_ids(input, context)
                .map(MlxTensor::from_array)
                .map_err(|error| Exception::custom(error.to_string()))?;
            let prepared_tokens = tokens.clone();
            operation(
                A::text_input(&prepared_tokens, None),
                tokens,
                input.cache_identity(),
            )
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
    type ExecutorTypes = MlxEmbeddedExecutorTypes;

    fn executor_context<'a>(
        context: <Self::ExecutorTypes as EmbeddedExecutorTypes>::Context<'a>,
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

    fn take_telemetry() -> Result<Self::Telemetry, Exception> {
        Ok(Self::Telemetry::default())
    }
}

impl SpeculativeTensorMechanisms for MlxEmbeddedPredictionMechanisms {
    type Tensor = MlxTensor;
    type Logits = Array;
    type Context<'a> = SpeculativeExecutionStreams<'a>;
    type Completion = super::speculative::MlxSpeculativeCompletion;
    type Error = Exception;

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

struct MlxSpeculativeLaneRuntime<'a, C> {
    input: MlxModelInput,
    config: SpeculativeConfig,
    prng_key: Option<Array>,
    sampler: MlxPreparedSampler<C>,
    semantic: Box<dyn SpeculativeSemanticState>,
    cancellation: GenerationCancellationToken,
    on_event: Box<dyn FnMut(SemanticEvent) + 'a>,
}

type MlxPreparedSampler<C> = ConstrainedSampler<GenerationSampler, C>;

fn validate_lane_proposal_capacity(
    config: &SpeculativeConfig,
    proposal_capacity: usize,
) -> Result<(), Error> {
    if config.max_draft_tokens > proposal_capacity {
        return Err(Error::Speculative(format!(
            "lane requests {} draft tokens, but the selected speculative realization admits at most {proposal_capacity}",
            config.max_draft_tokens
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
        let mut caches = (0..runner.lanes.len())
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
            .collect::<Result<Vec<_>, _>>()?;
        A::visit_executor::<
            crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
            _,
        >(
            target,
            assistant,
            runner.capture,
            MlxExternalExecutorRunner {
                lanes: runner.lanes,
                caches: &mut caches,
                streams: runner.streams,
                visitor: runner.visitor,
            },
        )
    }
}

struct MlxExternalExecutorRunner<'cache, 'lane, 'streams, C, V> {
    lanes: Vec<MlxSpeculativeLaneRuntime<'lane, C>>,
    caches: &'cache mut [super::speculative::external::MlxExternalPredictionCache],
    streams: SpeculativeExecutionStreams<'streams>,
    visitor: V,
}

impl<'cache, 'lane, 'streams, A, C, V>
    eredu_architectures::ExternalAssistantExecutorVisitor<
        A,
        crate::composition::mlx::speculative::external::MlxExternalAssistantMechanisms,
    > for MlxExternalExecutorRunner<'cache, 'lane, 'streams, C, V>
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
        .map_err(|error| Error::Speculative(error.to_string()))
    }
}

fn run_speculative_batch<'run, 'lane, 'streams, B, C, S>(
    backend: &'run mut B,
    lanes: Vec<MlxSpeculativeLaneRuntime<'lane, C>>,
    caches: &'run mut [B::Cache],
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
    if caches.len() != lanes.len() {
        return Err(Exception::custom(format!(
            "speculative cache has {} lanes but the request has {} lanes",
            caches.len(),
            lanes.len()
        )));
    }
    let topology = streams.topology();
    let component_timings_collected = component_timing_enabled() && backend.supports_telemetry();
    let mut prepared = Vec::with_capacity(lanes.len());
    for (lane, cache) in lanes.into_iter().zip(caches.iter_mut()) {
        let MlxSpeculativeLaneRuntime {
            input,
            config,
            prng_key,
            sampler,
            semantic,
            cancellation,
            on_event,
        } = lane;
        let sampling = MlxSpeculativeSampling::new(wrap_sampler(sampler)?);
        let randomness = <MlxSpeculativeSampling<S> as SpeculativeSampling>::initialize_randomness(
            prng_key,
            config.temperature,
            streams,
        )?;
        let sequence =
            GenerationSequence::new(config.max_tokens, config.eos_token_ids.iter().copied());
        prepared.push(PreparedSpeculativeLane::new(
            cache,
            input,
            config,
            SpeculativeOutputRuntime::new(
                sampling,
                sequence,
                SpeculativeSemanticConstraint::semantic(semantic),
                SpeculativeCallbackPublisher::semantic(on_event),
                cancellation,
            ),
            randomness,
        ));
    }
    visitor
        .run(
            backend,
            prepared,
            topology,
            streams.is_split(),
            component_timings_collected,
            streams,
        )
        .map_err(|error| Exception::custom(error.to_string()))
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
        let prng_key = (resolved.temperature != 0.0)
            .then(|| safemlx::random::key(generation.seed()))
            .transpose()?;
        Ok((
            prng_key,
            ConstrainedSampler::new(GenerationSampler::from_resolved(resolved), constraint),
        ))
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
            let (prng_key, sampler) =
                Self::prepare_mlx_speculative_sampling(generation, constraint)?;
            prepared_lanes.push(MlxSpeculativeLaneRuntime {
                input: prompt,
                config,
                prng_key,
                sampler,
                semantic,
                cancellation,
                on_event,
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
        let lanes = request.take_lanes();
        match drafting {
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
        let streams =
            SpeculativeExecutionStreams::bind(&target_stream, &draft_stream, drafter.topology())?;
        let prepared_lanes = Self::prepare_speculative_batch_lanes(lanes, proposal_capacity)?;
        let capture = drafter.capture().clone();
        let selected = drafter.selected().clone();
        let model = self.runtime.session_mut().speculative_model_mut();
        drafter.visit(MlxExternalBatchVisitor {
            runner: MlxExternalBatchRunner {
                target: model.erased_mut(),
                lanes: prepared_lanes,
                streams,
                visitor,
                capture,
                selected,
            },
        })
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
        let stream = self.runtime.backend().stream().clone();
        let streams = SpeculativeExecutionStreams::single(&stream);
        let model = self.runtime.session_mut().speculative_model_mut();
        let target = model.erased_mut();
        let mut continuation = MlxEmbeddedBatchContinuation {
            lanes,
            streams,
            visitor: Some(visitor),
        };
        target
            .with_embedded_prediction(&mut continuation)
            .ok_or_else(|| {
                Error::Speculative(
                    "neutral target has no installed prediction-extension contract".into(),
                )
            })?
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
        executor: &mut DynEmbeddedExecutor<'_, MlxEmbeddedExecutorTypes>,
    ) -> Result<SpeculativeGenerationBatchOutput, Error> {
        let proposal_capacity = selected.requirements().strategy().proposal_capacity().get();
        let lanes = std::mem::take(&mut self.lanes);
        let prepared_lanes =
            MlxSpeculativeSession::prepare_speculative_batch_lanes(lanes, proposal_capacity)?;
        let mut caches = (0..prepared_lanes.len())
            .map(|_| executor.new_cache())
            .collect::<Result<Vec<_>, _>>()?;
        run_speculative_batch(
            executor,
            prepared_lanes,
            &mut caches,
            Ok,
            self.streams,
            self.visitor
                .take()
                .expect("embedded executor continuation is invoked once"),
        )
        .map_err(|error| Error::Speculative(error.to_string()))
    }
}

#[cfg(test)]
mod mechanism_tests {
    use super::*;

    #[test]
    fn lane_proposal_width_cannot_exceed_selected_realization() {
        let config = SpeculativeConfig {
            max_draft_tokens: 5,
            ..SpeculativeConfig::default()
        };
        let error = validate_lane_proposal_capacity(&config, 4).unwrap_err();
        assert!(error.to_string().contains("admits at most 4"));
        validate_lane_proposal_capacity(&config, 5).unwrap();
    }

    #[test]
    fn fused_rows_are_selected_by_backend_mechanisms_without_family_policy() {
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let value = MlxTensor::from_array(Array::from_slice(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 2, 3],
        ));
        assert_eq!(
            MlxEmbeddedPredictionMechanisms::sequence_len(&value).unwrap(),
            2
        );
        let row = MlxEmbeddedPredictionMechanisms::fused_logits_row(
            &value,
            1,
            SpeculativeExecutionStreams::single(&stream),
        )
        .unwrap();
        assert_eq!(row.evaluated().unwrap().as_slice::<f32>(), &[4.0, 5.0, 6.0]);
    }
}
