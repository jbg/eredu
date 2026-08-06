//! Codec-free realtime speech-to-speech token APIs.
//!
//! Realtime speech models in this crate operate on discrete codec tokens rather
//! than PCM. Callers are expected to encode live audio into model-native
//! codebook frames before calling these APIs, and decode emitted codebook frames
//! with a codec implementation outside `safemlx-lm`.

use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    random::RandomState,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use std::path::Path;

use crate::{
    api::{ensure_executable_load_options, moshi, personaplex, ModelLoadOptions},
    architectures::moshi::layerwise::MoshiLayerwiseModel,
    error::Error,
    runtime::checkpoint::artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
    runtime::execution::layerwise::{LayerExecutionLoadOptions, WeightResidency},
    runtime::generation::sampler::{DefaultSampler, Sampler},
    runtime::scheduler::{
        FairScheduler, RequestId, RequestStatus, SchedulerLimits, SchedulerReport, WorkDescriptor,
        WorkId,
    },
};

/// Static token-stream metadata needed to pair a realtime model with a codec.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RealtimeSpeechConfig<'a> {
    /// Total number of audio codebooks consumed by the temporal model.
    pub total_audio_codebooks: i32,
    /// Number of live input-side codebooks expected per realtime step.
    pub input_audio_codebooks: i32,
    /// Number of generated-side codebooks emitted per realtime step.
    pub generated_audio_codebooks: i32,
    /// Number of depth-transformer codebooks sampled or teacher-forced per step.
    pub depth_audio_codebooks: i32,
    /// Text token used before any sampled text is available.
    pub text_padding_token: i32,
    /// Audio token used while delayed streams warm up.
    pub audio_padding_token: i32,
    /// Per-audio-codebook delays, excluding the leading text delay.
    pub audio_delays: &'a [i32],
}

impl RealtimeSpeechConfig<'_> {
    /// Largest audio delay in frames.
    pub fn max_audio_delay(&self) -> i32 {
        self.audio_delays.iter().copied().max().unwrap_or(0)
    }
}

/// One encoded input-side audio frame for a realtime model step.
#[derive(Debug, Clone)]
pub struct RealtimeStepInput {
    /// Encoded input-side audio tokens shaped `[batch, input_audio_codebooks]`.
    input_audio_tokens: Array,
    /// Optional generated-side codec tokens forced by a prompt frame.
    forced_generated_audio_tokens: Option<Array>,
    /// Optional text token forced by a prompt frame.
    forced_text_token: Option<Array>,
}

impl RealtimeStepInput {
    /// Creates an owned realtime step from encoded audio-codebook tokens.
    pub fn encoded_audio(input_audio_tokens: &Array) -> Self {
        Self {
            input_audio_tokens: input_audio_tokens.clone(),
            forced_generated_audio_tokens: None,
            forced_text_token: None,
        }
    }

    /// Forces the generated-side codec tokens for a prompt transition.
    pub fn with_forced_generated_audio(mut self, tokens: &Array) -> Self {
        self.forced_generated_audio_tokens = Some(tokens.clone());
        self
    }

    /// Forces the generated text token for a prompt transition.
    pub fn with_forced_text(mut self, token: &Array) -> Self {
        self.forced_text_token = Some(token.clone());
        self
    }
}

impl WorkDescriptor for RealtimeStepInput {
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        encode_array_descriptor(&self.input_audio_tokens, output)?;
        encode_optional_array_descriptor(self.forced_generated_audio_tokens.as_ref(), output)?;
        encode_optional_array_descriptor(self.forced_text_token.as_ref(), output)
    }
}

/// Request-owned sampling controls for a realtime session.
#[derive(Debug, Clone)]
pub struct RealtimeSampling {
    /// Text sampling temperature.
    text_temperature: f32,
    /// Audio sampling temperature.
    audio_temperature: f32,
    /// Optional request-local PRNG state for stochastic samplers.
    prng_state: Option<RandomState>,
}

impl RealtimeSampling {
    /// Creates validated request-local sampling controls.
    pub fn new(
        text_temperature: f32,
        audio_temperature: f32,
        prng_state: Option<RandomState>,
    ) -> Result<Self, Error> {
        if !text_temperature.is_finite()
            || text_temperature < 0.0
            || !audio_temperature.is_finite()
            || audio_temperature < 0.0
        {
            return Err(Error::Parallel(format!(
                "realtime sampling temperatures must be finite and non-negative, got text={text_temperature} audio={audio_temperature}"
            )));
        }
        Ok(Self {
            text_temperature,
            audio_temperature,
            prng_state,
        })
    }

    /// Deterministic greedy sampling controls.
    pub fn greedy() -> Self {
        Self {
            text_temperature: 0.0,
            audio_temperature: 0.0,
            prng_state: None,
        }
    }

    /// Returns the text sampling temperature.
    pub const fn text_temperature(&self) -> f32 {
        self.text_temperature
    }

    /// Returns the audio sampling temperature.
    pub const fn audio_temperature(&self) -> f32 {
        self.audio_temperature
    }

    /// Returns whether this request owns stochastic PRNG state.
    pub const fn is_stochastic(&self) -> bool {
        self.prng_state.is_some()
    }
}

/// Output from one encoded-audio realtime generation step.
pub struct RealtimeStepOutput {
    /// Text token sampled at this model step, shaped `[batch, 1]`.
    pub text_token: Array,
    /// Newly sampled generated-codebook tokens before delay alignment.
    pub sampled_audio_tokens: Array,
    /// Delay-aligned codec frame ready for decoding, shaped `[batch, generated_audio_codebooks]`.
    ///
    /// This is `None` while delayed generated streams are warming up.
    pub output_audio_tokens: Option<Array>,
}

/// Text tokens and delay-aligned codec tokens from offline generation.
pub struct EncodedAudioOutput {
    /// Sampled text tokens, shaped `[batch, input_frames]`.
    pub text_tokens: Array,
    /// Generated codec tokens, shaped `[batch, generated_audio_codebooks, output_frames]`.
    ///
    /// The output may have fewer frames than the input because delayed streams
    /// need future encoded input frames before a coherent output frame exists.
    pub audio_tokens: Array,
}

/// Supported realtime speech-to-speech model-family dispatch target.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RealtimeModelKind {
    /// Moshi-family realtime token model with a native Moshi/MLX checkpoint layout.
    Moshi,
    /// NVIDIA PersonaPlex realtime token model with its released PyTorch safetensors layout.
    PersonaPlex,
}

impl RealtimeModelKind {
    /// Returns the model type string used for user-facing dispatch messages.
    pub fn model_type(self) -> &'static str {
        match self {
            Self::Moshi => "moshi",
            Self::PersonaPlex => "personaplex",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RealtimeModelMetadata {
    #[serde(default)]
    model_type: Option<String>,
}

fn realtime_model_kind(model_dir: impl AsRef<Path>) -> Result<RealtimeModelKind, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    if !config_path.exists() {
        return Ok(RealtimeModelKind::Moshi);
    }

    let metadata: RealtimeModelMetadata =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;
    match metadata.model_type.as_deref() {
        None | Some("moshi") => Ok(RealtimeModelKind::Moshi),
        Some("personaplex") => Ok(RealtimeModelKind::PersonaPlex),
        Some(other) => Err(Error::UnsupportedArchitecture(format!(
            "{other} is not a realtime speech-to-speech token model"
        ))),
    }
}

fn realtime_artifact_identity(
    model_dir: &Path,
    kind: RealtimeModelKind,
) -> Result<LoadedArtifactIdentity, Error> {
    let index = model_dir.join("model.safetensors.index.json");
    let weight_files = if index.exists() {
        crate::runtime::checkpoint::load::safetensors_files(model_dir)?
    } else {
        match kind {
            RealtimeModelKind::Moshi => {
                let args = moshi::get_model_args(model_dir)?;
                vec![model_dir.join(args.moshi_name.as_deref().unwrap_or("model.safetensors"))]
            }
            RealtimeModelKind::PersonaPlex => {
                vec![model_dir.join(personaplex::MODEL_SAFETENSORS)]
            }
        }
    };
    let files = weight_files
        .into_iter()
        .map(|path| {
            let logical_name = path
                .strip_prefix(model_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            ArtifactFile::new(logical_name, path)
        })
        .collect::<Vec<_>>();
    fingerprint_artifact(kind.model_type(), files)
}

/// Loads a supported realtime speech-to-speech token model from a model directory.
///
/// This is the high-level realtime counterpart to [`crate::api::LoadedModel`].
/// It does not load a text tokenizer or audio codec: callers bring tokenization,
/// codec encode/decode, transport, and device I/O.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedRealtimeModel, Error> {
    load_model_with_options(
        model_dir,
        ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads a realtime model using the shared architecture-independent options.
///
/// Successful loads bind an immutable SHA-256 identity of the selected weight
/// files to the model. Realtime sessions use that identity, together with the
/// normalized execution configuration, when validating state handoff.
pub fn load_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedRealtimeModel, Error> {
    ensure_executable_load_options(options)?;
    let model_dir = model_dir.as_ref();
    let kind = realtime_model_kind(model_dir)?;
    let execution = match options.weight_residency {
        WeightResidency::FullyResident => LayerExecutionLoadOptions::FullyResident,
        WeightResidency::LayerwiseHost(options) => options.into(),
        WeightResidency::DenseDiskStream(options) => options.into(),
        WeightResidency::SparseExpertCache(_)
        | WeightResidency::SparseExpertCacheWithDenseLayers(_) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "{} does not contain routed experts",
                kind.model_type()
            )));
        }
    };
    let model = if let Some(quantization) = options.quantization {
        let transformed = match kind {
            RealtimeModelKind::Moshi => {
                moshi::load_model_quantized(model_dir, quantization, stream, weights_stream)?
            }
            RealtimeModelKind::PersonaPlex => {
                personaplex::load_model_quantized(model_dir, quantization, stream, weights_stream)?
            }
        };
        crate::architectures::moshi::layerwise::execute_transformed_model(
            transformed,
            stream,
            weights_stream,
        )?
    } else {
        match kind {
            RealtimeModelKind::Moshi => {
                crate::architectures::moshi::layerwise::load_moshi_layerwise_model(
                    model_dir,
                    execution,
                    stream,
                    weights_stream,
                )?
            }
            RealtimeModelKind::PersonaPlex => {
                crate::architectures::moshi::layerwise::load_personaplex_layerwise_model(
                    model_dir,
                    execution,
                    stream,
                    weights_stream,
                )?
            }
        }
    };
    let model = model.with_artifact_identity(realtime_artifact_identity(model_dir, kind)?);
    Ok(match kind {
        RealtimeModelKind::Moshi => LoadedRealtimeModel::Moshi(model),
        RealtimeModelKind::PersonaPlex => LoadedRealtimeModel::PersonaPlex(model),
    })
}

/// Loaded realtime speech-to-speech token model.
///
/// Both model families use the same scheduler, session, encoded-frame, and
/// forced-frame APIs; checkpoint layout is the only dispatch distinction.
pub enum LoadedRealtimeModel {
    /// Moshi-family model.
    Moshi(MoshiLayerwiseModel),
    /// PersonaPlex model.
    PersonaPlex(MoshiLayerwiseModel),
}

impl LoadedRealtimeModel {
    fn artifact_identity(&self) -> &LoadedArtifactIdentity {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.artifact_identity(),
        }
    }

    /// Returns the loaded realtime model family.
    pub fn kind(&self) -> RealtimeModelKind {
        match self {
            Self::Moshi(_) => RealtimeModelKind::Moshi,
            Self::PersonaPlex(_) => RealtimeModelKind::PersonaPlex,
        }
    }

    /// Returns the loaded realtime model family as a model type string.
    pub fn model_type(&self) -> &'static str {
        self.kind().model_type()
    }

    /// Returns the parsed Moshi-family token-model configuration.
    pub fn args(&self) -> &moshi::ModelArgs {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.args(),
        }
    }

    /// Returns current residency telemetry for the selected parameter policy.
    pub fn residency_report(
        &self,
    ) -> Result<Option<crate::runtime::residency::manager::ResidencyReport>, Error> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.residency_report().map(Some),
        }
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.dense_stream_report(),
        }
    }

    /// Returns per-group residency for the selected parameter policy.
    pub fn execution_group_reports(
        &self,
    ) -> Result<Option<Vec<crate::runtime::residency::manager::ResidentLayerGroupReport>>, Error>
    {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => {
                model.execution_group_reports().map(Some)
            }
        }
    }
}

impl LoadedRealtimeModel {
    /// Returns the codec-token stream configuration for this model.
    pub fn realtime_config(&self) -> RealtimeSpeechConfig<'_> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.realtime_config(),
        }
    }

    fn new_generation_state(&self) -> moshi::GenerationState {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.new_generation_state(),
        }
    }

    fn execute_realtime_step<TS, AS>(
        &mut self,
        state: &mut moshi::GenerationState,
        input: &RealtimeStepInput,
        text_sampler: &mut TS,
        audio_samplers: &mut [AS],
        sampling: &mut RealtimeSampling,
        stream: &Stream,
    ) -> Result<RealtimeStepOutput, Error>
    where
        TS: Sampler,
        AS: Sampler,
    {
        let prng_state = sampling.prng_state.as_mut();
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.generate_step_forced(
                state,
                &input.input_audio_tokens,
                input.forced_generated_audio_tokens.as_ref(),
                input.forced_text_token.as_ref(),
                text_sampler,
                audio_samplers,
                sampling.text_temperature,
                sampling.audio_temperature,
                prng_state,
                stream,
            ),
        }
        .map_err(Error::from)
    }
}

/// Request-local Moshi/PersonaPlex state owned by the canonical scheduler.
///
/// The state carries the immutable checkpoint artifact and normalized
/// execution identity of the model that created it. A scheduler accepts a
/// released session only when both identities match its loaded model exactly.
pub struct RealtimeSession<TS, AS> {
    model_identity: RealtimeModelIdentity,
    generation: moshi::GenerationState,
    text_sampler: TS,
    audio_samplers: Vec<AS>,
    sampling: RealtimeSampling,
    batch_size: Option<i32>,
}

impl<TS, AS> RealtimeSession<TS, AS> {
    /// Returns the number of encoded frames committed by this session.
    pub fn step(&self) -> usize {
        self.generation.step()
    }

    /// Returns the request-local temporal/depth generation state.
    ///
    /// Session state is exposed after [`RealtimeInferenceScheduler::release_request`]
    /// for explicit application-level persistence or diagnostic inspection.
    pub fn generation_state(&self) -> &moshi::GenerationState {
        &self.generation
    }

    /// Returns the request-owned text sampler mutably while the session is released.
    pub fn text_sampler_mut(&mut self) -> &mut TS {
        &mut self.text_sampler
    }

    /// Returns the request-owned audio samplers mutably while the session is released.
    pub fn audio_samplers_mut(&mut self) -> &mut [AS] {
        &mut self.audio_samplers
    }

    /// Replaces sampling temperatures and PRNG state while the session is released.
    pub fn set_sampling(&mut self, sampling: RealtimeSampling) {
        self.sampling = sampling;
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct RealtimeModelIdentity {
    artifact: LoadedArtifactIdentity,
    execution: RealtimeExecutionIdentity,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct RealtimeExecutionIdentity {
    kind: RealtimeModelKind,
    dim: i32,
    temporal_layers: i32,
    temporal_heads: i32,
    temporal_intermediate: i32,
    temporal_context: i32,
    temporal_rope_base_bits: u32,
    depth_dim: i32,
    depth_layers: i32,
    depth_heads: i32,
    depth_intermediate: i32,
    depth_context: i32,
    depth_rope_base_bits: u32,
    text_card: i32,
    text_padding_token: i32,
    audio_card: i32,
    total_codebooks: i32,
    depth_codebooks: i32,
    generated_codebooks: i32,
    delays: Vec<i32>,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
}

impl RealtimeModelIdentity {
    fn new(model: &LoadedRealtimeModel) -> Self {
        Self {
            artifact: model.artifact_identity().clone(),
            execution: RealtimeExecutionIdentity::new(model.kind(), model.args()),
        }
    }

    fn mismatch(&self, other: &Self) -> Option<&'static str> {
        if self.artifact != other.artifact {
            Some("checkpoint artifact fingerprint")
        } else if self.execution != other.execution {
            Some("normalized execution identity")
        } else {
            None
        }
    }
}

impl RealtimeExecutionIdentity {
    fn new(kind: RealtimeModelKind, args: &moshi::ModelArgs) -> Self {
        Self {
            kind,
            dim: args.dim,
            temporal_layers: args.num_layers,
            temporal_heads: args.num_heads,
            temporal_intermediate: args.dim_feedforward.unwrap_or(args.dim * 4),
            temporal_context: args.context,
            temporal_rope_base_bits: args.max_period.to_bits(),
            depth_dim: args.depformer_dim,
            depth_layers: args.depformer_num_layers,
            depth_heads: args.depformer_num_heads,
            depth_intermediate: args
                .depformer_dim_feedforward
                .unwrap_or(args.depformer_dim * 4),
            // These defaults mirror the depth-transformer constructor, not
            // the temporal transformer defaults in the checkpoint schema.
            depth_context: args.depformer_context.unwrap_or(args.dep_q),
            depth_rope_base_bits: args.depformer_max_period.unwrap_or(8.0).to_bits(),
            text_card: args.text_card,
            text_padding_token: args.text_padding_token(),
            audio_card: args.card,
            total_codebooks: args.n_q,
            depth_codebooks: args.dep_q,
            generated_codebooks: args.generated_audio_codebooks(),
            delays: args.delays.clone(),
            quantization: args.quantization,
        }
    }
}

/// One completed realtime transition and its stable work identity.
pub struct RealtimeCompletedStep {
    work: WorkId,
    output: RealtimeStepOutput,
}

impl RealtimeCompletedStep {
    /// Returns the scheduler-assigned work identity.
    pub const fn work(&self) -> WorkId {
        self.work
    }

    /// Returns the generated text/audio result.
    pub const fn output(&self) -> &RealtimeStepOutput {
        &self.output
    }

    /// Consumes this completion into its work identity and generated result.
    pub fn into_parts(self) -> (WorkId, RealtimeStepOutput) {
        (self.work, self.output)
    }
}

/// Bounded fair scheduler for independent Moshi/PersonaPlex realtime sessions.
///
/// The shared scheduler is the sole owner of request/work identity, queueing,
/// state isolation, lifecycle, cancellation, backpressure, poisoning, and
/// telemetry. This adapter contributes only realtime input validation and the
/// temporal/depth execution closure.
pub struct RealtimeInferenceScheduler<TS, AS> {
    model_identity: RealtimeModelIdentity,
    scheduler: FairScheduler<RealtimeStepInput, RealtimeSession<TS, AS>>,
}

impl<TS, AS> RealtimeInferenceScheduler<TS, AS>
where
    TS: Sampler,
    AS: Sampler,
{
    /// Binds an empty scheduler to one loaded realtime artifact and execution identity.
    pub fn new(model: &LoadedRealtimeModel, limits: SchedulerLimits) -> Result<Self, Error> {
        Ok(Self {
            model_identity: RealtimeModelIdentity::new(model),
            scheduler: FairScheduler::new(limits)?,
        })
    }

    /// Registers a request with fresh temporal/depth state and owned samplers.
    pub fn register_request(
        &mut self,
        model: &LoadedRealtimeModel,
        request: RequestId,
        text_sampler: TS,
        audio_samplers: Vec<AS>,
        sampling: RealtimeSampling,
    ) -> Result<(), Error> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        self.validate_audio_samplers(model, audio_samplers.len())?;
        self.scheduler.register(
            request,
            RealtimeSession {
                model_identity: self.model_identity.clone(),
                generation: model.new_generation_state(),
                text_sampler,
                audio_samplers,
                sampling,
                batch_size: None,
            },
        )
    }

    /// Registers a released request session.
    pub fn register_request_with_session(
        &mut self,
        model: &LoadedRealtimeModel,
        request: RequestId,
        session: RealtimeSession<TS, AS>,
    ) -> Result<(), Error> {
        self.validate_model(model)?;
        self.scheduler.validate_registration(request)?;
        if let Some(component) = session.model_identity.mismatch(&self.model_identity) {
            return Err(Error::Parallel(format!(
                "realtime session {component} does not match the scheduler model"
            )));
        }
        self.validate_audio_samplers(model, session.audio_samplers.len())?;
        validate_generation_state(model, &session.generation)?;
        self.scheduler.register(request, session)
    }

    /// Enqueues one encoded or forced prompt frame.
    pub fn enqueue(
        &mut self,
        model: &LoadedRealtimeModel,
        request: RequestId,
        input: RealtimeStepInput,
    ) -> Result<WorkId, Error> {
        Ok(self
            .enqueue_batch(model, request, vec![input])?
            .pop()
            .expect("one submitted realtime frame"))
    }

    /// Atomically enqueues an ordered batch of encoded or forced prompt frames.
    pub fn enqueue_batch(
        &mut self,
        model: &LoadedRealtimeModel,
        request: RequestId,
        inputs: Vec<RealtimeStepInput>,
    ) -> Result<Vec<WorkId>, Error> {
        self.validate_model(model)?;
        let state = self.scheduler.request_state(request).ok_or_else(|| {
            Error::Parallel(format!(
                "realtime request {} is not active",
                request.value()
            ))
        })?;
        let mut expected_batch = state.batch_size;
        for input in &inputs {
            validate_realtime_input(model, input)?;
            let batch = input.input_audio_tokens.dim(0);
            if let Some(expected) = expected_batch {
                if expected != batch {
                    return Err(Error::Parallel(format!(
                        "realtime request {} changed batch size from {expected} to {batch}",
                        request.value()
                    )));
                }
            } else {
                expected_batch = Some(batch);
            }
        }
        let work = self.scheduler.enqueue_batch(request, inputs)?;
        if let Some(batch) = expected_batch {
            self.scheduler
                .request_state_mut(request)?
                .batch_size
                .get_or_insert(batch);
        }
        Ok(work)
    }

    /// Drains queued frames in fair request order on the local model.
    pub fn run_queued(
        &mut self,
        model: &mut LoadedRealtimeModel,
        stream: &Stream,
    ) -> Result<Vec<RealtimeCompletedStep>, Error> {
        self.run_bounded(model, usize::MAX, stream)
    }

    /// Runs at most `max_frames` fair-ordered transitions.
    ///
    /// Applications use bounded drains to regain control for deadlines and
    /// cancellation between frame executions.
    pub fn run_bounded(
        &mut self,
        model: &mut LoadedRealtimeModel,
        max_frames: usize,
        stream: &Stream,
    ) -> Result<Vec<RealtimeCompletedStep>, Error> {
        self.validate_model(model)?;
        self.scheduler
            .drain_local_bounded(max_frames, |_, input, session| {
                model.execute_realtime_step(
                    &mut session.generation,
                    input,
                    &mut session.text_sampler,
                    &mut session.audio_samplers,
                    &mut session.sampling,
                    stream,
                )
            })
            .map(|completed| {
                completed
                    .into_iter()
                    .map(|completed| {
                        let (work, _, output) = completed.into_parts();
                        RealtimeCompletedStep { work, output }
                    })
                    .collect()
            })
    }

    /// Completes a request and drops its temporal/depth state and samplers.
    pub fn finish_request(&mut self, request: RequestId) -> Result<(), Error> {
        self.scheduler.finish(request)
    }

    /// Cancels a request and discards all queued frames and owned state.
    pub fn cancel_request(&mut self, request: RequestId) -> Result<(), Error> {
        self.scheduler.cancel(request)
    }

    /// Releases an idle request for explicit persistence or later resumption.
    pub fn release_request(
        &mut self,
        request: RequestId,
    ) -> Result<RealtimeSession<TS, AS>, Error> {
        self.scheduler.release(request)
    }

    /// Removes a terminal identity so it may be explicitly reused.
    pub fn forget_terminal_request(&mut self, request: RequestId) -> Result<RequestStatus, Error> {
        self.scheduler.forget_terminal(request)
    }

    /// Returns the lifecycle state for a known request.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.scheduler.request_status(request)
    }

    /// Returns the queued frame count for an active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.scheduler.queued_for_request(request)
    }

    /// Replaces sampling controls for an idle active request.
    ///
    /// Queued frames retain a deterministic submission contract, so controls
    /// may change only when the request queue is empty.
    pub fn set_request_sampling(
        &mut self,
        request: RequestId,
        sampling: RealtimeSampling,
    ) -> Result<(), Error> {
        let queued = self.scheduler.queued_for_request(request);
        if queued != 0 {
            return Err(Error::Parallel(format!(
                "realtime request {} has {queued} queued frames; drain or cancel them before changing sampling",
                request.value()
            )));
        }
        self.scheduler.request_state_mut(request)?.sampling = sampling;
        Ok(())
    }

    /// Returns generic scheduler occupancy and lifecycle telemetry.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }

    /// Returns the error that poisoned all request state, if any.
    pub fn poison_reason(&self) -> Option<&str> {
        self.scheduler.poison_reason()
    }

    fn validate_model(&self, model: &LoadedRealtimeModel) -> Result<(), Error> {
        let actual = RealtimeModelIdentity::new(model);
        if let Some(component) = actual.mismatch(&self.model_identity) {
            return Err(Error::Parallel(format!(
                "realtime scheduler model {component} {:?} does not match {:?}",
                actual, self.model_identity
            )));
        }
        Ok(())
    }

    fn validate_audio_samplers(
        &self,
        model: &LoadedRealtimeModel,
        actual: usize,
    ) -> Result<(), Error> {
        let expected = model.args().dep_q as usize;
        if actual != expected {
            return Err(Error::Parallel(format!(
                "realtime request requires {expected} audio samplers, got {actual}"
            )));
        }
        Ok(())
    }
}

fn encode_array_descriptor(array: &Array, output: &mut Vec<u32>) -> Result<(), Error> {
    output.extend_from_slice(&[array.dtype() as u32, array.ndim() as u32]);
    for dimension in array.shape() {
        output.push(u32::try_from(*dimension).map_err(|_| {
            Error::Parallel(format!(
                "realtime work dimension {dimension} exceeds descriptor range"
            ))
        })?);
    }
    Ok(())
}

fn encode_optional_array_descriptor(
    array: Option<&Array>,
    output: &mut Vec<u32>,
) -> Result<(), Error> {
    match array {
        Some(array) => {
            output.push(1);
            encode_array_descriptor(array, output)
        }
        None => {
            output.push(0);
            Ok(())
        }
    }
}

fn validate_realtime_input(
    model: &LoadedRealtimeModel,
    input: &RealtimeStepInput,
) -> Result<(), Error> {
    let args = model.args();
    let tokens = &input.input_audio_tokens;
    if tokens.dtype() != Dtype::Int32
        || tokens.ndim() != 2
        || tokens.dim(1) != args.input_audio_codebooks()
    {
        return Err(Error::Parallel(format!(
            "realtime input must be int32 [batch, {}], got {:?} {:?}",
            args.input_audio_codebooks(),
            tokens.dtype(),
            tokens.shape()
        )));
    }
    let batch = tokens.dim(0);
    if batch <= 0 {
        return Err(Error::Parallel(
            "realtime input batch size must be positive".into(),
        ));
    }
    if let Some(forced) = &input.forced_generated_audio_tokens {
        if forced.dtype() != Dtype::Int32
            || forced.shape() != [batch, args.generated_audio_codebooks()]
        {
            return Err(Error::Parallel(format!(
                "forced realtime audio must be int32 [batch, {}], got {:?} {:?}",
                args.generated_audio_codebooks(),
                forced.dtype(),
                forced.shape()
            )));
        }
    }
    if let Some(forced) = &input.forced_text_token {
        if forced.dtype() != Dtype::Int32 || forced.shape() != [batch, 1] {
            return Err(Error::Parallel(format!(
                "forced realtime text must be int32 [batch, 1], got {:?} {:?}",
                forced.dtype(),
                forced.shape()
            )));
        }
    }
    Ok(())
}

fn validate_generation_state(
    model: &LoadedRealtimeModel,
    state: &moshi::GenerationState,
) -> Result<(), Error> {
    let args = model.args();
    if state.cache.temporal.len() != args.num_layers as usize
        || state.cache.depth.len() != args.depformer_num_layers as usize
        || state
            .frames
            .iter()
            .any(|frame| frame.len() != args.n_q as usize + 1)
    {
        return Err(Error::Parallel(
            "realtime session state does not match the loaded temporal/depth geometry".into(),
        ));
    }
    Ok(())
}

/// Greedily generates delay-aligned codec tokens through the canonical scheduler.
///
/// Input and output use `[batch, codebooks, frames]` layout. This helper does
/// not append encoded silence, so delayed tail frames are not flushed after the
/// supplied input ends.
pub fn generate_encoded_greedy(
    model: &mut LoadedRealtimeModel,
    input_audio_tokens: &Array,
    stream: &Stream,
) -> Result<EncodedAudioOutput, Error> {
    let config = model.realtime_config();
    let input_audio_codebooks = config.input_audio_codebooks;
    let generated_audio_codebooks = config.generated_audio_codebooks;
    let depth_audio_codebooks = config.depth_audio_codebooks;
    if input_audio_tokens.shape().len() != 3 || input_audio_tokens.dim(1) != input_audio_codebooks {
        return Err(Error::Parallel(format!(
            "encoded input sequence must have shape [batch, {}, frames], got {:?}",
            input_audio_codebooks,
            input_audio_tokens.shape()
        )));
    }

    let batch = input_audio_tokens.dim(0);
    let request = RequestId::new(0);
    let mut scheduler = RealtimeInferenceScheduler::new(model, SchedulerLimits::new(1, 1)?)?;
    let audio_samplers = (0..depth_audio_codebooks)
        .map(|_| DefaultSampler)
        .collect::<Vec<_>>();
    scheduler.register_request(
        model,
        request,
        DefaultSampler,
        audio_samplers,
        RealtimeSampling::greedy(),
    )?;
    let mut text = Vec::with_capacity(input_audio_tokens.dim(2) as usize);
    let mut audio = Vec::new();
    for frame in 0..input_audio_tokens.dim(2) {
        let input = input_audio_tokens.try_index_device((.., .., frame), stream)?;
        scheduler.enqueue(model, request, RealtimeStepInput::encoded_audio(&input))?;
        let output = scheduler
            .run_queued(model, stream)?
            .pop()
            .expect("one queued realtime frame")
            .output;
        text.push(output.text_token.squeeze_axes(&[-1], stream)?);
        if let Some(tokens) = output.output_audio_tokens {
            audio.push(tokens);
        }
    }
    scheduler.finish_request(request)?;
    let text_tokens = if text.is_empty() {
        Array::zeros::<i32>(&[batch, 0], stream)?
    } else {
        stack_axis(&text, 1, stream)?
    };
    let audio_tokens = if audio.is_empty() {
        Array::zeros::<i32>(&[batch, generated_audio_codebooks, 0], stream)?
    } else {
        stack_axis(&audio, 2, stream)?
    };
    Ok(EncodedAudioOutput {
        text_tokens,
        audio_tokens,
    })
}

#[cfg(test)]
mod tests {
    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext, Stream};

    use super::*;

    fn tiny_args() -> moshi::ModelArgs {
        serde_json::from_value(serde_json::json!({
            "model_type": "moshi",
            "dim": 16,
            "text_card": 32,
            "n_q": 4,
            "dep_q": 2,
            "card": 8,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 32,
            "causal": true,
            "context": 16,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 8,
            "depformer_dim_feedforward": 16,
            "depformer_num_heads": 2,
            "depformer_num_layers": 2,
            "depformer_context": 2,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 0, 1]
        }))
        .unwrap()
    }

    fn tiny_model(stream: &Stream) -> LoadedRealtimeModel {
        let mut resident = moshi::Model::new(tiny_args(), stream).unwrap();
        for (name, parameter) in resident.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        LoadedRealtimeModel::Moshi(
            crate::architectures::moshi::layerwise::execute_transformed_model(
                resident,
                stream,
                &weights_stream,
            )
            .unwrap(),
        )
    }

    fn default_audio_samplers() -> Vec<DefaultSampler> {
        vec![DefaultSampler, DefaultSampler]
    }

    #[test]
    fn execution_identity_normalizes_defaults_and_tracks_quantization() {
        let mut implicit = tiny_args();
        implicit.dim_feedforward = None;
        implicit.depformer_dim_feedforward = None;
        implicit.depformer_context = None;
        implicit.depformer_max_period = None;
        let mut explicit = implicit.clone();
        explicit.dim_feedforward = Some(explicit.dim * 4);
        explicit.depformer_dim_feedforward = Some(explicit.depformer_dim * 4);
        explicit.depformer_context = Some(explicit.dep_q);
        explicit.depformer_max_period = Some(8.0);
        assert_eq!(
            RealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            RealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );

        explicit.quantization =
            Some(crate::runtime::checkpoint::quantization::AffineQuantization::default().into());
        assert_ne!(
            RealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            RealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );
    }

    #[test]
    fn released_session_rejects_a_different_same_geometry_model() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let first_model = tiny_model(context.stream());
        let second_model = tiny_model(context.stream());
        let first_identity = RealtimeModelIdentity::new(&first_model);
        let second_identity = RealtimeModelIdentity::new(&second_model);
        assert_eq!(first_identity.execution, second_identity.execution);
        assert_ne!(first_identity.artifact, second_identity.artifact);

        let request = RequestId::new(7);
        let mut first_scheduler =
            RealtimeInferenceScheduler::new(&first_model, SchedulerLimits::new(1, 1).unwrap())
                .unwrap();
        first_scheduler
            .register_request(
                &first_model,
                request,
                DefaultSampler,
                default_audio_samplers(),
                RealtimeSampling::greedy(),
            )
            .unwrap();
        let session = first_scheduler.release_request(request).unwrap();

        let mut second_scheduler =
            RealtimeInferenceScheduler::new(&second_model, SchedulerLimits::new(1, 1).unwrap())
                .unwrap();
        let error = second_scheduler
            .register_request_with_session(&second_model, request, session)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("checkpoint artifact fingerprint"));
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn realtime_adapter_uses_generic_capacity_lifecycle_and_state_handoff() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let model = tiny_model(stream);
        let mut scheduler =
            RealtimeInferenceScheduler::new(&model, SchedulerLimits::new(2, 2).unwrap()).unwrap();
        let first = RequestId::new(11);
        let second = RequestId::new(22);
        scheduler
            .register_request(
                &model,
                first,
                DefaultSampler,
                default_audio_samplers(),
                RealtimeSampling::greedy(),
            )
            .unwrap();
        assert!(scheduler
            .register_request(
                &model,
                second,
                DefaultSampler,
                vec![DefaultSampler],
                RealtimeSampling::greedy(),
            )
            .unwrap_err()
            .to_string()
            .contains("requires 2 audio samplers"));
        scheduler
            .register_request(
                &model,
                second,
                DefaultSampler,
                default_audio_samplers(),
                RealtimeSampling::greedy(),
            )
            .unwrap();

        let frame = Array::from_slice(&[1i32, 2], &[1, 2]);
        assert!(scheduler
            .enqueue_batch(
                &model,
                first,
                vec![
                    RealtimeStepInput::encoded_audio(&frame),
                    RealtimeStepInput::encoded_audio(&frame),
                    RealtimeStepInput::encoded_audio(&frame),
                ],
            )
            .unwrap_err()
            .to_string()
            .contains("queue capacity"));
        assert_eq!(scheduler.report().queued_work, 0);
        assert_eq!(scheduler.report().submitted_work, 0);
        scheduler
            .enqueue(&model, first, RealtimeStepInput::encoded_audio(&frame))
            .unwrap();
        scheduler
            .enqueue(&model, second, RealtimeStepInput::encoded_audio(&frame))
            .unwrap();
        assert!(scheduler
            .enqueue(&model, first, RealtimeStepInput::encoded_audio(&frame))
            .unwrap_err()
            .to_string()
            .contains("queue capacity"));
        scheduler.cancel_request(second).unwrap();
        let changed_batch = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        assert!(scheduler
            .enqueue(
                &model,
                first,
                RealtimeStepInput::encoded_audio(&changed_batch),
            )
            .unwrap_err()
            .to_string()
            .contains("changed batch size"));
        scheduler.finish_request(first).unwrap();
        assert_eq!(
            scheduler.request_status(first),
            Some(RequestStatus::Finished)
        );
        assert_eq!(
            scheduler.request_status(second),
            Some(RequestStatus::Cancelled)
        );
        let report = scheduler.report();
        assert_eq!(report.active_requests, 0);
        assert_eq!(report.discarded_work, 2);
        assert_eq!(report.peak_queued_work, 2);

        scheduler.forget_terminal_request(first).unwrap();
        scheduler
            .register_request(
                &model,
                first,
                DefaultSampler,
                default_audio_samplers(),
                RealtimeSampling::greedy(),
            )
            .unwrap();
        let session = scheduler.release_request(first).unwrap();
        assert_eq!(session.step(), 0);
        scheduler
            .register_request_with_session(&model, first, session)
            .unwrap();
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn realtime_scheduler_is_fair_and_matches_independent_reference_sessions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = tiny_model(stream);
        let resident = match &mut model {
            LoadedRealtimeModel::Moshi(model) => model,
            _ => unreachable!(),
        };
        let first_frame = Array::from_slice(&[1i32, 2], &[1, 2]);
        let second_frame = Array::from_slice(&[3i32, 4], &[1, 2]);
        let mut first_state = resident.new_generation_state();
        let mut second_state = resident.new_generation_state();
        let mut first_text = DefaultSampler;
        let mut second_text = DefaultSampler;
        let mut first_audio = default_audio_samplers();
        let mut second_audio = default_audio_samplers();
        let first_zero = resident
            .generate_step(
                &mut first_state,
                &first_frame,
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let first_one = resident
            .generate_step(
                &mut first_state,
                &first_frame,
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_zero = resident
            .generate_step(
                &mut second_state,
                &second_frame,
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_one = resident
            .generate_step(
                &mut second_state,
                &second_frame,
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let forced_audio = Array::from_slice(&[5i32, 6], &[1, 2]);
        let forced_text = Array::from_slice(&[7i32], &[1, 1]);
        let first_forced = resident
            .generate_step_forced(
                &mut first_state,
                &first_frame,
                Some(&forced_audio),
                Some(&forced_text),
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_forced = resident
            .generate_step_forced(
                &mut second_state,
                &second_frame,
                Some(&forced_audio),
                Some(&forced_text),
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let references = [
            first_zero,
            second_zero,
            first_one,
            second_one,
            first_forced,
            second_forced,
        ];

        let first = RequestId::new(11);
        let second = RequestId::new(22);
        let mut scheduler =
            RealtimeInferenceScheduler::new(&model, SchedulerLimits::new(2, 6).unwrap()).unwrap();
        for request in [first, second] {
            scheduler
                .register_request(
                    &model,
                    request,
                    DefaultSampler,
                    default_audio_samplers(),
                    RealtimeSampling::greedy(),
                )
                .unwrap();
        }
        for (request, frame) in [
            (first, &first_frame),
            (first, &first_frame),
            (first, &first_frame),
            (second, &second_frame),
            (second, &second_frame),
            (second, &second_frame),
        ] {
            let sequence = scheduler.queued_for_request(request);
            let input = if sequence == 2 {
                RealtimeStepInput::encoded_audio(frame)
                    .with_forced_generated_audio(&forced_audio)
                    .with_forced_text(&forced_text)
            } else {
                RealtimeStepInput::encoded_audio(frame)
            };
            scheduler.enqueue(&model, request, input).unwrap();
        }
        let mut output = scheduler.run_bounded(&mut model, 2, stream).unwrap();
        assert_eq!(scheduler.report().queued_work, 4);
        output.extend(scheduler.run_queued(&mut model, stream).unwrap());
        assert_eq!(
            output
                .iter()
                .map(|output| (output.work().request().value(), output.work().sequence()))
                .collect::<Vec<_>>(),
            vec![(11, 0), (22, 0), (11, 1), (22, 1), (11, 2), (22, 2)]
        );
        for (expected, actual) in references.iter().zip(&output) {
            assert_tokens_equal(&expected.text_token, &actual.output().text_token, stream);
            assert_tokens_equal(
                &expected.sampled_audio_tokens,
                &actual.output().sampled_audio_tokens,
                stream,
            );
        }
        assert_eq!(scheduler.report().drain_cycles, 2);
        assert_eq!(scheduler.release_request(first).unwrap().step(), 3);
        assert_eq!(scheduler.release_request(second).unwrap().step(), 3);
    }

    fn assert_tokens_equal(expected: &Array, actual: &Array, stream: &Stream) {
        let equal = expected
            .eq(actual, stream)
            .unwrap()
            .all(None, stream)
            .unwrap();
        assert!(equal.item::<bool>(stream));
    }
}
