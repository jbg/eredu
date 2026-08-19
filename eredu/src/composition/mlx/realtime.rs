//! MLX implementation of backend-neutral realtime loading and execution.

use eredu_checkpoint::WeightQuantization;

use std::path::Path;

use eredu_core::{
    backend::{Completion, Submission},
    realtime::{
        RealtimeBackend, RealtimeError, RealtimeModel, RealtimeModelLoadingBackend,
        RealtimeSampling, RealtimeScheduler, RealtimeSpeechConfig,
    },
    scheduler::{RequestId, SchedulerLimits, SemanticStateTransaction, WorkDescriptor},
};
use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    random::{self, RandomState},
    transforms::async_eval_with_event,
    Array, Dtype, Event, Stream,
};
use serde::Deserialize;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::{
        checkpoint::artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
        generation::sampler::DefaultSampler,
    },
    backend::mlx::{ensure_replicated_load_options, ModelLoadOptions},
    composition::mlx_architectures::moshi::{
        layerwise::{self, MoshiLayerwiseModel},
        model as moshi, personaplex,
    },
};

/// Supported MLX realtime speech-to-speech model-family dispatch target.
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

fn realtime_model_kind(model_dir: &Path) -> Result<RealtimeModelKind, Error> {
    let config_path = model_dir.join("config.json");
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
        crate::backend::mlx::runtime::checkpoint::load::safetensors_files(model_dir)?
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

/// Loaded MLX realtime speech-to-speech token model.
///
/// Both model families share the same backend session and frame contract;
/// checkpoint layout is the only dispatch distinction.
pub enum MlxRealtimeModel {
    /// Moshi-family model.
    Moshi(MoshiLayerwiseModel),
    /// PersonaPlex model.
    PersonaPlex(MoshiLayerwiseModel),
}

impl MlxRealtimeModel {
    pub(crate) fn artifact_identity(&self) -> &LoadedArtifactIdentity {
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
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.residency_report().map(Some),
        }
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => model.dense_stream_report(),
        }
    }

    /// Returns per-group residency for the selected parameter policy.
    pub fn execution_group_reports(
        &self,
    ) -> Result<Option<Vec<eredu_runtime::ResidentLayerGroupReport>>, Error> {
        match self {
            Self::Moshi(model) | Self::PersonaPlex(model) => {
                model.execution_group_reports().map(Some)
            }
        }
    }
}

/// MLX encoded input-side audio frame and optional forced prompt values.
#[derive(Debug, Clone)]
pub struct MlxRealtimeInput {
    /// Encoded input-side audio tokens shaped `[batch, input_audio_codebooks]`.
    pub(crate) input_audio_tokens: Array,
    /// Optional generated-side codec tokens forced by a prompt frame.
    pub(crate) forced_generated_audio_tokens: Option<Array>,
    /// Optional text token forced by a prompt frame.
    pub(crate) forced_text_token: Option<Array>,
}

impl MlxRealtimeInput {
    /// Creates an owned MLX realtime step from encoded audio-codebook tokens.
    pub fn encoded_audio(input_audio_tokens: &Array) -> Self {
        Self {
            input_audio_tokens: input_audio_tokens.clone(),
            forced_generated_audio_tokens: None,
            forced_text_token: None,
        }
    }

    /// Forces generated-side codec tokens for a prompt transition.
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

impl WorkDescriptor for MlxRealtimeInput {
    type Error = Error;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        encode_array_descriptor(&self.input_audio_tokens, output)?;
        encode_optional_array_descriptor(self.forced_generated_audio_tokens.as_ref(), output)?;
        encode_optional_array_descriptor(self.forced_text_token.as_ref(), output)
    }
}

/// MLX output from one encoded-audio realtime generation step.
pub struct MlxRealtimeOutput {
    /// Text token sampled at this model step, shaped `[batch, 1]`.
    pub text_token: Array,
    /// Newly sampled generated-codebook tokens before delay alignment.
    pub sampled_audio_tokens: Array,
    /// Delay-aligned codec frame ready for decoding.
    pub output_audio_tokens: Option<Array>,
}

/// MLX text tokens and delay-aligned codec tokens from offline generation.
pub struct MlxEncodedAudioOutput {
    /// Sampled text tokens shaped `[batch, input_frames]`.
    pub text_tokens: Array,
    /// Generated codec tokens shaped `[batch, generated_codebooks, output_frames]`.
    pub audio_tokens: Array,
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
    model: &MlxRealtimeModel,
    input: &MlxRealtimeInput,
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
    model: &MlxRealtimeModel,
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

/// MLX execution assignment for complete realtime model sessions.
#[derive(Clone)]
pub struct MlxRealtimeBackend {
    stream: Stream,
    weights_stream: Stream,
}

impl MlxRealtimeBackend {
    /// Selects execution and weight-materialization streams for one backend.
    pub fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
        }
    }

    /// Selected MLX execution stream.
    pub const fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Selected MLX checkpoint materialization stream.
    pub const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
    }
}

impl RealtimeModelLoadingBackend for MlxRealtimeBackend {
    type LoadOptions = ModelLoadOptions;

    fn prepare_realtime_model(
        &self,
        artifact: &Path,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error> {
        materialize_realtime_model(artifact, options, &self.stream, &self.weights_stream)
    }
}

fn materialize_realtime_model(
    model_dir: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxRealtimeModel, Error> {
    ensure_replicated_load_options(options)?;
    let kind = realtime_model_kind(model_dir)?;
    if options.weight_residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(format!(
            "{} does not contain routed experts",
            kind.model_type()
        )));
    }
    let execution = options.weight_residency.layers();
    let model = match kind {
        RealtimeModelKind::Moshi => layerwise::load_moshi_layerwise_model(
            model_dir,
            execution,
            options.quantization,
            stream,
            weights_stream,
        )?,
        RealtimeModelKind::PersonaPlex => layerwise::load_personaplex_layerwise_model(
            model_dir,
            execution,
            options.quantization,
            stream,
            weights_stream,
        )?,
    };
    let model = model.with_artifact_identity(realtime_artifact_identity(model_dir, kind)?);
    Ok(match kind {
        RealtimeModelKind::Moshi => MlxRealtimeModel::Moshi(model),
        RealtimeModelKind::PersonaPlex => MlxRealtimeModel::PersonaPlex(model),
    })
}

/// Greedily generates delay-aligned codec tokens through the canonical scheduler.
///
/// Input and output use `[batch, codebooks, frames]` layout. This helper does
/// not append encoded silence, so delayed tail frames are not flushed after the
/// supplied input ends.
pub fn generate_encoded_greedy(
    model: &mut RealtimeModel<MlxRealtimeBackend>,
    input_audio_tokens: &Array,
) -> Result<MlxEncodedAudioOutput, Error> {
    let stream = model.backend().stream().clone();
    let config = model.speech_config();
    let input_audio_codebooks = config.input_audio_codebooks() as i32;
    let generated_audio_codebooks = config.generated_audio_codebooks() as i32;
    if input_audio_tokens.shape().len() != 3 || input_audio_tokens.dim(1) != input_audio_codebooks {
        return Err(Error::Parallel(format!(
            "encoded input sequence must have shape [batch, {}, frames], got {:?}",
            input_audio_codebooks,
            input_audio_tokens.shape()
        )));
    }

    let batch = input_audio_tokens.dim(0);
    let request = RequestId::new(0);
    let mut scheduler =
        RealtimeScheduler::new(model, SchedulerLimits::new(1, 1)?).map_err(realtime_error)?;
    scheduler
        .register_request(model, request, RealtimeSampling::greedy())
        .map_err(realtime_error)?;
    let mut text = Vec::with_capacity(input_audio_tokens.dim(2) as usize);
    let mut audio = Vec::new();
    for frame in 0..input_audio_tokens.dim(2) {
        let input = input_audio_tokens.try_index_device((.., .., frame), &stream)?;
        scheduler
            .enqueue(model, request, MlxRealtimeInput::encoded_audio(&input))
            .map_err(realtime_error)?;
        let output = loop {
            if let Some(completed) = scheduler.run_queued(model).map_err(realtime_error)?.pop() {
                break completed.into_parts().1;
            }
            std::thread::yield_now();
        };
        text.push(output.text_token.squeeze_axes(&[-1], &stream)?);
        if let Some(tokens) = output.output_audio_tokens {
            audio.push(tokens);
        }
    }
    scheduler.finish_request(request).map_err(realtime_error)?;
    let text_tokens = if text.is_empty() {
        Array::zeros::<i32>(&[batch, 0], &stream)?
    } else {
        stack_axis(&text, 1, &stream)?
    };
    let audio_tokens = if audio.is_empty() {
        Array::zeros::<i32>(&[batch, generated_audio_codebooks, 0], &stream)?
    } else {
        stack_axis(&audio, 2, &stream)?
    };
    Ok(MlxEncodedAudioOutput {
        text_tokens,
        audio_tokens,
    })
}

fn realtime_error(error: RealtimeError<Error>) -> Error {
    Error::Parallel(error.to_string())
}

/// Complete artifact and execution identity for MLX realtime state handoff.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MlxRealtimeModelIdentity {
    artifact: LoadedArtifactIdentity,
    execution: MlxRealtimeExecutionIdentity,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct MlxRealtimeExecutionIdentity {
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
    quantization: Option<WeightQuantization>,
}

impl MlxRealtimeModelIdentity {
    fn new(model: &MlxRealtimeModel) -> Self {
        Self {
            artifact: model.artifact_identity().clone(),
            execution: MlxRealtimeExecutionIdentity::new(model.kind(), model.args()),
        }
    }

    fn mismatch(&self, actual: &Self) -> Option<&'static str> {
        if self.artifact != actual.artifact {
            Some("checkpoint artifact fingerprint")
        } else if self.execution != actual.execution {
            Some("normalized execution identity")
        } else {
            None
        }
    }
}

impl MlxRealtimeExecutionIdentity {
    pub(crate) fn new(kind: RealtimeModelKind, args: &moshi::ModelArgs) -> Self {
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

/// MLX-owned realtime cache, delayed streams, samplers, and randomness.
#[derive(Clone)]
pub struct MlxRealtimeSession {
    generation: moshi::GenerationState,
    text_sampler: DefaultSampler,
    audio_samplers: Vec<DefaultSampler>,
    sampling: RealtimeSampling,
    prng_state: Option<RandomState>,
}

impl MlxRealtimeSession {
    /// Number of committed encoded frames.
    pub fn step(&self) -> usize {
        self.generation.step()
    }

    /// MLX temporal/depth cache and delayed-token state.
    pub const fn generation_state(&self) -> &moshi::GenerationState {
        &self.generation
    }

    fn set_sampling(&mut self, sampling: RealtimeSampling) -> Result<(), Error> {
        self.prng_state = sampling
            .is_stochastic()
            .then(|| random::key(sampling.seed()).map(RandomState::from_key))
            .transpose()?;
        self.sampling = sampling;
        Ok(())
    }
}

impl SemanticStateTransaction for MlxRealtimeSession {
    type Branch = Self;
    type Error = Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(self.clone())
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        *self = branch;
        Ok(())
    }
}

/// Exact MLX event retaining generated token arrays.
pub struct MlxRealtimeCompletion {
    event: Event,
    retained: Vec<Array>,
}

impl MlxRealtimeCompletion {
    fn submit(output: &MlxRealtimeOutput) -> Result<Self, Error> {
        let retained = std::iter::once(output.text_token.clone())
            .chain(std::iter::once(output.sampled_audio_tokens.clone()))
            .chain(output.output_audio_tokens.iter().cloned())
            .collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self { event, retained })
    }

    /// Number of array handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}

impl Completion for MlxRealtimeCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete().map_err(Into::into)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize().map_err(Into::into)
    }
}

impl Drop for MlxRealtimeCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

impl RealtimeBackend for MlxRealtimeBackend {
    type Model = MlxRealtimeModel;
    type ModelIdentity = MlxRealtimeModelIdentity;
    type Input = MlxRealtimeInput;
    type Output = MlxRealtimeOutput;
    type Session = MlxRealtimeSession;
    type Completion = MlxRealtimeCompletion;
    type Error = Error;

    fn name(&self) -> &str {
        "mlx"
    }

    fn model_identity(&self, model: &Self::Model) -> Self::ModelIdentity {
        MlxRealtimeModelIdentity::new(model)
    }

    fn model_identity_mismatch(
        &self,
        expected: &Self::ModelIdentity,
        actual: &Self::ModelIdentity,
    ) -> Option<String> {
        expected.mismatch(actual).map(str::to_owned)
    }

    fn speech_config(&self, model: &Self::Model) -> RealtimeSpeechConfig {
        let args = model.args();
        RealtimeSpeechConfig::new(
            args.n_q as usize,
            args.input_audio_codebooks() as usize,
            args.generated_audio_codebooks() as usize,
            args.dep_q as usize,
            args.text_padding_token(),
            args.audio_padding_token(),
            args.audio_delays()
                .iter()
                .map(|delay| *delay as usize)
                .collect(),
        )
        .expect("loaded MLX realtime model has validated codec geometry")
    }

    fn create_session(
        &self,
        model: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error> {
        let generation = match model {
            MlxRealtimeModel::Moshi(model) | MlxRealtimeModel::PersonaPlex(model) => {
                model.new_generation_state()
            }
        };
        let mut session = MlxRealtimeSession {
            generation,
            text_sampler: DefaultSampler,
            audio_samplers: vec![DefaultSampler; model.args().dep_q as usize],
            sampling: RealtimeSampling::greedy(),
            prng_state: None,
        };
        session.set_sampling(sampling)?;
        Ok(session)
    }

    fn validate_session(
        &self,
        model: &Self::Model,
        session: &Self::Session,
    ) -> Result<(), Self::Error> {
        validate_generation_state(model, &session.generation)
    }

    fn validate_input(&self, model: &Self::Model, input: &Self::Input) -> Result<(), Self::Error> {
        validate_realtime_input(model, input)
    }

    fn input_batch_size(&self, input: &Self::Input) -> usize {
        input.input_audio_tokens.dim(0) as usize
    }

    fn set_sampling(
        &self,
        session: &mut Self::Session,
        sampling: RealtimeSampling,
    ) -> Result<(), Self::Error> {
        session.set_sampling(sampling)
    }

    fn submit_step(
        &self,
        model: &mut Self::Model,
        session: &mut Self::Session,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error> {
        let output = match model {
            MlxRealtimeModel::Moshi(model) | MlxRealtimeModel::PersonaPlex(model) => model
                .generate_step_forced(
                    &mut session.generation,
                    &input.input_audio_tokens,
                    input.forced_generated_audio_tokens.as_ref(),
                    input.forced_text_token.as_ref(),
                    &mut session.text_sampler,
                    &mut session.audio_samplers,
                    session.sampling.text_temperature(),
                    session.sampling.audio_temperature(),
                    session.prng_state.as_mut(),
                    &self.stream,
                )
                .map_err(Error::from)?,
        };
        let completion = MlxRealtimeCompletion::submit(&output)?;
        Ok(Submission { output, completion })
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }
}

#[cfg(test)]
mod tests {
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
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );

        explicit.quantization = Some(eredu_checkpoint::AffineQuantization::default().into());
        assert_ne!(
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );
    }

    #[test]
    fn model_identity_rejects_equal_geometry_from_another_artifact() {
        let execution = MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &tiny_args());
        let expected = MlxRealtimeModelIdentity {
            artifact: LoadedArtifactIdentity::in_memory(),
            execution: execution.clone(),
        };
        let actual = MlxRealtimeModelIdentity {
            artifact: LoadedArtifactIdentity::in_memory(),
            execution,
        };
        assert_eq!(
            expected.mismatch(&actual),
            Some("checkpoint artifact fingerprint")
        );
    }
}
