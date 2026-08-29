//! MLX implementation of backend-neutral realtime loading and execution.

/// PersonaPlex prompt framing and forced-prompt enqueue protocol.
pub mod personaplex_prompt;

use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::moshi::{
    ArtifactProfile, EffectiveModelType, MoshiConfig, RealtimePreparationPlan,
};
use eredu_core::{
    backend::{Completion, Submission},
    realtime::{
        RealtimeBackend, RealtimeDecisionDiagnostics, RealtimeFrameForcing, RealtimeFrameSlot,
        RealtimeInputFrame, RealtimeModelLoadingBackend, RealtimeOutputFrame, RealtimeSampling,
        RealtimeSlotCoordinate, RealtimeSpeechConfig, RealtimeTargetSource, RealtimeTemporalSource,
    },
    scheduler::{SemanticStateTransaction, WorkDescriptor},
};
#[cfg(test)]
use eredu_core::{
    realtime::{RealtimeModel, RealtimeScheduler},
    scheduler::{RequestId, SchedulerLimits},
};
#[cfg(test)]
use eredu_runtime::DefaultSampler;
use eredu_runtime::{
    GenerationSampler, PredictionDirective, RealtimeGenerationBranch, RealtimeGenerationState,
    RuntimeState, SequentialDecisionPlan,
};
use safemlx::{
    distributed::Group,
    ops::{indexing::TryIndexOp, stack_axis},
    random::{self, RandomState},
    transforms::async_eval_with_event,
    Array, Dtype, Event, Stream,
};

use crate::{
    backend::runtime::{
        cache::state::{MlxKeyValueState, MlxKeyValueTransactionBranch},
        checkpoint::artifact::LoadedArtifactIdentity,
        generation::sampler::MlxSamplingBackend,
    },
    backend::ModelLoadOptions,
    backend::{
        error::Error,
        nn::tensor::{validate_token_domain, TokenValidationBatch, TokenValidationScope},
    },
    composition::moshi::{self as neutral_moshi, MoshiModel as NeutralMoshiModel},
    MlxTensor,
};

/// Loaded MLX realtime speech-to-speech token model.
pub struct MlxRealtimeModel {
    model: NeutralMoshiModel,
}

impl MlxRealtimeModel {
    /// Returns the stable identity of the loaded checkpoint artifact.
    pub fn artifact_identity(&self) -> &LoadedArtifactIdentity {
        self.model.artifact_identity()
    }

    /// Returns the architecture-owned effective model identity.
    pub fn effective_model_type(&self) -> EffectiveModelType {
        self.model.config().effective_model_type()
    }

    /// Returns the normalized Moshi-family configuration.
    pub fn config(&self) -> &MoshiConfig {
        self.model.config()
    }

    /// Returns the normalized source artifact profile.
    pub fn profile(&self) -> ArtifactProfile {
        self.model.source_config().artifact_profile()
    }

    /// Returns fail-closed capabilities of this realtime session route.
    pub const fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        realtime_session_capabilities()
    }

    /// Returns parameter topology, residency, quantization, and materialization metadata.
    pub fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        self.model.metadata()
    }

    /// Returns current residency telemetry for the selected parameter policy.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.model.residency_report().map(Some)
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.model.dense_stream_report()
    }

    /// Returns per-group residency for the selected parameter policy.
    pub fn execution_group_reports(
        &self,
    ) -> Result<Option<Vec<eredu_runtime::ResidentLayerGroupReport>>, Error> {
        self.model.execution_group_reports().map(Some)
    }
}

/// MLX encoded input-side audio frame and optional forced prompt values.
#[derive(Debug, Clone)]
pub struct MlxRealtimeInput {
    /// Encoded input-side audio tokens shaped `[batch, input_audio_codebooks]`.
    pub input_audio_tokens: Array,
    /// Optional generated-side codec tokens forced by a prompt frame.
    pub forced_generated_audio_tokens: Option<Array>,
    /// Optional per-generated-codebook forcing mask.
    pub forced_generated_audio_codebooks: Option<Vec<bool>>,
    /// Optional text token forced by a prompt frame.
    pub forced_text_token: Option<Array>,
    /// Whether complete decision logits are retained for host observation.
    pub retain_diagnostics: bool,
}

impl MlxRealtimeInput {
    /// Creates an owned MLX realtime step from encoded audio-codebook tokens.
    pub fn encoded_audio(input_audio_tokens: &Array) -> Self {
        Self {
            input_audio_tokens: input_audio_tokens.clone(),
            forced_generated_audio_tokens: None,
            forced_generated_audio_codebooks: None,
            forced_text_token: None,
            retain_diagnostics: false,
        }
    }

    /// Forces generated-side codec tokens for a prompt transition.
    pub fn with_forced_generated_audio(mut self, tokens: &Array) -> Self {
        self.forced_generated_audio_tokens = Some(tokens.clone());
        self.forced_generated_audio_codebooks = None;
        self
    }

    /// Forces only selected generated codec decisions from the supplied frame.
    pub fn with_partially_forced_generated_audio(
        mut self,
        tokens: &Array,
        forced_codebooks: impl Into<Vec<bool>>,
    ) -> Self {
        self.forced_generated_audio_tokens = Some(tokens.clone());
        self.forced_generated_audio_codebooks = Some(forced_codebooks.into());
        self
    }

    /// Forces the generated text token for a prompt transition.
    pub fn with_forced_text(mut self, token: &Array) -> Self {
        self.forced_text_token = Some(token.clone());
        self
    }

    /// Retains complete text and depth-decision logits in the completed output.
    pub fn with_diagnostics(mut self) -> Self {
        self.retain_diagnostics = true;
        self
    }
}

impl WorkDescriptor for MlxRealtimeInput {
    type Error = Error;

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        encode_array_descriptor(&self.input_audio_tokens, output)?;
        encode_optional_array_descriptor(self.forced_generated_audio_tokens.as_ref(), output)?;
        match &self.forced_generated_audio_codebooks {
            Some(mask) => {
                output.push(1);
                output.push(u32::try_from(mask.len()).map_err(|_| {
                    Error::Parallel("realtime forcing mask length exceeds descriptor range".into())
                })?);
                output.extend(mask.iter().map(|forced| u32::from(*forced)));
            }
            None => output.push(0),
        }
        encode_optional_array_descriptor(self.forced_text_token.as_ref(), output)?;
        output.push(u32::from(self.retain_diagnostics));
        Ok(())
    }
}

/// MLX output from one encoded-audio realtime generation step.
pub struct MlxRealtimeOutput {
    /// Text token sampled at this model step, shaped `[batch, 1]`.
    pub text_token: Array,
    /// Audio tokens resolved at every depth decision, shaped `[batch, depth]`.
    pub decision_audio_tokens: Array,
    /// Newly sampled generated-codebook tokens before delay alignment.
    pub sampled_audio_tokens: Array,
    /// Delay-aligned codec frame ready for decoding.
    pub output_audio_tokens: Option<Array>,
    /// Complete text-then-depth decision logits when requested by the input.
    pub diagnostics: Vec<Array>,
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
    let schedule = model.config().frame_schedule();
    let tokens = &input.input_audio_tokens;
    if tokens.dtype() != Dtype::Int32
        || tokens.ndim() != 2
        || tokens.dim(1) != schedule.input_audio_codebooks() as i32
    {
        return Err(Error::Parallel(format!(
            "realtime input must be int32 [batch, {}], got {:?} {:?}",
            schedule.input_audio_codebooks(),
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
            || forced.shape() != [batch, schedule.generated_audio_codebooks() as i32]
        {
            return Err(Error::Parallel(format!(
                "forced realtime audio must be int32 [batch, {}], got {:?} {:?}",
                schedule.generated_audio_codebooks(),
                forced.dtype(),
                forced.shape()
            )));
        }
    }
    if let Some(mask) = &input.forced_generated_audio_codebooks {
        if input.forced_generated_audio_tokens.is_none()
            || mask.len() != schedule.generated_audio_codebooks()
        {
            return Err(Error::Parallel(format!(
                "partial realtime forcing requires {} token columns and mask entries",
                schedule.generated_audio_codebooks()
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

/// MLX execution assignment for complete realtime model sessions.
#[derive(Clone)]
pub struct MlxRealtimeBackend {
    stream: Stream,
    weights_stream: Stream,
    tensor_parallel_group: Option<Arc<Group>>,
}

impl MlxRealtimeBackend {
    /// Selects execution and weight-materialization streams for one backend.
    pub fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            tensor_parallel_group: None,
        }
    }

    /// Supplies the rank-local tensor-parallel collective group.
    pub fn with_tensor_parallel_group(mut self, group: Arc<Group>) -> Self {
        self.tensor_parallel_group = Some(group);
        self
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
    type Preparation = RealtimePreparationPlan;
    type LoadOptions = ModelLoadOptions;

    fn materialize_realtime_model(
        &self,
        preparation: Self::Preparation,
        options: Self::LoadOptions,
    ) -> Result<Self::Model, Self::Error> {
        validate_realtime_session_requirements(&options)?;
        let model =
            materialize_realtime_model(preparation, options, &self.stream, &self.weights_stream)?;
        if let Some(topology) = model.model.topology() {
            let group = self.tensor_parallel_group.as_deref().ok_or_else(|| {
                Error::Parallel(
                    "tensor-parallel realtime loading requires a TP collective group".into(),
                )
            })?;
            if group.rank() != topology.tensor_parallel_rank
                || group.size() != topology.tensor_parallel_size
            {
                return Err(Error::Parallel(format!(
                    "realtime TP group rank/size {}/{} does not match model topology {}/{}",
                    group.rank(),
                    group.size(),
                    topology.tensor_parallel_rank,
                    topology.tensor_parallel_size
                )));
            }
        }
        Ok(model)
    }
}

const fn realtime_session_capabilities() -> eredu_core::SessionCapabilities {
    eredu_core::SessionCapabilities {
        persistent_cache: true,
        output_observation: true,
        activation_inspection: false,
    }
}

fn validate_realtime_session_requirements(options: &ModelLoadOptions) -> Result<(), Error> {
    options
        .required_session_capabilities
        .validate(&realtime_session_capabilities())?;
    Ok(())
}

fn realtime_sampler(top_k: Option<usize>) -> Result<GenerationSampler, Error> {
    let top_k = top_k
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::Parallel("realtime top-k exceeds i32".into()))?
        .unwrap_or(0);
    Ok(GenerationSampler::new().top_k(top_k).top_p(1.0).min_p(0.0))
}

fn realtime_samplers(
    schedule: &RealtimeSpeechConfig,
    sampling: RealtimeSampling,
) -> Result<Vec<GenerationSampler>, Error> {
    std::iter::once(realtime_sampler(sampling.text_top_k()))
        .chain(std::iter::repeat_with(|| {
            realtime_sampler(sampling.audio_top_k())
        }))
        .take(schedule.depth_audio_codebooks() + 1)
        .collect()
}

fn materialize_realtime_model(
    preparation: RealtimePreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxRealtimeModel, Error> {
    if options.weight_residency.expert_cache().is_some() {
        return Err(Error::ArchitectureModel(
            "Moshi does not contain routed experts".into(),
        ));
    }
    neutral_moshi::load(preparation, options, stream, weights_stream)
        .map(|model| MlxRealtimeModel { model })
}

fn array_i32_host(array: &Array) -> Result<Vec<i32>, Error> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated.as_slice::<i32>().to_vec()),
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Int64 => evaluated
            .as_slice::<i64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        dtype => Err(Error::Parallel(format!(
            "realtime token observation expected integer values, got {dtype:?}"
        ))),
    }
}

fn array_f32_host(array: &Array, stream: &Stream) -> Result<Vec<f32>, Error> {
    let array = if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32, stream)?
    };
    Ok(array.evaluated()?.as_slice::<f32>().to_vec())
}

/// Complete artifact and execution identity for MLX realtime state handoff.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MlxRealtimeModelIdentity {
    artifact: LoadedArtifactIdentity,
    source_architecture: String,
    execution_architecture: String,
}

impl MlxRealtimeModelIdentity {
    fn new(model: &MlxRealtimeModel) -> Self {
        Self {
            artifact: model.artifact_identity().clone(),
            source_architecture: model.model.identity().source_architecture().into(),
            execution_architecture: model.model.identity().execution_architecture().into(),
        }
    }

    fn mismatch(&self, actual: &Self) -> Option<&'static str> {
        if self.artifact != actual.artifact {
            Some("checkpoint artifact fingerprint")
        } else if self.source_architecture != actual.source_architecture {
            Some("normalized source architecture")
        } else if self.execution_architecture != actual.execution_architecture {
            Some("normalized execution identity")
        } else {
            None
        }
    }
}

/// MLX cache plus backend-native payloads keyed by portable coordinates.
#[derive(Debug)]
pub struct MlxRealtimeModelState {
    cache: MlxKeyValueState,
    tokens: BTreeMap<RealtimeSlotCoordinate, Array>,
}

/// Unpublished MLX cache/payload branch.
#[derive(Debug)]
pub struct MlxRealtimeModelStateBranch {
    cache: MlxKeyValueTransactionBranch,
    tokens: BTreeMap<RealtimeSlotCoordinate, Array>,
}

impl SemanticStateTransaction for MlxRealtimeModelState {
    type Branch = MlxRealtimeModelStateBranch;
    type Error = Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(MlxRealtimeModelStateBranch {
            cache: self.cache.branch()?,
            tokens: self.tokens.clone(),
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.cache.commit_branch(branch.cache)?;
        self.tokens = branch.tokens;
        Ok(())
    }

    fn permits_parallel_branches(&self) -> bool {
        self.cache.permits_parallel_branches()
    }
}

/// Atomic MLX realtime cache, schedule, samplers, and randomness.
pub struct MlxRealtimeSession {
    generation: RealtimeGenerationState<
        MlxRealtimeModelState,
        GenerationSampler,
        RandomState,
        MlxRealtimeCompletion,
    >,
    sampling: RealtimeSampling,
}

impl MlxRealtimeSession {
    /// Number of committed encoded frames.
    pub fn step(&self) -> usize {
        self.generation.schedule_state().frontier()
    }

    /// Atomic portable realtime generation state.
    pub const fn generation_state(
        &self,
    ) -> &RealtimeGenerationState<
        MlxRealtimeModelState,
        GenerationSampler,
        RandomState,
        MlxRealtimeCompletion,
    > {
        &self.generation
    }

    fn set_sampling(&mut self, sampling: RealtimeSampling) -> Result<(), Error> {
        let random = sampling
            .is_stochastic()
            .then(|| random::key(sampling.seed()).map(RandomState::from_key))
            .transpose()?;
        self.generation
            .set_samplers(realtime_samplers(
                self.generation.schedule_state().schedule(),
                sampling,
            )?)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.generation.set_random_state(random);
        self.sampling = sampling;
        Ok(())
    }
}

/// Unpublished atomic realtime transition.
pub struct MlxRealtimeSessionBranch {
    generation: RealtimeGenerationBranch<
        MlxRealtimeModelStateBranch,
        GenerationSampler,
        RandomState,
        MlxRealtimeCompletion,
    >,
    sampling: RealtimeSampling,
}

impl SemanticStateTransaction for MlxRealtimeSession {
    type Branch = MlxRealtimeSessionBranch;
    type Error = Error;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(MlxRealtimeSessionBranch {
            generation: self
                .generation
                .branch()
                .map_err(|error| Error::Parallel(error.to_string()))?,
            sampling: self.sampling,
        })
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.generation
            .commit_branch(branch.generation)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.sampling = branch.sampling;
        Ok(())
    }

    fn permits_parallel_branches(&self) -> bool {
        self.generation.permits_parallel_branches()
    }
}

/// Exact MLX event retaining generated token arrays.
#[derive(Clone)]
pub struct MlxRealtimeCompletion {
    inner: Arc<MlxRealtimeCompletionInner>,
}

struct MlxRealtimeCompletionInner {
    event: Event,
    retained: Vec<Array>,
    token_validations: TokenValidationBatch,
}

impl MlxRealtimeCompletion {
    fn submit(
        output: &MlxRealtimeOutput,
        state_values: impl IntoIterator<Item = Array>,
        token_validations: TokenValidationBatch,
    ) -> Result<Self, Error> {
        let mut retained = std::iter::once(output.text_token.clone())
            .chain(std::iter::once(output.decision_audio_tokens.clone()))
            .chain(std::iter::once(output.sampled_audio_tokens.clone()))
            .chain(output.output_audio_tokens.iter().cloned())
            .chain(output.diagnostics.iter().cloned())
            .chain(state_values)
            .collect::<Vec<_>>();
        retained.extend(token_validations.arrays().cloned());
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self {
            inner: Arc::new(MlxRealtimeCompletionInner {
                event,
                retained,
                token_validations,
            }),
        })
    }

    /// Number of array handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.inner.retained.len()
    }
}

impl Completion for MlxRealtimeCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.inner.event.is_complete().map_err(Into::into)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.inner.event.synchronize()?;
        self.inner.token_validations.validate_completed()?;
        Ok(())
    }
}

impl Drop for MlxRealtimeCompletionInner {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

fn forcing_mask(
    input: &MlxRealtimeInput,
    schedule: &eredu_core::RealtimeSpeechConfig,
) -> RealtimeFrameForcing {
    let generated_audio = match &input.forced_generated_audio_tokens {
        None => vec![false; schedule.generated_audio_codebooks()],
        Some(_) => input
            .forced_generated_audio_codebooks
            .clone()
            .unwrap_or_else(|| vec![true; schedule.generated_audio_codebooks()]),
    };
    RealtimeFrameForcing::new(input.forced_text_token.is_some(), generated_audio)
}

fn token_column(array: &Array, column: usize, stream: &Stream) -> Result<Array, Error> {
    array
        .try_index_device((.., column as i32..column as i32 + 1), stream)
        .map_err(Into::into)
}

fn padding_token(
    slot: RealtimeFrameSlot,
    config: &eredu_architectures::moshi::MoshiConfig,
    batch: i32,
    stream: &Stream,
) -> Result<Array, Error> {
    let token = match slot {
        RealtimeFrameSlot::Text => config.frame_schedule().text_padding_token(),
        RealtimeFrameSlot::Audio(_) => config.frame_schedule().audio_padding_token(),
    };
    Array::full::<i32>(&[batch, 1], Array::from_int(token), stream).map_err(Into::into)
}

fn forced_token(
    slot: RealtimeFrameSlot,
    input: &MlxRealtimeInput,
    stream: &Stream,
) -> Result<Array, Error> {
    match slot {
        RealtimeFrameSlot::Text => input
            .forced_text_token
            .clone()
            .ok_or_else(|| Error::Parallel("forced text target has no payload".into())),
        RealtimeFrameSlot::Audio(codebook) => token_column(
            input
                .forced_generated_audio_tokens
                .as_ref()
                .ok_or_else(|| Error::Parallel("forced audio target has no payload".into()))?,
            codebook,
            stream,
        ),
    }
}

fn validate_realtime_token_payloads(
    config: &eredu_architectures::moshi::MoshiConfig,
    input: &MlxRealtimeInput,
    stream: &Stream,
) -> Result<(), Error> {
    let text_with_padding = config
        .text_vocabulary_size()
        .checked_add(1)
        .ok_or_else(|| Error::Parallel("text token domain overflowed int32".into()))?;
    let audio_with_padding = config
        .audio_vocabulary_size()
        .checked_add(1)
        .ok_or_else(|| Error::Parallel("audio token domain overflowed int32".into()))?;
    validate_token_domain(&input.input_audio_tokens, audio_with_padding, None, stream)?;
    if let Some(text) = &input.forced_text_token {
        validate_token_domain(text, text_with_padding, None, stream)?;
    }
    if let Some(audio) = &input.forced_generated_audio_tokens {
        let forced = input
            .forced_generated_audio_codebooks
            .as_deref()
            .map_or_else(
                || vec![true; config.frame_schedule().generated_audio_codebooks()],
                <[bool]>::to_vec,
            );
        for (codebook, forced) in forced.into_iter().enumerate() {
            if forced {
                let token = token_column(audio, codebook, stream)?;
                validate_token_domain(&token, audio_with_padding, None, stream)?;
            }
        }
    }
    Ok(())
}

fn submit_neutral_step(
    model: &mut MlxRealtimeModel,
    branch: &mut MlxRealtimeSessionBranch,
    input: &MlxRealtimeInput,
    stream: &Stream,
    tensor_parallel_group: Option<&Group>,
) -> Result<Submission<MlxRealtimeOutput, MlxRealtimeCompletion>, Error> {
    let token_validation_scope = TokenValidationScope::begin()?;
    let config = model.config().clone();
    validate_realtime_token_payloads(&config, input, stream)?;
    let schedule = config.frame_schedule().clone();
    let forcing = forcing_mask(input, &schedule);
    let transition = branch
        .generation
        .schedule_state_mut()
        .advance(&schedule, &forcing)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let batch = input.input_audio_tokens.dim(0);
    {
        let state = branch.generation.model_state_mut();
        for (column, coordinate) in transition.input_placements().iter().copied().enumerate() {
            state.tokens.insert(
                coordinate,
                token_column(&input.input_audio_tokens, column, stream)?,
            );
        }
        for coordinate in transition.forced_placements().iter().copied() {
            state
                .tokens
                .insert(coordinate, forced_token(coordinate.slot(), input, stream)?);
        }
        for coordinate in transition.warmup_padding().iter().copied() {
            state.tokens.insert(
                coordinate,
                padding_token(coordinate.slot(), &config, batch, stream)?,
            );
        }
    }

    if !transition.model_call_required() {
        let output = MlxRealtimeOutput {
            text_token: padding_token(RealtimeFrameSlot::Text, &config, batch, stream)?,
            decision_audio_tokens: Array::zeros::<i32>(&[batch, 0], stream)?,
            sampled_audio_tokens: Array::full::<i32>(
                &[batch, schedule.generated_audio_codebooks() as i32],
                Array::from_int(schedule.audio_padding_token()),
                stream,
            )?,
            output_audio_tokens: None,
            diagnostics: Vec::new(),
        };
        let retained = {
            let state = branch.generation.model_state_mut();
            state
                .cache
                .retained_arrays()
                .into_iter()
                .cloned()
                .chain(state.tokens.values().cloned())
                .collect::<Vec<_>>()
        };
        let completion =
            MlxRealtimeCompletion::submit(&output, retained, token_validation_scope.finish())?;
        branch
            .generation
            .attach_submission_completion(completion.clone())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        return Ok(Submission { output, completion });
    }

    let temporal = {
        let state = branch.generation.model_state_mut();
        transition
            .temporal_inputs()
            .iter()
            .map(|source| match source {
                RealtimeTemporalSource::Padding(slot) => {
                    padding_token(*slot, &config, batch, stream)
                }
                RealtimeTemporalSource::Occupied { coordinate, .. } => {
                    state.tokens.get(coordinate).cloned().ok_or_else(|| {
                        Error::Parallel(format!(
                            "realtime payload is missing occupied coordinate {coordinate:?}"
                        ))
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let text_input = temporal
        .first()
        .ok_or_else(|| Error::Parallel("realtime transition has no text input".into()))?;
    let audio_inputs = temporal.iter().skip(1).collect::<Vec<_>>();

    let directives = {
        let state = branch.generation.model_state_mut();
        transition
            .targets()
            .iter()
            .map(|target| match target.source() {
                RealtimeTargetSource::Sampled => Ok(PredictionDirective::Sample),
                RealtimeTargetSource::Forced => forced_token(target.slot(), input, stream)
                    .map(MlxTensor::from_array)
                    .map(PredictionDirective::Force),
                RealtimeTargetSource::Existing(_) => target
                    .coordinate()
                    .and_then(|coordinate| state.tokens.get(&coordinate))
                    .cloned()
                    .map(MlxTensor::from_array)
                    .map(PredictionDirective::Force)
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "existing realtime target {:?} has no backend payload",
                            target.coordinate()
                        ))
                    }),
            })
            .collect::<Result<Vec<_>, Error>>()?
    };
    let plan = SequentialDecisionPlan::new(
        directives,
        input.retain_diagnostics,
        !input.retain_diagnostics,
    )
    .map_err(|error| Error::Parallel(error.to_string()))?;
    let temperatures = std::iter::once(branch.sampling.text_temperature())
        .chain(std::iter::repeat_n(
            branch.sampling.audio_temperature(),
            schedule.depth_audio_codebooks(),
        ))
        .collect::<Vec<_>>();
    let mut driver = branch
        .generation
        .decision_driver::<MlxSamplingBackend>(plan, temperatures)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    {
        let state = branch.generation.model_state_mut();
        let temporal = eredu_architectures::moshi::Input {
            text: text_input,
            audio: &audio_inputs,
            mask: None,
        };
        match (model.model.topology(), tensor_parallel_group) {
            (Some(_), Some(group)) => model.model.forward_realtime_parallel(
                temporal,
                &mut state.cache,
                &mut driver,
                group,
                stream,
            )?,
            (Some(_), None) => {
                return Err(Error::Parallel(
                    "tensor-parallel realtime execution has no TP collective group".into(),
                ))
            }
            (None, _) => {
                model
                    .model
                    .forward_realtime(temporal, &mut state.cache, &mut driver, stream)?
            }
        };
    }
    let decisions = driver
        .decisions()
        .iter()
        .map(|decision| decision.token().as_array().clone())
        .collect::<Vec<_>>();
    let diagnostics = driver
        .diagnostics()
        .iter()
        .map(|diagnostic| diagnostic.logits().as_array().clone())
        .collect::<Vec<_>>();
    branch
        .generation
        .adopt_decision_driver(driver)
        .map_err(|error| Error::Parallel(error.to_string()))?;

    let text_token = decisions
        .first()
        .cloned()
        .ok_or_else(|| Error::Parallel("realtime text decision is missing".into()))?;
    let sampled_audio_tokens = if schedule.generated_audio_codebooks() == 0 {
        Array::zeros::<i32>(&[batch, 0], stream)?
    } else {
        stack_axis(
            &decisions[1..1 + schedule.generated_audio_codebooks()],
            1,
            stream,
        )?
        .squeeze_axes(&[-1], stream)?
    };
    let decision_audio_tokens = if decisions.len() <= 1 {
        Array::zeros::<i32>(&[batch, 0], stream)?
    } else {
        stack_axis(&decisions[1..], 1, stream)?.squeeze_axes(&[-1], stream)?
    };
    let (output_audio_tokens, retained) = {
        let state = branch.generation.model_state_mut();
        for (target, token) in transition.targets().iter().zip(&decisions) {
            if let Some(coordinate) = target.coordinate() {
                state.tokens.insert(coordinate, token.clone());
            }
        }
        let output = transition
            .output()
            .map(|coordinates| {
                coordinates
                    .iter()
                    .map(|coordinate| {
                        state.tokens.get(coordinate).cloned().ok_or_else(|| {
                            Error::Parallel(format!(
                                "aligned realtime output is missing coordinate {coordinate:?}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .and_then(|tokens| {
                        stack_axis(&tokens, 1, stream)
                            .and_then(|tokens| tokens.squeeze_axes(&[-1], stream))
                            .map_err(Into::into)
                    })
            })
            .transpose()?;
        let minimum = transition
            .next_frontier()
            .saturating_sub(schedule.max_delay().saturating_add(2));
        state
            .tokens
            .retain(|coordinate, _| coordinate.position() >= minimum);
        let retained = state
            .cache
            .retained_arrays()
            .into_iter()
            .cloned()
            .chain(state.tokens.values().cloned())
            .collect::<Vec<_>>();
        (output, retained)
    };
    let output = MlxRealtimeOutput {
        text_token,
        decision_audio_tokens,
        sampled_audio_tokens,
        output_audio_tokens,
        diagnostics,
    };
    let completion =
        MlxRealtimeCompletion::submit(&output, retained, token_validation_scope.finish())?;
    branch
        .generation
        .attach_submission_completion(completion.clone())
        .map_err(|error| Error::Parallel(error.to_string()))?;
    Ok(Submission { output, completion })
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

    fn session_capabilities(&self, model: &Self::Model) -> eredu_core::SessionCapabilities {
        model.session_capabilities()
    }

    fn model_identity_mismatch(
        &self,
        expected: &Self::ModelIdentity,
        actual: &Self::ModelIdentity,
    ) -> Option<String> {
        expected.mismatch(actual).map(str::to_owned)
    }

    fn speech_config(&self, model: &Self::Model) -> RealtimeSpeechConfig {
        model.config().frame_schedule().clone()
    }

    fn materialize_input(
        &self,
        model: &Self::Model,
        frame: &RealtimeInputFrame,
    ) -> Result<Self::Input, Self::Error> {
        let schedule = model.config().frame_schedule();
        let batch = frame.batch();
        if batch == 0 {
            return Err(Error::Parallel(
                "portable realtime input batch must be positive".into(),
            ));
        }
        let batch_i32 = i32::try_from(batch)
            .map_err(|_| Error::Parallel("realtime input batch exceeds i32".into()))?;
        let input_codebooks = schedule.input_audio_codebooks();
        let expected_input = batch
            .checked_mul(input_codebooks)
            .ok_or_else(|| Error::Parallel("realtime input shape overflow".into()))?;
        if frame.input_audio_tokens().len() != expected_input {
            return Err(Error::Parallel(format!(
                "portable realtime input has {} audio tokens, expected {expected_input}",
                frame.input_audio_tokens().len()
            )));
        }
        let input_audio_tokens = Array::from_slice(
            frame.input_audio_tokens(),
            &[batch_i32, input_codebooks as i32],
        )
        .copy(&self.stream)?;
        let mut input = MlxRealtimeInput::encoded_audio(&input_audio_tokens);
        if let Some(tokens) = frame.forced_generated_audio_tokens() {
            let generated = schedule.generated_audio_codebooks();
            let expected = batch
                .checked_mul(generated)
                .ok_or_else(|| Error::Parallel("realtime forcing shape overflow".into()))?;
            if tokens.len() != expected {
                return Err(Error::Parallel(format!(
                    "portable realtime generated-audio forcing has {} tokens, expected {expected}",
                    tokens.len()
                )));
            }
            let tokens =
                Array::from_slice(tokens, &[batch_i32, generated as i32]).copy(&self.stream)?;
            input = match frame.forced_generated_audio_codebooks() {
                Some(mask) => {
                    if mask.len() != generated {
                        return Err(Error::Parallel(format!(
                            "portable realtime forcing mask has {} entries, expected {generated}",
                            mask.len()
                        )));
                    }
                    input.with_partially_forced_generated_audio(&tokens, mask.to_vec())
                }
                None => input.with_forced_generated_audio(&tokens),
            };
        } else if frame.forced_generated_audio_codebooks().is_some() {
            return Err(Error::Parallel(
                "portable realtime forcing mask has no generated-audio tokens".into(),
            ));
        }
        if let Some(tokens) = frame.forced_text_tokens() {
            if tokens.len() != batch {
                return Err(Error::Parallel(format!(
                    "portable realtime text forcing has {} tokens, expected {batch}",
                    tokens.len()
                )));
            }
            let tokens = Array::from_slice(tokens, &[batch_i32, 1]).copy(&self.stream)?;
            input = input.with_forced_text(&tokens);
        }
        if frame.retains_diagnostics() {
            input = input.with_diagnostics();
        }
        Ok(input)
    }

    fn observe_output(&self, output: &Self::Output) -> Result<RealtimeOutputFrame, Self::Error> {
        let batch = usize::try_from(output.text_token.dim(0))
            .map_err(|_| Error::Parallel("negative realtime output batch".into()))?;
        let diagnostics = output
            .diagnostics
            .iter()
            .enumerate()
            .map(|(prediction, logits)| {
                let shape = logits
                    .shape()
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            Error::Parallel("negative realtime diagnostic dimension".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RealtimeDecisionDiagnostics::new(
                    prediction,
                    shape,
                    array_f32_host(logits, &self.stream)?,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(RealtimeOutputFrame::new(
            batch,
            array_i32_host(&output.text_token)?,
            array_i32_host(&output.decision_audio_tokens)?,
            array_i32_host(&output.sampled_audio_tokens)?,
            output
                .output_audio_tokens
                .as_ref()
                .map(array_i32_host)
                .transpose()?,
            diagnostics,
        ))
    }

    fn create_session(
        &self,
        model: &Self::Model,
        sampling: RealtimeSampling,
    ) -> Result<Self::Session, Self::Error> {
        let schedule = model.config().frame_schedule().clone();
        let model_state = MlxRealtimeModelState {
            cache: model.model.new_realtime_state()?,
            tokens: BTreeMap::new(),
        };
        let random_state = sampling
            .is_stochastic()
            .then(|| random::key(sampling.seed()).map(RandomState::from_key))
            .transpose()?;
        let generation = RealtimeGenerationState::new(
            model_state,
            schedule.clone(),
            realtime_samplers(&schedule, sampling)?,
            random_state,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let session = MlxRealtimeSession {
            generation,
            sampling,
        };
        Ok(session)
    }

    fn validate_session(
        &self,
        model: &Self::Model,
        session: &Self::Session,
    ) -> Result<(), Self::Error> {
        session
            .generation
            .schedule_state()
            .validate_schedule(model.config().frame_schedule())
            .map_err(|error| Error::Parallel(error.to_string()))?;
        if session.generation.model_state().cache.layout() != model.model.state_layout() {
            return Err(Error::Parallel(
                "realtime session state does not match the loaded neutral model".into(),
            ));
        }
        Ok(())
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
        session: &mut <Self::Session as SemanticStateTransaction>::Branch,
        input: &Self::Input,
    ) -> Result<Submission<Self::Output, Self::Completion>, Self::Error> {
        submit_neutral_step(
            model,
            session,
            input,
            &self.stream,
            self.tensor_parallel_group.as_deref(),
        )
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{
        schema::StoredDtypeConstraint, AffineQuantization, StoredDtype, WeightQuantization,
    };
    use eredu_core::{
        load_realtime_model_with_options, RealtimeFrameConvention, RealtimeFrameScheduleState,
    };
    use eredu_runtime::{
        CompositeLayeredTraversalHook, DenseDiskStreamLoadOptions, ExecutionResidency,
        LayeredTraversalHook, LayeredUnitAction, LayerwiseLoadOptions, SequentialDecisionDriver,
        SequentialDecisionTraversal, WeightResidency,
    };
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::path::Path;

    fn prepare(path: &Path) -> RealtimePreparationPlan {
        eredu_architectures::moshi::prepare_realtime_model(path)
            .unwrap_or_else(|error| panic!("prepare realtime artifact {}: {error}", path.display()))
    }

    #[test]
    fn realtime_session_capabilities_fail_closed_for_activation_inspection() {
        let available = realtime_session_capabilities();
        assert!(available.persistent_cache);
        assert!(available.output_observation);
        assert!(!available.activation_inspection);

        let options = ModelLoadOptions::default().with_required_session_capabilities(
            eredu_core::SessionCapabilities {
                activation_inspection: true,
                ..eredu_core::SessionCapabilities::default()
            },
        );
        let error = validate_realtime_session_requirements(&options).unwrap_err();
        match error {
            Error::SessionCapability(error) => {
                assert_eq!(error.capability(), "activation_inspection")
            }
            error => panic!("expected session capability error, got {error:?}"),
        }
    }

    const TINY_NATIVE_CONFIG: &str = r#"{
        "model_type": "moshi",
        "dim": 32,
        "text_card": 32,
        "n_q": 2,
        "dep_q": 1,
        "generated_audio_codebooks": 1,
        "card": 32,
        "num_heads": 4,
        "num_layers": 1,
        "dim_feedforward": 48,
        "causal": true,
        "context": 7,
        "max_period": 10000.0,
        "positional_embedding": "rope",
        "depformer_dim": 32,
        "depformer_dim_feedforward": 48,
        "depformer_num_heads": 4,
        "depformer_num_layers": 1,
        "depformer_context": 3,
        "depformer_max_period": 10000.0,
        "depformer_pos_emb": "none",
        "delays": [0, 0, 1]
    }"#;

    #[derive(Debug, Eq, PartialEq)]
    struct TinyFrameTokens {
        text: Vec<i32>,
        sampled_audio: Vec<i32>,
        output_audio: Option<Vec<i32>>,
    }

    #[derive(Default)]
    struct TeacherObservationCapture {
        values: Vec<(eredu_architectures::moshi::ObservationPoint, Array)>,
    }

    impl
        LayeredTraversalHook<
            crate::backend::nn::shared::MlxNeuralBackend,
            eredu_architectures::moshi::ForwardContext<crate::MlxTensor>,
            eredu_nn::Error,
        > for TeacherObservationCapture
    {
        fn before_unit(
            &mut self,
            group: usize,
            index: usize,
            _remaining_units: usize,
            value: &MlxTensor,
            _forward: &mut eredu_architectures::moshi::ForwardContext<crate::MlxTensor>,
            _context: &Stream,
        ) -> Result<LayeredUnitAction, eredu_nn::Error> {
            if group == 0 && index == 0 {
                self.values.push((
                    eredu_architectures::moshi::ObservationPoint::TemporalInput,
                    value.as_array().clone(),
                ));
            }
            Ok(LayeredUnitAction::Execute)
        }

        fn after_unit(
            &mut self,
            group: usize,
            index: usize,
            value: &MlxTensor,
            _forward: &mut eredu_architectures::moshi::ForwardContext<crate::MlxTensor>,
            _context: &Stream,
        ) -> Result<(), eredu_nn::Error> {
            let point = match group {
                0 => eredu_architectures::moshi::ObservationPoint::TemporalLayer { layer: index },
                1 => {
                    eredu_architectures::moshi::ObservationPoint::DepthSliceLogits { slice: index }
                }
                _ => return Ok(()),
            };
            self.values.push((point, value.as_array().clone()));
            Ok(())
        }

        fn after_group(
            &mut self,
            group: usize,
            _value: &MlxTensor,
            forward: &mut eredu_architectures::moshi::ForwardContext<crate::MlxTensor>,
            _context: &Stream,
        ) -> Result<(), eredu_nn::Error> {
            if group == 0 {
                self.values.push((
                    eredu_architectures::moshi::ObservationPoint::TextLogits,
                    forward
                        .text_logits()
                        .expect("completed temporal group owns text logits")
                        .as_array()
                        .clone(),
                ));
            }
            Ok(())
        }
    }

    fn write_tiny_native_artifact(directory: &Path, quantization: Option<WeightQuantization>) {
        let mut config_json = serde_json::from_str::<serde_json::Value>(TINY_NATIVE_CONFIG)
            .expect("tiny native JSON");
        if let Some(quantization) = quantization {
            config_json.as_object_mut().unwrap().insert(
                "quantization".into(),
                serde_json::to_value(quantization).expect("serialize tiny quantization"),
            );
        }
        let config_json = serde_json::to_string_pretty(&config_json).unwrap();
        let config = eredu_architectures::moshi::MoshiConfig::from_json(&config_json)
            .expect("tiny native Moshi config");
        let plan = eredu_architectures::moshi::safetensors_plan(&config)
            .expect("tiny native SafeTensors plan");
        assert!(plan.layout_groups.is_empty());

        // Derive every physical name and shape from the strict architecture
        // catalog. Zero matrices make greedy decisions exact across dense and
        // load-time packed execution; unit normalization scales remain valid.
        let tensors = plan
            .common_tensors
            .iter()
            .map(|constraint| {
                let dtype = match &constraint.dtype {
                    StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                    StoredDtypeConstraint::Floating => StoredDtype::F32,
                    StoredDtypeConstraint::OneOf(dtypes) => dtypes
                        .iter()
                        .find(|dtype| **dtype == StoredDtype::F32)
                        .or_else(|| dtypes.first())
                        .cloned()
                        .expect("validated catalog dtype set"),
                };
                let elements = constraint.shape.iter().product::<usize>();
                let (dtype, bytes) = match dtype {
                    StoredDtype::F32 => {
                        let value = if constraint.key.contains("norm")
                            || constraint.key.ends_with(".scales")
                        {
                            1.0f32
                        } else {
                            0.0f32
                        };
                        (
                            SafeDtype::F32,
                            std::iter::repeat_n(value, elements)
                                .flat_map(f32::to_le_bytes)
                                .collect::<Vec<_>>(),
                        )
                    }
                    StoredDtype::U32 => (
                        SafeDtype::U32,
                        std::iter::repeat_n(0u32, elements)
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    ),
                    StoredDtype::U8 => (
                        SafeDtype::U8,
                        vec![
                            if constraint.key.ends_with(".scales") {
                                127
                            } else {
                                0
                            };
                            elements
                        ],
                    ),
                    dtype => panic!("tiny native writer does not support {dtype:?}"),
                };
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    dtype,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        let views = tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).expect("catalog-derived tensor view"),
            )
        });
        std::fs::write(directory.join("config.json"), config_json)
            .expect("write tiny native config");
        serialize_to_file(views, None, &directory.join("model.safetensors"))
            .expect("write tiny native SafeTensors artifact");
    }

    fn artifact_files(directory: &Path) -> std::collections::BTreeSet<String> {
        std::fs::read_dir(directory)
            .expect("read tiny artifact directory")
            .map(|entry| {
                entry
                    .expect("tiny artifact entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    fn host_i32(array: &Array) -> Vec<i32> {
        array
            .evaluated()
            .expect("evaluate realtime token array")
            .as_slice::<i32>()
            .to_vec()
    }

    fn run_tiny_realtime_frames(
        model: &mut RealtimeModel<MlxRealtimeBackend>,
    ) -> Vec<TinyFrameTokens> {
        let request = RequestId::new(8);
        let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1).unwrap())
            .expect("tiny realtime scheduler");
        scheduler
            .register_request(model, request, RealtimeSampling::greedy())
            .expect("tiny realtime session");

        let inputs = [
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[1i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[2i32], &[1, 1]))
                .with_forced_text(&Array::from_slice(&[7i32], &[1, 1]))
                .with_forced_generated_audio(&Array::from_slice(&[9i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[3i32], &[1, 1]))
                .with_forced_text(&Array::from_slice(&[11i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[4i32], &[1, 1]))
                .with_forced_generated_audio(&Array::from_slice(&[13i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[5i32], &[1, 1])),
        ];
        let mut frames = Vec::with_capacity(inputs.len());
        for input in inputs {
            scheduler
                .enqueue(model, request, input)
                .expect("enqueue tiny realtime frame");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            let output = loop {
                if let Some(completed) = scheduler
                    .run_queued(model)
                    .expect("execute tiny realtime frame")
                    .pop()
                {
                    break completed.into_parts().1;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "tiny realtime completion exceeded 60 seconds"
                );
                std::thread::yield_now();
            };
            frames.push(TinyFrameTokens {
                text: host_i32(&output.text_token),
                sampled_audio: host_i32(&output.sampled_audio_tokens),
                output_audio: output.output_audio_tokens.as_ref().map(host_i32),
            });
        }
        scheduler
            .finish_request(request)
            .expect("finish tiny realtime request");
        frames
    }

    fn verify_tiny_native_hardware_matrix() {
        let directory = tempfile::tempdir().expect("tiny native artifact directory");
        write_tiny_native_artifact(directory.path(), None);
        let original_files = artifact_files(directory.path());
        assert_eq!(
            original_files,
            std::collections::BTreeSet::from([
                "config.json".to_string(),
                "model.safetensors".to_string(),
            ])
        );

        let device = safemlx::Device::new(safemlx::DeviceType::Gpu, 0);
        let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let weights_stream = Stream::new_with_device(&weights_device);
        let policies = [
            (
                WeightResidency::fully_resident(),
                ExecutionResidency::FullyResident,
            ),
            (
                WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
                ExecutionResidency::LayerwiseHost,
            ),
            (
                WeightResidency::dense_disk_stream(
                    DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1)
                        .expect("tiny dense stream policy"),
                ),
                ExecutionResidency::DenseDiskStream,
            ),
        ];
        let expected = vec![
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![0],
                output_audio: None,
            },
            TinyFrameTokens {
                text: vec![7],
                sampled_audio: vec![9],
                output_audio: Some(vec![0]),
            },
            TinyFrameTokens {
                text: vec![11],
                sampled_audio: vec![0],
                output_audio: Some(vec![9]),
            },
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![13],
                output_audio: Some(vec![0]),
            },
            TinyFrameTokens {
                text: vec![0],
                sampled_audio: vec![0],
                output_audio: Some(vec![13]),
            },
        ];

        for (residency, execution) in policies {
            let backend = MlxRealtimeBackend::new(&stream, &weights_stream);
            let mut model = load_realtime_model_with_options(
                backend,
                prepare(directory.path()),
                ModelLoadOptions::default().with_weight_residency(residency),
            )
            .unwrap_or_else(|error| panic!("load tiny {execution:?} model: {error}"));
            assert_eq!(model.model().metadata().residency(), execution);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            let report = model
                .model()
                .residency_report()
                .expect("tiny residency report")
                .expect("MLX realtime models expose residency telemetry");
            assert!(report.initialized());
            assert!(report.weight_store().physical_reads > 0);
            if execution == ExecutionResidency::DenseDiskStream {
                let dense = model
                    .model()
                    .dense_stream_report()
                    .expect("tiny dense stream report")
                    .expect("selected dense-stream policy has a report");
                assert!(dense.planned_layer_count() > 0);
                assert!(dense.decode_forwards() > 0);
            }
        }

        for (request, quantization) in [
            (
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
                WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            ),
            (
                eredu_core::QuantizationRequest::MxFp4,
                WeightQuantization::MxFp4,
            ),
        ] {
            let backend = MlxRealtimeBackend::new(&stream, &weights_stream);
            let mut model = load_realtime_model_with_options(
                backend,
                prepare(directory.path()),
                ModelLoadOptions::with_quantization(request),
            )
            .unwrap_or_else(|error| panic!("load-time {quantization:?} tiny model: {error}"));
            let metadata = model.model().metadata();
            assert_eq!(metadata.quantization(), Some(quantization));
            let materialization = metadata
                .materialization()
                .expect("load-time quantization telemetry");
            assert!(materialization.transformed_weights > 0);
            assert!(materialization.source_bytes_read > 0);
            assert!(materialization.output_bytes > 0);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            drop(model);
            assert_eq!(
                artifact_files(directory.path()),
                original_files,
                "load-time {quantization:?} created a disk artifact"
            );
        }

        for quantization in [
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
            WeightQuantization::MxFp4,
        ] {
            let packed_directory = tempfile::tempdir().expect("tiny packed artifact directory");
            write_tiny_native_artifact(packed_directory.path(), Some(quantization));
            let original_files = artifact_files(packed_directory.path());
            let backend = MlxRealtimeBackend::new(&stream, &weights_stream);
            let mut model = load_realtime_model_with_options(
                backend,
                prepare(packed_directory.path()),
                ModelLoadOptions::default(),
            )
            .unwrap_or_else(|error| panic!("load checkpoint-native {quantization:?}: {error}"));
            let metadata = model.model().metadata();
            assert_eq!(metadata.quantization(), Some(quantization));
            assert_eq!(metadata.materialization(), None);
            assert_eq!(run_tiny_realtime_frames(&mut model), expected);
            drop(model);
            assert_eq!(artifact_files(packed_directory.path()), original_files);
        }
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn moshi_mlx_scheduler_transaction_rollback_release_resume() {
        let directory = tempfile::tempdir().expect("tiny scheduler artifact directory");
        write_tiny_native_artifact(directory.path(), None);
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxRealtimeBackend::new(execution.stream(), &weights);
        let mut model = load_realtime_model_with_options(
            backend,
            prepare(directory.path()),
            ModelLoadOptions::default(),
        )
        .expect("load tiny scheduler model");
        let request = RequestId::new(81);
        let mut scheduler = RealtimeScheduler::new(&model, SchedulerLimits::new(1, 1).unwrap())
            .expect("tiny scheduler");
        scheduler
            .register_request(&model, request, RealtimeSampling::greedy())
            .expect("register tiny scheduler request");

        let drive_one = |scheduler: &mut RealtimeScheduler<MlxRealtimeBackend>,
                         model: &mut RealtimeModel<MlxRealtimeBackend>| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            loop {
                if let Some(completed) = scheduler.run_queued(model).unwrap().pop() {
                    return completed.into_parts().1;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
        };

        scheduler
            .enqueue(
                &model,
                request,
                MlxRealtimeInput::encoded_audio(&Array::from_slice(&[1_i32], &[1, 1])),
            )
            .unwrap();
        drive_one(&mut scheduler, &mut model);
        let released = scheduler.release_request(request).unwrap();
        assert_eq!(released.state().step(), 1);
        scheduler
            .register_request_with_session(&model, request, released)
            .unwrap();

        scheduler
            .enqueue(
                &model,
                request,
                MlxRealtimeInput::encoded_audio(&Array::from_slice(&[2_i32], &[1, 1]))
                    .with_forced_text(&Array::from_slice(&[7_i32], &[1, 1]))
                    .with_forced_generated_audio(&Array::from_slice(&[9_i32], &[1, 1])),
            )
            .unwrap();
        drive_one(&mut scheduler, &mut model);
        let released = scheduler.release_request(request).unwrap();
        assert_eq!(released.state().step(), 2);
        let (backend, mut backend_model) = model.into_parts();
        let mut failed_branch = released.state().branch().unwrap();
        let failed = backend
            .submit_step(
                &mut backend_model,
                &mut failed_branch,
                &MlxRealtimeInput::encoded_audio(&Array::from_slice(&[-1_i32], &[1, 1])),
            )
            .unwrap();
        assert!(failed.completion.wait().is_err());
        MlxRealtimeSession::discard_branch(failed_branch).unwrap();
        assert_eq!(released.state().step(), 2);
        let mut model = RealtimeModel::new(backend, backend_model);
        scheduler
            .register_request_with_session(&model, request, released)
            .unwrap();

        scheduler
            .enqueue(
                &model,
                request,
                MlxRealtimeInput::encoded_audio(&Array::from_slice(&[3_i32], &[1, 1])),
            )
            .unwrap();
        drive_one(&mut scheduler, &mut model);
        let released = scheduler.release_request(request).unwrap();
        assert_eq!(released.state().step(), 3);
    }

    #[test]
    #[ignore = "runs the MLX operator, transaction, and tiny native model conformance suite"]
    fn moshi_mlx_conformance_suite() {
        verify_tiny_native_hardware_matrix();
        const TESTS: &[&str] = &[
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_dense_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_affine_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_mxfp4_fused_projection_equivalence",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_sentinel_embedding_validation",
            "backend::nn::shared::neutral_semantic_operator_tests::mlx_multi_table_embedding_sum_is_ordered_and_sentinel_safe",
            "backend::runtime::cache::state::semantic_transaction_tests::paged_depth_segment_reset_preserves_temporal_pages_and_later_rollback",
            "backend::runtime::cache::state::semantic_transaction_tests::mlx_realtime_transaction_paged_rollback_release_resume",
            "backend::runtime::residency::manager::tests::cross_unit_alias_reacquisition_reuses_one_pinned_owner_read",
            "backend::runtime::generation::backend::tests::mlx_token_domain_validation_is_deferred_to_completion",
            "composition::mlx::realtime::tests::mlx_realtime_input_domains_are_deferred_and_strict",
        ];
        let executable = std::env::current_exe().expect("current unit-test executable");
        for test in TESTS {
            let output = std::process::Command::new(&executable)
                .args(["--exact", test, "--ignored", "--nocapture"])
                .output()
                .unwrap_or_else(|error| {
                    panic!("failed to launch MLX conformance test {test}: {error}")
                });
            assert!(
                output.status.success(),
                "MLX conformance test {test} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    #[test]
    fn partial_forcing_and_initialization_only_transition_are_portable() {
        let schedule = RealtimeSpeechConfig::new(
            4,
            2,
            2,
            3,
            100,
            64,
            RealtimeFrameConvention::AbsoluteDelayedSlots,
            vec![0, 0, 1, 0, 1],
        )
        .unwrap();
        let mut state = RealtimeFrameScheduleState::new(schedule.clone());
        let transition = state
            .advance(
                &schedule,
                &RealtimeFrameForcing::new(false, vec![true, false]),
            )
            .unwrap();
        assert!(!transition.model_call_required());
        assert_eq!(transition.forced_placements().len(), 1);
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_realtime_input_domains_are_deferred_and_strict() {
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = execution.stream();
        let config = eredu_architectures::moshi::MoshiConfig::from_json(TINY_NATIVE_CONFIG)
            .expect("tiny native Moshi config");

        let validate = |input: &MlxRealtimeInput| {
            let scope = TokenValidationScope::begin().unwrap();
            validate_realtime_token_payloads(&config, input, stream)
                .expect("payload validation must remain lazy during submission");
            let validations = scope.finish();
            let event = async_eval_with_event(validations.arrays()).unwrap();
            event.synchronize().unwrap();
            validations.validate_completed()
        };

        let valid = MlxRealtimeInput::encoded_audio(&Array::from_slice(&[32_i32], &[1, 1]))
            .with_forced_text(&Array::from_slice(&[32_i32], &[1, 1]))
            .with_forced_generated_audio(&Array::from_slice(&[32_i32], &[1, 1]));
        validate(&valid).unwrap();

        for input in [
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[-1_i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[33_i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[0_i32], &[1, 1]))
                .with_forced_text(&Array::from_slice(&[33_i32], &[1, 1])),
            MlxRealtimeInput::encoded_audio(&Array::from_slice(&[0_i32], &[1, 1]))
                .with_forced_generated_audio(&Array::from_slice(&[33_i32], &[1, 1])),
        ] {
            assert!(validate(&input).is_err());
        }
    }

    #[test]
    fn model_identity_rejects_equal_geometry_from_another_artifact() {
        let expected = MlxRealtimeModelIdentity {
            artifact: LoadedArtifactIdentity::Content([1; 32]),
            source_architecture: "source".into(),
            execution_architecture: "execution".into(),
        };
        let actual = MlxRealtimeModelIdentity {
            artifact: LoadedArtifactIdentity::Content([2; 32]),
            source_architecture: "source".into(),
            execution_architecture: "execution".into(),
        };
        assert_eq!(
            expected.mismatch(&actual),
            Some("checkpoint artifact fingerprint")
        );
    }

    fn required_fixture_array<'a>(
        fixture: &'a std::collections::HashMap<String, Array>,
        key: &str,
    ) -> &'a Array {
        fixture
            .get(key)
            .unwrap_or_else(|| panic!("teacher-forced fixture is missing tensor {key}"))
    }

    fn teacher_observation_key(
        step: usize,
        point: eredu_architectures::moshi::ObservationPoint,
    ) -> String {
        use eredu_architectures::moshi::ObservationPoint;
        match point {
            ObservationPoint::TemporalInput => format!("expected.{step}.temporal_input"),
            ObservationPoint::TemporalLayer { layer } => {
                format!("expected.{step}.temporal_layer.{layer}")
            }
            ObservationPoint::TextLogits => format!("expected.{step}.text_logits"),
            ObservationPoint::DepthSliceLogits { slice } => {
                format!("expected.{step}.audio_logits.{slice}")
            }
        }
    }

    fn assert_teacher_observation_close(
        actual: &Array,
        expected: &Array,
        label: &str,
        stream: &Stream,
    ) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "shape mismatch for {label}"
        );
        assert!(
            actual
                .all_close(expected, 2e-2, 2e-2, None, stream)
                .unwrap_or_else(|error| panic!("compare {label}: {error}"))
                .item::<bool>(stream),
            "teacher-forced observation differs at {label}"
        );
    }

    fn run_teacher_forced_fixture(
        model: &mut MlxRealtimeModel,
        fixture_path: &Path,
        stream: &Stream,
    ) {
        let fixture = Array::load_safetensors(fixture_path, stream)
            .unwrap_or_else(|error| panic!("load {}: {error}", fixture_path.display()));
        let text = required_fixture_array(&fixture, "input.text");
        let audio = required_fixture_array(&fixture, "input.audio");
        let depth = required_fixture_array(&fixture, "input.depth");
        assert_eq!(text.ndim(), 3);
        assert_eq!(audio.ndim(), 3);
        assert_eq!(depth.ndim(), 3);
        assert_eq!(text.dim(0), audio.dim(0));
        assert_eq!(text.dim(0), depth.dim(0));
        assert_eq!(
            audio.dim(2),
            model.config().frame_schedule().total_audio_codebooks() as i32
        );
        assert_eq!(
            depth.dim(2),
            model.config().frame_schedule().depth_audio_codebooks() as i32
        );

        let mut state = model
            .model
            .new_realtime_state()
            .expect("teacher-forced cache state");
        let expected_points = eredu_architectures::moshi::observation_points(model.config());
        for step in 0..text.dim(0) as usize {
            let text_step = text
                .try_index_device(step as i32, stream)
                .expect("teacher text step");
            let audio_step = audio
                .try_index_device(step as i32, stream)
                .expect("teacher audio step");
            let depth_step = depth
                .try_index_device(step as i32, stream)
                .expect("teacher depth step");
            let audio_tokens = (0..audio_step.dim(1) as usize)
                .map(|codebook| token_column(&audio_step, codebook, stream))
                .collect::<Result<Vec<_>, _>>()
                .expect("teacher audio columns");
            let mut directives = (0..depth_step.dim(1) as usize)
                .map(|prediction| {
                    token_column(&depth_step, prediction, stream)
                        .map(MlxTensor::from_array)
                        .map(PredictionDirective::Force)
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("teacher decision columns");
            directives.push(PredictionDirective::Sample);
            let prediction_count = directives.len();
            let plan = SequentialDecisionPlan::new(directives, true, false).unwrap();
            let mut driver = SequentialDecisionDriver::<MlxSamplingBackend, DefaultSampler>::new(
                plan,
                vec![DefaultSampler; prediction_count],
                vec![0.0; prediction_count],
                None,
            )
            .unwrap();
            let validation_scope = TokenValidationScope::begin().unwrap();
            let mut boundary = eredu_architectures::moshi::DecisionBoundary::new(model.config())
                .expect("teacher decision boundary");
            let decision = SequentialDecisionTraversal::new(&mut driver, &mut boundary);
            let capture = TeacherObservationCapture::default();
            let mut hook = CompositeLayeredTraversalHook::new(decision, capture);
            let audio_refs = audio_tokens.iter().collect::<Vec<_>>();
            model
                .model
                .forward_realtime_with_traversal_hook(
                    eredu_architectures::moshi::Input {
                        text: &text_step,
                        audio: &audio_refs,
                        mask: None,
                    },
                    &mut state,
                    &mut hook,
                    stream,
                )
                .unwrap_or_else(|error| panic!("teacher-forced frame {step}: {error}"));
            let (_, capture) = hook.into_parts();
            driver.finish().expect("every teacher decision resolved");
            assert_eq!(
                capture
                    .values
                    .iter()
                    .map(|(point, _)| *point)
                    .collect::<Vec<_>>(),
                expected_points
            );
            let validations = validation_scope.finish();
            let retained = capture
                .values
                .iter()
                .map(|(_, value)| value)
                .chain(validations.arrays());
            async_eval_with_event(retained)
                .expect("teacher observation submission")
                .synchronize()
                .expect("teacher observation completion");
            validations
                .validate_completed()
                .expect("teacher token domains");
            for (point, actual) in capture.values {
                let key = teacher_observation_key(step, point);
                assert_teacher_observation_close(
                    &actual,
                    required_fixture_array(&fixture, &key),
                    &key,
                    stream,
                );
            }
        }
    }

    fn run_released_teacher_fixture(
        model_env: &str,
        reference_env: &str,
        expected_model_type: EffectiveModelType,
    ) {
        let model_path = std::env::var_os(model_env).unwrap_or_else(|| {
            panic!(
                "{model_env} must point at a released artifact when this ignored fixture test is explicitly enabled"
            )
        });
        let reference_path = std::env::var_os(reference_env)
            .unwrap_or_else(|| panic!("{reference_env} must be set when {model_env} is set"));
        assert!(
            Path::new(&model_path).exists(),
            "{model_env} does not exist"
        );
        assert!(
            Path::new(&reference_path).is_file(),
            "{reference_env} is not a file"
        );
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxRealtimeBackend::new(execution.stream(), &weights);
        let mut model = backend
            .materialize_realtime_model(
                prepare(Path::new(&model_path)),
                ModelLoadOptions::default(),
            )
            .unwrap_or_else(|error| panic!("load {model_env}: {error}"));
        assert_eq!(model.effective_model_type(), expected_model_type);
        run_teacher_forced_fixture(&mut model, Path::new(&reference_path), execution.stream());
    }

    fn assert_token_array_equal(actual: &Array, expected: &Array, label: &str, stream: &Stream) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "shape mismatch for {label}"
        );
        assert!(
            actual
                .eq(expected, stream)
                .expect("token comparison")
                .all(None, stream)
                .expect("token comparison reduction")
                .item::<bool>(stream),
            "token fixture differs at {label}"
        );
    }

    fn run_personaplex_frame_fixture(
        model: &mut RealtimeModel<MlxRealtimeBackend>,
        fixture: &std::collections::HashMap<String, Array>,
        prefix: &str,
        forced: bool,
    ) {
        let stream = model.backend().stream().clone();
        let user_key = if forced {
            format!("{prefix}.user_audio")
        } else {
            format!("{prefix}.input_audio")
        };
        let user = required_fixture_array(fixture, &user_key);
        let agent =
            forced.then(|| required_fixture_array(fixture, &format!("{prefix}.agent_audio")));
        let text = forced.then(|| required_fixture_array(fixture, &format!("{prefix}.text")));
        let request = RequestId::new(91);
        let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1).unwrap())
            .expect("PersonaPlex fixture scheduler");
        scheduler
            .register_request(model, request, RealtimeSampling::greedy())
            .expect("PersonaPlex fixture session");
        let mut sampled = Vec::new();
        let mut output_audio = Vec::new();
        let mut emitted_steps = Vec::new();
        for step in 0..user.dim(2) {
            let user_step = user
                .try_index_device((.., .., step), &stream)
                .expect("PersonaPlex user frame");
            let mut input = MlxRealtimeInput::encoded_audio(&user_step);
            let agent_step;
            let text_step;
            if let (Some(agent), Some(text)) = (agent, text) {
                agent_step = agent
                    .try_index_device((.., .., step), &stream)
                    .expect("PersonaPlex agent frame");
                text_step = text
                    .try_index_device((.., .., step), &stream)
                    .expect("PersonaPlex text frame");
                input = input
                    .with_forced_generated_audio(&agent_step)
                    .with_forced_text(&text_step);
            }
            scheduler.enqueue(model, request, input).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
            let output = loop {
                if let Some(completed) = scheduler.run_queued(model).unwrap().pop() {
                    break completed.into_parts().1;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            };
            if step > 0 {
                sampled.push(
                    safemlx::ops::concatenate_axis(
                        &[&output.text_token, &output.sampled_audio_tokens],
                        1,
                        &stream,
                    )
                    .expect("PersonaPlex sampled frame"),
                );
            }
            if let Some(audio) = output.output_audio_tokens {
                output_audio.push(audio);
                emitted_steps.push(step);
            }
        }
        scheduler.finish_request(request).unwrap();
        let sampled = stack_axis(&sampled, 2, &stream).expect("PersonaPlex sampled transcript");
        let output_audio =
            stack_axis(&output_audio, 2, &stream).expect("PersonaPlex delayed audio transcript");
        let emitted_steps = Array::from_slice(&emitted_steps, &[output_audio.dim(2)]);
        async_eval_with_event([&sampled, &output_audio, &emitted_steps])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_token_array_equal(
            &sampled,
            required_fixture_array(fixture, &format!("{prefix}.expected_sampled")),
            &format!("{prefix}.expected_sampled"),
            &stream,
        );
        assert_token_array_equal(
            &output_audio,
            required_fixture_array(fixture, &format!("{prefix}.expected_output_audio")),
            &format!("{prefix}.expected_output_audio"),
            &stream,
        );
        assert_token_array_equal(
            &emitted_steps,
            required_fixture_array(fixture, &format!("{prefix}.expected_emitted_steps")),
            &format!("{prefix}.expected_emitted_steps"),
            &stream,
        );
    }

    fn run_native_seeded_fixture(
        model: &mut RealtimeModel<MlxRealtimeBackend>,
        fixture: &std::collections::HashMap<String, Array>,
    ) {
        let stream = model.backend().stream().clone();
        let input = required_fixture_array(fixture, "generation.input_audio");
        let seed = required_fixture_array(fixture, "generation.seeded.seed")
            .clone()
            .item::<i64>(&stream) as u64;
        let text_temperature =
            required_fixture_array(fixture, "generation.seeded.text_temperature")
                .clone()
                .item::<f32>(&stream);
        let audio_temperature =
            required_fixture_array(fixture, "generation.seeded.audio_temperature")
                .clone()
                .item::<f32>(&stream);
        let request = RequestId::new(92);
        let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1).unwrap())
            .expect("native seeded scheduler");
        scheduler
            .register_request(
                model,
                request,
                RealtimeSampling::new(text_temperature, audio_temperature, seed).unwrap(),
            )
            .expect("native seeded session");
        let mut text = Vec::new();
        let mut audio = Vec::new();
        for step in 0..input.dim(2) {
            let frame = input
                .try_index_device((.., .., step), &stream)
                .expect("native seeded input frame");
            scheduler
                .enqueue(model, request, MlxRealtimeInput::encoded_audio(&frame))
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
            let output = loop {
                if let Some(completed) = scheduler.run_queued(model).unwrap().pop() {
                    break completed.into_parts().1;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            };
            text.push(output.text_token.squeeze_axes(&[-1], &stream).unwrap());
            if let Some(tokens) = output.output_audio_tokens {
                audio.push(tokens);
            }
        }
        scheduler.finish_request(request).unwrap();
        let text = stack_axis(&text, 1, &stream).unwrap();
        let audio = if audio.is_empty() {
            Array::zeros::<i32>(
                &[
                    input.dim(0),
                    model.speech_config().generated_audio_codebooks() as i32,
                    0,
                ],
                &stream,
            )
            .unwrap()
        } else {
            stack_axis(&audio, 2, &stream).unwrap()
        };
        async_eval_with_event([&text, &audio])
            .unwrap()
            .synchronize()
            .unwrap();
        assert_token_array_equal(
            &text,
            required_fixture_array(fixture, "generation.seeded.expected_text"),
            "generation.seeded.expected_text",
            &stream,
        );
        assert_token_array_equal(
            &audio,
            required_fixture_array(fixture, "generation.seeded.expected_audio"),
            "generation.seeded.expected_audio",
            &stream,
        );
    }

    #[test]
    #[ignore = "requires released native Moshi artifact and teacher-forced fixture"]
    fn moshi_native_teacher_forced_fixture_parity() {
        run_released_teacher_fixture(
            "EREDU_MOSHI_FIXTURE",
            "EREDU_MOSHI_TEACHER_FIXTURE",
            EffectiveModelType::Moshi,
        );
    }

    #[test]
    #[ignore = "requires released PersonaPlex artifact and PyTorch teacher fixture"]
    fn moshi_personaplex_teacher_forced_fixture_parity() {
        run_released_teacher_fixture(
            "EREDU_PERSONAPLEX_FIXTURE",
            "EREDU_PERSONAPLEX_TEACHER_FIXTURE",
            EffectiveModelType::PersonaPlex,
        );
    }

    #[test]
    #[ignore = "requires released PersonaPlex artifact and PyTorch realtime fixture"]
    fn moshi_personaplex_prompt_realtime_and_residency_parity() {
        let model_path = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
            "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        let fixture_path = std::env::var_os("EREDU_PERSONAPLEX_TEACHER_FIXTURE")
            .expect("EREDU_PERSONAPLEX_TEACHER_FIXTURE must accompany the model fixture");
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
            .expect("load PersonaPlex parity fixture");
        for residency in [
            WeightResidency::fully_resident(),
            WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
            WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(1 << 30, 1 << 30, 1, 1).unwrap(),
            ),
        ] {
            let backend = MlxRealtimeBackend::new(execution.stream(), &weights);
            let mut model = load_realtime_model_with_options(
                backend,
                prepare(Path::new(&model_path)),
                ModelLoadOptions::default().with_weight_residency(residency),
            )
            .expect("load PersonaPlex residency mode");
            assert_eq!(
                model.model().effective_model_type(),
                EffectiveModelType::PersonaPlex
            );
            run_personaplex_frame_fixture(&mut model, &fixture, "generation", false);
            run_personaplex_frame_fixture(&mut model, &fixture, "prompt", true);
        }
    }

    #[test]
    #[ignore = "requires released native Moshi artifact and seeded MLX fixture"]
    fn moshi_native_multiframe_seeded_realtime_parity() {
        let model_path = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
            "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        let fixture_path = std::env::var_os("EREDU_MOSHI_TEACHER_FIXTURE")
            .expect("EREDU_MOSHI_TEACHER_FIXTURE must accompany the model fixture");
        let execution =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let backend = MlxRealtimeBackend::new(execution.stream(), &weights);
        let mut model = load_realtime_model_with_options(
            backend,
            prepare(Path::new(&model_path)),
            ModelLoadOptions::default(),
        )
        .expect("load native seeded fixture model");
        let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
            .expect("load native seeded fixture");
        run_native_seeded_fixture(&mut model, &fixture);
    }

    #[test]
    #[ignore = "requires EREDU_MOSHI_FIXTURE and an MLX runtime"]
    fn moshi_neutral_session_hook() {
        let fixture = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
            "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_MOSHI_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let backend = MlxRealtimeBackend::new(&stream, &stream);
        let model = backend
            .materialize_realtime_model(prepare(Path::new(&fixture)), ModelLoadOptions::default())
            .unwrap();
        let session = backend
            .create_session(&model, RealtimeSampling::greedy())
            .unwrap();
        backend.validate_session(&model, &session).unwrap();
    }

    #[test]
    #[ignore = "requires EREDU_PERSONAPLEX_FIXTURE and an MLX runtime"]
    fn moshi_personaplex_fixture_session_hook() {
        let fixture = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
            "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
        );
        assert!(
            Path::new(&fixture).exists(),
            "EREDU_PERSONAPLEX_FIXTURE does not exist: {}",
            Path::new(&fixture).display()
        );
        let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        let stream = Stream::new_with_device(&device);
        let backend = MlxRealtimeBackend::new(&stream, &stream);
        let model = backend
            .materialize_realtime_model(prepare(Path::new(&fixture)), ModelLoadOptions::default())
            .unwrap();
        assert_eq!(
            model.effective_model_type(),
            EffectiveModelType::PersonaPlex
        );
        let session = backend
            .create_session(&model, RealtimeSampling::greedy())
            .unwrap();
        backend.validate_session(&model, &session).unwrap();
    }
}
