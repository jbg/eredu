//! Architecture-erased MLX model-session execution.

use eredu_core::{
    BackendSession, Completion, InputModality, InspectableBackendSession, InspectedOutput,
    ModelRuntime, ObservationRequest, ObservationSet, ObservationValue, Submission,
    TensorObservation, TensorObservationData, TextGenerationBackend, TextGenerationConfig,
    TextSamplingStrategy, TokenFilter, TokenOutput,
};
use eredu_nn::Tensor as _;
use eredu_runtime::{
    ActivationObserver as RuntimeActivationObserver, GenerationSampler, MirostatV2Sampler, Sampler,
    SamplingBackend,
};
use ref_cast::RefCast;
use safemlx::{
    error::Exception,
    ops::indexing::{NewAxis, TryIndexOp},
    Array, Dtype, Stream,
};
use std::path::Path;

use crate::{
    backend::nn::tensor::{TokenValidationBatch, TokenValidationScope},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    backend::runtime::media::input,
    backend::{error::Error, MlxModelKind},
    composition::mlx::distributed::pipeline::{
        PipelineCache, PipelineModel, PipelineStageCompletion, PipelineStep,
    },
    MlxTensor,
};
#[cfg(any(feature = "image", feature = "audio"))]
use crate::{backend::runtime::media::PreparedModelInput, composition::mlx::ModelProcessor};
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::SpeculativeCapability;
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use super::{
    execution::{
        decode_model, decode_model_tensor_parallel, forward_model_tensor_parallel_with_observer,
        prefill_model, prefill_model_tensor_parallel, prefill_model_tensor_parallel_with_observer,
    },
    speculative::{MlxDrafter, MlxDrafterKind},
    Executable, MlxBackend, MlxCompletion, MlxDistributedSession, MlxModel,
};

pub(super) struct ArrayObserverAdapter<'a, O: ?Sized> {
    pub(super) inner: &'a mut O,
}

struct InspectionCollector<'a> {
    request: &'a ObservationRequest,
    values: Vec<(String, MlxTensor)>,
}

impl<'a> InspectionCollector<'a> {
    fn new(request: &'a ObservationRequest) -> Self {
        Self {
            request,
            values: Vec::new(),
        }
    }

    fn capture(&mut self, path: &str, value: &MlxTensor) {
        if self.request.matches(path) {
            self.values.push((path.into(), value.clone()));
        }
    }

    fn materialize(self, stream: &Stream) -> Result<ObservationSet, Error> {
        let mut observations = ObservationSet::new();
        for (path, value) in self.values {
            observations
                .insert(
                    path,
                    ObservationValue::Tensor(observe_tensor(&value, stream)?),
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(observations)
    }
}

impl RuntimeActivationObserver<MlxTensor, Exception> for InspectionCollector<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), Exception> {
        self.capture(path, value);
        Ok(())
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, MlxTensor>,
    ) -> Result<(), Exception> {
        let root = format!("{}.routing", routing.path);
        self.capture(
            &format!("{root}.selected_experts"),
            routing.selected_experts,
        );
        self.capture(&format!("{root}.selected_scores"), routing.selected_scores);
        self.capture(&format!("{root}.coefficients"), routing.coefficients);
        self.capture(&format!("{root}.routed_output"), routing.routed_output);
        if let Some(value) = routing.local_routed_output {
            self.capture(&format!("{root}.local_routed_output"), value);
        }
        if let Some(value) = routing.reduced_routed_output {
            self.capture(&format!("{root}.reduced_routed_output"), value);
        }
        if let Some(value) = routing.shared_output {
            self.capture(&format!("{root}.shared_output"), value);
        }
        if let Some(value) = routing.combined_output {
            self.capture(&format!("{root}.combined_output"), value);
        }
        Ok(())
    }
}

fn observe_tensor(value: &MlxTensor, stream: &Stream) -> Result<TensorObservation, Error> {
    let shape = value
        .shape()
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Error::ArchitectureModel(format!(
                    "observed tensor has negative dimension {dimension}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let data = match value.as_array().dtype() {
        Dtype::Bool => {
            TensorObservationData::Bool(value.as_array().evaluated()?.as_slice::<bool>().to_vec())
        }
        Dtype::Uint8 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u8>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint16 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u16>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint32 => TensorObservationData::U64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<u32>()
                .iter()
                .copied()
                .map(u64::from)
                .collect(),
        ),
        Dtype::Uint64 => {
            TensorObservationData::U64(value.as_array().evaluated()?.as_slice::<u64>().to_vec())
        }
        Dtype::Int8 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i8>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int16 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i16>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int32 => TensorObservationData::I64(
            value
                .as_array()
                .evaluated()?
                .as_slice::<i32>()
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
        ),
        Dtype::Int64 => {
            TensorObservationData::I64(value.as_array().evaluated()?.as_slice::<i64>().to_vec())
        }
        Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16 => {
            TensorObservationData::F32(value.to_f32_vec(stream)?)
        }
        Dtype::Complex64 => {
            return Err(Error::ArchitectureModel(
                "complex activation observation is unsupported".into(),
            ))
        }
    };
    TensorObservation::new(shape, data).map_err(|error| Error::ArchitectureModel(error.to_string()))
}

impl<O> RuntimeActivationObserver<Array, Exception> for ArrayObserverAdapter<'_, O>
where
    O: RuntimeActivationObserver<MlxTensor, Exception> + ?Sized,
{
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Exception> {
        self.inner.observe(path, MlxTensor::ref_cast(value))
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
        self.inner
            .intervene(path, MlxTensor::ref_cast(value))
            .map(|replacement| replacement.map(MlxTensor::into_array))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), Exception> {
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: MlxTensor::ref_cast(routing.selected_experts),
                selected_scores: MlxTensor::ref_cast(routing.selected_scores),
                coefficients: MlxTensor::ref_cast(routing.coefficients),
                routed_output: MlxTensor::ref_cast(routing.routed_output),
                local_routed_output: routing.local_routed_output.map(MlxTensor::ref_cast),
                reduced_routed_output: routing.reduced_routed_output.map(MlxTensor::ref_cast),
                shared_output: routing.shared_output.map(MlxTensor::ref_cast),
                combined_output: routing.combined_output.map(MlxTensor::ref_cast),
                expert_count: routing.expert_count,
            })
    }
}

/// Backend-owned output of one MLX model-session submission.
///
/// Pipeline ranks that do not own the output projection complete with no local
/// logits. The final rank and every non-pipeline session complete with logits.
#[derive(Debug, Clone)]
pub struct MlxModelOutput {
    logits: Option<MlxTensor>,
}

impl MlxModelOutput {
    const fn new(logits: Option<MlxTensor>) -> Self {
        Self { logits }
    }

    /// Borrows local logits when this rank owns them.
    pub const fn logits(&self) -> Option<&MlxTensor> {
        self.logits.as_ref()
    }

    /// Consumes the output and returns local logits when present.
    pub fn into_logits(self) -> Option<MlxTensor> {
        self.logits
    }
}

/// Opaque exact completion for any MLX model-session submission.
pub struct MlxSessionCompletion {
    inner: MlxSessionCompletionKind,
}

/// MLX token handle yielded by backend-generic text generation.
#[derive(Clone)]
pub struct MlxTextToken {
    value: Array,
    stream: Stream,
}

impl TokenOutput for MlxTextToken {
    type Error = Error;

    fn token_id(&self) -> Result<u32, Self::Error> {
        self.value
            .clone()
            .try_item::<u32>(&self.stream)
            .map_err(Into::into)
    }
}

/// Exact completion retaining both model execution and sampled token output.
pub struct MlxTextCompletion {
    model: MlxSessionCompletion,
    token: MlxCompletion,
}

impl Completion for MlxTextCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self.model.is_complete()? && self.token.is_complete()?)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.model.wait()?;
        self.token.wait()
    }
}

/// MLX sampling and randomness state for backend-generic text generation.
pub struct MlxTextGenerationState {
    temperature: f32,
    prng: Option<RandomState>,
    sampler: MlxTextSampler,
}

enum MlxTextSampler {
    Standard(GenerationSampler),
    MirostatV2(MirostatV2Sampler),
}

impl MlxTextSampler {
    fn sample(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        match self {
            Self::Standard(sampler) => {
                Sampler::<MlxSamplingBackend>::sample(sampler, logits, temperature, random, stream)
            }
            Self::MirostatV2(sampler) => {
                Sampler::<MlxSamplingBackend>::sample(sampler, logits, temperature, random, stream)
            }
        }
    }
}

struct FilteredTextSampler<'a> {
    sampler: &'a mut MlxTextSampler,
    filter: &'a TokenFilter,
}

impl Sampler<MlxSamplingBackend> for FilteredTextSampler<'_> {
    fn sample(
        &mut self,
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let logits = MlxSamplingBackend::apply_token_filter(logits, self.filter, stream)?;
        self.sampler.sample(&logits, temperature, random, stream)
    }
}

enum MlxSessionCompletionKind {
    Model {
        completion: MlxCompletion,
        token_validations: TokenValidationBatch,
    },
    Pipeline(PipelineStageCompletion),
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model { completion, .. } => completion.is_complete(),
            MlxSessionCompletionKind::Pipeline(completion) => completion.is_complete(),
        }
    }

    fn wait(&self) -> Result<(), Self::Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model {
                completion,
                token_validations,
            } => {
                completion.wait()?;
                token_validations.validate_completed().map_err(Into::into)
            }
            MlxSessionCompletionKind::Pipeline(completion) => completion.synchronize(),
        }
    }
}

/// MLX-owned prefill input.
///
/// Arrays are cloned handles, not copied tensor storage. Owning the handles
/// makes submission independent of the caller's temporary `ModelInput` view.
#[derive(Debug, Clone)]
pub struct MlxModelInput {
    parts: Vec<input::InputPart>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: input.parts.to_vec(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        input.with_model_input(|input| Self::from(input))
    }

    /// Borrows the owned input parts as a model-input view for one operation.
    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        execute(input::ModelInput::new(&self.parts))
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession<'a> {
    inner: MlxSessionKind,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    distributed: Option<MlxDistributedSession<'a>>,
    capabilities: eredu_core::SessionCapabilities,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

enum MlxSessionKind {
    Complete(Executable),
    Pipeline(
        crate::composition::mlx::distributed::pipeline::PipelineModel,
        PipelineCache,
    ),
}

pub(super) enum MlxSpeculativeSessionParts<'session, 'world> {
    Complete {
        model: &'session mut Executable,
        execution: Option<&'session MlxDistributedSession<'world>>,
    },
    Pipeline {
        model: &'session mut PipelineModel,
        execution: &'session MlxDistributedSession<'world>,
    },
}

impl<'a> MlxModelSession<'a> {
    /// Creates a session and validates that its communicator matches the model topology.
    pub(crate) fn from_model(
        model: MlxModel,
        distributed: Option<MlxDistributedSession<'a>>,
        admitted_capabilities: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        let floating_state_dtype_bytes = model.floating_state_dtype_bytes();
        let topology = model.topology();
        match (topology, distributed.as_ref()) {
            (None, None) => {}
            (Some(expected), Some(session)) => {
                super::distributed::topology::validate_session(expected, session)?;
            }
            (Some(_), None) => {
                return Err(Error::Parallel(
                    "distributed MLX model has no session-owned communication".into(),
                ))
            }
            (None, Some(_)) => {
                return Err(Error::Parallel(
                    "replicated MLX model cannot own distributed communication".into(),
                ))
            }
        }
        #[cfg(any(feature = "image", feature = "audio"))]
        let mut model = model;
        #[cfg(any(feature = "image", feature = "audio"))]
        let processor = model.take_processor();
        let inner = match model.into_kind() {
            MlxModelKind::Complete(model) => MlxSessionKind::Complete(model),
            MlxModelKind::Pipeline(model) => {
                let cache = model.new_cache()?;
                MlxSessionKind::Pipeline(model, cache)
            }
        };
        let realized_capabilities = eredu_core::SessionCapabilities {
            persistent_cache: true,
            output_observation: true,
            activation_inspection: true,
        };
        if admitted_capabilities != realized_capabilities {
            return Err(Error::ArchitectureModel(format!(
                "realized MLX session capabilities {realized_capabilities:?} do not match pre-materialization admission {admitted_capabilities:?}"
            )));
        }
        if let MlxSessionKind::Complete(model) = &inner {
            if let Some(selected) = model.selected_session_binding() {
                selected
                    .validate_bound_mechanisms(realized_capabilities, true, true)
                    .map_err(Error::ArchitectureModel)?;
            }
        }
        Ok(Self {
            inner,
            floating_state_dtype_bytes,
            distributed,
            capabilities: realized_capabilities,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor,
        })
    }

    pub(super) const fn floating_state_dtype_bytes(&self) -> std::num::NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    /// Returns the canonical architecture family of the session-owned model.
    pub fn model_family(&self) -> eredu_architectures::ModelKind {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.model_family(),
            MlxSessionKind::Pipeline(model, _) => model.model_family(),
        }
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.effective_model_type(),
            MlxSessionKind::Pipeline(model, _) => model.effective_model_type(),
        }
    }

    /// Reports how the session-owned model exposes speculative weights.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.speculative_capability(),
            MlxSessionKind::Pipeline(model, _) => model.speculative_capability(),
        }
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.residency_report(),
            MlxSessionKind::Pipeline(model, _) => model.parameter_residency_report(),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.dense_stream_report(),
            MlxSessionKind::Pipeline(model, _) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.parameter_bank_report(),
            MlxSessionKind::Pipeline(model, _) => model.parameter_bank_report(),
        }
    }

    /// Returns the complete model-derived identity for a reusable prompt cache.
    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => {
                model.prompt_cache_model_identity().map_err(Into::into)
            }
            MlxSessionKind::Pipeline(model, _) => model.prompt_cache_model_identity(),
        }
    }

    pub(super) fn validate_external_drafter(&self, drafter: &MlxDrafter) -> Result<(), Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => match model {
                Executable::Gemma4(_, target, _)
                    if drafter.kind() == MlxDrafterKind::Gemma4Assistant =>
                {
                    let _compatibility = drafter
                        .gemma4()
                        .config
                        .prove_compatibility(&target.args().text)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                }
                Executable::MuseGlimmer(_, target, _)
                    if drafter.kind() == MlxDrafterKind::MuseGlimmerDFlash =>
                {
                    let _compatibility = drafter
                        .muse_glimmer()
                        .config
                        .prove_compatibility(target.args())
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                }
                model @ (Executable::DeepSeek(_, _, _)
                | Executable::Gemma4(_, _, _)
                | Executable::GptOss(_, _, _)
                | Executable::Inkling(_, _, _)
                | Executable::KimiLinear(_, _, _)
                | Executable::Lfm2(_, _, _)
                | Executable::PartitionedLlama(_, _, _)
                | Executable::ReplicatedText(_, _)
                | Executable::MuseGlimmer(_, _, _)
                | Executable::NemotronH(_, _, _)
                | Executable::Qwen(_, _, _)
                | Executable::Qwen3Next(_, _, _)
                | Executable::Qwen3Vl(_, _, _)
                | Executable::Qwen3VlMoe(_, _, _)
                | Executable::Qwen35(_, _, _)) => {
                    return Err(Error::ArchitectureModel(format!(
                        "drafter {:?} is incompatible with target {} ({:?})",
                        drafter.kind(),
                        model.effective_model_type(),
                        model.speculative_capability()
                    )))
                }
            },
            MlxSessionKind::Pipeline(_, _) => {
                return Err(Error::Speculative(
                    "external drafting is unavailable for pipeline sessions".into(),
                ))
            }
        }
        Ok(())
    }

    pub(super) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.architecture_capability_estimate(),
            MlxSessionKind::Pipeline(model, _) => model.capability_estimate(),
        }
    }

    pub(super) fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.prepared_input_part_plan(input),
            MlxSessionKind::Pipeline(model, _) => model.prepared_input_part_plan(input),
        }
    }

    pub(super) fn speculative_parts_mut(
        &mut self,
    ) -> Result<MlxSpeculativeSessionParts<'_, 'a>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => Ok(MlxSpeculativeSessionParts::Complete {
                model,
                execution: self.distributed.as_ref(),
            }),
            MlxSessionKind::Pipeline(model, _) => {
                let execution = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "pipeline speculative execution requires session-owned communication"
                            .into(),
                    )
                })?;
                Ok(MlxSpeculativeSessionParts::Pipeline { model, execution })
            }
        }
    }

    /// Clears all MLX cache state while preserving session topology.
    pub fn reset(&mut self) -> Result<(), Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => model.reset_cache().map_err(Into::into),
            MlxSessionKind::Pipeline(_, cache) => cache.reset(),
        }
    }

    /// Submits one cached decode position from a portable token id.
    pub fn submit_token_decode(
        &mut self,
        backend: &MlxBackend<'a>,
        token_id: u32,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        let token_ids = [token_id];
        let input =
            Array::from(token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        self.decode(backend, input)
    }

    /// Returns aggregate cache-residency telemetry for this session.
    pub fn cache_residency_report(
        &self,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => model
                .cache_residency_report()
                .map_err(|error| Error::Parallel(error.to_string())),
            MlxSessionKind::Pipeline(model, cache) => model.cache_residency_report(cache),
        }
    }

    /// Replaces this session's cache with one allocated under `policy`.
    ///
    /// Cache representation remains an MLX backend detail for complete,
    /// pipeline, and expert models alike.
    pub fn configure_cache(&mut self, policy: CacheResidencyPolicy) -> Result<(), Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => model.reset_cache_with_options(policy)?,
            MlxSessionKind::Pipeline(model, cache) => {
                *cache = model.new_cache_with_options(policy)?;
            }
        }
        Ok(())
    }

    /// Atomically persists the completed prefix owned by this session.
    pub fn save_prompt_cache(
        &mut self,
        backend: &MlxBackend<'a>,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => model
                .save_prompt_cache(
                    root,
                    descriptor,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )
                .map_err(Into::into),
            MlxSessionKind::Pipeline(model, cache) => model.save_prompt_cache(
                cache,
                root,
                descriptor,
                prefix_token_ids,
                options,
                backend.stream(),
            ),
        }
    }

    /// Opens a compatible persisted prefix and replaces this session's cache.
    pub fn load_prompt_cache(
        &mut self,
        backend: &MlxBackend<'a>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let manifest = match &mut self.inner {
            MlxSessionKind::Complete(model) => model.load_prompt_cache(
                root,
                expected,
                prefix_token_ids,
                options,
                backend.stream(),
            )?,
            MlxSessionKind::Pipeline(model, cache) => {
                let (loaded_cache, manifest) = model.load_prompt_cache(
                    root,
                    expected,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )?;
                *cache = loaded_cache;
                manifest
            }
        };
        Ok(manifest)
    }

    /// Returns communication when this is a distributed session.
    pub const fn distributed(&self) -> Option<&MlxDistributedSession<'a>> {
        self.distributed.as_ref()
    }

    /// Samples on the canonical rank and synchronizes the result for this
    /// distributed model session.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        let distributed = self.distributed.as_ref().ok_or_else(|| {
            Error::Parallel("sampling synchronization requires a distributed model session".into())
        })?;
        match &self.inner {
            MlxSessionKind::Pipeline(model, _) => model.sample_and_synchronize_token(
                logits.map(MlxTensor::as_array),
                batch_size,
                sampler,
                temperature,
                prng_state,
                finished,
                distributed,
            ),
            MlxSessionKind::Complete(model) => {
                let topology = model
                    .parallel_info()
                    .map(|info| info.topology())
                    .ok_or_else(|| {
                        Error::Parallel(
                            "distributed complete-model sampling requires selected topology".into(),
                        )
                    })?;
                let sampling_rank = topology.global_rank_for(eredu_core::ParallelCoordinates {
                    tensor: 0,
                    pipeline: topology.pipeline_parallel_size - 1,
                    expert: 0,
                    data: topology.data_parallel_rank,
                })?;
                distributed.sample_and_synchronize_on_rank(
                    logits,
                    batch_size,
                    sampler,
                    temperature,
                    prng_state,
                    finished,
                    sampling_rank,
                )
            }
        }
    }

    /// Runs one MLX instrumented pass through the architecture-erased adapter.
    pub(crate) fn forward_with_observer(
        model: &mut Executable,
        input_tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Array, Error> {
        let mut observer = ArrayObserverAdapter { inner: observer };
        let result = match model {
            Executable::DeepSeek(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::Gemma4(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Gemma observer mask is unsupported; the adapter constructs its per-layer masks from cache state".into(),
                    ));
                }
                model
                    .forward_tokens_with_observer(
                        MlxTensor::ref_cast(input_tokens),
                        cache,
                        stream,
                        &mut observer,
                    )
                    .map(MlxTensor::into_array)
            }
            Executable::GptOss(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::Inkling(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit Inkling observer masks are unsupported".into(),
                    ));
                }
                model
                    .forward_tokens_with_observer(
                        MlxTensor::ref_cast(input_tokens),
                        cache,
                        stream,
                        &mut observer,
                    )
                    .map(MlxTensor::into_array)
            }
            Executable::KimiLinear(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Kimi Linear observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::Lfm2(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::PartitionedLlama(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::ReplicatedText(_, model) => {
                model.forward_with_observer(input_tokens, mask, stream, &mut observer)
            }
            Executable::MuseGlimmer(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit Muse-Glimmer observer masks are unsupported".into(),
                    ));
                }
                model
                    .forward_tokens_with_observer(
                        MlxTensor::ref_cast(input_tokens),
                        cache,
                        stream,
                        &mut observer,
                    )
                    .map(MlxTensor::into_array)
            }
            Executable::NemotronH(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::Qwen(_, model, cache) => {
                model.forward_with_observer(input_tokens, mask, cache, stream, &mut observer)
            }
            Executable::Qwen3Next(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Qwen hybrid observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, &mut observer)
            }
            Executable::Qwen3Vl(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit Qwen3-VL observer masks are unsupported".into(),
                    ));
                }
                model.forward_tokens_with_observer(input_tokens, cache, stream, &mut observer)
            }
            Executable::Qwen3VlMoe(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit Qwen3-VL-MoE observer masks are unsupported".into(),
                    ));
                }
                model.forward_tokens_with_observer(input_tokens, cache, stream, &mut observer)
            }
            Executable::Qwen35(_, model, cache) => {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "an explicit Qwen hybrid observer mask is unsupported; the adapter constructs the causal mask from cache state".into(),
                    ));
                }
                model.forward_with_observer(input_tokens, cache, stream, &mut observer)
            }
        };
        result.map_err(|error| safemlx::error::Exception::custom(error.to_string()).into())
    }

    /// Submits instrumented prefill through this selected MLX session.
    pub fn submit_prefill_with_observer(
        &mut self,
        backend: &MlxBackend<'a>,
        input: MlxModelInput,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => match &self.distributed {
                Some(distributed) => {
                    let output = input.with_borrowed(|input| {
                        prefill_model_tensor_parallel_with_observer(
                            model,
                            input,
                            distributed,
                            backend.stream(),
                            observer,
                        )
                    })?;
                    MlxCompletion::submission(output)
                }
                None => Self::submit_complete_prefill_with_observer(
                    model,
                    input,
                    backend.stream(),
                    observer,
                ),
            },
            MlxSessionKind::Pipeline(_, _) => unreachable!(
                "pipeline prefill inspection is dispatched through the pipeline executor"
            ),
        }
    }

    /// Submits an instrumented prefill for a non-pipeline executable.
    pub(crate) fn submit_complete_prefill_with_observer(
        model: &mut Executable,
        input: MlxModelInput,
        stream: &Stream,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        let output = input.with_borrowed(|input| {
            let logits = match model {
                Executable::DeepSeek(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::Gemma4(_, family, cache) => family
                    .prefill_with_observer(
                        input,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                    .into_array(),
                Executable::GptOss(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::Inkling(_, family, cache) => family
                    .forward_input_with_observer(
                        input,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                    .into_array(),
                Executable::KimiLinear(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::Lfm2(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::PartitionedLlama(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::ReplicatedText(_, family) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::MuseGlimmer(_, family, cache) => family
                    .forward_input_with_observer(
                        input,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                    .into_array(),
                Executable::NemotronH(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::Qwen(_, family, cache) => {
                    let tokens = input::text_token_ids(input, stream)?;
                    family.forward_with_observer(
                        &tokens,
                        None,
                        cache,
                        stream,
                        &mut ArrayObserverAdapter { inner: observer },
                    )?
                }
                Executable::Qwen3Next(_, family, cache) => family.prefill_input_with_observer(
                    input,
                    cache,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                )?,
                Executable::Qwen3Vl(_, family, cache) => family.prefill_with_observer(
                    input,
                    cache,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                )?,
                Executable::Qwen3VlMoe(_, family, cache) => family.prefill_with_observer(
                    input,
                    cache,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                )?,
                Executable::Qwen35(_, family, cache) => family.prefill_input_with_observer(
                    input,
                    cache,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                )?,
            };
            logits
                .try_index_device((.., -1, ..), stream)
                .map_err(Error::from)
        })?;
        MlxCompletion::submission(output)
    }

    fn submit_decode_with_observer(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Array,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => {
                let output = match &self.distributed {
                    Some(distributed) => forward_model_tensor_parallel_with_observer(
                        model,
                        &input,
                        distributed
                            .selected_group(crate::backend::distributed::SHARD_GROUP_ID)
                            .ok_or_else(|| {
                                Error::Parallel(
                                    "tensor-parallel model session has no tensor communicator"
                                        .into(),
                                )
                            })?,
                        backend.stream(),
                        observer,
                    )?,
                    None => Self::forward_with_observer(
                        model,
                        &input,
                        None,
                        backend.stream(),
                        observer,
                    )?,
                }
                .try_index_device((.., -1, ..), backend.stream())?;
                MlxCompletion::submission(output)
            }
            MlxSessionKind::Pipeline(_, _) => unreachable!(
                "pipeline decode inspection is dispatched through the pipeline executor"
            ),
        }
    }
}

impl<'a> BackendSession<MlxBackend<'a>> for MlxModelSession<'a> {
    type PrefillInput = MlxModelInput;
    type DecodeInput = Array;
    type Output = MlxModelOutput;
    type Completion = MlxSessionCompletion;

    fn capabilities(&self) -> eredu_core::SessionCapabilities {
        self.capabilities
    }

    fn prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => {
                let token_validation_scope = TokenValidationScope::begin()?;
                let output = match &self.distributed {
                    Some(distributed) => input.with_borrowed(|input| {
                        prefill_model_tensor_parallel(model, input, distributed, backend.stream())
                    })?,
                    None => input
                        .with_borrowed(|input| prefill_model(model, input, backend.stream()))?,
                };
                model_submission(output, token_validation_scope.finish())
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let completion = input.with_borrowed(|borrowed| {
                    if model.requires_embedded_mtp_prefill() {
                        return model.prefill_distributed_with_embedded_mtp(
                            borrowed,
                            cache,
                            distributed,
                            None,
                        );
                    }
                    let multimodal = borrowed
                        .parts
                        .iter()
                        .any(|part| part.modality() != InputModality::Text);
                    if multimodal {
                        let step = model.prepared_input_step(borrowed)?;
                        model.prefill_distributed(
                            model.stage_info().owns_input.then_some(borrowed),
                            step,
                            None,
                            cache,
                            distributed,
                        )
                    } else {
                        let tokens = input::text_token_ids(borrowed, backend.stream())?;
                        let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))?;
                        model.forward_distributed(
                            model.stage_info().owns_input.then_some(&tokens),
                            step,
                            None,
                            cache,
                            distributed,
                        )
                    }
                })?;
                pipeline_submission(completion)
            }
        }
    }

    fn decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => {
                let token_validation_scope = TokenValidationScope::begin()?;
                let output = match &self.distributed {
                    Some(distributed) => {
                        decode_model_tensor_parallel(model, &input, distributed, backend.stream())?
                    }
                    None => decode_model(model, &input, backend.stream())?,
                };
                model_submission(output, token_validation_scope.finish())
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let step = PipelineStep::new(input.dim(0), input.dim(1))?;
                let completion = model.forward_distributed(
                    model.stage_info().owns_input.then_some(&input),
                    step,
                    None,
                    cache,
                    distributed,
                )?;
                pipeline_submission(completion)
            }
        }
    }

    fn observe_output(
        &self,
        backend: &MlxBackend<'a>,
        output: &Self::Output,
    ) -> Result<ObservationSet, Error> {
        let mut observations = ObservationSet::new();
        if let Some(logits) = output.logits() {
            observations
                .insert(
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                    ObservationValue::Tensor(observe_tensor(logits, backend.stream())?),
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(observations)
    }
}

impl<'a> InspectableBackendSession<MlxBackend<'a>> for MlxModelSession<'a> {
    fn inspect_prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let output = match &mut self.inner {
            MlxSessionKind::Complete(_) => {
                let submission =
                    self.submit_prefill_with_observer(backend, input, &mut collector)?;
                let logits = submission.wait()?;
                MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let completion = input.with_borrowed(|borrowed| {
                    let mut observer = ArrayObserverAdapter {
                        inner: &mut collector,
                    };
                    if model.requires_embedded_mtp_prefill() {
                        return model.prefill_distributed_with_embedded_mtp(
                            borrowed,
                            cache,
                            distributed,
                            Some(&mut observer),
                        );
                    }
                    let multimodal = borrowed
                        .parts
                        .iter()
                        .any(|part| part.modality() != InputModality::Text);
                    if multimodal {
                        let step = model.prepared_input_step(borrowed)?;
                        model.prefill_distributed_with_observer(
                            model.stage_info().owns_input.then_some(borrowed),
                            step,
                            None,
                            cache,
                            distributed,
                            &mut observer,
                        )
                    } else {
                        let tokens = input::text_token_ids(borrowed, backend.stream())?;
                        let step = PipelineStep::new(tokens.dim(0), tokens.dim(1))?;
                        model.forward_distributed_with_observer(
                            model.stage_info().owns_input.then_some(&tokens),
                            step,
                            None,
                            cache,
                            distributed,
                            &mut observer,
                        )
                    }
                })?;
                completion.synchronize()?;
                MlxModelOutput::new(completion.logits().cloned().map(MlxTensor::from_array))
            }
        };
        let observations = collector.materialize(backend.stream())?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }

    fn inspect_decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let output = match &mut self.inner {
            MlxSessionKind::Complete(_) => {
                let submission =
                    self.submit_decode_with_observer(backend, input, &mut collector)?;
                let logits = submission.wait()?;
                MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
            }
            MlxSessionKind::Pipeline(model, cache) => {
                let distributed = self.distributed.as_ref().ok_or_else(|| {
                    Error::Parallel("pipeline model session has no communication".into())
                })?;
                let step = PipelineStep::new(input.dim(0), input.dim(1))?;
                let mut observer = ArrayObserverAdapter {
                    inner: &mut collector,
                };
                let completion = model.forward_distributed_with_observer(
                    model.stage_info().owns_input.then_some(&input),
                    step,
                    None,
                    cache,
                    distributed,
                    &mut observer,
                )?;
                completion.synchronize()?;
                MlxModelOutput::new(completion.logits().cloned().map(MlxTensor::from_array))
            }
        };
        let observations = collector.materialize(backend.stream())?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }
}

impl<'a> TextGenerationBackend for MlxBackend<'a> {
    type Prompt = MlxModelInput;
    type Token = MlxTextToken;
    type TextGenerationState = MlxTextGenerationState;
    type TextCompletion = MlxTextCompletion;

    fn start_text_generation(
        _: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Error> {
        let sampling = config.sampling();
        let prng = if sampling.temperature == 0.0 {
            None
        } else {
            Some(RandomState::from_key(safemlx::random::key(config.seed())?))
        };
        let sampler = match config.strategy() {
            TextSamplingStrategy::Standard => {
                MlxTextSampler::Standard(GenerationSampler::from_resolved(sampling))
            }
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                let sampler = MirostatV2Sampler::new(tau, eta)
                    .map_err(|error| eredu_core::BackendError::Execution {
                        session: "text-generation".into(),
                        operation: "configure Mirostat V2".into(),
                        message: error.to_string(),
                    })?
                    .penalties(
                        sampling.repetition_penalty,
                        sampling.repeat_last_n,
                        sampling.frequency_penalty,
                        sampling.presence_penalty,
                    );
                MlxTextSampler::MirostatV2(sampler)
            }
        };
        Ok(MlxTextGenerationState {
            temperature: sampling.temperature,
            prng,
            sampler,
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::ArchitectureModel(
                "text generation requires at least one prompt token".into(),
            ));
        }
        let tokens =
            Array::from(prompt_token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        let parts = [input::input_part(
            InputModality::Text,
            input::InputPayload::TokenIds(tokens),
            [],
            [],
        )?];
        Ok(MlxModelInput::from(input::ModelInput::new(&parts)))
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let submission = runtime.prefill(prompt)?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let input = token.value.try_index_device((.., NewAxis), &stream)?;
        let submission = runtime.decode(input)?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
    }
}

fn sample_text_submission(
    session: &MlxModelSession<'_>,
    submission: Submission<MlxModelOutput, MlxSessionCompletion>,
    filter: &TokenFilter,
    state: &mut MlxTextGenerationState,
    stream: Stream,
) -> Result<Submission<MlxTextToken, MlxTextCompletion>, Error> {
    let MlxTextGenerationState {
        temperature,
        prng,
        sampler,
    } = state;
    let mut sampler = FilteredTextSampler { sampler, filter };
    let token = if session.distributed().is_some() {
        session
            .sample_and_synchronize(
                submission.output.logits(),
                1,
                &mut sampler,
                *temperature,
                prng.as_mut(),
                false,
            )?
            .token
            .try_index_device((.., 0), &stream)?
    } else {
        let logits = submission
            .output
            .logits()
            .ok_or_else(|| Error::Parallel("local text generation requires model logits".into()))?;
        Sampler::<MlxSamplingBackend>::sample(
            &mut sampler,
            logits,
            *temperature,
            prng.as_mut(),
            &stream,
        )?
        .into_array()
    };
    let sampled = MlxCompletion::submission(token)?;
    Ok(Submission {
        output: MlxTextToken {
            value: sampled.output,
            stream,
        },
        completion: MlxTextCompletion {
            model: submission.completion,
            token: sampled.completion,
        },
    })
}

fn model_submission(
    output: Array,
    token_validations: TokenValidationBatch,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let submission =
        MlxCompletion::submission_retaining(output, token_validations.arrays().cloned())?;
    Ok(Submission {
        output: MlxModelOutput::new(Some(MlxTensor::from_array(submission.output))),
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model {
                completion: submission.completion,
                token_validations,
            },
        },
    })
}

fn pipeline_submission(
    completion: PipelineStageCompletion,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let output = MlxModelOutput::new(completion.logits().cloned().map(MlxTensor::from_array));
    Ok(Submission {
        output,
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Pipeline(completion),
        },
    })
}

#[cfg(test)]
mod tests {
    use crate::backend::ExecutionContext;
    use safemlx::{Array, Device, DeviceType};

    use crate::backend::nn::tensor::validate_token_domain;

    use super::*;

    #[test]
    fn owned_input_preserves_multimodal_parts_and_metadata() {
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let image = Array::from_slice(&[0.0_f32; 8], &[1, 2, 4]);
        let grid = Array::from_slice(&[1_i32, 2, 2], &[1, 3]);
        let parts = [
            input::input_part(
                InputModality::Text,
                input::InputPayload::TokenIds(tokens),
                [],
                [],
            )
            .unwrap(),
            input::input_part(
                InputModality::Image,
                input::InputPayload::Tensor(image),
                [(eredu_core::InputMetadataKey::PatchGrid, grid)],
                [],
            )
            .unwrap(),
        ];

        let owned = MlxModelInput::from(input::ModelInput::new(&parts));
        owned.with_borrowed(|borrowed| {
            assert_eq!(borrowed.parts.len(), 2);
            assert_eq!(borrowed.parts[0].modality(), InputModality::Text);
            assert_eq!(borrowed.parts[1].modality(), InputModality::Image);
            assert_eq!(
                borrowed.parts[1]
                    .metadata_value(eredu_core::InputMetadataKey::PatchGrid)
                    .expect("grid metadata")
                    .shape(),
                &[1, 3]
            );
        });
    }

    #[test]
    fn model_session_is_the_backend_session_implementation() {
        fn assert_session<T: BackendSession<MlxBackend<'static>>>() {}
        fn assert_inspectable<T: InspectableBackendSession<MlxBackend<'static>>>() {}
        assert_session::<MlxModelSession>();
        assert_inspectable::<MlxModelSession>();
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn complete_model_submission_completes_token_validation() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let submit = |token| {
            let scope = TokenValidationScope::begin().unwrap();
            validate_token_domain(&Array::from_int(token), 4, None, stream).unwrap();
            model_submission(Array::from_int(0), scope.finish()).unwrap()
        };

        submit(3).completion.wait().unwrap();
        let error = submit(4).completion.wait().unwrap_err();
        assert!(error.to_string().contains("outside 0..4"));
    }

    #[test]
    fn inspection_collector_filters_and_materializes_portable_values() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let requested = ObservationRequest::selected([eredu_core::ObservationSelector::Exact(
            "model.layers.1.output".into(),
        )]);
        let mut collector = InspectionCollector::new(&requested);
        collector.capture(
            "model.layers.0.output",
            &MlxTensor::from_array(Array::from_slice(&[1.0f32], &[1])),
        );
        collector.capture(
            "model.layers.1.output",
            &MlxTensor::from_array(Array::from_slice(&[2.0f32, 3.0], &[1, 2])),
        );
        let observations = collector.materialize(stream).unwrap();
        assert_eq!(observations.len(), 1);
        let Some(ObservationValue::Tensor(tensor)) = observations.get("model.layers.1.output")
        else {
            panic!("selected activation must be a tensor");
        };
        assert_eq!(tensor.shape(), [1, 2]);
        assert_eq!(tensor.data(), &TensorObservationData::F32(vec![2.0, 3.0]));
    }
}
