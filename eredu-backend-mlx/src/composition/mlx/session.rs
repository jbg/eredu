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
    MlxTensor,
};
#[cfg(any(feature = "image", feature = "audio"))]
use crate::{backend::runtime::media::PreparedModelInput, composition::mlx::ModelProcessor};
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::SpeculativeCapability;
use eredu_runtime::CacheResidencyPolicy;

use super::{
    execution::{decode_model, prefill_model},
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
}

impl Completion for MlxSessionCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        match &self.inner {
            MlxSessionCompletionKind::Model { completion, .. } => completion.is_complete(),
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
    cache_identity: Option<eredu_runtime::PreparedInputCacheIdentity>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: input.parts.to_vec(),
            cache_identity: input.cache_identity().cloned(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        let mut owned = input.with_model_input(|borrowed| Self::from(borrowed));
        owned.cache_identity = input.cache_identity().cloned();
        owned
    }

    /// Returns the exact semantic identity carried by processor-produced input.
    pub const fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Couples manually prepared tensors to caller-owned semantic content.
    pub fn with_semantic_content_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Result<Self, Error> {
        let prepared = eredu_runtime::PreparedModelInput::new(self.parts.clone(), |array| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, array)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.cache_identity = Some(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Ok(self)
    }

    /// Borrows the owned input parts as a model-input view for one operation.
    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        let input = match self.cache_identity.as_ref() {
            Some(identity) => input::ModelInput::with_cache_identity(&self.parts, identity),
            None => input::ModelInput::new(&self.parts),
        };
        execute(input)
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession {
    inner: MlxSessionKind,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    distributed: Option<MlxDistributedSession>,
    capabilities: eredu_core::SessionCapabilities,
    state_residency: CacheResidencyPolicy,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

enum MlxSessionKind {
    Complete(Executable),
}

pub(super) enum MlxSpeculativeSessionParts<'session> {
    Complete { model: &'session mut Executable },
}

impl MlxModelSession {
    /// Creates a session and validates that its communicator matches the model topology.
    pub(crate) fn from_model(
        model: MlxModel,
        admitted_capabilities: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        let floating_state_dtype_bytes = model.floating_state_dtype_bytes();
        let state_residency = model.state_residency().clone();
        #[cfg(any(feature = "image", feature = "audio"))]
        let mut model = model;
        #[cfg(any(feature = "image", feature = "audio"))]
        let processor = model.take_processor();
        let inner = match model.into_kind() {
            MlxModelKind::Complete(mut model) => {
                model
                    .reset_cache_with_options(state_residency.clone())
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                MlxSessionKind::Complete(model)
            }
        };
        let realized_capabilities = eredu_core::SessionCapabilities::new(true, true, true);
        if admitted_capabilities != realized_capabilities {
            return Err(Error::ArchitectureModel(format!(
                "realized MLX session capabilities {realized_capabilities:?} do not match pre-materialization admission {admitted_capabilities:?}"
            )));
        }
        Ok(Self {
            inner,
            floating_state_dtype_bytes,
            distributed: None,
            capabilities: realized_capabilities,
            state_residency,
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

    #[cfg(test)]
    pub(crate) fn model_family(&self) -> eredu_architectures::ModelKind {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.model_family(),
        }
    }

    pub(crate) fn effective_model_type(&self) -> &str {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.effective_model_type(),
        }
    }

    /// Reports how the session-owned model exposes speculative weights.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.speculative_capability(),
        }
    }

    /// Installs causal observers on this session's selected embedded-prediction executor.
    pub fn install_embedded_prediction_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: RuntimeActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: RuntimeActivationObserver<Array, Exception> + 'static,
    {
        let observers =
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                tensors, logits,
            );
        let MlxSessionKind::Complete(model) = &mut self.inner;
        if model.install_embedded_prediction_observers(observers) {
            Ok(())
        } else {
            Err(Error::ArchitectureModel(
                "session has no selected embedded-prediction executor".into(),
            ))
        }
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.residency_report(),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.dense_stream_report(),
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
        }
    }

    pub(super) fn external_assistant_target_profile(
        &self,
    ) -> Result<eredu_architectures::external_assistant::ExternalAssistantTargetProfile, Error>
    {
        match &self.inner {
            MlxSessionKind::Complete(Executable::ReplicatedText(_, target)) => target
                .external_prediction()
                .map(|target| target.external_assistant_target_profile())
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "selected target does not publish an external-assistant profile".into(),
                    )
                }),
            MlxSessionKind::Complete(model) => Err(Error::ArchitectureModel(format!(
                "external assistants are unavailable for target {}",
                model.effective_model_type()
            ))),
        }
    }

    pub(super) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.architecture_capability_estimate(),
        }
    }

    pub(super) fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        match &self.inner {
            MlxSessionKind::Complete(model) => model.prepared_input_part_plan(input),
        }
    }

    pub(super) fn speculative_parts_mut(
        &mut self,
    ) -> Result<MlxSpeculativeSessionParts<'_>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => Ok(MlxSpeculativeSessionParts::Complete { model }),
        }
    }

    #[cfg(test)]
    pub(crate) fn neutral_prediction_target_mut(
        &mut self,
    ) -> Result<&mut dyn super::replicated_text::ErasedReplicatedTextExecutable, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(Executable::ReplicatedText(_, target)) => Ok(target.as_mut()),
            MlxSessionKind::Complete(model) => Err(Error::ArchitectureModel(format!(
                "test session retained a non-neutral prediction target for {}",
                model.effective_model_type()
            ))),
        }
    }

    /// Clears all MLX cache state under the authoritative selected policy.
    pub fn reset(&mut self) -> Result<(), Error> {
        let MlxSessionKind::Complete(model) = &mut self.inner;
        if model.has_neutral_partitioned_control() {
            return model.reset_cache_distributed().map_err(Into::into);
        }
        model
            .reset_cache_with_options(self.state_residency.clone())
            .map_err(Into::into)
    }

    /// Submits one cached decode position from a portable token id.
    pub fn submit_token_decode(
        &mut self,
        backend: &MlxBackend<'_>,
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
        }
    }

    /// Atomically persists the completed prefix owned by this session.
    pub fn save_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let root = root.as_ref();
        match &mut self.inner {
            MlxSessionKind::Complete(model) if model.has_neutral_partitioned_control() => model
                .save_prompt_cache_distributed(root, descriptor, prefix_token_ids, options)?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                }),
            MlxSessionKind::Complete(model) => model
                .save_prompt_cache(
                    root,
                    descriptor,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )
                .map_err(Into::into),
        }
    }

    /// Opens a compatible persisted prefix and replaces this session's cache.
    pub fn load_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        let root = root.as_ref();
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let options = options.clone();
        let manifest = match &mut self.inner {
            MlxSessionKind::Complete(model) if model.has_neutral_partitioned_control() => model
                .load_prompt_cache_distributed(root, expected, prefix_token_ids)?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                })?,
            MlxSessionKind::Complete(model) => model.load_prompt_cache(
                root,
                expected,
                prefix_token_ids,
                options,
                backend.stream(),
            )?,
        };
        Ok(manifest)
    }

    /// Opens a persisted prefix only when it matches an exact prepared input.
    pub fn load_prompt_cache_for_input(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input: &MlxModelInput,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = input.cache_identity.clone().ok_or_else(|| {
            Error::ArchitectureModel(
                "prompt-cache loading requires prepared-input semantic identity".into(),
            )
        })?;
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let root = root.as_ref();
        match &mut self.inner {
            MlxSessionKind::Complete(model) if model.has_neutral_partitioned_control() => model
                .load_prompt_cache_for_input_distributed(
                    root,
                    expected,
                    prefix_token_ids,
                    identity,
                )?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                }),
            MlxSessionKind::Complete(model) => model
                .load_prompt_cache_for_input(
                    root,
                    expected,
                    prefix_token_ids,
                    identity,
                    options.clone(),
                    backend.stream(),
                )
                .map_err(Into::into),
        }
    }

    /// Returns communication when this is a distributed session.
    pub const fn distributed(&self) -> Option<&MlxDistributedSession> {
        self.distributed.as_ref()
    }

    fn synchronizes_sampling(&self) -> bool {
        self.distributed.is_some()
            || matches!(
                &self.inner,
                MlxSessionKind::Complete(Executable::ReplicatedText(_, executable))
                    if executable.partition_sampling_context().is_some()
            )
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
        match &self.inner {
            MlxSessionKind::Complete(model) => {
                if self.distributed.is_some() {
                    return Err(Error::Parallel(
                        "complete-model execution has no distributed production path".into(),
                    ));
                }
                let Executable::ReplicatedText(_, executable) = model else {
                    return Err(Error::Parallel(
                        "sampling synchronization requires a distributed model session".into(),
                    ));
                };
                let (group, authority, stream, sampling_rank) =
                    executable.partition_sampling_context().ok_or_else(|| {
                        Error::Parallel(
                            "sampling synchronization requires a distributed model session".into(),
                        )
                    })?;
                crate::backend::runtime::distributed::parallel::sample_and_synchronize_bounded(
                    logits,
                    batch_size,
                    sampler,
                    temperature,
                    prng_state,
                    finished,
                    sampling_rank,
                    group,
                    authority,
                    stream,
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
        backend: &MlxBackend<'_>,
        input: MlxModelInput,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => {
                if self.distributed.is_some() {
                    return Err(Error::Parallel(
                        "complete-model execution has no distributed production path".into(),
                    ));
                }
                Self::submit_complete_prefill_with_observer(
                    model,
                    input,
                    backend.stream(),
                    observer,
                )
            }
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
            if let Executable::ReplicatedText(_, family) = model {
                return family.prefill_with_observer(
                    input,
                    None,
                    stream,
                    &mut ArrayObserverAdapter { inner: observer },
                );
            }
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
                Executable::ReplicatedText(_, _) => unreachable!(
                    "replicated observed prefill returned through neutral output selection"
                ),
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
        backend: &MlxBackend<'_>,
        input: Array,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxCompletion>, Error> {
        match &mut self.inner {
            MlxSessionKind::Complete(model) => {
                if self.distributed.is_some() {
                    return Err(Error::Parallel(
                        "complete-model execution has no distributed production path".into(),
                    ));
                }
                let output = match model {
                    Executable::ReplicatedText(_, family) => family.decode_with_observer(
                        &input,
                        backend.stream(),
                        &mut ArrayObserverAdapter { inner: observer },
                    )?,
                    other => Self::forward_with_observer(
                        other,
                        &input,
                        None,
                        backend.stream(),
                        observer,
                    )?
                    .try_index_device((.., -1, ..), backend.stream())?,
                };
                MlxCompletion::submission(output)
            }
        }
    }
}

impl<'a> BackendSession<MlxBackend<'a>> for MlxModelSession {
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
                if self.distributed.is_some() {
                    return Err(Error::Parallel(
                        "complete-model execution has no distributed production path".into(),
                    ));
                }
                let token_validation_scope = TokenValidationScope::begin()?;
                let output =
                    input.with_borrowed(|input| prefill_model(model, input, backend.stream()))?;
                let public_output = match model {
                    Executable::ReplicatedText(_, executable) => {
                        executable.partition_public_output()
                    }
                    _ => true,
                };
                model_submission(output, token_validation_scope.finish(), public_output)
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
                if self.distributed.is_some() {
                    return Err(Error::Parallel(
                        "complete-model execution has no distributed production path".into(),
                    ));
                }
                let token_validation_scope = TokenValidationScope::begin()?;
                let output = decode_model(model, &input, backend.stream())?;
                let public_output = match model {
                    Executable::ReplicatedText(_, executable) => {
                        executable.partition_public_output()
                    }
                    _ => true,
                };
                model_submission(output, token_validation_scope.finish(), public_output)
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

impl<'a> InspectableBackendSession<MlxBackend<'a>> for MlxModelSession {
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
        };
        let output = match &self.inner {
            MlxSessionKind::Complete(Executable::ReplicatedText(_, executable))
                if !executable.partition_public_output() =>
            {
                MlxModelOutput::new(None)
            }
            _ => output,
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
        };
        let output = match &self.inner {
            MlxSessionKind::Complete(Executable::ReplicatedText(_, executable))
                if !executable.partition_public_output() =>
            {
                MlxModelOutput::new(None)
            }
            _ => output,
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
        MlxModelInput::from(input::ModelInput::new(&parts)).with_semantic_content_fingerprint(
            eredu_core::cache::prompt_cache_token_fingerprint(&prompt_token_ids),
        )
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
    session: &MlxModelSession,
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
    let token = if session.synchronizes_sampling() {
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
    public_output: bool,
) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
    let submission =
        MlxCompletion::submission_retaining(output, token_validations.arrays().cloned())?;
    Ok(Submission {
        output: MlxModelOutput::new(
            public_output.then(|| MlxTensor::from_array(submission.output)),
        ),
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model {
                completion: submission.completion,
                token_validations,
            },
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
            model_submission(Array::from_int(0), scope.finish(), true).unwrap()
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
