//! MLX mechanisms and generic binding for replicated text composition.

use std::{marker::PhantomData, path::Path, sync::Arc};

use eredu_checkpoint::{store::CheckpointSource, LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
};
use eredu_nn::NeuralBackend;
use eredu_runtime::{
    BackendMechanismCapabilities, CacheResidencyPolicy, CacheResidencyReport,
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseRuntime, PagedCacheOptions,
    ReplicatedTextArchitecture, ReplicatedTextRequirements, ReplicatedTextSelectionRequest,
    ResidencyReport, RuntimeState, SelectedReplicatedTextRealization, StateResidencyMechanism,
    WeightLoweringCapability, WeightLoweringKind, WeightResidencyMechanism,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::{
                residency::{open_prompt_cache, CacheResidencyManager},
                state::MlxKeyValueState,
            },
            execution::generic::{prepare_layerwise_policy, MlxLayerwisePolicy, MlxResidentPolicy},
            media::input,
        },
    },
    native_quantization::NativeQuantizationFormat,
    MlxTensor,
};

use crate::backend::runtime::execution::{
    generic::{architecture_execution_layout, construct_architecture_unit},
    layerwise::quantize_parameterized_store,
};

use eredu_architectures::replicated_text::{
    PreparedReplicatedTextArchitecture, ReplicatedTextArchitectureVisitor,
};

/// Reports the exact MLX mechanisms applicable to one neutral requirement set.
///
/// The report is derived only from source encodings, executable formats, and
/// implemented backend facilities. It does not receive architecture identity.
pub(crate) fn capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
) -> BackendMechanismCapabilities {
    let mut weight_lowerings = Vec::new();
    for parameter in requirements.parameters() {
        let requested = request
            .quantization()
            .and_then(|requested| parameter.transform_target(requested).ok().flatten())
            .map(|target| target.executable());
        for executable in std::iter::once(parameter.native_executable()).chain(requested) {
            let kind = if executable == parameter.native_executable()
                && supports_direct(parameter.source_encoding(), executable)
            {
                Some(WeightLoweringKind::Direct)
            } else if supports_transform(parameter.source_encoding(), executable) {
                Some(WeightLoweringKind::Transform)
            } else {
                None
            };
            if let Some(kind) = kind {
                let capability = WeightLoweringCapability::new(
                    parameter.source_encoding().clone(),
                    executable,
                    kind,
                );
                if !weight_lowerings.contains(&capability) {
                    weight_lowerings.push(capability);
                }
            }
        }
    }
    BackendMechanismCapabilities::new(
        MlxNeuralBackend::OPERATOR_CAPABILITIES,
        weight_lowerings,
        vec![
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ],
        vec![
            StateResidencyMechanism::Device,
            StateResidencyMechanism::Paged,
        ],
    )
    .with_session(eredu_core::SessionCapabilities {
        persistent_cache: true,
        output_observation: true,
        activation_inspection: true,
    })
    .with_prompt_cache(true)
    .with_exact_completion(true)
}

/// Backend-private erased operations for a paired architecture and mutable state.
pub trait ErasedReplicatedTextExecutable {
    fn effective_model_type(&self) -> &str;
    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate;
    fn selected_session_binding(&self) -> &SelectedSessionBinding;
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport>;
    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity;
    fn reset_cache(&mut self) -> Result<(), Exception>;
    fn reset_cache_with_options(&mut self, policy: CacheResidencyPolicy) -> Result<(), Error>;
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error>;
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error>;
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
}

/// Session mechanisms admitted by the authoritative realization.
pub struct SelectedSessionBinding {
    capabilities: eredu_core::SessionCapabilities,
    prompt_cache: bool,
    exact_completion: bool,
}

impl SelectedSessionBinding {
    fn from_selected(selected: &SelectedReplicatedTextRealization) -> Self {
        Self {
            capabilities: selected.session(),
            prompt_cache: selected.prompt_cache(),
            exact_completion: selected.exact_completion(),
        }
    }

    pub(crate) fn validate_bound_mechanisms(
        &self,
        capabilities: eredu_core::SessionCapabilities,
        prompt_cache: bool,
        exact_completion: bool,
    ) -> Result<(), String> {
        self.capabilities
            .validate(&capabilities)
            .map_err(|error| error.to_string())?;
        if self.prompt_cache && !prompt_cache {
            return Err("selected prompt-cache persistence was not bound by the session".into());
        }
        if self.exact_completion && !exact_completion {
            return Err("selected exact completion was not bound by the session".into());
        }
        Ok(())
    }
}

type ResidentRuntime<A> = LayerwiseRuntime<
    A,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<
        <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::Unit,
    >,
>;
type BoundedRuntime<A> = LayerwiseRuntime<
    A,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<
        <A as eredu_runtime::LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>>::Unit,
    >,
>;

enum Execution<A>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    Resident(ResidentRuntime<A>),
    Bounded(BoundedRuntime<A>),
}

struct BoundReplicatedText<A>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    execution: Execution<A>,
    state_layout: eredu_runtime::StateLayout,
    state: MlxKeyValueState,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    selected_session: SelectedSessionBinding,
}

impl<A> BoundReplicatedText<A>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn new(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
        options: LayerWeightResidency,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        let requirements = prepared.requirements().clone();
        let selected = prepared.selected().clone();
        let selected_session = SelectedSessionBinding::from_selected(&selected);
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let mut modules = prepared.into_modules();
        let mut architecture = modules.take_architecture();
        let source_architecture = modules.take_source_architecture();
        validate_residency(&selected, options)?;
        validate_architecture_contract(&architecture, &requirements)?;
        let (store, materialization) = match source_architecture {
            Some(source) => {
                let quantization = selected_transform_quantization(&selected)?;
                let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source)?;
                let target_layout =
                    architecture_execution_layout::<_, MlxKeyValueState>(&architecture)?;
                if source_layout != target_layout {
                    return Err(Error::Quantization(
                        "selected weight transform changed the execution-unit layout".into(),
                    ));
                }
                let unit_count = source_layout.len();
                let source_static = source.static_modules().clone();
                let target_static = architecture.static_modules().clone();
                let (store, report) = quantize_parameterized_store(
                    store,
                    &source_static,
                    &target_static,
                    |ordinal, context| {
                        construct_architecture_unit(
                            &source,
                            &source_layout,
                            ordinal,
                            context,
                            PhantomData::<MlxKeyValueState>,
                        )
                    },
                    |ordinal, context| {
                        construct_architecture_unit(
                            &architecture,
                            &target_layout,
                            ordinal,
                            context,
                            PhantomData::<MlxKeyValueState>,
                        )
                    },
                    unit_count,
                    quantization,
                    stream,
                )?;
                (store, Some(report))
            }
            None => (store, None),
        };
        let state_layout = architecture
            .state_layout()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let prompt_cache_identity = crate::composition::replicated_prompt_cache_identity(
            &architecture,
            eredu_core::cache::PromptCacheTopology::default(),
        )?;
        let (policy, _) = prepare_layerwise_policy(
            store,
            &mut architecture,
            (),
            PhantomData::<MlxKeyValueState>,
            options,
            stream,
            weights_stream,
            |_| false,
        )?;
        let execution = if options.is_fully_resident() {
            Execution::Resident(LayerwiseRuntime::new_policy_first(
                policy.into_resident(&architecture, stream, PhantomData::<MlxKeyValueState>)?,
                architecture,
            ))
        } else {
            Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
        };
        let state = match selected.state() {
            CacheResidencyPolicy::Device => MlxKeyValueState::device(state_layout.clone())?,
            CacheResidencyPolicy::Paged(options) => MlxKeyValueState::paged(
                state_layout.clone(),
                CacheResidencyManager::new(options.clone())
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )?,
        };
        Ok(Self {
            execution,
            state_layout,
            state,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            materialization,
            selected_session,
        })
    }

    fn forward(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.validate_state()?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let input = A::text_input(&tokens, mask.as_ref());
        let output = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, &mut self.state, stream),
            Execution::Bounded(runtime) => runtime.forward(input, &mut self.state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    fn validate_state(&self) -> Result<(), Error> {
        if self.state.layout() != &self.state_layout {
            return Err(Error::ArchitectureModel(
                "replicated text state layout does not match its paired architecture".into(),
            ));
        }
        Ok(())
    }

    fn new_state(&self, policy: CacheResidencyPolicy) -> Result<MlxKeyValueState, Error> {
        match policy {
            CacheResidencyPolicy::Device => {
                MlxKeyValueState::device(self.state_layout.clone()).map_err(Into::into)
            }
            CacheResidencyPolicy::Paged(options) => MlxKeyValueState::paged(
                self.state_layout.clone(),
                CacheResidencyManager::new(options)
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )
            .map_err(Into::into),
        }
    }
}

fn validate_architecture_contract<A>(
    architecture: &A,
    requirements: &ReplicatedTextRequirements,
) -> Result<(), Error>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if &graph != requirements.execution_graph() {
        return Err(Error::ArchitectureModel(
            "constructed execution graph differs from selected replicated requirements".into(),
        ));
    }
    let units = architecture_execution_layout::<_, MlxKeyValueState>(architecture)?;
    if &units != requirements.execution_units() {
        return Err(Error::ArchitectureModel(
            "constructed execution-unit geometry differs from selected replicated requirements"
                .into(),
        ));
    }
    let transports = (0..graph.groups().len())
        .map(|group| architecture.group_transport(group))
        .collect::<Vec<_>>();
    if transports != requirements.group_transports() {
        return Err(Error::ArchitectureModel(
            "constructed group transport differs from selected replicated requirements".into(),
        ));
    }
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if &state_layout != requirements.state_layout() {
        return Err(Error::ArchitectureModel(
            "constructed mutable-state layout differs from selected replicated requirements".into(),
        ));
    }
    Ok(())
}

impl<A> ErasedReplicatedTextExecutable for BoundReplicatedText<A>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    fn selected_session_binding(&self) -> &SelectedSessionBinding {
        &self.selected_session
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report()?,
            Execution::Bounded(runtime) => runtime.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.materialization.as_ref()
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.state = MlxKeyValueState::device(self.state_layout.clone())?;
        Ok(())
    }

    fn reset_cache_with_options(&mut self, policy: CacheResidencyPolicy) -> Result<(), Error> {
        self.state = self.new_state(policy)?;
        Ok(())
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(
            expected,
            &self.prompt_cache_identity,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            directory,
            expected,
            &self.prompt_cache_identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        self.state = MlxKeyValueState::paged(self.state_layout.clone(), manager, None)?;
        Ok(manifest)
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(
            &descriptor,
            &self.prompt_cache_identity,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        self.state
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.state.residency_report()
    }

    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, None, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        self.forward(tokens, None, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        self.validate_state()?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let input = A::text_input(&tokens, mask.as_ref());
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_observer(input, &mut self.state, stream, &mut observer)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_observer(input, &mut self.state, stream, &mut observer)
            }
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_runtime::observe_model_logits(&mut observer, &output)
            .map(MlxTensor::into_array)
            .map_err(Into::into)
    }
}

fn validate_residency(
    selected: &SelectedReplicatedTextRealization,
    options: LayerWeightResidency,
) -> Result<(), Error> {
    let actual = match options {
        LayerWeightResidency::FullyResident => WeightResidencyMechanism::Resident,
        LayerWeightResidency::LayerwiseHost(_) => WeightResidencyMechanism::Windowed,
        LayerWeightResidency::DenseDiskStream(_) => WeightResidencyMechanism::DiskStreamed,
    };
    if selected.residency() != actual {
        return Err(Error::ArchitectureModel(format!(
            "selected replicated text residency {:?} disagrees with runtime policy {actual:?}",
            selected.residency()
        )));
    }
    Ok(())
}

fn selected_transform_quantization(
    selected: &SelectedReplicatedTextRealization,
) -> Result<eredu_checkpoint::WeightQuantization, Error> {
    let mut quantization = None;
    for parameter in selected.parameters() {
        if parameter.lowering() != WeightLoweringKind::Transform {
            continue;
        }
        let current = parameter
            .executable()
            .weight_quantization()
            .ok_or_else(|| {
                Error::Quantization(format!(
                    "selected transform for {:?} has no materializable packed format",
                    parameter.name()
                ))
            })?;
        if quantization
            .replace(current)
            .is_some_and(|prior| prior != current)
        {
            return Err(Error::Quantization(
                "one replicated text realization selected multiple transform formats".into(),
            ));
        }
    }
    quantization.ok_or_else(|| {
        Error::Quantization("selected transform contains no transformed parameters".into())
    })
}

/// Family-agnostic MLX visitor that binds neutral parameter topology.
pub(crate) struct BindingVisitor<'a> {
    pub store: Arc<dyn CheckpointSource>,
    pub options: LayerWeightResidency,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
}

impl ReplicatedTextArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState> for BindingVisitor<'_> {
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxKeyValueState, Error = eredu_nn::Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        BoundReplicatedText::new(
            prepared,
            self.store,
            self.options,
            self.stream,
            self.weights_stream,
        )
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }
}

fn supports_direct(source: &SourceTensorEncoding, executable: LinearFormat) -> bool {
    match (source, executable) {
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32,
            ),
            LinearFormat::Dense,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3),
            LinearFormat::E4M3BlockFp8(format),
        ) => format.validate().is_ok(),
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Dense) => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Affine(format)) => {
            gguf_affine(*ggml_type).is_some_and(|native| native == format)
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::MxFp4) => {
            *ggml_type == eredu_gguf::GgmlType::MxFp4
        }
        (
            SourceTensorEncoding::Gguf { ggml_type, endian },
            LinearFormat::GgufIQuant {
                ggml_type: executable,
                endian: executable_endian,
            },
        ) => {
            *ggml_type == executable
                && *endian == executable_endian
                && NativeQuantizationFormat::from_ggml_type(executable).is_some()
        }
        _ => false,
    }
}

fn supports_transform(source: &SourceTensorEncoding, executable: LinearFormat) -> bool {
    let decodable = match source {
        SourceTensorEncoding::Safetensors(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::Gguf { ggml_type, .. } => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        _ => false,
    };
    decodable
        && match executable {
            LinearFormat::Affine(format) => format.validate().is_ok(),
            LinearFormat::MxFp4 => true,
            LinearFormat::Dense
            | LinearFormat::GgufIQuant { .. }
            | LinearFormat::E4M3BlockFp8(_) => false,
        }
}

fn gguf_affine(ggml_type: eredu_gguf::GgmlType) -> Option<eredu_checkpoint::AffineQuantization> {
    let (bits, group_size) = match ggml_type {
        eredu_gguf::GgmlType::Q2K => (2, 16),
        eredu_gguf::GgmlType::Q3K => (3, 16),
        eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::Q4_1 | eredu_gguf::GgmlType::Q4K => {
            (4, 32)
        }
        eredu_gguf::GgmlType::Q5_0 | eredu_gguf::GgmlType::Q5_1 | eredu_gguf::GgmlType::Q5K => {
            (5, 32)
        }
        eredu_gguf::GgmlType::Q6K => (6, 16),
        eredu_gguf::GgmlType::Q8_0 => (8, 32),
        _ => return None,
    };
    eredu_checkpoint::AffineQuantization::new(group_size, bits).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::SourceTensorEncoding;
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ModelConfigurationResolver,
    };
    use eredu_runtime::{ReplicatedTextParameterRequirement, StateLayout};

    #[test]
    fn report_distinguishes_native_and_transforming_lowerings() {
        let parameter = ReplicatedTextParameterRequirement::new(
            "projection.weight",
            vec!["projection.weight".into()],
            SourceTensorEncoding::Safetensors(StoredDtype::F16),
            LinearFormat::Dense,
        )
        .unwrap()
        .with_affine_transforms();
        let graph = eredu_runtime::ExecutionGraph::chain(["decoder"]).unwrap();
        let requirements = ReplicatedTextRequirements::new(
            eredu_nn::NeuralOperatorCapabilities::NONE,
            graph.clone(),
            eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap(),
            vec![eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            vec![parameter],
        )
        .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_core::ResidencyRequest::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_quantization(eredu_core::QuantizationRequest::Affine {
            group_size: 64,
            bits: 4,
        });
        let report = capabilities(&requirements, &request);
        assert!(report.weight_lowerings().iter().any(|lowering| {
            lowering.executable() == LinearFormat::Dense
                && lowering.kind() == WeightLoweringKind::Direct
        }));
        assert!(report.weight_lowerings().iter().any(|lowering| {
            matches!(lowering.executable(), LinearFormat::Affine(_))
                && lowering.kind() == WeightLoweringKind::Transform
        }));
    }

    fn tiny_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, false)
    }

    fn tiny_sharded_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
        tiny_safetensors_artifact(model_type, tied, true)
    }

    fn tiny_safetensors_artifact(model_type: &str, tied: bool, sharded: bool) -> tempfile::TempDir {
        use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

        let root = tempfile::tempdir().unwrap();
        let architecture = match model_type {
            "llama" => "LlamaForCausalLM",
            "mistral" => "MistralForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            _ => unreachable!(),
        };
        let config = serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tied
        });
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap();
        let plan = resolved
            .architecture_plan
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
        constraints.extend(
            plan.layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        );
        let tensors = constraints
            .into_iter()
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let elements = constraint.shape.iter().product::<usize>();
                (
                    constraint.key.clone(),
                    constraint.shape.clone(),
                    vec![0_u8; elements * 4],
                )
            })
            .collect::<Vec<_>>();
        let write = |path: &Path, tensors: &[(String, Vec<usize>, Vec<u8>)]| {
            let views = tensors
                .iter()
                .map(|(name, shape, bytes)| {
                    (
                        name.as_str(),
                        TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, path).unwrap();
        };
        if sharded {
            let split = tensors.len() / 2;
            let first = "model-00001-of-00002.safetensors";
            let second = "model-00002-of-00002.safetensors";
            write(&root.path().join(first), &tensors[..split]);
            write(&root.path().join(second), &tensors[split..]);
            let weight_map = tensors
                .iter()
                .enumerate()
                .map(|(index, (name, _, _))| {
                    (name.clone(), if index < split { first } else { second })
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            std::fs::write(
                root.path().join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
            )
            .unwrap();
        } else {
            write(&root.path().join("model.safetensors"), &tensors);
        }
        root
    }

    fn execution_streams() -> (Stream, Stream) {
        let execution_device = if cfg!(feature = "metal") {
            safemlx::Device::new(safemlx::DeviceType::Gpu, 0)
        } else {
            safemlx::Device::new(safemlx::DeviceType::Cpu, 0)
        };
        let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
        (
            Stream::new_with_device(&execution_device),
            Stream::new_with_device(&weights_device),
        )
    }

    fn tiny_llama_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
        ]);
        let tensors = [
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            (
                name.to_string(),
                Array::zeros::<f32>(&shape, stream).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    fn tiny_qwen_gguf(
        architecture: &str,
        packed: Option<eredu_gguf::GgmlType>,
        stream: &Stream,
    ) -> crate::test_utils::SyntheticGguf {
        use std::collections::HashMap;

        use eredu_gguf::MetadataValue;

        let key = |suffix: &str| format!("{architecture}.{suffix}");
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String(architecture.into()),
            ),
            ("general.file_type".into(), MetadataValue::Uint32(0)),
            (key("block_count"), MetadataValue::Uint32(1)),
            (key("embedding_length"), MetadataValue::Uint32(32)),
            (key("attention.head_count"), MetadataValue::Uint32(4)),
            (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
            (key("feed_forward_length"), MetadataValue::Uint32(64)),
            (
                key("attention.layer_norm_rms_epsilon"),
                MetadataValue::Float32(1e-5),
            ),
            (key("vocab_size"), MetadataValue::Uint32(64)),
            (key("context_length"), MetadataValue::Uint32(32)),
            (key("rope.freq_base"), MetadataValue::Float32(1_000_000.0)),
        ]);
        let mut tensors = vec![
            ("token_embd.weight", vec![64, 32]),
            ("output_norm.weight", vec![32]),
            ("blk.0.attn_norm.weight", vec![32]),
            ("blk.0.ffn_norm.weight", vec![32]),
            ("blk.0.attn_q.weight", vec![32, 32]),
            ("blk.0.attn_k.weight", vec![8, 32]),
            ("blk.0.attn_v.weight", vec![8, 32]),
            ("blk.0.attn_output.weight", vec![32, 32]),
            ("blk.0.ffn_gate.weight", vec![64, 32]),
            ("blk.0.ffn_up.weight", vec![64, 32]),
            ("blk.0.ffn_down.weight", vec![32, 64]),
        ];
        if architecture == "qwen2" {
            tensors.extend([
                ("blk.0.attn_q.bias", vec![32]),
                ("blk.0.attn_k.bias", vec![8]),
                ("blk.0.attn_v.bias", vec![8]),
            ]);
        } else {
            tensors.extend([
                ("blk.0.attn_q_norm.weight", vec![8]),
                ("blk.0.attn_k_norm.weight", vec![8]),
            ]);
        }
        let tensors = tensors
            .into_iter()
            .map(|(name, shape)| {
                (
                    name.to_string(),
                    Array::zeros::<f32>(&shape, stream).unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();
        crate::test_utils::SyntheticGguf::with_packed_tensors(&tensors, &metadata, |name, array| {
            packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2)
        })
    }

    #[test]
    fn public_handoff_executes_llama_and_dense_qwen_with_repeated_decode() {
        let (stream, weights_stream) = execution_streams();
        for (model_type, tied) in [
            ("llama", true),
            ("mistral", false),
            ("qwen2", false),
            ("qwen3", true),
        ] {
            let root = tiny_artifact(model_type, tied);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = super::super::loading::materialize_model_plan(
                plan,
                crate::backend::ModelLoadOptions::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            for token in [1_u32, 2] {
                let logits = executable
                    .decode(&Array::from_slice(&[token], &[1, 1]), &stream)
                    .unwrap();
                assert_eq!(logits.shape(), &[1, 64]);
                logits.evaluated().unwrap();
            }
        }
    }

    #[test]
    fn both_text_architectures_retain_exact_sharded_admission_with_one_cached_payload() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["mistral", "qwen3"] {
            let root = tiny_sharded_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let streaming = eredu_runtime::DenseDiskStreamLoadOptions {
                max_cached_shards: 1,
                ..Default::default()
            };
            let options = crate::backend::ModelLoadOptions::default().with_weight_residency(
                eredu_runtime::WeightResidency::dense_disk_stream(streaming),
            );
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            std::fs::remove_file(root.path().join("model.safetensors.index.json")).unwrap();

            let model = super::super::loading::materialize_model_plan(
                plan,
                options,
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{model_type}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary sharded text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let report = executable.dense_stream_report().unwrap().unwrap();
            assert!(report.residency().weight_store().currently_cached_shards <= 1);
            assert!(!report
                .residency()
                .weight_store()
                .payload_shard_paths
                .is_empty());
        }
    }

    #[test]
    fn unsupported_topology_fails_before_checkpoint_payload_or_module_construction() {
        let root = tiny_artifact("llama", false);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let topology = crate::backend::MlxParallelContext::for_rank(
            0,
            2,
            1,
            1,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let requirements =
            eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                .unwrap();
        let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_core::ResidencyRequest::FullyResident,
            CacheResidencyPolicy::Device,
        );
        let request = request.with_topology(topology.topology());
        std::fs::remove_file(root.path().join("model.safetensors")).unwrap();

        let error = eredu_runtime::select_replicated_text_realization(
            &requirements,
            &request,
            &capabilities(&requirements, &request),
        )
        .expect_err("unsupported topology was admitted");
        let message = error.to_string();
        assert!(
            message.contains("replicated execution topology"),
            "{message}"
        );
        assert!(!message.contains("No such file"), "{message}");
    }

    #[test]
    fn selected_paged_state_controls_generic_construction() {
        let (stream, weights_stream) = execution_streams();
        for model_type in ["llama", "qwen3"] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let policy = eredu_core::PreparationPolicy::default();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            let state = CacheResidencyPolicy::Paged(
                PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            );
            let request = eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_core::ResidencyRequest::FullyResident,
                state.clone(),
            );
            let selected = eredu_runtime::select_replicated_text_realization(
                &requirements,
                &request,
                &capabilities(&requirements, &request),
            )
            .unwrap();
            assert_eq!(selected.state(), &state);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                policy,
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let (artifact, architecture_plan, _, _) = plan.into_parts();
            let eredu_core::ModelArtifact::SafeTensors {
                configuration,
                tensors,
                shards,
                ..
            } = artifact
            else {
                panic!("expected SafeTensors fixture")
            };
            let prepared = super::super::artifact::PreparedSafetensorsArtifact::open(
                configuration,
                super::super::loading::prepared_safetensors_architecture(&architecture_plan)
                    .unwrap()
                    .clone(),
                tensors,
                shards,
                1,
            )
            .unwrap();
            let executable =
                eredu_architectures::replicated_text::visit_replicated_text_architecture(
                    &architecture_plan,
                    requirements,
                    selected,
                    &stream,
                    BindingVisitor {
                        store: prepared.store(),
                        options: LayerWeightResidency::FullyResident,
                        stream: &stream,
                        weights_stream: &weights_stream,
                    },
                )
                .unwrap();
            assert!(executable.cache_residency_report().unwrap().is_some());
        }
    }

    #[test]
    fn public_handoff_executes_selected_load_time_transform() {
        let (stream, weights_stream) = execution_streams();
        for (model_type, request) in [
            (
                "llama",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("llama", eredu_core::QuantizationRequest::MxFp4),
            (
                "qwen3",
                eredu_core::QuantizationRequest::Affine {
                    group_size: 32,
                    bits: 4,
                },
            ),
            ("qwen3", eredu_core::QuantizationRequest::MxFp4),
        ] {
            let root = tiny_artifact(model_type, false);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let options = crate::backend::ModelLoadOptions::with_quantization(request);
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = super::super::loading::materialize_model_plan(
                plan,
                options,
                &stream,
                &weights_stream,
            )
            .unwrap();
            assert!(model.materialization_report().is_some());
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn public_handoff_executes_admitted_gguf_mapping() {
        let (stream, weights_stream) = execution_streams();
        let artifacts = [
            tiny_llama_gguf("llama", None, &stream),
            tiny_llama_gguf("mistral", None, &stream),
            tiny_qwen_gguf("qwen2", None, &stream),
            tiny_qwen_gguf("qwen3", None, &stream),
        ];
        for artifact in artifacts {
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = super::super::loading::materialize_model_plan(
                plan,
                crate::backend::ModelLoadOptions::default(),
                &stream,
                &weights_stream,
            )
            .unwrap();
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary replicated GGUF text must use the generic executable")
            };
            let logits = executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap();
            assert_eq!(logits.shape(), &[1, 64]);
            logits.evaluated().unwrap();
        }
    }

    #[test]
    fn both_text_architectures_execute_checkpoint_native_packed_gguf_formats() {
        let (stream, weights_stream) = execution_streams();
        for (architecture, format) in [
            ("llama", eredu_gguf::GgmlType::Q4_0),
            ("llama", eredu_gguf::GgmlType::MxFp4),
            ("llama", eredu_gguf::GgmlType::IQ4NL),
            ("qwen2", eredu_gguf::GgmlType::Q4_0),
            ("qwen2", eredu_gguf::GgmlType::MxFp4),
            ("qwen2", eredu_gguf::GgmlType::IQ4NL),
        ] {
            let artifact = if architecture == "llama" {
                tiny_llama_gguf(architecture, Some(format), &stream)
            } else {
                tiny_qwen_gguf(architecture, Some(format), &stream)
            };
            let checkpoint = eredu_gguf::Checkpoint::open(artifact.path()).unwrap();
            let translated = if architecture == "llama" {
                checkpoint
                    .translated_outputs(eredu_architectures::llama::translate_gguf_weight_name)
                    .unwrap()
            } else {
                checkpoint
                    .translated_outputs(|name| {
                        eredu_architectures::qwen::translate_gguf_weight_name(name, false)
                    })
                    .unwrap()
            };
            if matches!(
                format,
                eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::MxFp4
            ) {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".scales")
                        && mapping.layout.name.ends_with(".scales")
                        && mapping.layout.name.starts_with("model.")
                }));
            }
            if format == eredu_gguf::GgmlType::Q4_0 {
                assert!(translated.iter().any(|mapping| {
                    mapping.original_name.ends_with(".biases")
                        && mapping.layout.name.ends_with(".biases")
                }));
            }
            let inspection =
                eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
            let requirements =
                eredu_architectures::replicated_text::replicated_text_requirements(&inspection)
                    .unwrap();
            assert!(requirements.parameters().iter().any(|parameter| {
                matches!(
                    (format, parameter.native_executable()),
                    (eredu_gguf::GgmlType::Q4_0, LinearFormat::Affine(_))
                        | (eredu_gguf::GgmlType::MxFp4, LinearFormat::MxFp4)
                        | (eredu_gguf::GgmlType::IQ4NL, LinearFormat::GgufIQuant { .. })
                )
            }));
            let plan = eredu_core::plan_model_preparation(
                inspection,
                eredu_core::PreparationPolicy::default(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = super::super::loading::materialize_model_plan(
                plan,
                crate::backend::ModelLoadOptions::default(),
                &stream,
                &weights_stream,
            )
            .unwrap_or_else(|error| panic!("{architecture} {format:?}: {error}"));
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, executable) = &mut executable else {
                panic!("ordinary packed GGUF text must use the generic executable")
            };
            executable
                .decode(&Array::from_slice(&[1_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }

    #[test]
    fn generic_controls_cover_residency_cache_persistence_and_observation() {
        struct Observer {
            activation: bool,
            logits: bool,
        }
        impl eredu_runtime::ActivationObserver<Array, Exception> for Observer {
            fn observe(&mut self, path: &str, _value: &Array) -> Result<(), Exception> {
                self.logits |= path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                self.activation |= path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, Exception> {
                Ok((path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH).then(|| value.clone()))
            }
        }

        let (stream, weights_stream) = execution_streams();
        for (model_type, residency) in ["llama", "qwen2"].into_iter().flat_map(|family| {
            [
                eredu_runtime::WeightResidency::fully_resident(),
                eredu_runtime::WeightResidency::layerwise_host(Default::default()),
                eredu_runtime::WeightResidency::dense_disk_stream(Default::default()),
            ]
            .into_iter()
            .map(move |residency| (family, residency))
        }) {
            let root = tiny_artifact(model_type, false);
            let options =
                crate::backend::ModelLoadOptions::default().with_weight_residency(residency);
            let inspection =
                eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
            let plan = eredu_core::plan_model_preparation(
                inspection,
                options.preparation_policy().unwrap(),
                eredu_core::SessionCapabilities::default(),
            )
            .unwrap();
            let model = super::super::loading::materialize_model_plan(
                plan,
                options,
                &stream,
                &weights_stream,
            )
            .unwrap();
            assert!(model.residency_report().unwrap().is_some());
            assert_eq!(
                model.dense_stream_report().unwrap().is_some(),
                matches!(
                    residency,
                    eredu_runtime::WeightResidency::Layers(
                        eredu_runtime::LayerWeightResidency::DenseDiskStream(_)
                    )
                )
            );
            let mut executable = model.into_complete().unwrap();
            let super::super::Executable::ReplicatedText(_, generic) = &mut executable else {
                panic!("ordinary replicated text must use the generic executable")
            };
            generic
                .decode(&Array::from_slice(&[1_u32, 2], &[1, 2]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();

            let mut observer = Observer {
                activation: false,
                logits: false,
            };
            generic
                .forward_with_observer(
                    &Array::from_slice(&[3_u32], &[1, 1]),
                    None,
                    &stream,
                    &mut observer,
                )
                .unwrap()
                .evaluated()
                .unwrap();
            assert!(observer.logits);
            assert!(observer.activation);

            let identity = generic.prompt_cache_model_identity().clone();
            let descriptor = PromptCacheDescriptor::from_model_identity(
                identity,
                "tiny-checkpoint",
                "tokens:1,2,3",
                1,
            )
            .unwrap();
            let cache_root = tempfile::tempdir().unwrap();
            let destination = cache_root.path().join("cache");
            let prefix = [1_u32, 2, 3];
            let paged = PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true);
            generic
                .reset_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
                .unwrap();
            generic
                .decode(&Array::from_slice(&prefix, &[1, 3]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
            let manifest = generic
                .save_prompt_cache(
                    &destination,
                    descriptor.clone(),
                    &prefix,
                    &PromptCacheOptions::default(),
                )
                .unwrap();
            assert_eq!(manifest.block_size_tokens, paged.block_size_tokens());
            let mut incompatible = descriptor.clone();
            incompatible.architecture_fingerprint.push_str("-different");
            assert!(generic
                .load_prompt_cache(&destination, &incompatible, &prefix, paged.clone())
                .is_err());
            generic
                .load_prompt_cache(&destination, &descriptor, &prefix, paged)
                .unwrap();
            assert!(generic.cache_residency_report().unwrap().is_some());
            generic
                .decode(&Array::from_slice(&[4_u32], &[1, 1]), &stream)
                .unwrap()
                .evaluated()
                .unwrap();
        }
    }
}
