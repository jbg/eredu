// MLX artifact and residency binding for the neutral Qwen hybrid graph.

use std::{collections::BTreeSet, path::Path, sync::Arc};

use eredu_architectures::qwen::{
    hybrid::{
        self, ConditionalInput, ConditionalLayeredModel, ConditionalUnit, EmbeddedInput,
        HybridConfig, ParsedHybridConfig, Unit,
    },
    vl::InputPart,
};
use eredu_checkpoint::{
    store::{CheckpointSource, CompositeCheckpointSource},
    WeightQuantization,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, LayerWeightResidency,
    LayeredArchitecture, LayerwiseRuntime, PagedCacheOptions, ParameterRole, ResidencyReport,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

fn prepare_conditional_input(
    parsed: &ParsedHybridConfig,
    input: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<hybrid::PreparedConditionalInput<crate::MlxTensor>, Exception> {
    let prepared = crate::composition::mlx::replicated_text::prepared_composite_input(input)
        .map_err(|error| Exception::custom(error.to_string()))?;
    let admitted = eredu_architectures::media_plan::admit_qwen_hybrid_input(
        parsed,
        &prepared,
        &input::MlxTensorInputInspector,
    )
    .map_err(|error| Exception::custom(error.to_string()))?;
    let input =
        eredu_architectures::composite_execution::PreparedCompositeInput::new(&prepared, &admitted)
            .map_err(Exception::custom)?;
    hybrid::prepare_conditional_input(input, stream)
        .map_err(|error| Exception::custom(error.to_string()))
}

use crate::composition::grouped_provider::*;

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::{
            residency::{
                load_prompt_cache_state_tensors, open_prompt_cache, CacheResidencyManager,
            },
            state::MlxHybridState,
        },
        checkpoint::binding::{
            build_module_bindings_with_recipes, build_module_bindings_with_recipes_excluding,
            parameter_name_in_targets, populate_module_from_lease_excluding,
        },
        checkpoint::{
            load::gguf_quantization_configs, quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::generic::{
            construct_architecture_unit, prepare_layerwise_policy_with_bindings,
            MlxLayerwisePolicy, MlxResidentPolicy, MlxUnitPopulator,
        },
        execution::layerwise::quantize_parameterized_store,
        media::input,
        residency::parameter_bank::{AddressableParameterBank, ParameterBankEntry},
    },
};

#[cfg(test)]
use crate::backend::runtime::execution::generic::architecture_execution_layout;

type Architecture = hybrid::LayeredModel<MlxNeuralBackend>;
type Block = Unit<MlxNeuralBackend>;

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenHybridCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub units: Vec<Block>,
}

#[cfg(test)]
impl QwenHybridCheckpointTemplate {
    pub fn new(config: HybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = Architecture::new(config, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let units = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: architecture.into_static_modules(),
            units,
        })
    }
}

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct QwenConditionalCheckpointTemplate {
    pub static_modules: hybrid::ConditionalStaticModules<MlxNeuralBackend>,
    pub units: Vec<hybrid::ConditionalUnit<MlxNeuralBackend>>,
}

#[cfg(test)]
impl QwenConditionalCheckpointTemplate {
    pub fn new(parsed: ParsedHybridConfig, stream: &Stream) -> Result<Self, Error> {
        let architecture = ConditionalArchitecture::new(parsed, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let units = (0..layout.len())
            .map(|ordinal| {
                construct_architecture_unit(
                    &architecture,
                    &layout,
                    ordinal,
                    stream,
                    std::marker::PhantomData::<MlxHybridState>,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: <ConditionalArchitecture as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::static_modules(&architecture)
            .clone(),
            units,
        })
    }
}

#[derive(Clone)]
struct UnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

#[derive(Clone)]
struct ConditionalUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<ConditionalUnit<MlxNeuralBackend>> for ConditionalUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<ConditionalUnit<MlxNeuralBackend>>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

/// Canonical independent-expert catalog for selected architecture-owned units.
pub fn expert_catalog_selected(
    config: &HybridConfig,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog =
        hybrid::expert_residency_catalog(store, config).map_err(Error::ArchitectureModel)?;
    let units = catalog.into_units_selected_by_owner(owns_unit);
    crate::composition::architecture_expert_units(units, store, layout)
}

const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _config: &HybridConfig,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}

fn prepare_hybrid_gguf_store(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    max_cached_shards: usize,
) -> Result<(ParsedHybridConfig, Arc<dyn CheckpointSource>), Error> {
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::QwenHybrid(primary) = source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen hybrid GGUF loader received a different prepared model".into(),
        ));
    };
    let parsed = match projector {
        Some(projector) => {
            let eredu_architectures::gguf_companion::GgufMediaProjectorConfig::Qwen35(parsed) =
                projector.model()
            else {
                return Err(Error::ArchitectureModel(
                    "Qwen hybrid GGUF loader received a mismatched media-projector plan".into(),
                ));
            };
            parsed.clone()
        }
        None => primary.clone(),
    };
    let text_formats = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let text: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        max_cached_shards,
    )?);
    if parsed.text.variant == hybrid::HybridVariant::Qwen3Next {
        let parsed = hybrid::conditional_with_checkpoint_formats(&parsed, text_formats, None)
            .map_err(Error::ArchitectureModel)?;
        return Ok((parsed, text));
    }
    let Some(projector) = projector else {
        let parsed = hybrid::conditional_with_checkpoint_formats(&parsed, text_formats, None)
            .map_err(Error::ArchitectureModel)?;
        return Ok((parsed, text));
    };
    parsed.vision.as_ref().ok_or_else(|| {
        Error::ArchitectureModel("admitted Qwen3.5 projector omitted its vision geometry".into())
    })?;
    let vision_formats =
        gguf_quantization_configs(projector.checkpoint(), projector.plan().tensor_mapping())?;
    let vision_source: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        projector.checkpoint().clone(),
        projector.plan().checkpoint(),
        projector.plan().tensor_mapping(),
        max_cached_shards,
    )?);
    let parsed =
        hybrid::conditional_with_checkpoint_formats(&parsed, text_formats, Some(vision_formats))
            .map_err(Error::ArchitectureModel)?;
    Ok((
        parsed,
        Arc::new(CompositeCheckpointSource::new([text, vision_source])?),
    ))
}

/// Loads a llama.cpp Qwen3-Next/Qwen3.5 text artifact through the same
/// neutral resident/bounded execution graph as SafeTensors.
pub(crate) fn load_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    projector: Option<&crate::composition::mlx::structural::AdmittedGgufProjector>,
    _route: &crate::composition::mlx::loading::ExcludedFamilyRoute,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Qwen35
            | eredu_architectures::GgufArchitecture::Qwen35Moe
            | eredu_architectures::GgufArchitecture::Qwen3Next
    ) {
        return Err(Error::ArchitectureModel(format!(
            "Qwen hybrid GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let expert_options = residency.parameter_bank_cache();
    let options = residency.layers();
    let (mut parsed, store) =
        prepare_hybrid_gguf_store(source, projector, options.max_cached_shards())?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen hybrid GGUF", None, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = if let Some(quantization) = quantize_on_load {
        if parsed.vision.is_some() {
            let (store, target, _) =
                quantize_conditional_store(store, &parsed, quantization, stream)?;
            parsed = target;
            store
        } else {
            let (store, target, _) = quantize_store(store, &parsed.text, quantization, stream)?;
            parsed.text = target;
            store
        }
    } else {
        store
    };
    let mut model = if parsed.vision.is_some() {
        load_conditional_store(
            store,
            parsed,
            options,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?
    } else {
        load_store(
            store,
            parsed,
            options,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?
    };
    if let Some(expert_options) = expert_options {
        attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

impl MlxUnitPopulator<Block> for UnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<Block>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

type Resident =
    LayerwiseRuntime<Architecture, MlxNeuralBackend, MlxHybridState, MlxResidentPolicy<Block>>;
type Bounded = LayerwiseRuntime<
    Architecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<Block, UnitPopulator>,
>;
type ConditionalArchitecture = ConditionalLayeredModel<MlxNeuralBackend>;
type ConditionalResident = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<ConditionalUnit<MlxNeuralBackend>>,
>;
type ConditionalBounded = LayerwiseRuntime<
    ConditionalArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<ConditionalUnit<MlxNeuralBackend>, ConditionalUnitPopulator>,
>;

enum Execution {
    Resident(Box<Resident>),
    Bounded(Box<Bounded>),
    ConditionalResident(Box<ConditionalResident>),
    ConditionalBounded(Box<ConditionalBounded>),
}

struct PreparedConditionalOutput {
    logits: crate::MlxTensor,
    hidden: Option<crate::MlxTensor>,
    tokens: crate::MlxTensor,
}

/// Neutral Qwen3-Next/Qwen3.5 text model bound to MLX storage policy.
pub struct QwenHybridModel {
    parsed: ParsedHybridConfig,
    state_layout: eredu_runtime::StateLayout,
    execution: Execution,
    parameter_bank: Option<AddressableParameterBank>,
}

impl QwenHybridModel {
    pub(crate) fn requires_family_executable(&self) -> bool {
        self.parsed.text.is_moe()
            || self.parsed.text.mtp_num_hidden_layers > 0
            || self.parsed.vision.is_some()
            || self.parameter_bank.is_some()
    }

    /// Complete validated text and optional vision architecture policy.
    pub fn parsed_args(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Validated neutral text policy.
    pub fn args(&self) -> &HybridConfig {
        &self.parsed.text
    }

    /// Conditional vision is handled by the composite binder.
    pub fn vision_config(&self) -> Option<&eredu_architectures::qwen::vision::VisionConfig> {
        self.parsed.vision.as_ref()
    }

    /// Embedded prediction depth declared by the constructed architecture graph.
    pub fn mtp_len(&self) -> usize {
        match &self.execution {
            Execution::Resident(runtime) => runtime.architecture().mtp_len(),
            Execution::Bounded(runtime) => runtime.architecture().mtp_len(),
            Execution::ConditionalResident(runtime) => runtime.architecture().mtp_len(),
            Execution::ConditionalBounded(runtime) => runtime.architecture().mtp_len(),
        }
    }
    /// Allocates the declared recurrent, convolution, KV, and MTP state.
    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone()).expect("validated Qwen hybrid state")
    }

    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                MlxHybridState::paged(self.state_layout.clone(), manager, None).map_err(Into::into)
            }
        }
    }

    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(self
            .parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()?)
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::Bounded(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalResident(runtime) => runtime.policy().checkpoint_store_arc(),
            Execution::ConditionalBounded(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        let topology = eredu_core::cache::PromptCacheTopology::default();
        match &self.execution {
            Execution::Resident(runtime) => crate::composition::replicated_prompt_cache_identity(
                runtime.architecture(),
                topology,
            ),
            Execution::Bounded(runtime) => crate::composition::replicated_prompt_cache_identity(
                runtime.architecture(),
                topology,
            ),
            Execution::ConditionalResident(runtime) => {
                crate::composition::replicated_prompt_cache_identity(
                    runtime.architecture(),
                    topology,
                )
            }
            Execution::ConditionalBounded(runtime) => {
                crate::composition::replicated_prompt_cache_identity(
                    runtime.architecture(),
                    topology,
                )
            }
        }
    }

    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &eredu_core::cache::PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<eredu_core::cache::PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(
            &descriptor,
            &self.prompt_cache_model_identity()?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &eredu_core::cache::PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, eredu_core::cache::PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache = MlxHybridState::paged(self.state_layout.clone(), manager, None)?;
        cache.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Exception::custom("prompt-cache prefix exceeds i32"))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((cache, manifest))
    }
    /// Logical residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            Execution::Resident(runtime) => runtime.policy().residency_report(),
            Execution::Bounded(runtime) => runtime.policy().residency_report(),
            Execution::ConditionalResident(runtime) => runtime.policy().residency_report(),
            Execution::ConditionalBounded(runtime) => runtime.policy().residency_report(),
        }
    }
    /// Dense disk-stream telemetry when selected.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            Execution::Resident(_) | Execution::ConditionalResident(_) => Ok(None),
            Execution::Bounded(runtime) => runtime.policy().dense_stream_report(),
            Execution::ConditionalBounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }
    /// Executes the one neutral target group.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let config = self.parsed.text.clone();
            let result = {
                let mut provider = cached_provider(&parameter_bank, &config);
                self.forward_with_provider(tokens, cache, &mut provider, stream)
            };
            self.parameter_bank = Some(parameter_bank);
            return result;
        }
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            return match &mut self.execution {
                Execution::ConditionalResident(runtime) => runtime.forward(input, cache, stream),
                Execution::ConditionalBounded(runtime) => runtime.forward(input, cache, stream),
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()));
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward(input, cache, stream),
            Execution::Bounded(runtime) => runtime.forward(input, cache, stream),
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn forward_with_provider<P>(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            let hook = |architecture: &mut ConditionalArchitecture,
                        group: usize,
                        index: usize,
                        unit: &mut ConditionalUnit<MlxNeuralBackend>,
                        hidden: &crate::MlxTensor,
                        state: &mut MlxHybridState,
                        forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                        context: &Stream| {
                architecture.forward_unit_with_provider(
                    group, index, unit, hidden, state, forward, provider, context,
                )
            };
            return match &mut self.execution {
                Execution::ConditionalResident(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        cache,
                        stream,
                        hook,
                        |_, _, _| Ok(()),
                    )
                    .map(|(output, _)| output),
                Execution::ConditionalBounded(runtime) => runtime
                    .forward_with_unit_executor_and_context_hook(
                        input,
                        cache,
                        stream,
                        hook,
                        |_, _, _| Ok(()),
                    )
                    .map(|(output, _)| output),
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map(crate::MlxTensor::into_array)
            .map_err(|error| Error::Parallel(error.to_string()));
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Block,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            architecture.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            )
        };
        match &mut self.execution {
            Execution::Resident(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map(|(output, _)| output),
            Execution::Bounded(runtime) => runtime
                .forward_with_unit_executor_and_context_hook(
                    input,
                    cache,
                    stream,
                    hook,
                    |_, _, _| Ok(()),
                )
                .map(|(output, _)| output),
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map(crate::MlxTensor::into_array)
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let mut neutral_observer = crate::composition::NeutralActivationObserver::new(observer);
        if self.parsed.vision.is_some() {
            let parts = [InputPart::Text(crate::composition::tensor_ref(tokens))];
            let input = ConditionalInput::Target {
                parts: &parts,
                pixels: None,
                mask: None,
            };
            let output = match &mut self.execution {
                Execution::ConditionalResident(runtime) => {
                    runtime.forward_with_observer(input, cache, stream, &mut neutral_observer)
                }
                Execution::ConditionalBounded(runtime) => {
                    runtime.forward_with_observer(input, cache, stream, &mut neutral_observer)
                }
                _ => unreachable!("conditional policy uses conditional execution"),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            return eredu_runtime::observe_model_logits(&mut neutral_observer, &output)
                .map(crate::MlxTensor::into_array)
                .map_err(Error::from);
        }
        let input = EmbeddedInput::target(crate::composition::tensor_ref(tokens), None);
        let output = match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_observer(input, cache, stream, &mut neutral_observer)
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_observer(input, cache, stream, &mut neutral_observer)
            }
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => unreachable!(),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        eredu_runtime::observe_model_logits(&mut neutral_observer, &output)
            .map(crate::MlxTensor::into_array)
            .map_err(Error::from)
    }

    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        if self.parsed.vision.is_some() {
            return self
                .prepared_conditional_forward(input, cache, stream, Some(observer))
                .map(|output| output.logits.into_array())
                .map_err(Error::Exception);
        }
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_with_observer(&tokens, cache, stream, observer)
    }

    fn prepared_conditional_forward(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PreparedConditionalOutput, Exception> {
        let prepared = prepare_conditional_input(&self.parsed, input, stream)?;
        let tokens = prepared
            .token_ids(stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        prepared.with_target_input(|model_input| {
            if let Some(observer) = observer {
                let mut neutral_observer =
                    crate::composition::NeutralActivationObserver::new(observer);
                let logits = match &mut self.execution {
                    Execution::ConditionalResident(runtime) => runtime.forward_with_observer(
                        model_input,
                        cache,
                        stream,
                        &mut neutral_observer,
                    ),
                    Execution::ConditionalBounded(runtime) => runtime.forward_with_observer(
                        model_input,
                        cache,
                        stream,
                        &mut neutral_observer,
                    ),
                    _ => return Err(Exception::custom("Qwen3.5 model is not conditional")),
                }
                .map_err(|error| Exception::custom(error.to_string()))?;
                let logits = eredu_runtime::observe_model_logits(&mut neutral_observer, &logits)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                return Ok(PreparedConditionalOutput {
                    logits,
                    hidden: None,
                    tokens,
                });
            }
            let (logits, forward) = match &mut self.execution {
                Execution::ConditionalResident(runtime) => {
                    runtime.forward_with_context_hook(model_input, cache, stream, |_, _, _| Ok(()))
                }
                Execution::ConditionalBounded(runtime) => {
                    runtime.forward_with_context_hook(model_input, cache, stream, |_, _, _| Ok(()))
                }
                _ => return Err(Exception::custom("Qwen3.5 model is not conditional")),
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            Ok(PreparedConditionalOutput {
                logits,
                hidden: forward.target_hidden().cloned(),
                tokens,
            })
        })
    }

    fn forward_mtp(
        &mut self,
        input: EmbeddedInput<'_, crate::MlxTensor>,
        tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<eredu_architectures::speculative_execution::EmbeddedPredictionOutput<crate::MlxTensor>, Exception> {
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let config = self.parsed.text.clone();
            let result = {
                let mut provider = cached_provider(&parameter_bank, &config);
                self.forward_mtp_with_provider(input, tokens, cache, &mut provider, stream)
            };
            self.parameter_bank = Some(parameter_bank);
            return result;
        }
        if self.parsed.vision.is_some() {
            let result = match input {
                EmbeddedInput::Target { tokens, mask } => {
                    let parts = [InputPart::Text(tokens)];
                    let input = ConditionalInput::Target {
                        parts: &parts,
                        pixels: None,
                        mask,
                    };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
                EmbeddedInput::Draft {
                    tokens,
                    hidden,
                    depth,
                } => {
                    let input = ConditionalInput::Draft {
                        tokens,
                        hidden,
                        depth,
                    };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(())),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            let hidden =
                result.1.target_hidden().cloned().ok_or_else(|| {
                    Exception::custom("conditional Qwen3.5 retained no hidden state")
                })?;
            return Ok(
                eredu_architectures::speculative_execution::EmbeddedPredictionOutput::<crate::MlxTensor> {
                    logits: result.0,
                    capture: hidden,
                    tokens: tokens.clone(),
                },
            );
        }
        let result = match &mut self.execution {
            Execution::Resident(runtime) => {
                runtime.forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            }
            Execution::Bounded(runtime) => {
                runtime.forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            }
            Execution::ConditionalResident(_) | Execution::ConditionalBounded(_) => {
                unreachable!("text policy uses text execution")
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = result
            .1
            .target_hidden()
            .cloned()
            .ok_or_else(|| Exception::custom("Qwen hybrid pass retained no hidden state"))?;
        Ok(
            eredu_architectures::speculative_execution::EmbeddedPredictionOutput::<crate::MlxTensor> {
                logits: result.0,
                capture: hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn forward_mtp_with_provider<P>(
        &mut self,
        input: EmbeddedInput<'_, crate::MlxTensor>,
        tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<eredu_architectures::speculative_execution::EmbeddedPredictionOutput<crate::MlxTensor>, Exception>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        if self.parsed.vision.is_some() {
            let conditional = match input {
                EmbeddedInput::Target { tokens, mask } => {
                    let parts = [InputPart::Text(tokens)];
                    let input = ConditionalInput::Target {
                        parts: &parts,
                        pixels: None,
                        mask,
                    };
                    let hook =
                        |architecture: &mut ConditionalArchitecture,
                         group: usize,
                         index: usize,
                         unit: &mut ConditionalUnit<MlxNeuralBackend>,
                         hidden: &crate::MlxTensor,
                         state: &mut MlxHybridState,
                         forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                         context: &Stream| {
                            architecture.forward_unit_with_provider(
                                group, index, unit, hidden, state, forward, provider, context,
                            )
                        };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
                EmbeddedInput::Draft {
                    tokens,
                    hidden,
                    depth,
                } => {
                    let input = ConditionalInput::Draft {
                        tokens,
                        hidden,
                        depth,
                    };
                    let hook =
                        |architecture: &mut ConditionalArchitecture,
                         group: usize,
                         index: usize,
                         unit: &mut ConditionalUnit<MlxNeuralBackend>,
                         hidden: &crate::MlxTensor,
                         state: &mut MlxHybridState,
                         forward: &mut hybrid::ConditionalForwardContext<crate::MlxTensor>,
                         context: &Stream| {
                            architecture.forward_unit_with_provider(
                                group, index, unit, hidden, state, forward, provider, context,
                            )
                        };
                    match &mut self.execution {
                        Execution::ConditionalResident(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        Execution::ConditionalBounded(runtime) => runtime
                            .forward_with_unit_executor_and_context_hook(
                                input,
                                cache,
                                stream,
                                hook,
                                |_, _, _| Ok(()),
                            ),
                        _ => unreachable!("conditional policy uses conditional execution"),
                    }
                }
            }
            .map_err(|error| Exception::custom(error.to_string()))?;
            let hidden =
                conditional.1.target_hidden().cloned().ok_or_else(|| {
                    Exception::custom("conditional Qwen3.5 retained no hidden state")
                })?;
            return Ok(
                eredu_architectures::speculative_execution::EmbeddedPredictionOutput::<crate::MlxTensor> {
                    logits: conditional.0,
                    capture: hidden,
                    tokens: tokens.clone(),
                },
            );
        }
        let hook = |architecture: &mut Architecture,
                    group: usize,
                    index: usize,
                    unit: &mut Block,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxHybridState,
                    forward: &mut hybrid::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            architecture.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, provider, context,
            )
        };
        let result = match &mut self.execution {
            Execution::Resident(runtime) => runtime.forward_with_unit_executor_and_context_hook(
                input,
                cache,
                stream,
                hook,
                |_, _, _| Ok(()),
            ),
            Execution::Bounded(runtime) => runtime.forward_with_unit_executor_and_context_hook(
                input,
                cache,
                stream,
                hook,
                |_, _, _| Ok(()),
            ),
            _ => unreachable!("text policy uses text execution"),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = result
            .1
            .target_hidden()
            .cloned()
            .ok_or_else(|| Exception::custom("Qwen hybrid pass retained no hidden state"))?;
        Ok(
            eredu_architectures::speculative_execution::EmbeddedPredictionOutput::<crate::MlxTensor> {
                logits: result.0,
                capture: hidden,
                tokens: tokens.clone(),
            },
        )
    }
}

impl CausalModel<MlxHybridState> for QwenHybridModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        cache.clear()?;
        let output = if self.parsed.vision.is_some() {
            let prepared = self.prepared_conditional_forward(input, cache, stream, None)?;
            let hidden = prepared.hidden.ok_or_else(|| {
                Exception::custom("conditional Qwen3.5 prefill retained no target hidden state")
            })?;
            eredu_architectures::speculative_execution::EmbeddedPredictionOutput::new(
                prepared.logits,
                hidden,
                prepared.tokens,
            )
        } else {
            let tokens = crate::MlxTensor::from_array(input::text_token_ids(input, stream)?);
            self.forward_mtp(EmbeddedInput::target(&tokens, None), &tokens, cache, stream)?
        };
        let tokens = output.tokens.clone();
        let sequence = tokens.dim(1);
        if sequence > 1 {
            let hidden = crate::MlxTensor::from_array(
                output
                    .capture
                    .as_array()
                    .try_index_device((.., ..sequence - 1, ..), stream)?,
            );
            let next =
                crate::MlxTensor::from_array(tokens.as_array().try_index_device((.., 1..), stream)?);
            for depth in 0..self.mtp_len() {
                let _ = self.forward_mtp(
                    EmbeddedInput::draft(&next, &hidden, depth),
                    &next,
                    cache,
                    stream,
                )?;
            }
        }
        output
            .logits
            .as_array()
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let output = self
            .forward(input_tokens.as_array(), cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        output
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    source: &HybridConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        HybridConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target =
        hybrid::load_time_quantization(source, quantization).map_err(Error::ArchitectureModel)?;
    let source_architecture = Architecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = Architecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = source_architecture
        .unit_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_layout = target_architecture
        .unit_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if source_layout != target_layout {
        return Err(Error::ArchitectureModel(
            "Qwen hybrid load-time quantization changed execution topology".into(),
        ));
    }
    let total = source_layout.len();
    let source_static =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &source_architecture,
        )
        .clone();
    let target_static =
        <Architecture as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::static_modules(
            &target_architecture,
        )
        .clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |flat, stream| {
            construct_architecture_unit(
                &source_architecture,
                &source_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |flat, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        total,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

fn quantize_conditional_store(
    store: Arc<dyn CheckpointSource>,
    source: &ParsedHybridConfig,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        ParsedHybridConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = hybrid::conditional_load_time_quantization(source, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source_architecture = ConditionalArchitecture::new(source.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_architecture = ConditionalArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = source_architecture
        .unit_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target_layout = target_architecture
        .unit_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if source_layout != target_layout {
        return Err(Error::ArchitectureModel(
            "conditional Qwen load-time quantization changed execution topology".into(),
        ));
    }
    let total = source_layout.len();
    let source_static = <ConditionalArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&source_architecture)
    .clone();
    let target_static = <ConditionalArchitecture as LayeredArchitecture<
        MlxNeuralBackend,
        MlxHybridState,
    >>::static_modules(&target_architecture)
    .clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |flat, stream| {
            construct_architecture_unit(
                &source_architecture,
                &source_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |flat, stream| {
            construct_architecture_unit(
                &target_architecture,
                &target_layout,
                flat,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        total,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Loads SafeTensors through the generic component residency engine.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    route: &crate::composition::mlx::loading::ExcludedFamilyRoute,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    load_safetensors_with_residency(
        artifact,
        route,
        eredu_runtime::WeightResidency::with_layers(options.into()),
        quantization,
        stream,
        weights_stream,
    )
}

pub fn load_safetensors_with_residency(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    _route: &crate::composition::mlx::loading::ExcludedFamilyRoute,
    residency: eredu_runtime::WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let eredu_architectures::configuration::SafetensorsModelConfig::QwenHybrid(parsed) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen hybrid loader received a different prepared architecture".into(),
        ));
    };
    let mut parsed = parsed.clone();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen hybrid", parsed.text.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_options = residency.parameter_bank_cache();
    let options = residency.layers();
    let store = artifact.store();
    if parsed.vision.is_some() {
        let store = if let Some(quantization) = quantize_on_load {
            let (store, target, _) =
                quantize_conditional_store(store, &parsed, quantization, stream)?;
            parsed = target;
            store
        } else {
            store
        };
        let mut model = load_conditional_store(
            store,
            parsed,
            options,
            expert_options.is_some(),
            stream,
            weights_stream,
        )?;
        if let Some(expert_options) = expert_options {
            attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let store = if let Some(quantization) = quantize_on_load {
        let (store, target, _) = quantize_store(store, &parsed.text, quantization, stream)?;
        parsed.text = target;
        store
    } else {
        store
    };
    let mut model = load_store(
        store,
        parsed,
        options,
        expert_options.is_some(),
        stream,
        weights_stream,
    )?;
    if let Some(expert_options) = expert_options {
        attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

fn load_conditional_store(
    store: Arc<dyn CheckpointSource>,
    parsed: ParsedHybridConfig,
    options: LayerWeightResidency,
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let mut architecture = ConditionalArchitecture::new(parsed.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let factory = ConditionalUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding = parsed.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings_with_recipes(
                &MlxModule::new(modules.clone()),
                "",
                store,
                hybrid::static_recipes(store).map_err(Error::ArchitectureModel)?,
            )
            .map_err(Into::into)
        },
        move |ordinal, _address, _path, unit, store, _| {
            let recipes = hybrid::conditional_unit_recipes(store, &binding, ordinal)
                .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let execution = if options.is_fully_resident() {
        Execution::ConditionalResident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::ConditionalBounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(QwenHybridModel {
        parsed,
        state_layout,
        execution,
        parameter_bank: None,
    })
}

fn load_store(
    store: Arc<dyn CheckpointSource>,
    parsed: ParsedHybridConfig,
    options: LayerWeightResidency,
    external_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridModel, Error> {
    let mut architecture = Architecture::new(parsed.text.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let factory = UnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding_config = parsed.text.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            let recipes = hybrid::static_recipes(store).map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes(&MlxModule::new(modules.clone()), "", store, recipes)
                .map_err(Into::into)
        },
        move |ordinal, _address, _path, unit, store, _| {
            let recipes = hybrid::unit_recipes(store, &binding_config, ordinal)
                .map_err(Error::ArchitectureModel)?;
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let execution = if options.is_fully_resident() {
        Execution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        Execution::Bounded(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(QwenHybridModel {
        parsed,
        state_layout,
        execution,
        parameter_bank: None,
    })
}

fn attach_parameter_bank(
    model: &mut QwenHybridModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = expert_catalog_selected(model.args(), store.as_ref(), None, |_, _| true)?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}
