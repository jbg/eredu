//! Neutral Kimi Linear/Kimi Linear-MoE composition over MLX execution policies.

use std::{collections::BTreeSet, path::Path, sync::Arc};

use eredu_architectures::kimi_linear::{Block, LayeredModel, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, DenseDiskStreamReport,
    LayerWeightResidency, LayerwiseRuntime, PagedCacheOptions, ParameterRole, ResidencyReport,
    WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

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
        checkpoint::{
            binding::{
                build_module_bindings, build_module_bindings_with_recipes_excluding,
                parameter_name_in_targets, populate_module_from_lease_excluding,
            },
            load::gguf_quantization_configs,
            quantization::should_quantize_on_load,
            store::open_gguf_checkpoint_source,
        },
        execution::{
            generic::{
                architecture_execution_layout, construct_architecture_unit,
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitPopulator,
            },
            layerwise::quantize_parameterized_store,
        },
        media::input,
        residency::parameter_bank::ParameterBankEntry,
        residency::parameter_bank::{AddressableParameterBank, ParameterBankResidencyReport},
    },
};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

type NeutralBlock = Block<MlxNeuralBackend>;
type NeutralArchitecture = LayeredModel<MlxNeuralBackend>;
type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxResidentPolicy<NeutralBlock>,
>;
type BoundedRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxHybridState,
    MlxLayerwisePolicy<NeutralBlock, KimiLinearUnitPopulator>,
>;
#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "crate::MlxTensor")]
#[doc(hidden)]
#[cfg(test)]
pub struct KimiLinearCheckpointTemplate {
    pub static_modules: eredu_architectures::decoder::StaticModules<MlxNeuralBackend>,
    pub layers: Vec<NeutralBlock>,
}

#[cfg(test)]
impl KimiLinearCheckpointTemplate {
    /// Builds one neutral full-parameter template for checkpoint tooling.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let layout = architecture_execution_layout::<_, MlxHybridState>(&architecture)?;
        let layers = (0..layout.len())
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
            static_modules: architecture.static_modules().clone(),
            layers,
        })
    }
}

#[derive(Clone)]
struct KimiLinearUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for KimiLinearUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &crate::backend::runtime::residency::manager::ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum KimiLinearExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<BoundedRuntime>),
}

impl KimiLinearExecution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Layerwise(runtime) => runtime.architecture(),
        }
    }
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    include_experts: bool,
) -> Result<std::collections::BTreeMap<String, DerivedWeightRecipe>, Error> {
    eredu_architectures::kimi_linear::unit_recipes(store, args, layer, include_experts)
        .map_err(Error::ArchitectureModel)
}

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = eredu_architectures::kimi_linear::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &ModelArgs,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}

fn load_neutral(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<KimiLinearModel, Error> {
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(
        architecture
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?
            .targets_for_role(ParameterRole::ExpertIntermediate),
    );
    let binding_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        KimiLinearUnitPopulator {
            external_experts,
            expert_targets: Arc::clone(&expert_targets),
        },
        std::marker::PhantomData::<MlxHybridState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _| {
            let index = address.index();
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                unit_recipes(store, &binding_args, index, !external_experts)?,
                |name| external_experts && parameter_name_in_targets(name, &binding_expert_targets),
            )
            .map_err(Into::into)
        },
    )?;
    let state_layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let execution = if options.is_fully_resident() {
        KimiLinearExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )?,
            architecture,
        )))
    } else {
        KimiLinearExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(KimiLinearModel {
        args,
        state_layout,
        execution,
        parameter_bank: None,
        parallel_rank: None,
    })
}

fn quantize_store(
    store: Arc<dyn CheckpointSource>,
    args: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = eredu_architectures::kimi_linear::load_time_quantization(args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let destination = NeutralArchitecture::new(target.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxHybridState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxHybridState>(&destination)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Kimi Linear quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = destination.static_modules().clone();
    let (store, report) = quantize_parameterized_store(
        store,
        &source_static,
        &target_static,
        move |ordinal, stream| {
            construct_architecture_unit(
                &source,
                &source_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &destination,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxHybridState>,
            )
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target, report))
}

/// Kimi Linear causal model whose equations are owned by `eredu-architectures`.
pub struct KimiLinearModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    execution: KimiLinearExecution,
    parameter_bank: Option<AddressableParameterBank>,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
}

impl KimiLinearModel {
    pub(crate) fn requires_family_executable(&self) -> bool {
        self.args.has_sparse_moe_layers() || self.parameter_bank.is_some()
    }

    /// Returns validated family policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Creates device-resident heterogeneous state.
    pub fn new_cache(&self) -> MlxHybridState {
        MlxHybridState::device(self.state_layout.clone())
            .expect("validated Kimi Linear state must be realizable by MLX")
    }

    /// Creates state with the requested attention residency.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxHybridState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)
                    .map_err(Into::into)
            }
        }
    }

    /// Returns weight residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().residency_report(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().residency_report(),
        }
    }

    /// Returns disk streaming telemetry when enabled.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            KimiLinearExecution::Resident(_) => Ok(None),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    /// Returns independent expert-cache telemetry when enabled.
    pub fn parameter_bank_report(&self) -> Result<Option<ParameterBankResidencyReport>, Error> {
        self.parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()
            .map_err(Into::into)
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            KimiLinearExecution::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    /// Returns the canonical prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            PromptCacheTopology::default(),
        )
    }

    /// Persists paged attention and all fixed components atomically.
    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.save_prompt_cache_with_identity(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            &self.prompt_cache_model_identity()?,
        )
    }

    pub fn save_prompt_cache_with_identity(
        &self,
        cache: &mut MlxHybridState,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        identity: &PromptCacheModelIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        eredu_core::cache::validate_prompt_cache_model_identity(&descriptor, identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    /// Opens paged attention and restores all fixed components.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        self.load_prompt_cache_with_identity(
            directory,
            expected,
            prefix_token_ids,
            options,
            &identity,
            stream,
        )
    }

    pub fn load_prompt_cache_with_identity(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(MlxHybridState, PromptCacheManifest), Error> {
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let tensors = load_prompt_cache_state_tensors(directory, &manifest, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut cache =
            MlxHybridState::paged(self.state_layout.clone(), manager, self.parallel_rank)?;
        cache.restore_prompt_cache_state(
            tensors,
            i32::try_from(prefix_token_ids.len())
                .map_err(|_| Exception::custom("prompt-cache prefix exceeds i32"))?,
            identity.layer_prefix_offsets(),
        )?;
        Ok((cache, manifest))
    }

    /// Executes embedding, the scheduled physical blocks, normalization, and logits.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = cached_provider(&parameter_bank, &args);
                self.forward_with_provider(tokens, None, cache, &mut provider, stream)
            };
            self.parameter_bank = Some(parameter_bank);
            return result;
        }
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: None,
        };
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => runtime.forward(input, cache, stream),
            KimiLinearExecution::Layerwise(runtime) => runtime.forward(input, cache, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    fn forward_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(tokens),
            mask: crate::composition::tensor_opt(mask),
        };
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut MlxHybridState,
             forward: &mut eredu_architectures::kimi_linear::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit_with_inferred_provider(
                    architecture,
                    group,
                    index,
                    block,
                    hidden,
                    state,
                    forward,
                    provider,
                    context,
                )
            };
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
            KimiLinearExecution::Layerwise(runtime) => {
                runtime.forward_with_unit_executor(input, cache, stream, hook)
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(output.into_array())
    }

    /// Executes a replicated neutral pass with activation intervention and
    /// normalized routing observations.
    pub fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let parameter_bank = self.parameter_bank.take();
        let result = {
            let mut observer = crate::composition::NeutralActivationObserver::new(observer);
            match parameter_bank.as_ref() {
                Some(parameter_bank) => {
                    let args = self.args.clone();
                    let mut provider = cached_provider(parameter_bank, &args);
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
                None => {
                    let mut provider = eredu_runtime::ResidentExpertProvider;
                    self.forward_observed_with_provider(
                        tokens,
                        mask,
                        cache,
                        &mut provider,
                        stream,
                        &mut observer,
                    )
                }
            }
        };
        self.parameter_bank = parameter_bank;
        result
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut MlxHybridState,
        provider: &mut P,
        stream: &Stream,
        observer: &mut crate::composition::NeutralActivationObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let output = match &mut self.execution {
            KimiLinearExecution::Resident(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(tokens),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                ),
            KimiLinearExecution::Layerwise(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(tokens),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                ),
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::observe_model_logits(observer, &output)
            .map(crate::MlxTensor::into_array)
            .map_err(Into::into)
    }
}

impl CausalModel<MlxHybridState> for KimiLinearModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut MlxHybridState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.forward(input_tokens.as_array(), cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
            .map(crate::MlxTensor::from_array)
    }
}

/// Loads SafeTensors Kimi Linear through one neutral model object.
pub fn load_kimi_linear_model(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    _route: &crate::composition::mlx::loading::ExcludedFamilyRoute,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let expert_options = residency.parameter_bank_cache();
    let options = residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::KimiLinear(args) =
        artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Kimi Linear loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize = quantization
        .map(|requested| {
            should_quantize_on_load("Kimi Linear", args.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    if let Some(quantization) = quantize {
        let (store, target, _) = quantize_store(store, &args, quantization, stream)?;
        let mut model = load_neutral(
            store,
            target,
            options,
            stream,
            weights_stream,
            expert_options.is_some(),
        )?;
        if let Some(expert_options) = expert_options {
            attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral(
        store,
        args,
        options,
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_parameter_bank(
    model: &mut KimiLinearModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = expert_catalog(&model.args, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors Kimi Linear through generalized tensor-parallel placement.
pub(crate) struct PreparedGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_gguf(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedGguf, Error> {
    if source.architecture() != eredu_architectures::GgufArchitecture::KimiLinear {
        return Err(Error::ArchitectureModel(format!(
            "Kimi Linear GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::KimiLinear(args) = source.model()
    else {
        return Err(Error::ArchitectureModel(
            "Kimi Linear GGUF loader received a different prepared model".into(),
        ));
    };
    let configs = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::kimi_linear::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedGguf { args })
}

/// Loads a GGUF checkpoint through the same neutral Kimi Linear model object.
pub(crate) fn load_kimi_linear_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    _route: &crate::composition::mlx::loading::ExcludedFamilyRoute,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gguf(source)?;
    let expert_options = residency.parameter_bank_cache();
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        residency.max_cached_shards(),
    )?);
    let (store, args) = match quantization {
        Some(quantization) => {
            let (store, args, _) = quantize_store(store, &prepared.args, quantization, stream)?;
            (store, args)
        }
        None => (store, prepared.args),
    };
    let mut model = load_neutral(
        store,
        args,
        residency.layers(),
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(expert_options) = expert_options {
        attach_parameter_bank(&mut model, expert_options, stream, weights_stream)?;
    }
    Ok(model)
}
