//! MLX loading and runtime binding for the backend-neutral GPT-OSS decoder.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::gpt_oss::ModelArgs;
use eredu_checkpoint::{store::CheckpointSource, WeightQuantization};
use eredu_nn::{ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized};
use eredu_runtime::{
    ArchitectureParameters, CacheResidencyPolicy, CausalModel, DenseDiskStreamReport,
    LayerWeightResidency, LayerwiseRuntime, PagedCacheOptions, ParameterRole, ResidencyReport,
    RuntimeState, WeightResidency,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        cache::{
            residency::{open_prompt_cache, CacheResidencyManager},
            state::MlxKeyValueState,
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
        residency::manager::ResidentUnitLease,
        residency::parameter_bank::{
            AddressableParameterBank, ParameterBankEntry, ParameterBankResidencyReport,
        },
    },
};
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

pub mod expert {
    include!("gpt_oss_expert.rs");
}

/// The architecture-erased cache representation used by GPT-OSS.
pub type Cache = MlxKeyValueState;

type NeutralBlock = eredu_architectures::gpt_oss::TransformerBlock<MlxNeuralBackend>;
type NeutralArchitecture = eredu_architectures::gpt_oss::LayeredModel<MlxNeuralBackend>;

fn expert_parameter_targets(
    architecture: &NeutralArchitecture,
    stream: &Stream,
) -> Result<BTreeSet<String>, Error> {
    let mut targets = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?
        .targets_for_role(ParameterRole::ExpertIntermediate);
    targets.extend(
        eredu_architectures::gpt_oss::safetensors_expert_tensors(architecture.args())
            .map_err(Error::ArchitectureModel)?
            .into_iter()
            .map(|tensor| tensor.key),
    );
    targets.extend(
        eredu_architectures::gpt_oss::gguf_expert_quantization_targets(architecture.args())
            .map_err(Error::ArchitectureModel)?,
    );
    Ok(targets)
}

type ResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type LayerwiseExecution = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, GptOssUnitPopulator>,
>;
#[derive(Clone)]
struct GptOssUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for GptOssUnitPopulator {
    fn populate(
        &mut self,
        unit: &mut MlxModule<NeutralBlock>,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        populate_module_from_lease_excluding(unit, lease, |name| {
            self.external_experts && parameter_name_in_targets(name, &self.expert_targets)
        })?;
        Ok(())
    }
}

enum GptOssExecution {
    Resident(Box<ResidentRuntime>),
    Layerwise(Box<LayerwiseExecution>),
}

impl GptOssExecution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Layerwise(runtime) => runtime.architecture(),
        }
    }
}

/// Parameter view used only to select ordinary dense matrices for load-time
/// quantization. Native expert matrices retain their exact MXFP4 recipes.
#[derive(Debug, Clone)]
struct DenseUnit {
    block: NeutralBlock,
    expert_targets: Arc<BTreeSet<String>>,
}

impl Parameterized<crate::MlxTensor> for DenseUnit {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V, &'v BTreeSet<String>);
        impl<'a, V: ParameterVisitor<'a, crate::MlxTensor>> ParameterVisitor<'a, crate::MlxTensor>
            for Filter<'_, V>
        {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a crate::MlxTensor) {
                if !parameter_name_in_targets(metadata.id.as_str(), self.1) {
                    self.0.visit(metadata, value);
                }
            }
        }
        self.block
            .visit_parameters(&mut Filter(visitor, &self.expert_targets));
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, crate::MlxTensor>,
    {
        struct Filter<'v, V>(&'v mut V, &'v BTreeSet<String>);
        impl<'a, V: ParameterVisitorMut<'a, crate::MlxTensor>>
            ParameterVisitorMut<'a, crate::MlxTensor> for Filter<'_, V>
        {
            fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut crate::MlxTensor) {
                if !parameter_name_in_targets(metadata.id.as_str(), self.1) {
                    self.0.visit_mut(metadata, value);
                }
            }
        }
        self.block
            .visit_parameters_mut(&mut Filter(visitor, &self.expert_targets));
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.block.set_trainable(trainable);
    }
}

fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    let recipes = eredu_architectures::gpt_oss::expert_recipes(store, args, layer)
        .map(|family| family.into_outputs().into_outputs())
        .map_err(Error::ArchitectureModel)?;
    recipes
        .into_iter()
        .map(|(name, mut recipe)| {
            if recipe.infer(store)?.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4 {
                recipe =
                    crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, store)?;
            }
            Ok((name, recipe))
        })
        .collect()
}

/// Builds one neutral GPT-OSS runtime from an already opened checkpoint store.
pub fn load_neutral_with_store(
    store: Arc<dyn CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<GptOssModel, Error> {
    let mut architecture =
        eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_targets = Arc::new(expert_parameter_targets(&architecture, stream)?);
    let factory = GptOssUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
    let binding_args = args.clone();
    let excluded_expert_targets = Arc::clone(&expert_targets);
    let binding_expert_targets = Arc::clone(&expert_targets);
    let (policy, _) = prepare_layerwise_policy_with_bindings(
        store,
        &mut architecture,
        factory,
        std::marker::PhantomData::<MlxKeyValueState>,
        options,
        stream,
        weights_stream,
        move |key| external_experts && parameter_name_in_targets(key, &excluded_expert_targets),
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |_ordinal, address, _path, unit, store, _stream| {
            let index = address.index();
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                unit_recipes(store, &binding_args, index)?
            };
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
        GptOssExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        GptOssExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(GptOssModel {
        args,
        state_layout,
        parallel_rank: None,
        planned_external_experts: None,
        prompt_cache_topology: PromptCacheTopology::default(),
        execution,
        parameter_bank: None,
    })
}

pub fn quantize_neutral_store(
    store: Arc<dyn CheckpointSource>,
    source_args: &ModelArgs,
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
    let target_args =
        eredu_architectures::gpt_oss::load_time_quantization(source_args, quantization)
            .map_err(Error::ArchitectureModel)?;
    let source = eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(
        source_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target = eredu_architectures::gpt_oss::new_layered_model::<MlxNeuralBackend>(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_expert_targets = Arc::new(expert_parameter_targets(&source, stream)?);
    let target_expert_targets = Arc::new(expert_parameter_targets(&target, stream)?);
    let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxKeyValueState>(&target)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "GPT-OSS quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = target.static_modules().clone();
    let source_unit_expert_targets = Arc::clone(&source_expert_targets);
    let target_unit_expert_targets = Arc::clone(&target_expert_targets);
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
                std::marker::PhantomData::<MlxKeyValueState>,
            )
            .map(|block| DenseUnit {
                block,
                expert_targets: Arc::clone(&source_unit_expert_targets),
            })
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &target,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )
            .map(|block| DenseUnit {
                block,
                expert_targets: Arc::clone(&target_unit_expert_targets),
            })
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Neutral GPT-OSS causal LM with resident or bounded layer execution.
pub struct GptOssModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    parallel_rank: Option<eredu_core::cache::CacheRankIdentity>,
    planned_external_experts: Option<Vec<ParameterBankEntry>>,
    prompt_cache_topology: PromptCacheTopology,
    execution: GptOssExecution,
    parameter_bank: Option<AddressableParameterBank>,
}

impl GptOssModel {
    /// Returns normalized model arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Builds expert-cache units with this rank's exact TP selections.
    pub fn external_expert_catalog(&self) -> Result<Vec<ParameterBankEntry>, Error> {
        self.planned_external_experts.clone().map_or_else(
            || expert::expert_catalog(&self.args, self.checkpoint_store(), None),
            Ok,
        )
    }

    /// Returns logical layer-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().residency_report(),
            GptOssExecution::Layerwise(execution) => execution.policy().residency_report(),
        }
    }

    /// Returns dense disk-stream telemetry when active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            GptOssExecution::Resident(_) => Ok(None),
            GptOssExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
        }
    }

    /// Returns independent expert-cache telemetry when configured.
    pub fn parameter_bank_report(&self) -> Result<Option<ParameterBankResidencyReport>, Error> {
        self.parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store used by either execution policy.
    pub fn checkpoint_store(&self) -> &dyn CheckpointSource {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
        }
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn CheckpointSource> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
        }
    }

    /// Creates empty device-resident state.
    pub fn new_cache(&self) -> Cache {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("MLX key/value state supports validated GPT-OSS geometry")
    }

    /// Creates device or explicitly bounded paged state.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                self.new_paged_cache(options, None, self.parallel_rank)
            }
        }
    }

    /// Lazily catalogs a compatible persisted prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        eredu_core::cache::validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (manager, manifest) = open_prompt_cache(
            directory.as_ref(),
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        let state =
            self.new_paged_cache_from_manager(manager, identity.topology().cache_rank_identity())?;
        let _ = stream;
        Ok((state, manifest))
    }

    /// Persists a completed prefix after validating model identity.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        eredu_core::cache::validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let _ = stream;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn new_paged_cache(
        &self,
        options: PagedCacheOptions,
        manager: Option<CacheResidencyManager>,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
    ) -> Result<Cache, Error> {
        let manager = match manager {
            Some(manager) => manager,
            None => CacheResidencyManager::new(options)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        self.new_paged_cache_from_manager(manager, rank)
    }

    fn new_paged_cache_from_manager(
        &self,
        manager: CacheResidencyManager,
        rank: Option<eredu_core::cache::CacheRankIdentity>,
    ) -> Result<Cache, Error> {
        MlxKeyValueState::paged(self.state_layout.clone(), manager, rank).map_err(Into::into)
    }

    /// Executes embedding, all neutral blocks, final normalization, and head.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(parameter_bank) = self.parameter_bank.take() {
            let args = self.args.clone();
            let result = {
                let mut provider = expert::cached_provider(&parameter_bank, &args);
                self.forward_with_grouped_provider(inputs, None, cache, &mut provider, stream)
            };
            self.parameter_bank = Some(parameter_bank);
            return result;
        }
        self.validate_cache(cache)?;
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: None,
        };
        let output = match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
        }?;
        Ok(output.into_array())
    }

    /// Runs the neutral decoder with runtime-owned expert residency.
    pub fn forward_with_grouped_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let hook =
            |architecture: &mut NeutralArchitecture,
             group: usize,
             index: usize,
             block: &mut NeutralBlock,
             hidden: &crate::MlxTensor,
             state: &mut Cache,
             forward: &mut eredu_architectures::gpt_oss::ForwardContext<crate::MlxTensor>,
             context: &Stream| {
                <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                    MlxNeuralBackend,
                    Cache,
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
        let input = eredu_architectures::decoder::LayeredInput {
            tokens: crate::composition::tensor_ref(inputs),
            mask: crate::composition::tensor_opt(mask),
        };
        let output = match &mut self.execution {
            GptOssExecution::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
        }?;
        Ok(output.into_array())
    }

    /// Runs with stable layer-input, layer-output, and logits observations.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        let parameter_bank = self.parameter_bank.take();
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let result = match parameter_bank.as_ref() {
            Some(parameter_bank) => {
                let args = self.args.clone();
                let mut provider = expert::cached_provider(parameter_bank, &args);
                self.forward_observed_with_provider(
                    inputs,
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
                    inputs,
                    mask,
                    cache,
                    &mut provider,
                    stream,
                    &mut observer,
                )
            }
        };
        self.parameter_bank = parameter_bank;
        result
    }

    fn forward_observed_with_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        provider: &mut P,
        stream: &Stream,
        observer: &mut crate::composition::NeutralActivationObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let output = match &mut self.execution {
            GptOssExecution::Resident(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
            GptOssExecution::Layerwise(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::decoder::LayeredInput {
                        tokens: crate::composition::tensor_ref(inputs),
                        mask: crate::composition::tensor_opt(mask),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        };
        eredu_runtime::observe_model_logits(observer, &output)
            .map(crate::MlxTensor::into_array)
            .map_err(Into::into)
    }

    /// Runs prompt prefill and returns final-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached decode and returns final-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            self.prompt_cache_topology.clone(),
        )
    }

    fn validate_cache(&self, cache: &Cache) -> Result<(), Error> {
        if cache.layout() != &self.state_layout {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match GPT-OSS layout {:?}",
                cache.layout(),
                self.state_layout
            ))
            .into());
        }
        Ok(())
    }
}

impl CausalModel<Cache> for GptOssModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &crate::MlxTensor,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.decode(input_tokens.as_array(), cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

fn attach_parameter_bank(
    model: &mut GptOssModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let store = model.checkpoint_store_arc();
    let entries = model.external_expert_catalog()?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

/// Loads SafeTensors GPT-OSS using the selected weight-residency policy.
pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let expert_options = weight_residency.parameter_bank_cache();
    let execution_options = weight_residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::GptOss(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "GPT-OSS loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("GPT-OSS", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    let (store, args) = match quantize_on_load {
        Some(quantization) => {
            let (store, args, _) = quantize_neutral_store(store, &args, quantization, stream)?;
            (store, args)
        }
        None => (store, args),
    };
    let mut model = load_neutral_with_store(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Loads SafeTensors or an inspected GGUF through the neutral GPT-OSS tensor-parallel graph.
/// Header-only results needed to open a portable GGUF GPT-OSS checkpoint.
pub(crate) struct PreparedGptOssGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_gpt_oss_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedGptOssGguf, Error> {
    if source.architecture() != eredu_architectures::GgufArchitecture::GptOss {
        return Err(Error::ArchitectureModel(format!(
            "GPT-OSS GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::GptOss(args) = source.model() else {
        return Err(Error::ArchitectureModel(
            "GPT-OSS GGUF loader received a different prepared model".into(),
        ));
    };
    let configs = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::gpt_oss::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedGptOssGguf { args })
}

/// Loads a GGUF checkpoint through the same neutral model/runtime object.
pub(crate) fn load_gpt_oss_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_gpt_oss_gguf_checkpoint(source)?;
    let store: Arc<dyn CheckpointSource> = Arc::new(open_gguf_checkpoint_source(
        checkpoint.clone(),
        source.plan().checkpoint(),
        source.plan().tensor_mapping(),
        residency.max_cached_shards(),
    )?);
    let expert_options = residency.parameter_bank_cache();
    let execution_options = residency.layers();
    let (store, args) = match quantization {
        Some(quantization) => {
            let (store, args, _) =
                quantize_neutral_store(store, &prepared.args, quantization, stream)?;
            (store, args)
        }
        None => (store, prepared.args),
    };
    let mut model = load_neutral_with_store(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}
