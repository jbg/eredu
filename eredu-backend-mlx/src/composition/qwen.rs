//! Unified Qwen loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{
    ArchitectureParameters, CausalModel, LayerWeightResidency, LayerwiseRuntime, RuntimeState,
    WeightResidency,
};

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use eredu_architectures::qwen::ModelArgs;
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::backend::runtime::checkpoint::load::gguf_quantization_configs;
use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::runtime::cache::state::MlxKeyValueState,
    backend::runtime::checkpoint::binding::{
        build_module_bindings, build_module_bindings_with_recipes_excluding,
        parameter_name_in_targets, populate_module_from_lease_excluding,
    },
    backend::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::runtime::execution::generic::{
        architecture_execution_layout, construct_architecture_unit,
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitPopulator,
    },
    backend::runtime::execution::layerwise::quantize_parameterized_store,
    backend::runtime::media::input,
    backend::runtime::residency::manager::ResidentUnitLease,
    backend::runtime::residency::parameter_bank::{
        AddressableParameterBank, ParameterBankResidencyReport,
    },
};

pub mod expert {
    include!("qwen_expert.rs");
}

pub mod hybrid {
    include!("qwen_hybrid.rs");
}

#[allow(
    dead_code,
    reason = "Qwen-VL keeps complete-model helpers beside the selected distributed checkpoint/store binder until those shared helpers are split"
)]
pub mod vl {
    include!("qwen_vl.rs");
}
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamReport, PagedCacheOptions, ParameterRole,
};

use eredu_runtime::ResidencyReport;

type NeutralBlock = eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>;

type NeutralArchitecture = eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>;

type NeutralResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxNeuralBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, QwenUnitPopulator>,
>;

#[derive(Clone)]
struct QwenUnitPopulator {
    external_experts: bool,
    expert_targets: Arc<BTreeSet<String>>,
}

impl MlxUnitPopulator<NeutralBlock> for QwenUnitPopulator {
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

enum QwenExecution {
    Resident(Box<NeutralResidentRuntime>),
    Layerwise(Box<NeutralLayerwiseRuntime>),
}

impl QwenExecution {
    fn architecture(&self) -> &NeutralArchitecture {
        match self {
            Self::Resident(runtime) => runtime.architecture(),
            Self::Layerwise(runtime) => runtime.architecture(),
        }
    }
}

fn qwen_unit_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    let resolved = eredu_architectures::qwen::expert_recipes(store, args, layer)
        .map_err(Error::ArchitectureModel)?;
    Ok(BTreeMap::from([
        (resolved.target_gate_up, resolved.gate_up),
        (resolved.target_down, resolved.down),
    ]))
}

fn load_neutral_qwen(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<QwenModel, Error> {
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
    let factory = QwenUnitPopulator {
        external_experts,
        expert_targets: Arc::clone(&expert_targets),
    };
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
            let recipes = if external_experts || !binding_args.is_moe() {
                BTreeMap::new()
            } else {
                qwen_unit_recipes(store, &binding_args, index)?
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
        QwenExecution::Resident(Box::new(LayerwiseRuntime::new_policy_first(
            policy.into_resident(
                &architecture,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )?,
            architecture,
        )))
    } else {
        QwenExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(QwenModel {
        state_layout,
        args,
        execution,
        parameter_bank: None,
    })
}

pub fn quantize_neutral_qwen_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &ModelArgs,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        ModelArgs,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target_args = eredu_architectures::qwen::load_time_quantization(source_args, quantization)
        .map_err(Error::ArchitectureModel)?;
    let source = NeutralArchitecture::new(source_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let target = NeutralArchitecture::new(target_args.clone(), stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_layout = architecture_execution_layout::<_, MlxKeyValueState>(&source)?;
    let target_layout = architecture_execution_layout::<_, MlxKeyValueState>(&target)?;
    if source_layout != target_layout {
        return Err(Error::Quantization(
            "Qwen quantization changed the architecture execution layout".into(),
        ));
    }
    let unit_count = source_layout.len();
    let source_static = source.static_modules().clone();
    let target_static = target.static_modules().clone();
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
        },
        move |ordinal, stream| {
            construct_architecture_unit(
                &target,
                &target_layout,
                ordinal,
                stream,
                std::marker::PhantomData::<MlxKeyValueState>,
            )
        },
        unit_count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Qwen causal LM whose execution engine follows its residency policy.
pub struct QwenModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    execution: QwenExecution,
    parameter_bank: Option<AddressableParameterBank>,
}

impl QwenModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().residency_report()?,
            QwenExecution::Layerwise(execution) => execution.policy().residency_report()?,
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            QwenExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            QwenExecution::Resident(_) => Ok(None),
        }
    }

    /// Returns independent expert residency telemetry when configured.
    pub fn parameter_bank_report(&self) -> Result<Option<ParameterBankResidencyReport>, Error> {
        self.parameter_bank
            .as_ref()
            .map(AddressableParameterBank::report)
            .transpose()
            .map_err(Error::from)
    }

    pub fn checkpoint_store_arc(&self) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            QwenExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
        }
    }

    /// Creates the cache representation required by the model configuration.
    pub fn new_cache(&self) -> MlxKeyValueState {
        MlxKeyValueState::device(self.state_layout.clone())
            .expect("MLX key/value state supports the validated Qwen layout")
    }

    /// Creates a device-resident or explicitly bounded paged model cache.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<MlxKeyValueState, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => self.new_paged_cache(options, None, None),
        }
    }

    /// Catalogs a compatible reusable prefix without loading all cache blocks.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MlxKeyValueState, PromptCacheManifest), Error> {
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

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut MlxKeyValueState,
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
    ) -> Result<MlxKeyValueState, Error> {
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
    ) -> Result<MlxKeyValueState, Error> {
        MlxKeyValueState::paged(self.state_layout.clone(), manager, rank).map_err(Into::into)
    }

    /// Runs embedding, decoder layers, final normalization, and projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
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
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: &inputs,
            mask: None,
        };
        let output = match &mut self.execution {
            QwenExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
            QwenExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string())),
        }?;
        Ok(output.into_array())
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        self.validate_cache(cache)?;
        let args = self.args.clone();
        let parameter_bank = self.parameter_bank.take();
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let result = match parameter_bank.as_ref() {
            Some(parameter_bank) => {
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
        let output = result?;
        eredu_runtime::observe_model_logits(observer.inner, &output).map_err(Error::from)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_observed_with_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        provider: &mut P,
        stream: &Stream,
        observer: &mut crate::composition::NeutralActivationObserver<'_>,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        let output = match &mut self.execution {
            QwenExecution::Resident(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: &inputs,
                        mask: mask.as_ref(),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                ),
            QwenExecution::Layerwise(runtime) => runtime
                .forward_with_inferred_provider_and_observer(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: &inputs,
                        mask: mask.as_ref(),
                    },
                    cache,
                    provider,
                    stream,
                    observer,
                ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(output.into_array())
    }

    fn forward_with_grouped_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let hook = |architecture: &mut NeutralArchitecture,
                    group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &crate::MlxTensor,
                    state: &mut MlxKeyValueState,
                    forward: &mut eredu_architectures::qwen::ForwardContext<crate::MlxTensor>,
                    context: &Stream| {
            <NeutralArchitecture as eredu_runtime::RoutedLayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
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
        let inputs = crate::MlxTensor::from_array(inputs.clone());
        let mask = mask.cloned().map(crate::MlxTensor::from_array);
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: &inputs,
            mask: mask.as_ref(),
        };
        let output = match &mut self.execution {
            QwenExecution::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
        }?;
        Ok(output.into_array())
    }

    /// Runs prompt prefill and returns last-token logits.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.forward(inputs, cache, stream)?
            .try_index_device((.., -1, ..), stream)
            .map_err(Into::into)
    }

    /// Runs cached decode and returns last-token logits.
    pub fn decode(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.prefill(input_tokens, cache, stream)
    }

    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let topology = PromptCacheTopology::default();
        crate::composition::replicated_prompt_cache_identity(
            self.execution.architecture(),
            topology,
        )
    }

    fn validate_cache(&self, cache: &MlxKeyValueState) -> Result<(), Error> {
        let expected = &self.state_layout;
        if cache.layout() != expected {
            return Err(Exception::custom(format!(
                "MLX key/value state layout {:?} does not match Qwen layout {expected:?}",
                cache.layout()
            ))
            .into());
        }
        Ok(())
    }
}

impl CausalModel<MlxKeyValueState> for QwenModel {
    type Tensor = crate::MlxTensor;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxKeyValueState,
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
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<crate::MlxTensor, Exception> {
        self.decode(input_tokens.as_array(), cache, stream)
            .map(crate::MlxTensor::from_array)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

pub fn load_safetensors(
    artifact: &crate::composition::mlx::artifact::PreparedSafetensorsArtifact,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let expert_options = weight_residency.parameter_bank_cache();
    let execution_options = weight_residency.layers();
    let eredu_architectures::configuration::SafetensorsModelConfig::Qwen(args) = artifact.model()
    else {
        return Err(Error::ArchitectureModel(
            "Qwen loader received a different prepared architecture".into(),
        ));
    };
    let args = args.clone();
    if !args.is_moe() {
        return Err(Error::ArchitectureModel(
            "ordinary replicated Qwen must use replicated text composition".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = artifact.store();
    if let Some(quantization) = quantize_on_load {
        let (store, args, _) = quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        let mut model = load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            expert_options.is_some(),
        )?;
        if let Some(options) = expert_options {
            attach_qwen_parameter_bank(&mut model, options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral_qwen(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_qwen_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_qwen_parameter_bank(
    model: &mut QwenModel,
    options: eredu_runtime::ParameterBankLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    let topology = eredu_core::ParallelTopology::new(1, 1, 1, 1)
        .and_then(|topology| eredu_core::ParallelRankTopology::new(topology, 0))
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let expert_realization = eredu_architectures::qwen::expert_realization_plan(
        model.execution.architecture(),
        topology,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if expert_realization.is_none() {
        return Err(Error::ArchitectureModel(
            "independent expert caching requires an architecture expert realization".into(),
        ));
    }
    let store = model.checkpoint_store_arc();
    let entries = expert::expert_catalog(&model.args, store.as_ref())?;
    model.parameter_bank = Some(AddressableParameterBank::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

pub(crate) struct PreparedQwenGguf {
    pub args: ModelArgs,
}

pub(crate) fn prepare_qwen_gguf_checkpoint(
    source: &crate::composition::mlx::structural::AdmittedGguf,
) -> Result<PreparedQwenGguf, Error> {
    if !matches!(
        source.architecture(),
        eredu_architectures::GgufArchitecture::Qwen2
            | eredu_architectures::GgufArchitecture::Qwen3
            | eredu_architectures::GgufArchitecture::Qwen3Moe
    ) {
        return Err(Error::ArchitectureModel(format!(
            "Qwen GGUF loader received architecture {:?}",
            source.architecture()
        )));
    }
    let checkpoint = source.checkpoint();
    let eredu_architectures::configuration::GgufModelConfig::Qwen(args) = source.model() else {
        return Err(Error::ArchitectureModel(
            "Qwen GGUF loader received a different prepared model".into(),
        ));
    };
    let configs = gguf_quantization_configs(checkpoint, source.plan().tensor_mapping())?;
    let args = eredu_architectures::qwen::with_checkpoint_formats(args, configs)
        .map_err(Error::ArchitectureModel)?;
    Ok(PreparedQwenGguf { args })
}

/// Loads a Qwen GGUF checkpoint using the selected residency policy.
pub(crate) fn load_qwen_gguf_model(
    source: &crate::composition::mlx::structural::AdmittedGguf,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let checkpoint = source.checkpoint();
    let prepared = prepare_qwen_gguf_checkpoint(source)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            source.plan().checkpoint(),
            source.plan().tensor_mapping(),
            residency.max_cached_shards(),
        )?);
    let args = prepared.args;
    if !args.is_moe() {
        return Err(Error::ArchitectureModel(
            "ordinary replicated Qwen must use replicated text composition".into(),
        ));
    }
    let expert_options = residency.parameter_bank_cache();
    let execution_options = residency.layers();
    let model = if let Some(quantization) = quantization {
        let (store, args, _) = quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            expert_options.is_some(),
        )?
    } else {
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            expert_options.is_some(),
        )?
    };
    let mut model = model;
    if let Some(options) = expert_options {
        attach_qwen_parameter_bank(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

/// Structured failures at the unified Qwen model boundary.
#[derive(Debug, thiserror::Error)]
pub enum QwenModelError {
    /// The normalized decoder count cannot be represented by this runtime.
    #[error("invalid Qwen decoder layer count {count}")]
    InvalidLayerCount {
        /// Invalid configured count.
        count: i32,
    },
}

impl From<QwenModelError> for crate::backend::error::Error {
    fn from(error: QwenModelError) -> Self {
        Self::ArchitectureModel(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::store::MemoryWeightStore;
    use safemlx::{Device, DeviceType};

    use super::*;

    fn fixture_args() -> ModelArgs {
        eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }))
        .unwrap()
    }

    fn f32_tensor(
        name: &str,
        shape: Vec<usize>,
    ) -> (String, safetensors::Dtype, Vec<usize>, Vec<u8>) {
        let elements = shape.iter().product::<usize>();
        (
            name.into(),
            safetensors::Dtype::F32,
            shape,
            vec![0; elements * size_of::<f32>()],
        )
    }

    #[test]
    fn qwen_static_bindings_use_architecture_parameter_identities() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let architecture = NeutralArchitecture::new(fixture_args(), &stream).unwrap();
        let store = MemoryWeightStore::from_safetensors([
            f32_tensor("model.embed_tokens.weight", vec![32, 8]),
            f32_tensor("model.norm.weight", vec![8]),
            f32_tensor("lm_head.weight", vec![32, 8]),
        ])
        .unwrap();

        let units = crate::composition::architecture_static_units(&architecture, &store).unwrap();
        let actual = units
            .iter()
            .map(|unit| {
                (
                    unit.id().as_str(),
                    unit.bindings()[0].name(),
                    unit.bindings()[0].logical_target(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [
                (
                    "embedding",
                    "model.embed_tokens.weight",
                    Some("model.embed_tokens.weight")
                ),
                ("norm", "model.norm.weight", Some("model.norm.weight")),
                ("output", "lm_head.weight", Some("lm_head.weight")),
            ]
        );

        let input_stage = MemoryWeightStore::from_safetensors([f32_tensor(
            "model.embed_tokens.weight",
            vec![32, 8],
        )])
        .unwrap();
        let selected = crate::composition::architecture_static_units_for_roles(
            &architecture,
            &input_stage,
            &["embedding"],
        )
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id().as_str(), "embedding");
    }

    #[test]
    fn replicated_identity_uses_neutral_architecture_state_contract() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let architecture = NeutralArchitecture::new(fixture_args(), &stream).unwrap();
        let layout = architecture.state_layout().unwrap();

        let identity = crate::composition::replicated_prompt_cache_identity(
            &architecture,
            PromptCacheTopology::default(),
        )
        .unwrap();

        assert_eq!(identity.global_layer_start(), 0);
        assert_eq!(identity.layer_count(), layout.len());
        assert_eq!(identity.layer_layout(), layout.layers());
        assert_eq!(
            identity.architecture_fingerprint(),
            eredu_architectures::qwen::prompt_cache_architecture_fingerprint(architecture.args())
        );
    }
}
