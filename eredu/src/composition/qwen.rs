//! Unified Qwen/Mistral loading across weight-residency policies.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::ActivationObserver as RuntimeActivationObserver;
use eredu_runtime::{
    CausalModel, ExecutionGraph, ExecutionResidency, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, LayeredForwardState, LayerwiseRuntime, ParallelLayeredArchitecture,
    RuntimeState, WeightResidency,
};

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use eredu_architectures::qwen::ModelArgs;
use eredu_nn::{
    NormalizationOperator, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized,
    RoutedNeuralBackend, SwiGluExpertBankSpec, SwiGluExpertLayout, SwiGluExpertProjection,
};
use safemlx::{
    error::Exception,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::backend::mlx::runtime::checkpoint::load::gguf_quantization_configs;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::parallel::{VocabParallelEmbedding, VocabParallelLmHead},
    backend::mlx::nn::shared::{MlxBackend, MlxModule, MlxNamedModule},
    backend::mlx::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::mlx::runtime::cache::state::MlxKeyValueState,
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes,
        build_module_binding_plan_with_recipes_excluding, build_module_bindings,
        build_module_bindings_with_recipes_excluding,
    },
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, store::open_gguf_checkpoint_source,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_parameterized_store, shard_layer_bindings,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{ExpertCache, ExpertCacheReport},
};
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamReport, LayerwiseModelMetadata, PagedCacheOptions,
    ParallelModelInfo, StaticUnitBindings,
};

use eredu_runtime::{ResidencyReport, WeightBinding};

type NeutralBlock = eredu_architectures::qwen::TransformerBlock<MlxBackend>;

#[derive(Clone)]
struct QwenUnitFactory {
    args: ModelArgs,
}

impl MlxUnitFactory<NeutralBlock> for QwenUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        eredu_architectures::qwen::new_block::<MlxBackend>(&self.args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

#[derive(Clone)]
struct QwenParallelUnitFactory {
    local_args: Arc<Vec<ModelArgs>>,
}

impl MlxUnitFactory<NeutralBlock> for QwenParallelUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<NeutralBlock, Error> {
        let args = self.local_args.get(index).ok_or_else(|| {
            Error::Parallel(format!(
                "parallel Qwen unit {index} is outside {} local layouts",
                self.local_args.len()
            ))
        })?;
        eredu_architectures::qwen::new_block::<MlxBackend>(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }
}

type NeutralArchitecture = eredu_architectures::qwen::LayeredModel<MlxBackend>;
type NeutralResidentRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralLayerwiseRuntime = LayerwiseRuntime<
    NeutralArchitecture,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, QwenUnitFactory>,
>;

type NeutralParallelResidentRuntime = LayerwiseRuntime<
    QwenParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxResidentPolicy<NeutralBlock>,
>;
type NeutralParallelLayerwiseRuntime = LayerwiseRuntime<
    QwenParallelComposition,
    MlxBackend,
    MlxKeyValueState,
    MlxLayerwisePolicy<NeutralBlock, QwenParallelUnitFactory>,
>;

enum QwenExecution {
    Resident(Box<NeutralResidentRuntime>),
    Layerwise(Box<NeutralLayerwiseRuntime>),
    TensorParallelResident(Box<NeutralParallelResidentRuntime>),
    TensorParallelLayerwise(Box<NeutralParallelLayerwiseRuntime>),
}

struct NeutralQwenObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
}

impl eredu_runtime::ActivationObserver<Array, eredu_nn::Error> for NeutralQwenObserver<'_> {
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(&mut self, path: &str, value: &Array) -> Result<Option<Array>, eredu_nn::Error> {
        self.inner
            .intervene(path, value)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, Array>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_routing(routing)
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_external_experts<P>(
    architecture: &mut NeutralArchitecture,
    index: usize,
    block: &mut NeutralBlock,
    hidden: &Array,
    state: &mut MlxKeyValueState,
    forward: &mut eredu_architectures::qwen::ForwardContext<Array>,
    pass: eredu_runtime::ExpertPass,
    parallel: Option<&safemlx::distributed::Group>,
    stream: &Stream,
    provider: &mut P,
) -> Result<Array, eredu_nn::Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
    P::Error: std::fmt::Display,
{
    let feed_forward = |policy: &mut eredu_architectures::qwen::FeedForward<MlxBackend>,
                        normalized: &Array,
                        context: &Stream| {
        let shape = normalized.shape().to_vec();
        let flat = normalized
            .reshape(&[-1, normalized.dim(-1)], context)
            .map_err(eredu_nn::Error::backend)?;
        policy
            .forward_with_provider(index, pass, &flat, context, provider)?
            .reshape(&shape, context)
            .map_err(eredu_nn::Error::backend)
    };
    match parallel {
        Some(group) => architecture.forward_block_parallel_with_feed_forward(
            index,
            block,
            hidden,
            state,
            forward,
            group,
            stream,
            feed_forward,
        ),
        None => architecture.forward_block_with_feed_forward(
            index,
            block,
            hidden,
            state,
            forward,
            stream,
            feed_forward,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_observed_unit(
    architecture: &mut NeutralArchitecture,
    args: &ModelArgs,
    group: usize,
    index: usize,
    block: &mut NeutralBlock,
    hidden: &Array,
    state: &mut MlxKeyValueState,
    forward: &mut eredu_architectures::qwen::ForwardContext<Array>,
    stream: &Stream,
    observer: &mut NeutralQwenObserver<'_>,
) -> Result<Array, eredu_nn::Error> {
    if group != 0 {
        return Err(eredu_nn::Error::backend(format!(
            "Qwen decoder received execution group {group}"
        )));
    }
    let path = format!("{}.layers.{index}", args.parameter_root);
    observer.observe(&format!("{path}.input"), hidden)?;
    let mut output = architecture.forward_block_with_feed_forward(
        index,
        block,
        hidden,
        state,
        forward,
        stream,
        |policy, normalized, context| {
            policy.forward_observed(
                &format!("{path}.mlp"),
                args.num_experts,
                normalized,
                context,
                observer,
            )
        },
    )?;
    output = eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &output)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_cached_observed_unit(
    architecture: &mut NeutralArchitecture,
    args: &ModelArgs,
    expert_cache: &ExpertCache,
    pass: eredu_runtime::ExpertPass,
    group: usize,
    index: usize,
    block: &mut NeutralBlock,
    hidden: &Array,
    state: &mut MlxKeyValueState,
    forward: &mut eredu_architectures::qwen::ForwardContext<Array>,
    stream: &Stream,
    observer: &mut NeutralQwenObserver<'_>,
) -> Result<Array, eredu_nn::Error> {
    if group != 0 {
        return Err(eredu_nn::Error::backend(format!(
            "Qwen decoder received execution group {group}"
        )));
    }
    let path = format!("{}.layers.{index}", args.parameter_root);
    let mut provider = crate::composition::qwen_expert::cached_provider(expert_cache, args);
    observer.observe(&format!("{path}.input"), hidden)?;
    let mut output = architecture.forward_block_with_feed_forward(
        index,
        block,
        hidden,
        state,
        forward,
        stream,
        |policy, normalized, context| {
            let shape = normalized.shape().to_vec();
            let flat = normalized
                .reshape(&[-1, normalized.dim(-1)], context)
                .map_err(eredu_nn::Error::backend)?;
            policy
                .forward_observed_with_provider(
                    &format!("{path}.mlp"),
                    index,
                    pass,
                    args.num_experts,
                    &flat,
                    context,
                    observer,
                    &mut provider,
                )?
                .reshape(&shape, context)
                .map_err(eredu_nn::Error::backend)
        },
    )?;
    output = eredu_runtime::observe_and_intervene(observer, &format!("{path}.output"), &output)?;
    Ok(output)
}

fn decoder_unit_layout(layer_count: usize) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["decoder"])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    ExecutionUnitLayout::new(&graph, [layer_count])
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn load_model_args(model_dir: &Path) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    eredu_architectures::qwen::model_args_from_config_reader(file)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn resolve_qwen_safetensors_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = eredu_architectures::qwen::safetensors_plan(args)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "{} checkpoint contract did not resolve: {validation:?}",
                args.model_type
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn qwen_unit_recipes(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>, Error> {
    if !args.is_moe() {
        return Ok(BTreeMap::new());
    }
    let resolved =
        eredu_architectures::qwen::expert_recipes(store, args, &args.parameter_root, layer)
            .map_err(Error::UnsupportedArchitecture)?;
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
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    external_experts: bool,
) -> Result<QwenModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen layer count".into()))?;
    let mut architecture = NeutralArchitecture::new(args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let factory = QwenUnitFactory { args: args.clone() };
    let unit_layout = decoder_unit_layout(layer_count)?;
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        store,
        architecture.static_modules_mut(),
        factory,
        unit_layout,
        options,
        stream,
        weights_stream,
        move |key| {
            key.starts_with("rope_freqs.")
                || key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        |modules, store| {
            build_module_bindings(&MlxModule::new(modules.clone()), "", store).map_err(Into::into)
        },
        move |index, unit, store, _stream| {
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                qwen_unit_recipes(store, &binding_args, index)?
            };
            build_module_bindings_with_recipes_excluding(
                &MlxModule::new(unit),
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".mlp.experts."),
            )
            .map_err(Into::into)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        QwenExecution::Resident(Box::new(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        )))
    } else {
        QwenExecution::Layerwise(Box::new(LayerwiseRuntime::new(architecture, policy)))
    };
    Ok(QwenModel {
        state_layout: eredu_architectures::qwen::state_layout(&args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
        args,
        metadata,
        parallel_info: None,
        parallel_rank: None,
        execution,
        expert_cache: None,
    })
}

pub(crate) fn quantize_neutral_qwen_store(
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
    let mut target_args = source_args.clone();
    target_args.quantization = Some(quantization);
    target_args.quantization_config = None;
    target_args.quantized_weights = None;
    target_args.quantized_weight_configs = None;
    let source = NeutralArchitecture::new(source_args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let target = NeutralArchitecture::new(target_args.clone(), stream)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let count = usize::try_from(source_args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen layer count".into()))?;
    let source_unit_args = source_args.clone();
    let target_unit_args = target_args.clone();
    let (store, report) = quantize_parameterized_store(
        store,
        source.static_modules(),
        target.static_modules(),
        move |index, stream| {
            eredu_architectures::qwen::new_block::<MlxBackend>(&source_unit_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        move |index, stream| {
            eredu_architectures::qwen::new_block::<MlxBackend>(&target_unit_args, index, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        },
        count,
        quantization,
        stream,
    )?;
    Ok((store, target_args, report))
}

/// Qwen/Mistral causal LM whose execution engine follows its residency policy.
pub struct QwenModel {
    args: ModelArgs,
    state_layout: eredu_runtime::StateLayout,
    metadata: LayerwiseModelMetadata,
    parallel_info: Option<ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_rank: Option<crate::CacheRankIdentity>,
    execution: QwenExecution,
    expert_cache: Option<ExpertCache>,
}

impl QwenModel {
    /// Returns normalized model arguments regardless of execution engine.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        eredu_architectures::qwen::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns whether all parameters use the eager execution-device engine.
    pub fn is_fully_resident(&self) -> bool {
        self.metadata.residency() == ExecutionResidency::FullyResident
    }

    /// Returns canonical parameter and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Returns logical residency and transfer telemetry for a layerwise model.
    pub fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        let report = match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().residency_report()?,
            QwenExecution::Layerwise(execution) => execution.policy().residency_report()?,
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().residency_report()?
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().residency_report()?
            }
        };
        Ok(Some(report))
    }

    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match &self.execution {
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().dense_stream_report()
            }
            QwenExecution::TensorParallelResident(_) => Ok(None),
            QwenExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
            QwenExecution::Resident(_) => Ok(None),
        }
    }

    /// Returns independent expert residency telemetry when configured.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store used by a layerwise model.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().checkpoint_store(),
            QwenExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store()
            }
        }
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            QwenExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().checkpoint_store_arc()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().checkpoint_store_arc()
            }
        }
    }

    /// Returns the number of pinned static leases used by the layerwise engine.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            QwenExecution::Resident(execution) => execution.policy().static_lease_count(),
            QwenExecution::Layerwise(execution) => execution.policy().static_lease_count(),
            QwenExecution::TensorParallelResident(execution) => {
                execution.policy().static_lease_count()
            }
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().static_lease_count()
            }
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
            CacheResidencyPolicy::Paged(options) => {
                self.new_paged_cache(options, None, self.parallel_rank)
            }
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
            self.new_paged_cache_from_manager(manager, identity.topology.cache_rank_identity())?;
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
        rank: Option<crate::CacheRankIdentity>,
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
        rank: Option<crate::CacheRankIdentity>,
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
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider =
                    crate::composition::qwen_expert::cached_provider(&expert_cache, &args);
                self.forward_with_expert_provider(inputs, None, cache, &mut provider, stream)
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        self.validate_cache(cache)?;
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: inputs,
            mask: None,
        };
        match &mut self.execution {
            QwenExecution::Resident(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            QwenExecution::Layerwise(execution) => execution
                .forward(input, cache, stream)
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel Qwen requires its collective execution context".into(),
            )),
        }
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
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let expert_cache = self.expert_cache.take();
        let mut observer = NeutralQwenObserver { inner: observer };
        let result = match &mut self.execution {
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel observation requires its collective execution context".into(),
            )),
            QwenExecution::Resident(execution) => execution
                .forward_with_unit_executor(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: inputs,
                        mask,
                    },
                    cache,
                    stream,
                    |architecture, group, index, block, hidden, state, forward, stream| {
                        match expert_cache.as_ref() {
                            Some(expert_cache) => forward_qwen_cached_observed_unit(
                                architecture,
                                &args,
                                expert_cache,
                                pass,
                                group,
                                index,
                                block,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut observer,
                            ),
                            None => forward_qwen_observed_unit(
                                architecture,
                                &args,
                                group,
                                index,
                                block,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut observer,
                            ),
                        }
                    },
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
            QwenExecution::Layerwise(execution) => execution
                .forward_with_unit_executor(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: inputs,
                        mask,
                    },
                    cache,
                    stream,
                    |architecture, group, index, block, hidden, state, forward, stream| {
                        match expert_cache.as_ref() {
                            Some(expert_cache) => forward_qwen_cached_observed_unit(
                                architecture,
                                &args,
                                expert_cache,
                                pass,
                                group,
                                index,
                                block,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut observer,
                            ),
                            None => forward_qwen_observed_unit(
                                architecture,
                                &args,
                                group,
                                index,
                                block,
                                hidden,
                                state,
                                forward,
                                stream,
                                &mut observer,
                            ),
                        }
                    },
                )
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string())),
        };
        self.expert_cache = expert_cache;
        let output = result?;
        observer
            .inner
            .observe("model.logits", &output)
            .map_err(Error::from)?;
        Ok(output)
    }

    /// Runs a rank-local tensor-parallel forward pass.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(expert_cache) = self.expert_cache.take() {
            let args = self.args.clone();
            let result = {
                let mut provider =
                    crate::composition::qwen_expert::cached_provider(&expert_cache, &args);
                self.forward_tensor_expert_provider(
                    inputs,
                    None,
                    cache,
                    group,
                    &mut provider,
                    stream,
                )
            };
            self.expert_cache = Some(expert_cache);
            return result;
        }
        self.validate_cache(cache)?;
        match &mut self.execution {
            QwenExecution::TensorParallelResident(execution) => execution
                .forward_parallel(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelLayerwise(execution) => execution
                .forward_parallel(
                    eredu_architectures::qwen::LayeredInput {
                        tokens: inputs,
                        mask: None,
                    },
                    cache,
                    group,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Resident(_) | QwenExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel execution".into(),
            )),
        }
    }

    /// Runs the neutral decoder while delegating routed experts to a runtime
    /// provider such as bounded expert residency or expert-parallel exchange.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        self.forward_with_expert_provider(inputs, mask, cache, &mut provider, stream)
    }

    fn forward_with_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let hook = |architecture: &mut NeutralArchitecture,
                    group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &Array,
                    state: &mut MlxKeyValueState,
                    forward: &mut eredu_architectures::qwen::ForwardContext<Array>,
                    context: &Stream| {
            if group != 0 {
                return Err(eredu_nn::Error::backend(format!(
                    "Qwen decoder received execution group {group}"
                )));
            }
            forward_qwen_external_experts(
                architecture,
                index,
                block,
                hidden,
                state,
                forward,
                pass,
                None,
                context,
                provider,
            )
        };
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: inputs,
            mask,
        };
        match &mut self.execution {
            QwenExecution::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelResident(_)
            | QwenExecution::TensorParallelLayerwise(_) => Err(Error::Parallel(
                "tensor-parallel Qwen requires its collective expert executor".into(),
            )),
        }
    }

    /// Runs tensor-parallel attention and projections while delegating routed
    /// experts to the matching-coordinate expert-parallel provider.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut provider =
            crate::backend::mlx::runtime::residency::expert_provider::ExpertExecutorProvider::new(
                &mut execute,
            );
        self.forward_tensor_expert_provider(inputs, mask, cache, group, &mut provider, stream)
    }

    fn forward_tensor_expert_provider<P>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        provider: &mut P,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        P: eredu_runtime::RoutedExpertProvider<MlxBackend>,
        P::Error: std::fmt::Display,
    {
        self.validate_cache(cache)?;
        let pass = if inputs.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let hook = |composition: &mut QwenParallelComposition,
                    execution_group: usize,
                    index: usize,
                    block: &mut NeutralBlock,
                    hidden: &Array,
                    state: &mut MlxKeyValueState,
                    forward: &mut eredu_architectures::qwen::ForwardContext<Array>,
                    parallel: &safemlx::distributed::Group,
                    context: &Stream| {
            if execution_group != 0 {
                return Err(Error::Parallel(format!(
                    "Qwen decoder received execution group {execution_group}"
                )));
            }
            forward_qwen_external_experts(
                &mut composition.architecture,
                index,
                block,
                hidden,
                state,
                forward,
                pass,
                Some(parallel),
                context,
                provider,
            )
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
        };
        let input = eredu_architectures::qwen::LayeredInput {
            tokens: inputs,
            mask,
        };
        match &mut self.execution {
            QwenExecution::TensorParallelResident(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::TensorParallelLayerwise(runtime) => runtime
                .forward_parallel_with_unit_executor(input, cache, group, stream, hook)
                .map_err(|error| Error::Parallel(error.to_string())),
            QwenExecution::Resident(_) | QwenExecution::Layerwise(_) => Err(Error::Parallel(
                "model was not loaded for tensor-parallel expert execution".into(),
            )),
        }
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

    /// Clears temporary execution-device decoder copies when layerwise residency is active.
    ///
    /// Returns `true` when a layerwise window was cleared and `false` for the
    /// fully resident engine.
    pub fn clear_device_layer_window(&self) -> Result<bool, Error> {
        if self.is_fully_resident() {
            return Ok(false);
        }
        match &self.execution {
            QwenExecution::Layerwise(_) => {}
            QwenExecution::TensorParallelLayerwise(execution) => {
                execution.policy().clear_device_window()?
            }
            QwenExecution::Resident(_) | QwenExecution::TensorParallelResident(_) => {
                return Ok(false)
            }
        }
        Ok(true)
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layout = self.state_layout.clone();
        let topology = self
            .parallel_info
            .as_ref()
            .map_or_else(PromptCacheTopology::default, |info| {
                crate::backend::mlx::cache::prompt_cache_topology(info.topology())
            });
        let identity = eredu_architectures::qwen::state_identity(self.args(), &layout, 0, topology)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        identity
            .prompt_cache_identity(&layout)
            .map_err(|error| Error::Parallel(error.to_string()))
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
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.prefill(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut MlxKeyValueState,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

pub(crate) fn load_qwen_safetensors_mlx(
    model_dir: impl AsRef<Path>,
    weight_residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let model_dir = model_dir.as_ref();
    let expert_options = weight_residency.expert_cache();
    let execution_options = weight_residency.layers();
    let args = load_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Qwen", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, execution_options.max_mapped_shards())?;
    let store = resolve_qwen_safetensors_store(store, &args)?;
    if let Some(quantization) = quantize_on_load {
        let (store, args, report) =
            quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        let mut model = load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?;
        if let Some(options) = expert_options {
            attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
        }
        return Ok(model);
    }
    let mut model = load_neutral_qwen(
        store,
        args,
        execution_options,
        stream,
        weights_stream,
        None,
        expert_options.is_some(),
    )?;
    if let Some(options) = expert_options {
        attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok(model)
}

fn attach_qwen_expert_cache(
    model: &mut QwenModel,
    options: eredu_runtime::ExpertCacheLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(), Error> {
    if !model.args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires Qwen3-MoE arguments".into(),
        ));
    }
    let store = model.checkpoint_store_arc();
    let entries = crate::composition::qwen_expert::expert_catalog(&model.args, store.as_ref())?;
    model.expert_cache = Some(ExpertCache::new_shared(
        store,
        entries,
        options,
        weights_stream.clone(),
        stream.clone(),
    )?);
    Ok(())
}

fn load_neutral_qwen_parallel(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
    external_experts: bool,
) -> Result<QwenModel, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid Qwen layer count".into()))?;
    let mut composition = if external_experts {
        QwenParallelComposition::new_external_experts(args.clone(), stream)?
    } else {
        QwenParallelComposition::new(args.clone(), stream)?
    };
    let mut planner = build.planner();
    let static_modules = composition.architecture.static_modules();
    for group in eredu_architectures::qwen::static_parallel_parameter_groups::<MlxBackend>(
        &static_modules.embeddings,
        &static_modules.norm,
        static_modules.lm_head.as_ref(),
        &args.parameter_root,
    )? {
        planner.register(group)?;
    }
    for index in 0..layer_count {
        let unit = eredu_architectures::qwen::new_block::<MlxBackend>(&args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for group in eredu_architectures::qwen::layer_parallel_parameter_groups::<MlxBackend>(
            &unit, &args, index,
        )? {
            planner.register(group)?;
        }
    }
    let (_, layout) = planner.finish()?;
    if layout.is_empty() {
        return Err(Error::Parallel(
            "Qwen declared no tensor-parallel parameters".into(),
        ));
    }
    composition.configure_parallel(build, &layout, stream)?;
    let state_layout = composition.local_state_layout()?;
    let factory = composition.unit_factory()?;

    let global_static_bindings = build_module_bindings(
        &MlxModule::new(composition.architecture.static_modules().clone()),
        "",
        store.as_ref(),
    )?;
    let mut global_parameter_bytes = binding_bytes(&global_static_bindings)?;
    for index in 0..layer_count {
        let unit = eredu_architectures::qwen::new_block::<MlxBackend>(&args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let recipes = if external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store.as_ref(), &args, index)?
        };
        let bytes = binding_bytes(&build_module_bindings_with_recipes_excluding(
            &MlxModule::new(unit),
            "",
            store.as_ref(),
            recipes,
            |name| external_experts && name.contains(".mlp.experts."),
        )?)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("global Qwen parameter bytes overflowed".into()))?;
    }

    let binding_args = args.clone();
    let unit_layout = decoder_unit_layout(layer_count)?;
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        &mut composition,
        factory,
        unit_layout,
        options,
        stream,
        weights_stream,
        move |key| {
            key.starts_with("rope_freqs.")
                || key.ends_with(".rotary_emb.inv_freq")
                || (external_experts && key.contains(".mlp.experts."))
        },
        |modules, store| {
            let global = MlxModule::new(modules.architecture.static_modules().clone());
            let bindings = build_module_bindings(&global, "", store)?;
            shard_layer_bindings(bindings, "", store, &layout)
        },
        |index, _local, store, stream| {
            let global =
                eredu_architectures::qwen::new_block::<MlxBackend>(&binding_args, index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let recipes = if external_experts {
                BTreeMap::new()
            } else {
                qwen_unit_recipes(store, &binding_args, index)?
            };
            let bindings = build_module_bindings_with_recipes_excluding(
                &MlxModule::new(global),
                "",
                store,
                recipes,
                |name| external_experts && name.contains(".mlp.experts."),
            )?;
            shard_layer_bindings(
                bindings,
                &format!("{}.layers.{index}", binding_args.parameter_root),
                store,
                &layout,
            )
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("local Qwen parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("device Qwen parameter bytes overflowed".into()))?;
    let parallel_info = ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        layout
            .tensors()
            .map(|(target, _)| target.to_owned())
            .collect(),
        local_parameter_bytes,
        global_parameter_bytes,
        if options.is_fully_resident() {
            local_parameter_bytes
        } else {
            metadata.static_device_bytes()
        },
        maximum_device_parameter_bytes,
    );
    let parallel_rank =
        crate::backend::mlx::cache::prompt_cache_topology(build.topology()).cache_rank_identity();
    let execution = if options.is_fully_resident() {
        QwenExecution::TensorParallelResident(Box::new(LayerwiseRuntime::new(
            composition,
            policy.into_resident(stream)?,
        )))
    } else {
        QwenExecution::TensorParallelLayerwise(Box::new(LayerwiseRuntime::new(composition, policy)))
    };
    Ok(QwenModel {
        args,
        state_layout,
        metadata,
        parallel_info: Some(parallel_info),
        parallel_rank,
        execution,
        expert_cache: None,
    })
}

/// Loads only Qwen static and nonexpert layer weights, leaving routed expert
/// materialization to the runtime expert provider.
pub(crate) fn load_qwen_external_experts_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: Option<crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    if !args.is_moe() {
        return Err(Error::Parallel(
            "external Qwen expert residency requires a routed Qwen model".into(),
        ));
    }
    let store = resolve_qwen_safetensors_store(store, &args)?;
    match build {
        Some(build) if build.topology().tensor_parallel_size > 1 => {
            load_neutral_qwen_parallel(store, args, options, build, stream, weights_stream, true)
        }
        _ => load_neutral_qwen(store, args, options, stream, weights_stream, None, true),
    }
}

/// Loads Qwen/Mistral through the generalized tensor-parallel execution engine.
pub(crate) fn load_qwen_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_qwen_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = load_model_args(model_dir)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let store = resolve_qwen_safetensors_store(store, &args)?;
    load_neutral_qwen_parallel(store, args, options, build, stream, weights_stream, false)
}

struct QwenGgufCatalog<'a>(&'a GgufCheckpoint);

impl eredu_architectures::qwen::GgufTensorCatalog for QwenGgufCatalog<'_> {
    fn contains(&self, name: &str) -> bool {
        crate::backend::mlx::runtime::checkpoint::load::GgufTensorNames::contains_gguf_tensor(
            self.0, name,
        )
    }
}

pub(crate) struct PreparedQwenGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_qwen_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<PreparedQwenGguf, Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => value.as_str(),
        _ => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF general.architecture must be a string".into(),
            ))
        }
    };
    let gguf_architecture = crate::core::GgufArchitecture::resolve(architecture)?;
    crate::composition::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let mut args = eredu_architectures::qwen::model_args_from_gguf_catalog(
        &QwenGgufCatalog(checkpoint),
        metadata,
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let is_moe = args.is_moe();
    let translate =
        |name: &str| eredu_architectures::qwen::translate_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    if is_moe {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj.weight")) {
                configs.remove(&format!("{prefix}.up_proj.weight"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    Ok(PreparedQwenGguf {
        args,
        eos_token_ids: crate::backend::mlx::gguf_eos_token_ids(metadata)?,
    })
}

pub(crate) fn load_qwen_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = prepare_qwen_gguf_checkpoint(checkpoint, metadata)?;
    let gguf_plan = eredu_architectures::qwen::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            |name| {
                eredu_architectures::qwen::translate_gguf_weight_name(name, prepared.args.is_moe())
            },
            options.max_mapped_shards(),
        )?);
    let model = load_neutral_qwen_parallel(
        store,
        prepared.args,
        options,
        build,
        stream,
        weights_stream,
        false,
    )?;
    Ok((model, prepared.eos_token_ids))
}

/// Loads a Qwen/Mistral GGUF checkpoint using the selected residency policy.
pub(crate) fn load_qwen_gguf_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenModel, Vec<u32>), Error> {
    let prepared = prepare_qwen_gguf_checkpoint(checkpoint, metadata)?;
    let gguf_plan = eredu_architectures::qwen::gguf_plan(&prepared.args)
        .map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            |name| {
                eredu_architectures::qwen::translate_gguf_weight_name(name, prepared.args.is_moe())
            },
            residency.max_mapped_shards(),
        )?);
    let args = prepared.args;
    let expert_options = residency.expert_cache();
    let execution_options = residency.layers();
    let model = if let Some(quantization) = quantization {
        let (store, args, report) =
            quantize_neutral_qwen_store(store, &args, quantization, stream)?;
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            Some(report),
            expert_options.is_some(),
        )?
    } else {
        load_neutral_qwen(
            store,
            args,
            execution_options,
            stream,
            weights_stream,
            None,
            expert_options.is_some(),
        )?
    };
    let mut model = model;
    if let Some(options) = expert_options {
        attach_qwen_expert_cache(&mut model, options, stream, weights_stream)?;
    }
    Ok((model, prepared.eos_token_ids))
}

/// Qwen implementation of the generic layerwise model-family contract.
pub struct QwenParallelComposition {
    architecture: NeutralArchitecture,
    parallel_embedding: Option<MlxNamedModule<VocabParallelEmbedding>>,
    parallel_lm_head: Option<MlxNamedModule<VocabParallelLmHead>>,
    parallel_kv_heads: Option<Vec<i32>>,
    local_args: Option<Arc<Vec<ModelArgs>>>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
    external_experts: bool,
}

impl QwenParallelComposition {
    /// Creates metadata-only static modules for a normalized Qwen configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let architecture = NeutralArchitecture::new(args, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        Ok(Self {
            architecture,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            local_args: None,
            topology: None,
            external_experts: false,
        })
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut composition = Self::new(args, stream)?;
        composition.external_experts = true;
        Ok(composition)
    }

    /// Returns normalized Qwen arguments.
    pub const fn args(&self) -> &ModelArgs {
        self.architecture.args()
    }

    pub(crate) fn model_type(&self) -> &str {
        &self.args().model_type
    }

    pub(crate) fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    pub(crate) fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let roles = [
            ("embedding", "qwen.static.embedding"),
            ("norm", "qwen.static.norm"),
            ("output", "qwen.static.output"),
        ]
        .into_iter()
        .filter_map(|(role, unit)| select(unit).then_some(role))
        .collect::<Vec<_>>();
        self.selected_static_units_for_roles(store, &roles)
    }

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let static_modules = self.architecture.static_modules();
        for group in eredu_architectures::qwen::static_parallel_parameter_groups::<MlxBackend>(
            &static_modules.embeddings,
            &static_modules.norm,
            static_modules.lm_head.as_ref(),
            &self.args().parameter_root,
        )? {
            planner.register(group)?;
        }
        for index in 0..self.args().num_hidden_layers as usize {
            let unit =
                eredu_architectures::qwen::new_block::<MlxBackend>(self.args(), index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            for group in eredu_architectures::qwen::layer_parallel_parameter_groups::<MlxBackend>(
                &unit,
                self.args(),
                index,
            )? {
                planner.register(group)?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MlxModule<NeutralBlock>, Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "Qwen decoder has no execution group {group}"
            )));
        }
        let local;
        let args = match layout {
            Some(layout) => {
                local = eredu_architectures::qwen::local_block_args(self.args(), index, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                &local
            }
            None => self.args(),
        };
        let mut block = eredu_architectures::qwen::new_block::<MlxBackend>(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        if let (Some(assignment), eredu_architectures::qwen::FeedForward::Routed(moe)) =
            (assignment, &mut block.mlp)
        {
            let count = i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?;
            if count > 0 {
                let prefix = format!("model.layers.{index}.mlp.experts");
                let gate_up = format!("{prefix}.gate_up_proj");
                let down = format!("{prefix}.down_proj");
                moe.experts = <MlxBackend as RoutedNeuralBackend>::swiglu_expert_bank(
                    SwiGluExpertBankSpec {
                        expert_count: count,
                        input_dimensions: args.hidden_size,
                        intermediate_dimensions: args.moe_intermediate_size,
                        output_dimensions: args.hidden_size,
                        limit: None,
                        layout: SwiGluExpertLayout::Packed {
                            gate_up: SwiGluExpertProjection {
                                weight: ParameterSpec::trainable(&gate_up)
                                    .map_err(|error| Error::Parallel(error.to_string()))?,
                                format: args.weight_quantization_for(&gate_up).into(),
                            },
                            down: SwiGluExpertProjection {
                                weight: ParameterSpec::trainable(&down)
                                    .map_err(|error| Error::Parallel(error.to_string()))?,
                                format: args.weight_quantization_for(&down).into(),
                            },
                        },
                    },
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(MlxModule::new(block))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "Qwen decoder has no execution group {group}"
            )));
        }
        let recipes = if self.external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store, self.args(), index)?
        };
        // Checkpoint recipes describe the global bank. Build their initial
        // bindings against global geometry, then apply EP selection and TP
        // sharding before the result is populated into the local block.
        let global_layer;
        let binding_layer = if layout.is_some() || assignment.is_some() {
            global_layer = MlxModule::new(
                eredu_architectures::qwen::new_block::<MlxBackend>(self.args(), index, stream)
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            &global_layer
        } else {
            layer
        };
        let mut bindings = build_module_binding_plan_with_recipes_excluding(
            binding_layer,
            "",
            store,
            recipes,
            |name| self.external_experts && name.contains(".mlp.experts."),
        )?
        .build_bindings(store)?;
        if let Some(assignment) = assignment {
            let indices = assignment.local_global_expert_ids().to_vec();
            bindings = bindings
                .into_iter()
                .map(|binding| {
                    let target = binding.logical_target().unwrap_or_else(|| binding.name());
                    if target.contains(".mlp.experts.") {
                        binding
                            .select_bounded_output(
                                store,
                                eredu_checkpoint::store::TensorSelection::Indices {
                                    axis: 0,
                                    indices: indices.clone(),
                                },
                            )
                            .map_err(Error::from)
                    } else {
                        Ok(binding)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
        }
        match layout {
            Some(layout) => {
                shard_layer_bindings(bindings, &format!("model.layers.{index}"), store, layout)
            }
            None => Ok(bindings),
        }
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self.args().is_moe() {
            return Err(Error::Parallel(
                "Qwen has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args().num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<NeutralBlock>, Error> {
        self.new_cartesian_layer(group, index, None, None, stream)
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MlxModule<NeutralBlock>,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "Qwen decoder has no execution group {group}"
            )));
        }
        let recipes = if self.external_experts {
            BTreeMap::new()
        } else {
            qwen_unit_recipes(store, self.args(), index)?
        };
        Ok(
            build_module_binding_plan_with_recipes_excluding(layer, "", store, recipes, |name| {
                self.external_experts && name.contains(".mlp.experts.")
            })?
            .build_bindings(store)?,
        )
    }

    fn selected_static_units_for_roles(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        roles: &[&str],
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let selected = |role: &str| roles.contains(&role);
        let static_modules = self.architecture.static_modules();
        let mut units = Vec::new();
        if selected("embedding") {
            let prefix = format!("{}.embed_tokens", self.args().parameter_root);
            units.push(StaticUnitBindings::new(
                "qwen.static.embedding",
                build_module_binding_plan_with_recipes(
                    &static_modules.embeddings,
                    &prefix,
                    store,
                    Default::default(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if selected("norm") {
            let prefix = format!("{}.norm", self.args().parameter_root);
            units.push(StaticUnitBindings::new(
                "qwen.static.norm",
                build_module_binding_plan_with_recipes(
                    &static_modules.norm,
                    &prefix,
                    store,
                    Default::default(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if selected("output") {
            if let Some(head) = &static_modules.lm_head {
                units.push(StaticUnitBindings::new(
                    "qwen.static.output",
                    build_module_binding_plan_with_recipes(
                        head,
                        "lm_head",
                        store,
                        Default::default(),
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }
}

impl QwenParallelComposition {
    fn configure_parallel(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_kv_heads = Some(
            eredu_architectures::qwen::local_key_value_heads(self.args(), layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        self.parallel_embedding = Some(MlxNamedModule::new(
            VocabParallelEmbedding::unloaded(
                self.args().vocab_size as usize,
                self.args().hidden_size,
                self.args()
                    .weight_quantization_for("model.embed_tokens.weight"),
                context,
                stream,
            )?,
            ParameterSpec::trainable("model.embed_tokens.weight")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            None,
        )?);
        if self.architecture.static_modules().lm_head.is_some() {
            self.parallel_lm_head = Some(MlxNamedModule::new(
                VocabParallelLmHead::unloaded(
                    self.args().hidden_size,
                    self.args().vocab_size as usize,
                    self.args().weight_quantization_for("lm_head.weight"),
                    context,
                    stream,
                )?,
                ParameterSpec::trainable("lm_head.weight")
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                None,
            )?);
        }
        self.local_args = Some(Arc::new(
            (0..self.args().num_hidden_layers as usize)
                .map(|index| {
                    eredu_architectures::qwen::local_block_args(self.args(), index, layout)
                        .map_err(|error| Error::Parallel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ));
        self.topology = Some(context.topology());
        Ok(())
    }

    fn unit_factory(&self) -> Result<QwenParallelUnitFactory, Error> {
        Ok(QwenParallelUnitFactory {
            local_args: Arc::clone(self.local_args.as_ref().ok_or_else(|| {
                Error::Parallel("parallel Qwen unit layout is not configured".into())
            })?),
        })
    }

    fn local_state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        let layout = eredu_architectures::qwen::cache_layout_with_key_value_heads(
            self.args(),
            self.parallel_kv_heads.clone().ok_or_else(|| {
                Error::Parallel("parallel Qwen cache layout is not configured".into())
            })?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        eredu_runtime::StateLayout::new(layout).map_err(|error| Error::Parallel(error.to_string()))
    }

    fn execution_context<'a>(
        &self,
        group: &'a safemlx::distributed::Group,
        stream: &'a Stream,
    ) -> Result<
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'a>,
        Error,
    > {
        let topology = self
            .topology
            .ok_or_else(|| Error::Parallel("parallel Qwen topology is not configured".into()))?;
        crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            topology, group, stream,
        )
    }
}

impl Parameterized<Array> for QwenParallelComposition {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, Array>,
    {
        if let Some(embedding) = &self.parallel_embedding {
            embedding.visit_parameters(visitor);
        }
        self.architecture
            .static_modules()
            .norm
            .visit_parameters(visitor);
        if let Some(head) = &self.parallel_lm_head {
            head.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, Array>,
    {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.visit_parameters_mut(visitor);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .visit_parameters_mut(visitor);
        if let Some(head) = &mut self.parallel_lm_head {
            head.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(embedding) = &mut self.parallel_embedding {
            embedding.set_trainable(trainable);
        }
        self.architecture
            .static_modules_mut()
            .norm
            .set_trainable(trainable);
        if let Some(head) = &mut self.parallel_lm_head {
            head.set_trainable(trainable);
        }
    }
}

impl LayeredArchitecture<MlxBackend, MlxKeyValueState> for QwenParallelComposition {
    type Input<'a> = eredu_architectures::qwen::LayeredInput<'a, Array>;
    type StaticModules = Self;
    type Unit = NeutralBlock;
    type ForwardContext = eredu_architectures::qwen::ForwardContext<Array>;
    type RetainedContextValues<'a> = std::option::Iter<'a, Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args().model_type
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "parallel Qwen execution group {group} is outside the decoder"
            )));
        }
        usize::try_from(self.args().num_hidden_layers).map_err(|_| {
            QwenModelError::InvalidLayerCount {
                count: self.args().num_hidden_layers,
            }
            .into()
        })
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::Parallel(format!(
                "parallel Qwen unit {index} is outside the decoder"
            )));
        }
        Ok(format!("model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::Parallel(format!(
                "parallel Qwen execution group {group} is outside the decoder"
            )));
        }
        let args = self
            .local_args
            .as_ref()
            .and_then(|args| args.get(index))
            .ok_or_else(|| {
                Error::Parallel(format!("parallel Qwen unit {index} is not configured"))
            })?;
        eredu_architectures::qwen::new_block::<MlxBackend>(args, index, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        _input: Self::Input<'a>,
        _state: &mut MlxKeyValueState,
        _stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        Err(Error::Parallel(
            "parallel Qwen composition requires a collective context".into(),
        ))
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _unit: &mut Self::Unit,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Qwen composition requires a collective context".into(),
        ))
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _state: &mut MlxKeyValueState,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::Parallel(format!(
                "parallel Qwen decoder group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn finish_forward(
        &mut self,
        _hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::Parallel(
            "parallel Qwen composition requires a collective context".into(),
        ))
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        <NeutralArchitecture as LayeredArchitecture<MlxBackend, MlxKeyValueState>>::retained_context_values(
            &self.architecture,
            forward,
            group,
            index,
        )
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MlxKeyValueState> for QwenParallelComposition {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut MlxKeyValueState,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .parallel_embedding
            .as_mut()
            .ok_or_else(|| Error::Parallel("parallel Qwen embedding is not configured".into()))?
            .forward(input.tokens, &execution)?;
        let expected = self.local_state_layout()?;
        self.architecture
            .begin_embedded_with_layout(hidden, input.mask, state, &expected, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn forward_unit_parallel(
        &mut self,
        _group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        state: &mut MlxKeyValueState,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        self.architecture
            .forward_block_parallel(index, unit, hidden, state, forward, group, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _state: &mut MlxKeyValueState,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        let execution = self.execution_context(group, stream)?;
        let hidden = self
            .architecture
            .static_modules_mut()
            .norm
            .forward(hidden, stream)?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, &execution)?,
            None => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Error::Parallel("parallel Qwen embedding is not configured".into()))?
                .project_logits(&hidden, &execution)?,
        };
        logits.all_gather(&execution)
    }
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

impl From<QwenModelError> for crate::backend::mlx::error::Error {
    fn from(error: QwenModelError) -> Self {
        Self::ArchitectureModel(error.to_string())
    }
}
