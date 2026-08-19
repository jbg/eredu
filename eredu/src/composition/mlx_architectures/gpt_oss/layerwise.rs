//! Unified fully resident and bounded layer execution for GPT-OSS.

use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, ExpertCacheLoadOptions, ExpertIdentity, ExpertPass,
    LayerWeightResidency, LayeredArchitecture, LayeredForwardState, LayerwiseRuntime,
    NonExpertWeightResidency, ParallelLayeredArchitecture, StaticUnitBindings, WeightResidency,
};

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::{
    MemberSharding, OffloadUnit, ParameterGroupSpec, ParameterRole, WeightBinding,
};

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::parallel::{
        gqa_projection_members, planned_kv_head_layout, GqaProjectionNames, VocabParallelEmbedding,
        VocabParallelLmHead,
    },
    backend::mlx::nn::shared::{MlxBackend, MlxParameterTree},
    backend::mlx::nn::{self as common},
    backend::mlx::runtime::cache::residency::{open_prompt_cache, CacheResidencyManager},
    backend::mlx::runtime::cache::{KeyValueCache, PagedKeyValueCache},
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes_excluding, build_module_bindings,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    backend::mlx::runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    backend::mlx::runtime::checkpoint::store::{open_gguf_checkpoint_source, TensorSelection},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load,
        recipe::{DerivedWeightRecipe, RecipeDtype},
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_projection_module,
        register_replicated_module, ParallelPlanBuilder, ProjectionSharding,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_module_store_with_bindings, shard_layer_bindings,
        LoadTimeQuantizableAdapter, MlxArchitectureSemantics,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheReport, ExpertCatalogEntry, ExpertRouteBatch,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    composition::mlx_architectures::gpt_oss::model::{
        self as resident, Cache, Experts, LayerCache, ModelArgs, TransformerBlock,
    },
    core::attention::{AttentionPolicy, LayerSchedule},
};
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "gpt_oss.static.embedding";
const NORM_UNIT: &str = "gpt_oss.static.norm";
const HEAD_UNIT: &str = "gpt_oss.static.output";

type GptOssUnit = MlxParameterTree<TransformerBlock>;
type GptOssStatic = MlxParameterTree<GptOssStaticModules>;
type GptOssResidentRuntime =
    LayerwiseRuntime<GptOssArchitecture, MlxBackend, Cache, MlxResidentPolicy<GptOssUnit>>;
type GptOssBoundedRuntime = LayerwiseRuntime<
    GptOssArchitecture,
    MlxBackend,
    Cache,
    MlxLayerwisePolicy<GptOssUnit, GptOssUnitFactory>,
>;

enum GptOssExecution {
    Resident(GptOssResidentRuntime),
    Layerwise(GptOssBoundedRuntime),
}

/// GPT-OSS causal LM using bounded residency for complete decoder blocks.
pub struct GptOssLayerwiseModel {
    execution: GptOssExecution,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl GptOssLayerwiseModel {
    fn architecture(&self) -> &GptOssArchitecture {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.architecture(),
            GptOssExecution::Layerwise(execution) => execution.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut GptOssArchitecture {
        match &mut self.execution {
            GptOssExecution::Resident(execution) => execution.architecture_mut(),
            GptOssExecution::Layerwise(execution) => execution.architecture_mut(),
        }
    }

    fn parallel_kv_heads(&self) -> Option<&[i32]> {
        self.architecture().parallel_kv_heads.as_deref()
    }

    fn prompt_cache_rank_identity(&self) -> Option<crate::core::cache::CacheRankIdentity> {
        self.parallel_topology
            .map(crate::backend::mlx::cache::prompt_cache_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    fn prompt_cache_directory(&self, root: &Path) -> std::path::PathBuf {
        match self.parallel_topology {
            Some(topology) => root.join(format!("rank-{:05}", topology.global_rank)),
            None => root.to_path_buf(),
        }
    }

    /// Returns the validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.architecture().args(),
            GptOssExecution::Layerwise(execution) => execution.architecture().args(),
        }
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.parallel_topology = Some(topology);
    }

    /// Returns the canonical cache-relevant architecture identity.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        gpt_oss_prompt_cache_identity(
            self.args(),
            self.parallel_topology,
            self.parallel_kv_heads(),
        )
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Returns generalized parameter-residency and encoding metadata.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    /// Creates caches matching the canonical per-layer attention schedule.
    pub fn new_cache(&self) -> Cache {
        Cache::new_device(self.args()).expect("validated GPT-OSS cache geometry remains valid")
    }

    /// Creates scheduled attention caches independently of weight residency.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => {
                let manager = CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Cache::new_paged(self.args(), manager, self.prompt_cache_rank_identity())
                    .map_err(Into::into)
            }
        }
    }

    /// Lazily catalogs a compatible persisted scheduled-attention prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        let args = self.args();
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let directory = self.prompt_cache_directory(directory.as_ref());
        let (manager, manifest) =
            open_prompt_cache(&directory, expected, &identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let layers = args
            .attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                let window = policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated GPT-OSS sliding window fits i32")
                });
                PagedKeyValueCache::new(manager.clone(), layer, window).map(LayerCache::Paged)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            Cache {
                layout: Some(resident::state_layout(args)?),
                layers,
            },
            manifest,
        ))
    }

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        validate_prompt_cache_model_identity(&descriptor, &self.prompt_cache_model_identity()?)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let directory = self.prompt_cache_directory(destination.as_ref());
        cache
            .save_prompt_cache(directory, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().residency_report(),
            GptOssExecution::Layerwise(execution) => execution.policy().residency_report(),
        }
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            GptOssExecution::Resident(_) => Ok(None),
            GptOssExecution::Layerwise(execution) => execution.policy().dense_stream_report(),
        }
    }

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.architecture()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store(),
        }
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            GptOssExecution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            GptOssExecution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
        }
    }

    /// Runs a rank-local tensor-parallel forward pass through the generalized engine.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward_parallel(inputs, cache, group, stream)
                .map_err(gpt_oss_layerwise_error),
            GptOssExecution::Layerwise(execution) => execution
                .forward_parallel(inputs, cache, group, stream)
                .map_err(gpt_oss_layerwise_error),
        }
    }

    /// Runs GPT-OSS while preserving its heterogeneous cache schedule.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward(inputs, cache, stream)
                .map_err(gpt_oss_layerwise_error),
            GptOssExecution::Layerwise(execution) => execution
                .forward(inputs, cache, stream)
                .map_err(gpt_oss_layerwise_error),
        }
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |architecture: &mut GptOssArchitecture,
                            _group: usize,
                            index: usize,
                            layer: &mut GptOssUnit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut GptOssForwardContext,
                            stream: &Stream| {
            architecture.forward_unit_with_expert_executor(
                index,
                layer,
                hidden,
                cache,
                context,
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )
        };
        match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward_with_unit_executor(inputs, cache, stream, execute_unit)
                .map_err(gpt_oss_layerwise_error),
            GptOssExecution::Layerwise(execution) => execution
                .forward_with_unit_executor(inputs, cache, stream, execute_unit)
                .map_err(gpt_oss_layerwise_error),
        }
    }

    /// Runs TP-sharded nonexpert layers while delegating routed experts to EP.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |architecture: &mut GptOssArchitecture,
                            _group: usize,
                            index: usize,
                            layer: &mut GptOssUnit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut GptOssForwardContext,
                            group: &safemlx::distributed::Group,
                            stream: &Stream| {
            architecture.forward_unit_parallel_with_expert_executor(
                index,
                layer,
                hidden,
                cache,
                context,
                group,
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )
        };
        match &mut self.execution {
            GptOssExecution::Resident(execution) => execution
                .forward_parallel_with_unit_executor(
                    inputs,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                )
                .map_err(gpt_oss_layerwise_error),
            GptOssExecution::Layerwise(execution) => execution
                .forward_parallel_with_unit_executor(
                    inputs,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                )
                .map_err(gpt_oss_layerwise_error),
        }
    }

    /// Clears temporary decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution {
            GptOssExecution::Resident(execution) => {
                execution.policy().clear_device_group("text_decoder")
            }
            GptOssExecution::Layerwise(execution) => {
                execution.policy().clear_device_group("text_decoder")
            }
        }
    }
}

impl CausalModel<Cache> for GptOssLayerwiseModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct GptOssStaticBody {
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
struct GptOssReplicatedStatic {
    #[param]
    model: GptOssStaticBody,
    #[param]
    lm_head: MaybeQuantized<nn::Linear>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct GptOssParallelStaticBody {
    #[param]
    embed_tokens: VocabParallelEmbedding,
    #[param]
    norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
struct GptOssParallelStatic {
    #[param]
    model: GptOssParallelStaticBody,
    #[param]
    lm_head: VocabParallelLmHead,
}

#[derive(Debug, Clone)]
enum GptOssStaticModules {
    Replicated(GptOssReplicatedStatic),
    Parallel(GptOssParallelStatic),
}

impl ModuleParameters for GptOssStaticModules {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Replicated(modules) => modules.num_parameters(),
            Self::Parallel(modules) => modules.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Replicated(modules) => modules.parameters(),
            Self::Parallel(modules) => modules.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Replicated(modules) => modules.parameters_mut(),
            Self::Parallel(modules) => modules.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Replicated(modules) => modules.trainable_parameters(),
            Self::Parallel(modules) => modules.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(modules) => modules.freeze_parameters(recursive),
            Self::Parallel(modules) => modules.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(modules) => modules.unfreeze_parameters(recursive),
            Self::Parallel(modules) => modules.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(modules) => modules.all_frozen(),
            Self::Parallel(modules) => modules.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(modules) => modules.any_frozen(),
            Self::Parallel(modules) => modules.any_frozen(),
        }
    }
}

impl GptOssStaticModules {
    fn replicated(args: &ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Ok(Self::Replicated(GptOssReplicatedStatic {
            model: GptOssStaticBody {
                embed_tokens: common::linear::unloaded_maybe_quantized_embedding(
                    args.vocab_size,
                    args.hidden_size,
                    args.weight_quantization_for("model.embed_tokens.weight"),
                    stream,
                )?,
                norm: nn::RmsNorm::unloaded(
                    args.hidden_size,
                    args.rms_norm_eps,
                    Dtype::Float32,
                    stream,
                )?,
            },
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
        }))
    }

    fn parallel(
        args: &ModelArgs,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Ok(Self::Parallel(GptOssParallelStatic {
            model: GptOssParallelStaticBody {
                embed_tokens: VocabParallelEmbedding::unloaded(
                    args.vocab_size as usize,
                    args.hidden_size,
                    args.weight_quantization_for("model.embed_tokens.weight"),
                    build,
                    stream,
                )?,
                norm: nn::RmsNorm::unloaded(
                    args.hidden_size,
                    args.rms_norm_eps,
                    Dtype::Float32,
                    stream,
                )?,
            },
            lm_head: VocabParallelLmHead::unloaded(
                args.hidden_size,
                args.vocab_size as usize,
                args.weight_quantization_for("lm_head.weight"),
                build,
                stream,
            )?,
        }))
    }
}

#[derive(Clone)]
struct GptOssUnitFactory {
    args: ModelArgs,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    sparse_experts: bool,
}

impl MlxUnitFactory<GptOssUnit> for GptOssUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<GptOssUnit, Error> {
        let layer = build_gpt_oss_unit(&self.args, index, self.parallel_layout.as_deref(), stream)?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_experts || !name.starts_with("mlp.experts.")
        })
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

struct GptOssArchitecture {
    args: ModelArgs,
    static_modules: GptOssStatic,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl GptOssArchitecture {
    fn new(args: ModelArgs, sparse_expert_cache: bool, stream: &Stream) -> Result<Self, Error> {
        args.validate()?;
        let static_modules = GptOssStaticModules::replicated(&args, stream)?;
        Ok(Self {
            args,
            static_modules: MlxParameterTree::new(static_modules, "")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            parallel_topology: None,
            parallel_kv_heads: None,
            sparse_expert_cache,
            expert_cache: None,
        })
    }

    const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = Cache::new_device(&self.args)?;
        }
        if cache.layers.len() != self.args.attention_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.attention_schedule.len()
            )));
        }
        for (index, (cache, policy)) in cache
            .layers
            .iter()
            .zip(self.args.attention_schedule.iter())
            .enumerate()
        {
            let actual = cache.attention_policy()?;
            if actual != *policy {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GPT-OSS cache policy mismatch at layer {index}: expected {policy:?}, got {actual:?}"
                )));
            }
        }
        Ok(())
    }

    fn forward_unit_with_expert_executor<F>(
        &mut self,
        index: usize,
        layer: &mut GptOssUnit,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut GptOssForwardContext,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .args
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        Ok(layer.forward_with_expert_executor(
            hidden,
            mask.as_ref(),
            layer_cache,
            stream,
            execute,
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_unit_parallel_with_expert_executor<F>(
        &mut self,
        index: usize,
        layer: &mut GptOssUnit,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut GptOssForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Error>
    where
        F: FnMut(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .args
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        Ok(layer.forward_tensor_with_expert_executor(
            hidden,
            mask.as_ref(),
            layer_cache,
            group,
            stream,
            execute,
        )?)
    }
}

fn build_gpt_oss_unit(
    args: &ModelArgs,
    index: usize,
    parallel_layout: Option<&eredu_runtime::LocalModelLayout>,
    stream: &Stream,
) -> Result<TransformerBlock, Error> {
    let Some(layout) = parallel_layout else {
        return Ok(TransformerBlock::new(args, index, stream)?);
    };
    let prefix = format!("model.layers.{index}");
    let find = |name: &str| {
        layout
            .tensor(&format!("{prefix}.{name}.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.{name}.inner.weight")))
    };
    let query = find("self_attn.q_proj")
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
    let key = find("self_attn.k_proj")
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
    let expert = layout
        .tensor(&format!("{prefix}.mlp.experts.gate_up_proj_bias"))
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} experts")))?;
    let mut local = args.clone();
    local.num_attention_heads = query.local_shape()[0] as i32 / local.head_dim;
    local.num_key_value_heads = key.local_shape()[0] as i32 / local.head_dim;
    local.intermediate_size = expert.local_shape()[1] as i32 / 2;
    Ok(TransformerBlock::new(&local, index, stream)?)
}

fn gpt_oss_layer_recipes(
    args: &ModelArgs,
    index: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let prefix = format!("model.layers.{index}.mlp.experts");
    if !store
        .source_keys()
        .contains(&format!("{prefix}.gate_proj.weight"))
    {
        return Ok(BTreeMap::new());
    }
    let experts = args.num_local_experts as usize;
    let hidden = args.hidden_size as usize;
    let intermediate = args.intermediate_size as usize;
    let source =
        |name: &str| DerivedWeightRecipe::source(format!("{prefix}.{name}"), TensorSelection::Full);
    let stack_reshape = |gate: &str, up: &str, shape: Vec<usize>| DerivedWeightRecipe::Reshape {
        input: Box::new(DerivedWeightRecipe::Stack {
            axis: 2,
            inputs: vec![source(gate), source(up)],
        }),
        shape,
    };
    let gate_up_u32 = stack_reshape(
        "gate_proj.weight",
        "up_proj.weight",
        vec![experts, 2 * intermediate, hidden / 8],
    );
    Ok(BTreeMap::from([
        (
            "mlp.experts.gate_up_proj_blocks".into(),
            DerivedWeightRecipe::View {
                input: Box::new(gate_up_u32),
                dtype: RecipeDtype::U8,
                shape: vec![experts, 2 * intermediate, hidden / 32, 16],
            },
        ),
        (
            "mlp.experts.gate_up_proj_scales".into(),
            stack_reshape(
                "gate_proj.scales",
                "up_proj.scales",
                vec![experts, 2 * intermediate, hidden / 32],
            ),
        ),
        (
            "mlp.experts.gate_up_proj_bias".into(),
            stack_reshape(
                "gate_proj.bias",
                "up_proj.bias",
                vec![experts, 2 * intermediate],
            ),
        ),
        (
            "mlp.experts.down_proj_blocks".into(),
            DerivedWeightRecipe::View {
                input: Box::new(source("down_proj.weight")),
                dtype: RecipeDtype::U8,
                shape: vec![experts, hidden, intermediate / 32, 16],
            },
        ),
        (
            "mlp.experts.down_proj_scales".into(),
            source("down_proj.scales"),
        ),
        (
            "mlp.experts.down_proj_bias".into(),
            source("down_proj.bias"),
        ),
    ]))
}

fn gpt_oss_unit_bindings(
    args: &ModelArgs,
    index: usize,
    layer: &TransformerBlock,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    sparse_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    Ok(build_module_binding_plan_with_recipes_excluding(
        layer,
        &format!("model.layers.{index}"),
        store,
        gpt_oss_layer_recipes(args, index, store)?,
        |name| sparse_experts && name.starts_with("mlp.experts."),
    )?
    .build_bindings(store)?)
}

impl LayeredArchitecture<MlxBackend, Cache> for GptOssArchitecture {
    type Input<'a> = &'a Array;
    type StaticModules = GptOssStatic;
    type Unit = GptOssUnit;
    type ForwardContext = GptOssForwardContext;
    type RetainedContextValues<'a> = std::iter::Empty<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.attention_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no execution group {group}"
            )))
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no decoder unit {index}"
            )));
        }
        Ok(format!("model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        self.group_unit_count(group)?;
        let layer = build_gpt_oss_unit(&self.args, index, None, stream)?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_expert_cache || !name.starts_with("mlp.experts.")
        })
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let GptOssStaticModules::Replicated(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "GPT-OSS replicated execution received parallel static modules".into(),
            ));
        };
        let hidden = modules.model.embed_tokens.forward(input, stream)?;
        Ok(LayeredForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _cache: &mut Cache,
        _context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS execution group {group} received {} dependencies",
                dependencies.len()
            ))),
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group)?;
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "GPT-OSS sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            let args = self.args.clone();
            let layer_cache = &mut cache.layers[index];
            let offset = layer_cache.offset();
            let policy = args
                .attention_schedule
                .get(index)
                .expect("validated GPT-OSS layer index");
            let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
            return Ok(layer.forward_with_expert_executor(
                hidden,
                mask.as_ref(),
                layer_cache,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_gpt_oss_experts(
                        expert_cache,
                        &args,
                        index,
                        pass,
                        flat,
                        indices,
                        weights,
                        stream,
                    )
                },
            )?);
        }
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .args
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        Ok(layer.forward(hidden, mask.as_ref(), layer_cache, stream)?)
    }

    fn retained_context_values<'a>(
        &self,
        _context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::empty()
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let GptOssStaticModules::Replicated(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "GPT-OSS replicated execution received parallel static modules".into(),
            ));
        };
        let hidden = modules.model.norm.forward(hidden, stream)?;
        Ok(modules.lm_head.forward(&hidden, stream)?)
    }
}

impl ParallelLayeredArchitecture<MlxBackend, Cache> for GptOssArchitecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("GPT-OSS parallel topology was not configured".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            topology,
            group,
            stream,
        )?;
        let GptOssStaticModules::Parallel(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "GPT-OSS parallel execution received replicated static modules".into(),
            ));
        };
        let hidden = modules.model.embed_tokens.forward(input, &execution)?;
        Ok(LayeredForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn forward_unit_parallel(
        &mut self,
        execution_group: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(execution_group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .args
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        Ok(layer.forward_tensor_parallel(hidden, mask.as_ref(), layer_cache, group, stream)?)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _context: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("GPT-OSS parallel topology was not configured".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
            topology,
            group,
            stream,
        )?;
        let GptOssStaticModules::Parallel(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "GPT-OSS parallel execution received replicated static modules".into(),
            ));
        };
        let hidden = modules.model.norm.forward(hidden, stream)?;
        modules
            .lm_head
            .forward(&hidden, &execution)?
            .all_gather(&execution)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_gpt_oss_experts(
    expert_cache: &ExpertCache,
    args: &ModelArgs,
    index: usize,
    pass: ExpertPass,
    flat: &Array,
    indices: &Array,
    weights: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    expert_cache
        .execute_routes_bounded(
            ExpertRouteBatch::new(index, flat, indices, weights, pass),
            stream,
            |flat, acquired, weights, stream| {
                let started = Instant::now();
                let mut compact_args = args.clone();
                compact_args.num_local_experts = acquired.identities().len() as i32;
                let mut bank = Experts::new(&compact_args, stream)?;
                bank.gate_up_proj_blocks = Param::new(
                    acquired
                        .compact_binding("gate_up_proj_blocks", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.gate_up_proj_scales = Param::new(
                    acquired
                        .compact_binding("gate_up_proj_scales", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.gate_up_proj_bias = Param::new(
                    acquired
                        .compact_binding("gate_up_proj_bias", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj_blocks = Param::new(
                    acquired
                        .compact_binding("down_proj_blocks", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj_scales = Param::new(
                    acquired
                        .compact_binding("down_proj_scales", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj_bias = Param::new(
                    acquired
                        .compact_binding("down_proj_bias", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                expert_cache.record_compact_bank(
                    pass,
                    acquired.scratch_bytes(),
                    started.elapsed(),
                )?;
                Ok(bank.forward(flat, acquired.compact_routes(), weights, stream)?)
            },
        )
        .map_err(|error| Exception::custom(error.to_string()))
}

fn resolve_gpt_oss_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = super::checkpoint::safetensors_plan(args).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "GPT-OSS checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_gpt_oss_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &ModelArgs,
    sparse_experts: bool,
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
    target_args.quantized_weight_configs = None;
    let source_static = GptOssStaticModules::replicated(source_args, stream)?;
    let target_static = GptOssStaticModules::replicated(&target_args, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let count = source_args.attention_schedule.len();
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| Ok(TransformerBlock::new(&source_units, index, stream)?),
        move |index, stream| Ok(TransformerBlock::new(&target_units, index, stream)?),
        count,
        quantization,
        stream,
        |modules, store| Ok(build_module_bindings(modules, "", store)?),
        move |index, layer, store| {
            gpt_oss_unit_bindings(&binding_args, index, layer, store, sparse_experts)
        },
    )?;
    Ok((store, target_args, report))
}

fn gpt_oss_execution_layout(args: &ModelArgs) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["text_decoder"])?;
    ExecutionUnitLayout::new(&graph, [args.attention_schedule.len()])
        .map_err(|error| Error::Parallel(error.to_string()))
}

fn load_gpt_oss_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let store = resolve_gpt_oss_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_gpt_oss_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut architecture = GptOssArchitecture::new(args.clone(), sparse_experts, stream)?;
    let factory = GptOssUnitFactory {
        args: args.clone(),
        parallel_layout: None,
        sparse_experts,
    };
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        gpt_oss_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        |modules, store| Ok(build_module_bindings(&**modules, "", store)?),
        move |index, unit, store, _| {
            gpt_oss_unit_bindings(&binding_args, index, &unit, store, sparse_experts)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        GptOssExecution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        GptOssExecution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(GptOssLayerwiseModel {
        execution,
        metadata,
        parallel_info: None,
        parallel_topology: None,
    })
}

fn register_gpt_oss_parallel_parameters(
    planner: &mut ParallelPlanBuilder,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    let GptOssStaticModules::Replicated(modules) = GptOssStaticModules::replicated(args, stream)?
    else {
        unreachable!()
    };
    planner.register(
        crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
            &modules.model.embed_tokens,
            "model.embed_tokens",
            args.vocab_size as usize,
            args.hidden_size,
            false,
        )?,
    )?;
    crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
        planner,
        &modules.model.norm,
        "model.norm",
    )?;
    planner.register(
        crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
            &modules.lm_head,
            "lm_head",
            args.hidden_size,
            args.vocab_size as usize,
            false,
        )?,
    )?;
    for index in 0..args.attention_schedule.len() {
        let layer = TransformerBlock::new(args, index, stream)?;
        register_gpt_oss_layer_parallel_plan(planner, &layer, args, index)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_gpt_oss_parallel_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let store = resolve_gpt_oss_store(store, &args)?;
    let mut planner = build.planner();
    register_gpt_oss_parallel_parameters(&mut planner, &args, stream)?;
    let (_, local_layout) = planner.finish()?;
    if local_layout.is_empty() {
        return Err(Error::Parallel(
            "GPT-OSS declared no tensor-parallel parameters".into(),
        ));
    }
    let mut architecture = GptOssArchitecture::new(args.clone(), sparse_experts, stream)?;
    architecture.static_modules =
        MlxParameterTree::new(GptOssStaticModules::parallel(&args, build, stream)?, "")
            .map_err(|error| Error::Parallel(error.to_string()))?;
    architecture.parallel_topology = Some(build.topology());
    architecture.parallel_kv_heads = Some(planned_kv_head_layout(
        &local_layout,
        args.attention_schedule.len(),
        args.head_dim,
        "model.layers",
    )?);

    let global_static = GptOssStaticModules::replicated(&args, stream)?;
    let static_bindings = build_module_bindings(&global_static, "", store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    for index in 0..args.attention_schedule.len() {
        let layer = TransformerBlock::new(&args, index, stream)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&gpt_oss_unit_bindings(
                &args,
                index,
                &layer,
                store.as_ref(),
                sparse_experts,
            )?)?)
            .ok_or_else(|| Error::Parallel("GPT-OSS global parameter bytes overflowed".into()))?;
    }
    let shared_layout = Arc::new(local_layout);
    let factory = GptOssUnitFactory {
        args: args.clone(),
        parallel_layout: Some(Arc::clone(&shared_layout)),
        sparse_experts,
    };
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        gpt_oss_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        move |_, store| shard_layer_bindings(static_bindings, "", store, &static_layout),
        move |index, _local, store, stream| {
            let global = TransformerBlock::new(&binding_args, index, stream)?;
            let bindings =
                gpt_oss_unit_bindings(&binding_args, index, &global, store, sparse_experts)?;
            shard_layer_bindings(
                bindings,
                &format!("model.layers.{index}"),
                store,
                &unit_layout,
            )
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("GPT-OSS local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("GPT-OSS device parameter bytes overflowed".into()))?;
    let info = eredu_runtime::ParallelModelInfo::new(
        build.topology(),
        args.model_type.clone(),
        shared_layout
            .tensors()
            .map(|(target, _)| target.to_string())
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
    let execution = if options.is_fully_resident() {
        GptOssExecution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        GptOssExecution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(GptOssLayerwiseModel {
        execution,
        metadata,
        parallel_info: Some(info),
        parallel_topology: Some(build.topology()),
    })
}

fn gpt_oss_prompt_cache_identity(
    args: &ModelArgs,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
    parallel_kv_heads: Option<&[i32]>,
) -> Result<PromptCacheModelIdentity, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Exception::custom("invalid GPT-OSS cache layer count"))?;
    let local_kv_heads = match topology {
        Some(topology) if topology.tensor_parallel_size > 1 => {
            let heads = parallel_kv_heads.ok_or_else(|| {
                Error::Parallel(
                    "GPT-OSS prompt-cache identity requires planner-derived KV geometry".into(),
                )
            })?;
            let first = *heads.first().ok_or_else(|| {
                Error::Parallel("GPT-OSS planner returned no KV-head geometry".into())
            })?;
            if heads.iter().any(|heads| *heads != first) {
                return Err(Error::Parallel(
                    "GPT-OSS layers unexpectedly received different local KV-head counts".into(),
                ));
            }
            first
        }
        _ => args.num_key_value_heads,
    };
    Ok(PromptCacheModelIdentity {
        model_family: "gpt_oss".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layer_count],
        topology: topology.map_or_else(
            PromptCacheTopology::default,
            crate::backend::mlx::cache::prompt_cache_topology,
        ),
        layer_layout: PromptCacheModelIdentity::key_value_layouts(
            args.attention_schedule
                .iter()
                .map(|policy| policy.window().map(|window| window.get() as i32)),
            local_kv_heads,
            args.head_dim,
        )
        .map_err(|error| Exception::custom(error.to_string()))?,
    })
}

fn gpt_oss_layerwise_error(error: impl std::fmt::Display) -> Error {
    Error::Parallel(error.to_string())
}

/// Loads GPT-OSS through the generalized execution engine.
pub fn load_gpt_oss_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("GPT-OSS", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    load_gpt_oss_with_store(
        store,
        args,
        options,
        quantize_on_load,
        false,
        stream,
        weights_stream,
    )
}

/// Loads GPT-OSS through the generalized tensor-parallel execution engine.
pub(crate) fn load_gpt_oss_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_gpt_oss_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    load_gpt_oss_parallel_with_store(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        resident::get_model_args(model_dir)?,
        options,
        build,
        false,
        stream,
        weights_stream,
    )
}

pub(crate) fn load_gpt_oss_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssLayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let model = load_gpt_oss_parallel_with_store(
        store,
        prepared.args,
        options,
        build,
        false,
        stream,
        weights_stream,
    )?;
    Ok((model, prepared.eos_token_ids))
}

pub(crate) fn load_gpt_oss_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(GptOssLayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let args = prepared.args;
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_gpt_oss_gguf_sparse_with_store(
                store,
                args,
                expert_options,
                residency.layers(),
                quantization,
                stream,
                weights_stream,
            )?,
            prepared.eos_token_ids,
        ));
    }
    let model = load_gpt_oss_with_store(
        store,
        args,
        residency.layers(),
        quantization,
        false,
        stream,
        weights_stream,
    )?;
    Ok((model, prepared.eos_token_ids))
}

fn load_gpt_oss_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let mut model = load_gpt_oss_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantization,
        true,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = model.checkpoint_store_arc();
    let entries = gpt_oss_expert_catalog(&args, checkpoint_store.as_ref())?;
    model.architecture_mut().expert_cache = Some(match quantization {
        Some(quantization) => ExpertCache::new_quantized_shared(
            checkpoint_store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            checkpoint_store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    });
    Ok(model)
}

/// Loads GPT-OSS with independently cached experts and bounded non-expert units.
pub fn load_gpt_oss_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "GPT-OSS independent expert cache",
                args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut model = load_gpt_oss_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantize_on_load,
        true,
        stream,
        weights_stream,
    )?;
    let store = model.checkpoint_store_arc();
    let entries = gpt_oss_expert_catalog(&args, store.as_ref())?;
    model.architecture_mut().expert_cache = Some(match quantize_on_load {
        Some(quantization) => ExpertCache::new_quantized_shared(
            store,
            entries,
            options,
            quantization,
            weights_stream.clone(),
            stream.clone(),
        )?,
        None => ExpertCache::new_shared(
            store,
            entries,
            options,
            weights_stream.clone(),
            stream.clone(),
        )?,
    });
    Ok(model)
}

/// Builds the streamed nonexpert GPT-OSS execution base used by distributed EP.
pub(crate) fn load_gpt_oss_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    load_gpt_oss_with_store(
        store,
        args,
        non_expert.into(),
        None,
        true,
        stream,
        weights_stream,
    )
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_gpt_oss_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<GptOssLayerwiseModel, Error> {
    load_gpt_oss_parallel_with_store(
        store,
        args,
        non_expert.into(),
        build,
        true,
        stream,
        weights_stream,
    )
}

/// Generalized adapter for GPT-OSS native MXFP4 sparse decoder blocks.
pub struct GptOssLayerwiseAdapter {
    args: ModelArgs,
    attention_schedule: LayerSchedule<AttentionPolicy>,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl GptOssLayerwiseAdapter {
    /// Creates metadata-only pinned modules for a validated configuration.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        args.validate()?;
        let attention_schedule = args.attention_schedule.clone();
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = common::linear::unloaded_maybe_quantized_linear(
            args.hidden_size,
            args.vocab_size,
            false,
            args.weight_quantization_for("lm_head.weight"),
            stream,
        )?;
        Ok(Self {
            args,
            attention_schedule,
            embedding,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Creates the semantic adapter with routed experts supplied externally.
    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns the validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    fn new_cache(&self) -> Cache {
        Cache::new_device(&self.args).expect("validated GPT-OSS cache geometry remains valid")
    }

    fn layer_recipes(
        &self,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        gpt_oss_layer_recipes(&self.args, index, store)
    }
}

/// GPT-OSS state shared across temporary decoder blocks.
pub struct GptOssForwardContext {
    sequence_length: i32,
}

fn register_gpt_oss_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &ModelArgs,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    let attention = &layer.self_attn;
    let attention_prefix = format!("{prefix}.self_attn");
    let (head_units, mut attention_members) = gqa_projection_members(
        &attention_prefix,
        GqaProjectionNames {
            query: "q_proj",
            key: "k_proj",
            value: "v_proj",
            output: "o_proj",
        },
        &attention.q_proj,
        &attention.k_proj,
        &attention.v_proj,
        &attention.o_proj,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
    )?;
    attention_members.push(array_parameter_member(
        format!("{prefix}.self_attn.sinks"),
        attention.sinks.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.self_attn.heads"),
        ParameterRole::AttentionHeads,
        head_units,
        attention_members,
    )?)?;
    register_replicated_module(
        planner,
        &layer.input_layernorm,
        &format!("{prefix}.input_layernorm"),
    )?;
    register_replicated_module(
        planner,
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm"),
    )?;
    register_projection_module(
        planner,
        &layer.mlp.router,
        &format!("{prefix}.mlp.router"),
        ProjectionSharding::Replicated,
    )?;
    let experts = &layer.mlp.experts;
    let intermediate = usize::try_from(args.intermediate_size)
        .map_err(|_| Error::Parallel("GPT-OSS expert width exceeds usize".into()))?;
    let intermediate_units = aligned_partition_units(
        &format!("{prefix}.mlp.experts.intermediate"),
        intermediate,
        1,
        32,
    )?;
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.mlp.experts.intermediate"),
        ParameterRole::ExpertIntermediate,
        intermediate_units,
        [
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_blocks"),
                experts.gate_up_proj_blocks.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_scales"),
                experts.gate_up_proj_scales.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                experts.gate_up_proj_bias.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_blocks"),
                experts.down_proj_blocks.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_scales"),
                experts.down_proj_scales.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?,
            array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj_bias"),
                experts.down_proj_bias.as_ref(),
                MemberSharding::Replicated,
            )?,
        ],
    )?)?;
    Ok(())
}

impl LoadTimeQuantizableAdapter for GptOssLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl MlxArchitectureSemantics for GptOssLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = GptOssForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        super::checkpoint::safetensors_plan(&self.args).map_err(Error::UnsupportedArchitecture)
    }

    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.args.quantization
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid GPT-OSS cache layer count"))?;
        let local_kv_heads = match topology {
            Some(topology) if topology.tensor_parallel_size > 1 => {
                let heads = self.parallel_kv_heads.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "GPT-OSS prompt-cache identity requires planner-derived KV geometry".into(),
                    )
                })?;
                let first = *heads.first().ok_or_else(|| {
                    Error::Parallel("GPT-OSS planner returned no KV-head geometry".into())
                })?;
                if heads.iter().any(|heads| *heads != first) {
                    return Err(Error::Parallel(
                        "GPT-OSS layers unexpectedly received different local KV-head counts"
                            .into(),
                    ));
                }
                first
            }
            _ => self.args.num_key_value_heads,
        };
        Ok(PromptCacheModelIdentity {
            model_family: "gpt_oss".into(),
            effective_model_type: self.args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(&self.args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
            layer_layout: PromptCacheModelIdentity::key_value_layouts(
                self.args
                    .attention_schedule
                    .iter()
                    .map(|policy| policy.window().map(|window| window.get() as i32)),
                local_kv_heads,
                self.args.head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        })
    }

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        let (manager, manifest) =
            open_prompt_cache(directory, expected, identity, prefix_token_ids, options)
                .map_err(|error| Exception::custom(error.to_string()))?;
        Ok((
            Cache::new_paged(&self.args, manager, identity.topology.cache_rank_identity())?,
            manifest,
        ))
    }

    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = Vec::new();
        if select(EMBEDDING_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_bindings(&self.embedding, "model.embed_tokens", store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_bindings(&self.norm, "model.norm", store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_bindings(&self.lm_head, "lm_head", store)?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        if leases.len() != 3 {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS adapter received {} static leases, expected 3",
                leases.len()
            )));
        }
        if let Some(v) = &mut self.parallel_embedding {
            populate_module_from_lease(v.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[0])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        if let Some(v) = &mut self.parallel_lm_head {
            populate_module_from_lease(v.inner_mut(), &leases[2])?;
        } else {
            populate_module_from_lease(&mut self.lm_head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.attention_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS cache has {} layers, expected {}",
                cache.layers.len(),
                self.attention_schedule.len()
            )));
        }
        for (index, (cache, policy)) in cache
            .layers
            .iter()
            .zip(self.attention_schedule.iter())
            .enumerate()
        {
            let actual = cache
                .attention_policy()
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            if actual != *policy {
                return Err(Error::UnsupportedArchitecture(format!(
                    "GPT-OSS cache policy mismatch at layer {index}: expected {policy:?}, got {actual:?}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
        Ok(eredu_runtime::LayeredForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Self::ForwardContext>, Error> {
        let Some(v) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = v.forward(input, execution)?;
        Ok(eredu_runtime::LayeredForwardState {
            context: GptOssForwardContext {
                sequence_length: hidden.dim(1),
            },
            hidden,
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.attention_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "GPT-OSS has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(TransformerBlock::new(&self.args, index, stream)?)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        let mut local_args = self.args.clone();
        local_args.num_local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local GPT-OSS expert count exceeds i32".into()))?
        };
        layer.mlp.experts = Experts::new(&local_args, stream)?;
        Ok(layer)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let prefix = format!("model.layers.{index}");
        let expert = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_up_proj_bias"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} experts")))?;
        let mut local_args = self.args.clone();
        local_args.intermediate_size = i32::try_from(expert.local_shape()[1] / 2)
            .map_err(|_| Error::Parallel("local GPT-OSS expert width exceeds i32".into()))?;
        local_args.num_local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local GPT-OSS expert count exceeds i32".into()))?
        };
        layer.mlp.experts = Experts::new(&local_args, stream)?;
        Ok(layer)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_local_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn register_parallel_parameters(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
                &self.embedding,
                "model.embed_tokens",
                self.args.vocab_size as usize,
                self.args.hidden_size,
                false,
            )?,
        )?;
        crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.norm",
        )?;
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
                &self.lm_head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?,
        )?;
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new(&self.args, index, stream)?;
            register_gpt_oss_layer_parallel_plan(planner, &layer, &self.args, index)?;
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args
                .weight_quantization_for("model.embed_tokens.weight"),
            context,
            stream,
        )?);
        self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
            self.args.hidden_size,
            self.args.vocab_size as usize,
            self.args.weight_quantization_for("lm_head.weight"),
            context,
            stream,
        )?);
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            self.attention_schedule.len(),
            self.args.head_dim,
            "model.layers",
        )?);
        Ok(())
    }
    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let find = |n: &str| {
            layout
                .tensor(&format!("{prefix}.{n}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{n}.inner.weight")))
        };
        let q = find("self_attn.q_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let k = find("self_attn.k_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        let expert = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_up_proj_bias"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} experts")))?;
        let mut args = self.args.clone();
        args.num_attention_heads = q.local_shape()[0] as i32 / args.head_dim;
        args.num_key_value_heads = k.local_shape()[0] as i32 / args.head_dim;
        args.intermediate_size = expert.local_shape()[1] as i32 / 2;
        Ok(TransformerBlock::new(&args, index, stream)?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            Ok(populate_module_from_lease_excluding(
                layer,
                lease,
                |name| name.starts_with("mlp.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = format!("model.layers.{index}");
        Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            &prefix,
            store,
            self.layer_recipes(index, store)?,
            |name| self.sparse_expert_cache && name.starts_with("mlp.experts."),
        )?
        .build_bindings(store)?)
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        let indices = assignment.local_global_expert_ids().to_vec();
        self.layer_bindings(group, index, &global, store)?
            .into_iter()
            .map(|binding| {
                let target = binding.logical_target().unwrap_or_else(|| binding.name());
                if target.contains("mlp.experts.") {
                    binding
                        .select_bounded_output(
                            store,
                            TensorSelection::Indices {
                                axis: 0,
                                indices: indices.clone(),
                            },
                        )
                        .map_err(Error::from)
                } else {
                    Ok(binding)
                }
            })
            .collect()
    }

    fn additional_consumed_checkpoint_keys(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .source_keys()
                .into_iter()
                .filter(|key| key.contains(".mlp.experts."))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("gpt_oss.layer.{index:05}")
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask = resident::attention_mask(policy, context.sequence_length, offset, stream)?;
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "GPT-OSS sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_with_expert_executor(
                hidden,
                mask.as_ref(),
                layer_cache,
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            ExpertRouteBatch::new(index, flat, indices, weights, pass),
                            stream,
                            |flat, acquired, weights, stream| {
                                let started = Instant::now();
                                let mut compact_args = self.args.clone();
                                compact_args.num_local_experts = acquired.identities().len() as i32;
                                let mut bank = Experts::new(&compact_args, stream)?;
                                bank.gate_up_proj_blocks = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_blocks", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_scales = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.gate_up_proj_bias = Param::new(
                                    acquired
                                        .compact_binding("gate_up_proj_bias", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_blocks = Param::new(
                                    acquired
                                        .compact_binding("down_proj_blocks", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_scales = Param::new(
                                    acquired
                                        .compact_binding("down_proj_scales", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                bank.down_proj_bias = Param::new(
                                    acquired
                                        .compact_binding("down_proj_bias", stream)
                                        .map_err(|error| Exception::custom(error.to_string()))?,
                                );
                                expert_cache.record_compact_bank(
                                    pass,
                                    acquired.scratch_bytes(),
                                    started.elapsed(),
                                )?;
                                Ok(bank.forward(
                                    flat,
                                    acquired.compact_routes(),
                                    weights,
                                    stream,
                                )?)
                            },
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
                },
            )?);
        }
        Ok(layer.forward(hidden, mask.as_ref(), layer_cache, stream)?)
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.forward_layer(
                group,
                index,
                layer,
                hidden,
                cache,
                context,
                execution.stream(),
            );
        };
        self.layer_count(group)?;
        let layer_cache = &mut cache.layers[index];
        let offset = layer_cache.offset();
        let policy = self
            .attention_schedule
            .get(index)
            .expect("validated GPT-OSS layer index");
        let mask =
            resident::attention_mask(policy, context.sequence_length, offset, execution.stream())?;
        Ok(layer.forward_tensor_parallel(
            hidden,
            mask.as_ref(),
            layer_cache,
            tp_group,
            execution.stream(),
        )?)
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        _group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(self.lm_head.forward(&hidden, stream)?)
    }
    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(head) = &mut self.parallel_lm_head else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        head.forward(&hidden, execution)?.all_gather(execution)
    }
}

pub(crate) fn gpt_oss_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    gpt_oss_expert_catalog_cartesian(args, store, None)
}

/// Builds expert-granular GPT-OSS bindings under an optional TP layout.
///
/// Expert selection is resolved before the shared semantic TP selection so
/// native MXFP4 blocks, E8M0 scales, and biases remain one atomic cache unit.
pub(crate) fn gpt_oss_expert_catalog_cartesian(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gguf = store
            .source_keys()
            .contains(&format!("{prefix}.gate_proj.weight"));
        for expert in 0..args.num_local_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            let selected = |name: &str| {
                DerivedWeightRecipe::source(
                    format!("{prefix}.{name}"),
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                )
            };
            let recipes = if gguf {
                let hidden = args.hidden_size as usize;
                let intermediate = args.intermediate_size as usize;
                let stack_reshape =
                    |gate: &str, up: &str, shape: Vec<usize>| DerivedWeightRecipe::Reshape {
                        input: Box::new(DerivedWeightRecipe::Stack {
                            axis: 2,
                            inputs: vec![selected(gate), selected(up)],
                        }),
                        shape,
                    };
                vec![
                    (
                        "gate_up_proj_blocks",
                        DerivedWeightRecipe::View {
                            input: Box::new(stack_reshape(
                                "gate_proj.weight",
                                "up_proj.weight",
                                vec![1, 2 * intermediate, hidden / 8],
                            )),
                            dtype: RecipeDtype::U8,
                            shape: vec![1, 2 * intermediate, hidden / 32, 16],
                        },
                    ),
                    (
                        "gate_up_proj_scales",
                        stack_reshape(
                            "gate_proj.scales",
                            "up_proj.scales",
                            vec![1, 2 * intermediate, hidden / 32],
                        ),
                    ),
                    (
                        "gate_up_proj_bias",
                        stack_reshape("gate_proj.bias", "up_proj.bias", vec![1, 2 * intermediate]),
                    ),
                    (
                        "down_proj_blocks",
                        DerivedWeightRecipe::View {
                            input: Box::new(selected("down_proj.weight")),
                            dtype: RecipeDtype::U8,
                            shape: vec![1, hidden, intermediate / 32, 16],
                        },
                    ),
                    ("down_proj_scales", selected("down_proj.scales")),
                    ("down_proj_bias", selected("down_proj.bias")),
                ]
            } else {
                [
                    "gate_up_proj_blocks",
                    "gate_up_proj_scales",
                    "gate_up_proj_bias",
                    "down_proj_blocks",
                    "down_proj_scales",
                    "down_proj_bias",
                ]
                .into_iter()
                .map(|name| (name, selected(name)))
                .collect()
            };
            let mut planned = Vec::with_capacity(recipes.len());
            for (name, recipe) in recipes {
                let metadata = recipe.infer(store)?;
                planned.push(PlannedBinding {
                    target_name: name.into(),
                    expected_shape: metadata.shape().to_vec(),
                    expected_dtype: metadata.dtype().clone(),
                    recipe,
                });
            }
            bindings.extend(
                BindingPlan::new(planned)
                    .and_then(|plan| plan.build_bindings(store))
                    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?,
            );
            let bindings = match layout {
                Some(layout) => {
                    crate::backend::mlx::runtime::execution::layerwise::shard_layer_bindings(
                        bindings, &prefix, store, layout,
                    )?
                }
                None => bindings,
            };
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("GPT-OSS expert byte total overflowed".into())
                })
            })?;
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod neutral_runtime_tests {
    #[test]
    fn production_model_uses_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let start = source
            .find("pub struct GptOssLayerwiseModel")
            .expect("GPT-OSS production wrapper");
        let end = source
            .find("/// Generalized adapter for GPT-OSS")
            .expect("pipeline-only legacy adapter marker");
        let production = &source[start..end];
        assert!(production.contains("LayerwiseRuntime"));
        for legacy in [
            "LayerwiseModel<",
            ".adapter()",
            "load_layerwise_model(",
            "load_tensor_parallel_layerwise_model(",
        ] {
            assert!(
                !production.contains(legacy),
                "production GPT-OSS wrapper still references {legacy}"
            );
        }
    }
}

/// GPT-OSS token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, GptOssLayerwiseModel, Cache, S>;
