//! Bounded layer execution for the shared dense-Qwen decoder.

use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, ExpertCacheLoadOptions, ExpertIdentity, ExpertPass,
    LayerWeightResidency, LayeredArchitecture, LayeredForwardState, LayerwiseRuntime,
    NonExpertWeightResidency, ParallelLayeredArchitecture, RuntimeState, StateError, StateLayout,
    StaticUnitBindings, WeightResidency,
};

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::{
    MemberSharding, OffloadUnit, ParameterGroupSpec, ParameterRole, WeightBinding,
};

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::{DecoderConfig, Experts, FeedForward, TransformerBlock};
use crate::composition::mlx_architectures::qwen::dense as resident;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        attention::AttentionInput,
        linear::{
            build_unloaded_maybe_quantized_lm_head_with_quantization,
            project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
        },
    },
    backend::mlx::nn::{
        parallel::{
            planned_kv_head_layout, register_gqa_projection_group,
            register_swiglu_projection_group, GqaProjectionNames, SwiGluProjectionNames,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        shared::{MlxBackend, MlxParameterTree},
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes,
        build_module_binding_plan_with_recipes_excluding, ModuleBindingPlan,
    },
    backend::mlx::runtime::checkpoint::store::{
        open_gguf_checkpoint_source, TensorSelection, WeightStoreBackend,
    },
    backend::mlx::runtime::checkpoint::{
        binding_plan::{BindingPlan, PlannedBinding},
        quantization::should_quantize_on_load,
        recipe::DerivedWeightRecipe,
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_replicated_module,
        ParallelPlanBuilder,
    },
    backend::mlx::runtime::execution::{
        generic::{
            prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            MlxUnitFactory,
        },
        layerwise::{
            open_safetensors_weight_store, quantize_module_store_with_bindings,
            shard_layer_bindings,
        },
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheReport, ExpertCatalogEntry, ExpertRouteBatch,
    },
    core::cache::{
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity,
        PromptCacheOptions, PromptCacheTopology,
    },
};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "dense_qwen.static.embedding";
const NORM_UNIT: &str = "dense_qwen.static.norm";
const HEAD_UNIT: &str = "dense_qwen.static.output";

/// Architecture-owned KV cache accepted by dense-Qwen execution.
#[derive(Clone)]
pub enum DenseQwenLayerwiseCache {
    /// Append-only device KV caches.
    Concat {
        layout: StateLayout,
        caches: Vec<Option<ConcatKeyValueCache>>,
    },
    /// Sliding device KV caches used by expert-parallel execution.
    Sliding {
        layout: StateLayout,
        caches: Vec<Option<SlidingKeyValueCache>>,
    },
    /// Paged KV caches used by expert-parallel execution.
    Paged {
        layout: StateLayout,
        caches: Vec<Option<PagedKeyValueCache>>,
    },
}

impl DenseQwenLayerwiseCache {
    pub(crate) fn concat(layout: StateLayout, caches: Vec<Option<ConcatKeyValueCache>>) -> Self {
        Self::Concat { layout, caches }
    }

    pub(crate) fn sliding(layout: StateLayout, caches: Vec<Option<SlidingKeyValueCache>>) -> Self {
        Self::Sliding { layout, caches }
    }

    pub(crate) fn paged(layout: StateLayout, caches: Vec<Option<PagedKeyValueCache>>) -> Self {
        Self::Paged { layout, caches }
    }
}

impl RuntimeState<crate::backend::mlx::nn::shared::MlxBackend> for DenseQwenLayerwiseCache {
    type RetainedValues<'a> = std::vec::IntoIter<&'a Array>;

    fn layout(&self) -> &StateLayout {
        match self {
            Self::Concat { layout, .. }
            | Self::Sliding { layout, .. }
            | Self::Paged { layout, .. } => layout,
        }
    }

    fn retained_values(
        &self,
        _ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        if address.group() != 0 {
            return Err(StateError::UnknownLayer {
                layer: address.group(),
                count: 1,
            });
        }
        let index = address.index();
        let count = self.layout().len();
        if index >= count {
            return Err(StateError::UnknownLayer {
                layer: index,
                count,
            });
        }
        Ok(match self {
            Self::Concat { caches, .. } => retained_cache_arrays(caches, index),
            Self::Sliding { caches, .. } => retained_cache_arrays(caches, index),
            Self::Paged { caches, .. } => retained_cache_arrays(caches, index),
        }
        .into_iter())
    }
}

type DenseQwenUnit = MlxParameterTree<TransformerBlock>;
type DenseQwenStatic = MlxParameterTree<DenseQwenStaticModules>;
type DenseQwenResidentRuntime = LayerwiseRuntime<
    DenseQwenArchitecture,
    MlxBackend,
    DenseQwenLayerwiseCache,
    MlxResidentPolicy<DenseQwenUnit>,
>;
type DenseQwenBoundedRuntime = LayerwiseRuntime<
    DenseQwenArchitecture,
    MlxBackend,
    DenseQwenLayerwiseCache,
    MlxLayerwisePolicy<DenseQwenUnit, DenseQwenUnitFactory>,
>;

enum DenseQwenRuntime {
    Resident(DenseQwenResidentRuntime),
    Layerwise(DenseQwenBoundedRuntime),
}

struct DenseQwenExecution {
    runtime: DenseQwenRuntime,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DenseQwenExecution {
    fn architecture(&self) -> &DenseQwenArchitecture {
        match &self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime.architecture(),
            DenseQwenRuntime::Layerwise(runtime) => runtime.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut DenseQwenArchitecture {
        match &mut self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime.architecture_mut(),
            DenseQwenRuntime::Layerwise(runtime) => runtime.architecture_mut(),
        }
    }

    fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime.policy().checkpoint_store(),
            DenseQwenRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store(),
        }
    }

    fn checkpoint_store_arc(&self) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            DenseQwenRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime.policy().residency_report(),
            DenseQwenRuntime::Layerwise(runtime) => runtime.policy().residency_report(),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.runtime {
            DenseQwenRuntime::Resident(_) => Ok(None),
            DenseQwenRuntime::Layerwise(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn forward(
        &mut self,
        input: DenseQwenAdapterInput<'_>,
        cache: &mut DenseQwenLayerwiseCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            DenseQwenRuntime::Layerwise(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    fn forward_parallel(
        &mut self,
        input: DenseQwenAdapterInput<'_>,
        cache: &mut DenseQwenLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            DenseQwenRuntime::Layerwise(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DenseQwenReplicatedStatic {
    #[param]
    embedding: MaybeQuantized<nn::Embedding>,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct DenseQwenParallelStatic {
    #[param]
    embedding: VocabParallelEmbedding,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: Option<VocabParallelLmHead>,
}

#[derive(Debug, Clone)]
enum DenseQwenStaticModules {
    Replicated(DenseQwenReplicatedStatic),
    Parallel(DenseQwenParallelStatic),
}

macro_rules! dense_qwen_static_parameters {
    ($self:ident, $method:ident $(, $arg:expr)?) => {
        match $self {
            DenseQwenStaticModules::Replicated(module) => module.$method($($arg)?),
            DenseQwenStaticModules::Parallel(module) => module.$method($($arg)?),
        }
    };
}

impl ModuleParameters for DenseQwenStaticModules {
    fn num_parameters(&self) -> usize {
        dense_qwen_static_parameters!(self, num_parameters)
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        dense_qwen_static_parameters!(self, parameters)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        dense_qwen_static_parameters!(self, parameters_mut)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        dense_qwen_static_parameters!(self, trainable_parameters)
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        dense_qwen_static_parameters!(self, freeze_parameters, recursive)
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        dense_qwen_static_parameters!(self, unfreeze_parameters, recursive)
    }
    fn all_frozen(&self) -> Option<bool> {
        dense_qwen_static_parameters!(self, all_frozen)
    }
    fn any_frozen(&self) -> Option<bool> {
        dense_qwen_static_parameters!(self, any_frozen)
    }
}

impl DenseQwenStaticModules {
    fn replicated(args: &DecoderConfig, stream: &Stream) -> Result<Self, Error> {
        Ok(Self::Replicated(DenseQwenReplicatedStatic {
            embedding: unloaded_maybe_quantized_embedding(
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
            lm_head: if args.tie_word_embeddings {
                None
            } else {
                Some(build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )?)
            },
        }))
    }
}

#[derive(Clone)]
struct DenseQwenUnitFactory {
    args: DecoderConfig,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    sparse_experts: bool,
}

impl MlxUnitFactory<DenseQwenUnit> for DenseQwenUnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<DenseQwenUnit, Error> {
        build_dense_qwen_unit(
            &self.args,
            index,
            self.parallel_layout.as_deref(),
            self.sparse_experts,
            stream,
        )
    }
}

struct DenseQwenArchitecture {
    args: DecoderConfig,
    static_modules: DenseQwenStatic,
    sparse_experts: bool,
    expert_cache: Option<ExpertCache>,
    parallel_kv_heads: Option<Vec<i32>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DenseQwenArchitecture {
    fn new(args: DecoderConfig, sparse_experts: bool, stream: &Stream) -> Result<Self, Error> {
        Ok(Self {
            static_modules: MlxParameterTree::new(
                DenseQwenStaticModules::replicated(&args, stream)?,
                "",
            )
            .map_err(|error| Error::Parallel(error.to_string()))?,
            args,
            sparse_experts,
            expert_cache: None,
            parallel_kv_heads: None,
            parallel_topology: None,
        })
    }

    fn validate_cache(&self, cache: &mut DenseQwenLayerwiseCache) -> Result<(), Error> {
        let expected = self.args.num_hidden_layers as usize;
        match cache {
            DenseQwenLayerwiseCache::Concat { caches, .. } => {
                if caches.is_empty() {
                    *caches = new_dense_qwen_concat_cache(&self.args);
                }
                validate_dense_qwen_cache(caches, expected)
            }
            DenseQwenLayerwiseCache::Sliding { caches, .. } => {
                validate_dense_qwen_cache(caches, expected)
            }
            DenseQwenLayerwiseCache::Paged { caches, .. } => {
                validate_dense_qwen_cache(caches, expected)
            }
        }
    }
}

/// Host-backed dense-Qwen causal LM.
pub struct LayerwiseDecoder {
    execution: DenseQwenExecution,
}

impl LayerwiseDecoder {
    /// Returns the normalized decoder configuration.
    pub fn args(&self) -> &DecoderConfig {
        &self.execution.architecture().args
    }

    pub(crate) fn cache_layout(&self) -> Result<StateLayout, Error> {
        resident::state_layout(
            self.args(),
            self.execution.architecture().parallel_kv_heads.as_deref(),
        )
        .map_err(Into::into)
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.topology = Some(topology);
        self.execution.architecture_mut().parallel_topology = Some(topology);
    }

    /// Creates one standard device-resident KV cache per decoder block.
    pub fn new_cache(&self) -> Vec<Option<ConcatKeyValueCache>> {
        self.args()
            .attention_schedule
            .iter()
            .map(|policy| {
                Some(match policy.window() {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                        i32::try_from(window.get())
                            .expect("validated dense-Qwen attention window fits i32"),
                    ),
                    None => ConcatKeyValueCache::new(),
                })
            })
            .collect()
    }

    /// Creates device-resident or globally budgeted paged KV state without
    /// changing decoder-weight residency.
    pub fn new_cache_with_options(
        &self,
        policy: CacheResidencyPolicy,
    ) -> Result<DenseQwenLayerwiseCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(DenseQwenLayerwiseCache::concat(
                self.cache_layout()?,
                self.new_cache(),
            )),
            CacheResidencyPolicy::Paged(options) => {
                let manager =
                    crate::backend::mlx::runtime::cache::residency::CacheResidencyManager::new(
                        options,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let caches = resident::new_paged_cache_with_manager(
                    self.args(),
                    manager,
                    self.execution
                        .topology
                        .map(crate::backend::mlx::cache::prompt_cache_topology)
                        .and_then(|topology| topology.cache_rank_identity()),
                )?;
                Ok(DenseQwenLayerwiseCache::paged(self.cache_layout()?, caches))
            }
        }
    }

    /// Returns aggregate live KV paging observations, if paging is enabled.
    pub fn cache_residency_report(
        &self,
        cache: &DenseQwenLayerwiseCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            DenseQwenLayerwiseCache::Paged { caches, .. } => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            DenseQwenLayerwiseCache::Concat { .. } | DenseQwenLayerwiseCache::Sliding { .. } => {
                Ok(None)
            }
        }
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.execution.parallel_info.as_ref()
    }

    /// Returns generalized parameter-residency and memory metadata.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.execution.metadata
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    /// Returns the architecture identity used to validate persisted prompt caches.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        dense_qwen_prompt_cache_identity(
            self.args(),
            self.execution.topology,
            self.execution.architecture().parallel_kv_heads.as_deref(),
        )
    }

    /// Persists a compatible standard prefix cache.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let mut owned =
            DenseQwenLayerwiseCache::concat(self.cache_layout()?, std::mem::take(cache));
        let result = match &mut owned {
            DenseQwenLayerwiseCache::Concat { caches, .. } => resident::save_prompt_cache(
                self.args(),
                caches,
                destination,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            )
            .map_err(Into::into),
            _ => unreachable!(),
        };
        let DenseQwenLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen prompt-cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Restores a compatible standard prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Vec<Option<ConcatKeyValueCache>>, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        let (cache, manifest) = resident::load_prompt_cache_with_identity(
            self.args(),
            directory,
            expected,
            prefix_token_ids,
            &identity,
            stream,
        )?;
        let _ = options;
        Ok((cache, manifest))
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns sparse expert-cache telemetry when that residency mode is active.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .architecture()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Runs a rank-local tensor-parallel forward pass through the generalized
    /// execution-group engine.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_parallel(
            DenseQwenAdapterInput { inputs, mask },
            cache,
            group,
            stream,
        )
    }

    /// Runs Qwen2/Qwen2.5 or Qwen3 with a standard KV cache.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned =
            DenseQwenLayerwiseCache::concat(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.execution
                .forward(DenseQwenAdapterInput { inputs, mask }, &mut owned, stream);
        let DenseQwenLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs Qwen2/Qwen2.5 or Qwen3 with an explicitly sliding KV cache.
    pub fn forward_sliding(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<SlidingKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned =
            DenseQwenLayerwiseCache::sliding(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.execution
                .forward(DenseQwenAdapterInput { inputs, mask }, &mut owned, stream);
        let DenseQwenLayerwiseCache::Sliding { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen sliding cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs dense Qwen through the canonical observer contract.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned =
            DenseQwenLayerwiseCache::concat(self.cache_layout()?, std::mem::take(cache));
        let result = self.forward_cache_with_observer(inputs, mask, &mut owned, stream, observer);
        let DenseQwenLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs paged dense Qwen through the canonical observer contract.
    pub fn forward_paged_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::paged(self.cache_layout()?, std::mem::take(cache));
        let result = self.forward_cache_with_observer(inputs, mask, &mut owned, stream, observer);
        let DenseQwenLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Runs Qwen2/Qwen2.5 or Qwen3 with a block-addressable paged KV cache.
    pub fn forward_paged(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned = DenseQwenLayerwiseCache::paged(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.execution
                .forward(DenseQwenAdapterInput { inputs, mask }, &mut owned, stream);
        let DenseQwenLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    fn forward_cache_with_observer(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let hook = |_architecture: &mut DenseQwenArchitecture,
                    _group: usize,
                    index: usize,
                    layer: &mut DenseQwenUnit,
                    hidden: &Array,
                    cache: &mut DenseQwenLayerwiseCache,
                    context: &mut DenseQwenForwardContext,
                    stream: &Stream| {
            let prefix = format!("model.layers.{index}");
            match cache {
                DenseQwenLayerwiseCache::Concat { caches, .. } => Ok(layer.forward_with_observer(
                    AttentionInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(caches[index].as_mut().expect("validated dense-Qwen cache")),
                    },
                    stream,
                    &prefix,
                    observer,
                )?),
                DenseQwenLayerwiseCache::Sliding { caches, .. } => Ok(layer
                    .forward_with_observer(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: Some(
                                caches[index].as_mut().expect("validated dense-Qwen cache"),
                            ),
                        },
                        stream,
                        &prefix,
                        observer,
                    )?),
                DenseQwenLayerwiseCache::Paged { caches, .. } => Ok(layer.forward_with_observer(
                    AttentionInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(caches[index].as_mut().expect("validated dense-Qwen cache")),
                    },
                    stream,
                    &prefix,
                    observer,
                )?),
            }
        };
        match &mut self.execution.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime
                .forward_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            DenseQwenRuntime::Layerwise(runtime) => runtime
                .forward_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned =
            DenseQwenLayerwiseCache::concat(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_with_sliding_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<SlidingKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned =
            DenseQwenLayerwiseCache::sliding(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Sliding { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen sliding cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_with_paged_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::paged(self.cache_layout()?, std::mem::take(cache));
        let result =
            self.forward_with_expert_executor_cache(inputs, mask, &mut owned, &mut execute, stream);
        let DenseQwenLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    fn forward_with_expert_executor_cache<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let hook = |_architecture: &mut DenseQwenArchitecture,
                    _group: usize,
                    index: usize,
                    layer: &mut DenseQwenUnit,
                    hidden: &Array,
                    cache: &mut DenseQwenLayerwiseCache,
                    context: &mut DenseQwenForwardContext,
                    stream: &Stream| match cache {
            DenseQwenLayerwiseCache::Concat { caches: cache, .. } => forward_sparse_with_executor(
                layer,
                hidden,
                cache[index].as_mut(),
                context,
                index,
                execute,
                stream,
            ),
            DenseQwenLayerwiseCache::Sliding { caches: cache, .. } => forward_sparse_with_executor(
                layer,
                hidden,
                cache[index].as_mut(),
                context,
                index,
                execute,
                stream,
            ),
            DenseQwenLayerwiseCache::Paged { caches: cache, .. } => forward_sparse_with_executor(
                layer,
                hidden,
                cache[index].as_mut(),
                context,
                index,
                execute,
                stream,
            ),
        };
        match &mut self.execution.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime
                .forward_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            DenseQwenRuntime::Layerwise(runtime) => runtime
                .forward_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    /// Runs the shared tensor-parallel model while delegating routed experts
    /// to a topology-scoped expert-parallel executor.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned =
            DenseQwenLayerwiseCache::concat(self.cache_layout()?, std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen concat cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_tensor_expert_parallel_sliding<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<SlidingKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned =
            DenseQwenLayerwiseCache::sliding(self.cache_layout()?, std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Sliding { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen sliding cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    pub(crate) fn forward_tensor_expert_parallel_paged<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        tensor_group: &safemlx::distributed::Group,
        execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let mut owned = DenseQwenLayerwiseCache::paged(self.cache_layout()?, std::mem::take(cache));
        let result = self.forward_tensor_expert_parallel_cache(
            inputs,
            mask,
            &mut owned,
            tensor_group,
            execute,
            stream,
        );
        let DenseQwenLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("dense-Qwen paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    fn forward_tensor_expert_parallel_cache<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut DenseQwenLayerwiseCache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let hook = |_architecture: &mut DenseQwenArchitecture,
                    _group: usize,
                    index: usize,
                    layer: &mut DenseQwenUnit,
                    hidden: &Array,
                    cache: &mut DenseQwenLayerwiseCache,
                    context: &mut DenseQwenForwardContext,
                    tp_group: &safemlx::distributed::Group,
                    stream: &Stream| {
            match cache {
                DenseQwenLayerwiseCache::Concat { caches: cache, .. } => {
                    forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        stream,
                    )
                }
                DenseQwenLayerwiseCache::Sliding { caches: cache, .. } => {
                    forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        stream,
                    )
                }
                DenseQwenLayerwiseCache::Paged { caches: cache, .. } => {
                    forward_sparse_tp_with_executor(
                        layer,
                        hidden,
                        cache[index].as_mut(),
                        context,
                        index,
                        tp_group,
                        &mut execute,
                        stream,
                    )
                }
            }
        };
        match &mut self.execution.runtime {
            DenseQwenRuntime::Resident(runtime) => runtime
                .forward_parallel_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    tensor_group,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
            DenseQwenRuntime::Layerwise(runtime) => runtime
                .forward_parallel_with_unit_executor(
                    DenseQwenAdapterInput { inputs, mask },
                    cache,
                    tensor_group,
                    stream,
                    hook,
                )
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    /// Clears temporary device decoder copies.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution.runtime {
            DenseQwenRuntime::Resident(runtime) => {
                runtime.policy().clear_device_group("text_decoder")
            }
            DenseQwenRuntime::Layerwise(runtime) => {
                runtime.policy().clear_device_group("text_decoder")
            }
        }
    }
}

fn forward_sparse_with_executor<C, F>(
    layer: &mut TransformerBlock,
    hidden: &Array,
    cache: Option<&mut C>,
    context: &DenseQwenForwardContext,
    index: usize,
    execute: &mut F,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache,
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
{
    Ok(layer.forward_sparse_experts(
        AttentionInput {
            x: hidden,
            mask: context.mask.as_ref(),
            cache,
        },
        stream,
        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
    )?)
}

#[allow(clippy::too_many_arguments)]
fn forward_sparse_tp_with_executor<C, F>(
    layer: &mut TransformerBlock,
    hidden: &Array,
    cache: Option<&mut C>,
    context: &DenseQwenForwardContext,
    index: usize,
    tensor_group: &safemlx::distributed::Group,
    execute: &mut F,
    stream: &Stream,
) -> Result<Array, Error>
where
    C: KeyValueCache,
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
{
    Ok(layer.forward_sparse_experts_tensor_parallel(
        AttentionInput {
            x: hidden,
            mask: context.mask.as_ref(),
            cache,
        },
        tensor_group,
        stream,
        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
    )?)
}

fn new_dense_qwen_concat_cache(args: &DecoderConfig) -> Vec<Option<ConcatKeyValueCache>> {
    args.attention_schedule
        .iter()
        .map(|policy| {
            Some(match policy.window() {
                Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                    i32::try_from(window.get())
                        .expect("validated dense-Qwen attention window fits i32"),
                ),
                None => ConcatKeyValueCache::new(),
            })
        })
        .collect()
}

fn build_dense_qwen_unit(
    args: &DecoderConfig,
    index: usize,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    sparse_experts: bool,
    stream: &Stream,
) -> Result<DenseQwenUnit, Error> {
    let mut local = args.clone();
    if let Some(layout) = layout {
        let prefix = format!("model.layers.{index}");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let query = tensor("self_attn.q_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let key = tensor("self_attn.k_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        local.num_attention_heads = i32::try_from(query.local_shape()[0])
            .map_err(|_| Error::Parallel("Qwen local query width exceeds i32".into()))?
            / local.head_dim;
        local.num_key_value_heads = i32::try_from(key.local_shape()[0])
            .map_err(|_| Error::Parallel("Qwen local key width exceeds i32".into()))?
            / local.head_dim;
        if local.is_moe() {
            let experts = layout
                .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                .ok_or_else(|| {
                    Error::Parallel(format!("missing TP layout for {prefix} experts"))
                })?;
            local.moe_intermediate_size = i32::try_from(experts.local_shape()[1] / 2)
                .map_err(|_| Error::Parallel("Qwen local expert width exceeds i32".into()))?;
        } else {
            let gate = tensor("mlp.gate_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLP")))?;
            local.intermediate_size = i32::try_from(gate.local_shape()[0])
                .map_err(|_| Error::Parallel("Qwen local MLP width exceeds i32".into()))?;
        }
    }
    let mut layer = TransformerBlock::new_for_layer(&local, index as i32, stream)?;
    if sparse_experts {
        replace_qwen_expert_bank(
            &mut layer,
            args,
            index,
            0,
            layout.map(|_| local.moe_intermediate_size),
            stream,
        )?;
    }
    MlxParameterTree::new_filtered(layer, "", |name| {
        !sparse_experts || !name.starts_with("mlp.experts.")
    })
    .map_err(|error| Error::Parallel(error.to_string()))
}

impl LayeredArchitecture<MlxBackend, DenseQwenLayerwiseCache> for DenseQwenArchitecture {
    type Input<'a> = DenseQwenAdapterInput<'a>;
    type StaticModules = DenseQwenStatic;
    type Unit = DenseQwenUnit;
    type ForwardContext = DenseQwenForwardContext;
    type RetainedContextValues<'a> = std::option::Iter<'a, Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no execution group {group}"
            )))
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no decoder unit {index}"
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
        build_dense_qwen_unit(&self.args, index, None, self.sparse_experts, stream)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut DenseQwenLayerwiseCache,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let DenseQwenStaticModules::Replicated(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "dense-Qwen replicated execution received parallel static modules".into(),
            ));
        };
        let hidden = modules.embedding.forward(input.inputs, stream)?;
        let mask = match cache {
            DenseQwenLayerwiseCache::Concat { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Sliding { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Paged { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: DenseQwenForwardContext { mask },
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        _cache: &mut DenseQwenLayerwiseCache,
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen group {group} received {} dependencies",
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
        cache: &mut DenseQwenLayerwiseCache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group)?;
        match cache {
            DenseQwenLayerwiseCache::Concat { caches, .. } => forward_dense_qwen_unit(
                &self.args,
                self.expert_cache.as_ref(),
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                forward,
                stream,
            ),
            DenseQwenLayerwiseCache::Sliding { caches, .. } => forward_dense_qwen_unit(
                &self.args,
                self.expert_cache.as_ref(),
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                forward,
                stream,
            ),
            DenseQwenLayerwiseCache::Paged { caches, .. } => forward_dense_qwen_unit(
                &self.args,
                self.expert_cache.as_ref(),
                index,
                layer,
                hidden,
                caches[index].as_mut().expect("validated dense-Qwen cache"),
                forward,
                stream,
            ),
        }
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        _cache: &mut DenseQwenLayerwiseCache,
        _forward: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let DenseQwenStaticModules::Replicated(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "dense-Qwen replicated execution received parallel static modules".into(),
            ));
        };
        let hidden = modules.norm.forward(hidden, stream)?;
        Ok(project_logits_maybe_quantized(
            &mut modules.lm_head,
            &mut modules.embedding,
            &hidden,
            stream,
        )?)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl ParallelLayeredArchitecture<MlxBackend, DenseQwenLayerwiseCache> for DenseQwenArchitecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut DenseQwenLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("dense-Qwen parallel topology was not configured".into())
        })?;
        let DenseQwenStaticModules::Parallel(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "dense-Qwen parallel execution received replicated static modules".into(),
            ));
        };
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        let hidden = modules.embedding.forward(input.inputs, &execution)?;
        let mask = match cache {
            DenseQwenLayerwiseCache::Concat { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Sliding { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
            DenseQwenLayerwiseCache::Paged { caches, .. } => {
                dense_qwen_attention_mask(&hidden, input.mask, caches, stream)?
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: DenseQwenForwardContext { mask },
        })
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut DenseQwenLayerwiseCache,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group_index)?;
        match cache {
            DenseQwenLayerwiseCache::Concat { caches, .. } => Ok(layer.forward_tensor_parallel(
                hidden,
                forward.mask.as_ref(),
                caches[index].as_mut(),
                group,
                stream,
            )?),
            DenseQwenLayerwiseCache::Sliding { caches, .. } => Ok(layer.forward_tensor_parallel(
                hidden,
                forward.mask.as_ref(),
                caches[index].as_mut(),
                group,
                stream,
            )?),
            DenseQwenLayerwiseCache::Paged { caches, .. } => Ok(layer.forward_tensor_parallel(
                hidden,
                forward.mask.as_ref(),
                caches[index].as_mut(),
                group,
                stream,
            )?),
        }
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _cache: &mut DenseQwenLayerwiseCache,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("dense-Qwen parallel topology was not configured".into())
        })?;
        let DenseQwenStaticModules::Parallel(modules) = &mut *self.static_modules else {
            return Err(Error::Parallel(
                "dense-Qwen parallel execution received replicated static modules".into(),
            ));
        };
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        let hidden = modules.norm.forward(hidden, stream)?;
        let logits = match &mut modules.lm_head {
            Some(head) => head.forward(&hidden, &execution)?,
            None => modules.embedding.project_logits(&hidden, &execution)?,
        };
        logits.all_gather(&execution)
    }
}

fn forward_dense_qwen_unit<C: KeyValueCache>(
    args: &DecoderConfig,
    expert_cache: Option<&ExpertCache>,
    index: usize,
    layer: &mut TransformerBlock,
    hidden: &Array,
    cache: &mut C,
    context: &DenseQwenForwardContext,
    stream: &Stream,
) -> Result<Array, Error> {
    if let Some(expert_cache) = expert_cache {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        return Ok(layer.forward_sparse_experts(
            AttentionInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(cache),
            },
            stream,
            |flat, indices, weights, stream| {
                execute_cached_qwen_experts(
                    expert_cache,
                    args,
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
    Ok(layer.forward(
        AttentionInput {
            x: hidden,
            mask: context.mask.as_ref(),
            cache: Some(cache),
        },
        stream,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_qwen_experts(
    expert_cache: &ExpertCache,
    args: &DecoderConfig,
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
                if acquired.is_empty() {
                    return Err(ExpertCacheError::EmptyRoutedBank {
                        architecture: "Qwen3",
                    });
                }
                let started = Instant::now();
                let prefix = format!("model.layers.{index}.mlp.experts");
                let load_time = expert_cache.weight_quantization();
                let mut bank = resident::Experts::new(
                    acquired.identities().len() as i32,
                    args.hidden_size,
                    args.moe_intermediate_size,
                    load_time.or_else(|| {
                        args.weight_quantization_for(&format!("{prefix}.gate_up_proj"))
                    }),
                    load_time
                        .or_else(|| args.weight_quantization_for(&format!("{prefix}.down_proj"))),
                    stream,
                )?;
                bank.gate_up_proj = Param::new(
                    acquired
                        .compact_binding("gate_up_proj", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.gate_up_proj_scales = Param::new(
                    acquired
                        .optional_compact_binding("gate_up_proj_scales", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.gate_up_proj_biases = Param::new(
                    acquired
                        .optional_compact_binding("gate_up_proj_biases", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj = Param::new(
                    acquired
                        .compact_binding("down_proj", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj_scales = Param::new(
                    acquired
                        .optional_compact_binding("down_proj_scales", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.down_proj_biases = Param::new(
                    acquired
                        .optional_compact_binding("down_proj_biases", stream)
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

impl CausalModel<Vec<Option<ConcatKeyValueCache>>> for LayerwiseDecoder {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward(&tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(input_tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

impl CausalModel<Vec<Option<PagedKeyValueCache>>> for LayerwiseDecoder {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        self.forward_paged(&tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward_paged(input_tokens, None, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

fn dense_qwen_execution_layout(args: &DecoderConfig) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["text_decoder"])?;
    ExecutionUnitLayout::new(&graph, [args.num_hidden_layers as usize])
        .map_err(|error| Error::Parallel(error.to_string()))
}

fn dense_qwen_prompt_cache_identity(
    args: &DecoderConfig,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
    parallel_kv_heads: Option<&[i32]>,
) -> Result<PromptCacheModelIdentity, Error> {
    let layer_count = args.num_hidden_layers as usize;
    let kv_heads = match topology {
        Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
            parallel_kv_heads
                .ok_or_else(|| {
                    Error::Parallel(
                        "dense-Qwen parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?
                .to_vec()
        }
        _ => vec![args.num_key_value_heads; layer_count],
    };
    Ok(PromptCacheModelIdentity {
        model_family: "dense_qwen".into(),
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
        layer_layout: resident::prompt_cache_layer_layout_with_kv_heads(args, &kv_heads)?,
    })
}

fn name_dense_qwen_binding(binding: WeightBinding, name: String) -> Result<WeightBinding, Error> {
    binding.with_name(name).map_err(Into::into)
}

fn dense_qwen_static_bindings(
    modules: &DenseQwenStaticModules,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let DenseQwenStaticModules::Replicated(modules) = modules else {
        return Err(Error::Parallel(
            "dense-Qwen global static modules are sharded".into(),
        ));
    };
    let mut bindings = build_module_binding_plan_with_recipes(
        &modules.embedding,
        "model.embed_tokens",
        store,
        BTreeMap::new(),
    )?
    .build_bindings(store)?
    .into_iter()
    .map(|binding| {
        let name = format!("embedding.{}", binding.name());
        name_dense_qwen_binding(binding, name)
    })
    .collect::<Result<Vec<_>, _>>()?;
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &modules.norm,
            "model.norm",
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?
        .into_iter()
        .map(|binding| {
            let name = format!("norm.{}", binding.name());
            name_dense_qwen_binding(binding, name)
        })
        .collect::<Result<Vec<_>, _>>()?,
    );
    if let Some(head) = &modules.lm_head {
        bindings.extend(
            build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
                .build_bindings(store)?
                .into_iter()
                .map(|binding| {
                    let name = format!("lm_head.{}", binding.name());
                    name_dense_qwen_binding(binding, name)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(bindings)
}

fn dense_qwen_unit_bindings(
    args: &DecoderConfig,
    index: usize,
    layer: &TransformerBlock,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    sparse_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    qwen_text_layer_bindings(
        layer,
        args,
        &format!("model.layers.{index}"),
        store,
        sparse_experts,
    )
}

fn resolve_dense_qwen_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &DecoderConfig,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend != WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = super::checkpoint::safetensors_plan(args).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "dense-Qwen checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_dense_qwen_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    source_args: &DecoderConfig,
    sparse_experts: bool,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        DecoderConfig,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let mut target_args = source_args.clone();
    target_args.quantization = Some(quantization);
    target_args.quantization_config = None;
    target_args.quantized_weight_configs = None;
    let source_static = DenseQwenStaticModules::replicated(source_args, stream)?;
    let target_static = DenseQwenStaticModules::replicated(&target_args, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| {
            TransformerBlock::new_for_layer(&source_units, index as i32, stream).map_err(Into::into)
        },
        move |index, stream| {
            TransformerBlock::new_for_layer(&target_units, index as i32, stream).map_err(Into::into)
        },
        source_args.num_hidden_layers as usize,
        quantization,
        stream,
        |modules, store| dense_qwen_static_bindings(modules, store),
        move |index, layer, store| {
            dense_qwen_unit_bindings(&binding_args, index, layer, store, sparse_experts)
        },
    )?;
    Ok((store, target_args, report))
}

fn load_dense_qwen_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: DecoderConfig,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DenseQwenExecution, Error> {
    let store = resolve_dense_qwen_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_dense_qwen_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut architecture = DenseQwenArchitecture::new(args.clone(), sparse_experts, stream)?;
    let factory = DenseQwenUnitFactory {
        args: args.clone(),
        parallel_layout: None,
        sparse_experts,
    };
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        dense_qwen_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        |modules, store| dense_qwen_static_bindings(&**modules, store),
        move |index, unit, store, _| {
            dense_qwen_unit_bindings(&binding_args, index, &unit, store, sparse_experts)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    metadata.set_materialization(materialization);
    let runtime = if options.is_fully_resident() {
        DenseQwenRuntime::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DenseQwenRuntime::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(DenseQwenExecution {
        runtime,
        metadata,
        parallel_info: None,
        topology: None,
    })
}

fn register_dense_qwen_parallel_parameters(
    planner: &mut ParallelPlanBuilder,
    args: &DecoderConfig,
    stream: &Stream,
) -> Result<(), Error> {
    let DenseQwenStaticModules::Replicated(modules) =
        DenseQwenStaticModules::replicated(args, stream)?
    else {
        unreachable!()
    };
    planner.register(
        crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
            &modules.embedding,
            "model.embed_tokens",
            args.vocab_size as usize,
            args.hidden_size,
            false,
        )?,
    )?;
    crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
        planner,
        &modules.norm,
        "model.norm",
    )?;
    if let Some(head) = &modules.lm_head {
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
                head,
                "lm_head",
                args.hidden_size,
                args.vocab_size as usize,
                false,
            )?,
        )?;
    }
    for index in 0..args.num_hidden_layers as usize {
        let layer = TransformerBlock::new_for_layer(args, index as i32, stream)?;
        register_qwen_layer_parallel_plan(planner, &layer, args, &format!("model.layers.{index}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_dense_qwen_parallel_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: DecoderConfig,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DenseQwenExecution, Error> {
    let store = resolve_dense_qwen_store(store, &args)?;
    let mut planner = build.planner();
    register_dense_qwen_parallel_parameters(&mut planner, &args, stream)?;
    let (_, local_layout) = planner.finish()?;
    let DenseQwenStaticModules::Replicated(global_modules) =
        DenseQwenStaticModules::replicated(&args, stream)?
    else {
        unreachable!()
    };
    let parallel_static = DenseQwenStaticModules::Parallel(DenseQwenParallelStatic {
        embedding: VocabParallelEmbedding::unloaded(
            args.vocab_size as usize,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            build,
            stream,
        )?,
        norm: global_modules.norm.clone(),
        lm_head: if global_modules.lm_head.is_some() {
            Some(VocabParallelLmHead::unloaded(
                args.hidden_size,
                args.vocab_size as usize,
                args.weight_quantization_for("lm_head.weight"),
                build,
                stream,
            )?)
        } else {
            None
        },
    });
    let mut architecture = DenseQwenArchitecture::new(args.clone(), sparse_experts, stream)?;
    architecture.static_modules = MlxParameterTree::new(parallel_static, "")
        .map_err(|error| Error::Parallel(error.to_string()))?;
    architecture.parallel_topology = Some(build.topology());
    architecture.parallel_kv_heads = Some(planned_kv_head_layout(
        &local_layout,
        args.num_hidden_layers as usize,
        args.head_dim,
        "model.layers",
    )?);

    let global_static = DenseQwenStaticModules::replicated(&args, stream)?;
    let static_bindings = dense_qwen_static_bindings(&global_static, store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    for index in 0..args.num_hidden_layers as usize {
        let global = TransformerBlock::new_for_layer(&args, index as i32, stream)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&dense_qwen_unit_bindings(
                &args,
                index,
                &global,
                store.as_ref(),
                sparse_experts,
            )?)?)
            .ok_or_else(|| {
                Error::Parallel("dense-Qwen global parameter bytes overflowed".into())
            })?;
    }
    let shared_layout = Arc::new(local_layout);
    let factory = DenseQwenUnitFactory {
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
        dense_qwen_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        move |_, store| shard_layer_bindings(static_bindings, "", store, &static_layout),
        move |index, _local, store, stream| {
            let global = TransformerBlock::new_for_layer(&binding_args, index as i32, stream)?;
            shard_layer_bindings(
                dense_qwen_unit_bindings(&binding_args, index, &global, store, sparse_experts)?,
                &format!("model.layers.{index}"),
                store,
                &unit_layout,
            )
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.weight_quantization());
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("dense-Qwen local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("dense-Qwen device parameter bytes overflowed".into()))?;
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
    let runtime = if options.is_fully_resident() {
        DenseQwenRuntime::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DenseQwenRuntime::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(DenseQwenExecution {
        runtime,
        metadata,
        parallel_info: Some(info),
        topology: Some(build.topology()),
    })
}

/// Loads Qwen2/Qwen2.5 or Qwen3 through the generalized residency engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::load_config(model_dir)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("dense Qwen", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    Ok(LayerwiseDecoder {
        execution: load_dense_qwen_with_store(
            store,
            args,
            options,
            quantize_on_load,
            false,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Qwen2/3 dense or MoE checkpoints through the generalized
/// tensor-parallel execution-group engine.
pub(crate) fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let architecture = match metadata.get("general.architecture") {
            Some(GgufMetadataValue::String(architecture)) => architecture.as_str(),
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF metadata key general.architecture has the wrong type".into(),
                ));
            }
            None => {
                return Err(Error::UnsupportedArchitecture(
                    "GGUF metadata is missing general.architecture".into(),
                ));
            }
        };
        return load_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            architecture,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = resident::load_config(model_dir)?;
    Ok(LayerwiseDecoder {
        execution: load_dense_qwen_parallel_with_store(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            args,
            options,
            build,
            false,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LayerwiseDecoder, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let is_moe = architecture == "qwen3moe";
    let (args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    let variant = match architecture {
        "qwen2" => super::checkpoint::GgufVariant::Qwen2,
        "qwen3moe" => super::checkpoint::GgufVariant::Qwen3Moe,
        _ => super::checkpoint::GgufVariant::Qwen3,
    };
    let gguf_plan =
        super::checkpoint::gguf_plan(&args, variant).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            move |name| resident::translate_gguf_weight_name(name, is_moe),
            options.max_mapped_shards(),
        )?);
    let execution = load_dense_qwen_parallel_with_store(
        store,
        args,
        options,
        build,
        false,
        stream,
        weights_stream,
    )?;
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

pub(crate) fn load_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LayerwiseDecoder, Vec<u32>), Error> {
    let is_moe = architecture == "qwen3moe";
    let (args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    let variant = match architecture {
        "qwen2" => super::checkpoint::GgufVariant::Qwen2,
        "qwen3moe" => super::checkpoint::GgufVariant::Qwen3Moe,
        _ => super::checkpoint::GgufVariant::Qwen3,
    };
    let gguf_plan =
        super::checkpoint::gguf_plan(&args, variant).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            |name| resident::translate_gguf_weight_name(name, is_moe),
            residency.max_mapped_shards(),
        )?);

    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_qwen3_gguf_sparse_with_store(
                store,
                args,
                expert_options,
                residency.layers(),
                quantization,
                stream,
                weights_stream,
            )?,
            eos_token_ids,
        ));
    }
    let execution = load_dense_qwen_with_store(
        store,
        args,
        residency.layers(),
        quantization,
        false,
        stream,
        weights_stream,
    )?;
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

/// Loads replicated Qwen3-MoE GGUF parameters for sparse expert-parallel
/// execution without materializing routed experts.
fn load_qwen3_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: DecoderConfig,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3 sparse-MoE GGUF checkpoint".into(),
        ));
    }
    let mut execution = load_dense_qwen_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantization,
        true,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = execution.checkpoint_store_arc();
    let entries = qwen3_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.architecture_mut().expert_cache = Some(match quantization {
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
    Ok(LayerwiseDecoder { execution })
}

/// Builds the streamed nonexpert Qwen3 execution base used by distributed EP.
pub(crate) fn load_qwen3_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: DecoderConfig,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "streamed sparse expert parallelism requires Qwen3 MoE".into(),
        ));
    }
    let execution = load_dense_qwen_with_store(
        store,
        args,
        non_expert.into(),
        None,
        true,
        stream,
        weights_stream,
    )?;
    Ok(LayerwiseDecoder { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_qwen3_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: DecoderConfig,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "combined tensor/expert parallelism requires Qwen3 MoE".into(),
        ));
    }
    let execution = load_dense_qwen_parallel_with_store(
        store,
        args,
        non_expert.into(),
        build,
        true,
        stream,
        weights_stream,
    )?;
    Ok(LayerwiseDecoder { execution })
}

/// Loads sparse Qwen3 with independently cached experts and bounded non-expert units.
pub fn load_qwen3_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::load_config(model_dir)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3 sparse-MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3 independent expert cache",
                args.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_dense_qwen_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantize_on_load,
        true,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = qwen3_expert_catalog(&args, store.as_ref())?;
    let cache = match quantize_on_load {
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
    };
    execution.architecture_mut().expert_cache = Some(cache);
    Ok(LayerwiseDecoder { execution })
}

/// Pipeline-staging legacy adapter retained until the old MLX runtime is deleted.
/// Dense-Qwen adapter sharing one complete-block execution path.
pub struct DenseQwenLayerwiseAdapter {
    args: DecoderConfig,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl DenseQwenLayerwiseAdapter {
    /// Creates metadata-only static dense-Qwen modules.
    pub fn new(args: DecoderConfig, stream: &Stream) -> Result<Self, Error> {
        let embedding = unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(build_unloaded_maybe_quantized_lm_head_with_quantization(
                args.hidden_size,
                args.vocab_size,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
            args,
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

    pub(crate) fn new_external_experts(
        args: DecoderConfig,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns normalized model arguments.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }
}

/// Attention mask shared by every temporary dense-Qwen decoder block.
pub struct DenseQwenForwardContext {
    mask: Option<Array>,
}

/// Dense-Qwen input consumed by the architecture-neutral layerwise engine.
pub struct DenseQwenAdapterInput<'a> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
}

fn replace_qwen_expert_bank(
    layer: &mut TransformerBlock,
    args: &DecoderConfig,
    layer_index: usize,
    experts: i32,
    intermediate: Option<i32>,
    stream: &Stream,
) -> Result<(), Error> {
    let FeedForward::Moe(moe) = &mut layer.mlp else {
        return Err(Error::Parallel(format!(
            "dense Qwen layer {layer_index} is not an MoE layer"
        )));
    };
    let prefix = format!("model.layers.{layer_index}.mlp.experts");
    moe.experts = Experts::new(
        experts,
        args.hidden_size,
        intermediate.unwrap_or(args.moe_intermediate_size),
        args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
        args.weight_quantization_for(&format!("{prefix}.down_proj")),
        stream,
    )?;
    Ok(())
}

pub(crate) fn register_qwen_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &DecoderConfig,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    register_gqa_projection_group(
        planner,
        &format!("{prefix}.self_attn"),
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
        attention.n_heads,
        attention.n_kv_heads,
        args.head_dim,
    )?;
    if let Some(norm) = &attention.q_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.q_norm"))?;
    }
    if let Some(norm) = &attention.k_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.k_norm"))?;
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    match &layer.mlp {
        resident::FeedForward::Dense(mlp) => {
            register_swiglu_projection_group(
                planner,
                &format!("{prefix}.mlp"),
                SwiGluProjectionNames {
                    gate: "gate_proj",
                    up: "up_proj",
                    down: "down_proj",
                },
                &mlp.gate_proj,
                &mlp.up_proj,
                &mlp.down_proj,
                args.intermediate_size,
            )?;
        }
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Qwen expert width exceeds usize".into()))?;
            let down_alignment =
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel("Qwen expert quantization group exceeds usize".into())
                        })
                    })?;
            let expert_units = aligned_partition_units(
                &format!("{prefix}.mlp.experts"),
                intermediate,
                1,
                down_alignment,
            )?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let mut members = vec![array_parameter_member(
                format!("{prefix}.mlp.experts.gate_up_proj"),
                experts.gate_up_proj.as_ref(),
                MemberSharding::PartitionedSegments {
                    axis: 1,
                    segments: segments.clone(),
                },
            )?];
            for (name, value) in [
                (
                    "gate_up_proj_scales",
                    experts.gate_up_proj_scales.as_ref().as_ref(),
                ),
                (
                    "gate_up_proj_biases",
                    experts.gate_up_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        },
                    )?);
                }
            }
            members.push(array_parameter_member(
                format!("{prefix}.mlp.experts.down_proj"),
                experts.down_proj.as_ref(),
                MemberSharding::Partitioned { axis: 2 },
            )?);
            for (name, value) in [
                (
                    "down_proj_scales",
                    experts.down_proj_scales.as_ref().as_ref(),
                ),
                (
                    "down_proj_biases",
                    experts.down_proj_biases.as_ref().as_ref(),
                ),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::partitioned(
                format!("{prefix}.mlp.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                expert_units,
                members,
            )?)?;
        }
    }
    register_replicated_module(
        planner,
        &layer.input_layernorm,
        &format!("{prefix}.input_layernorm"),
    )?;
    register_replicated_module(
        planner,
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm"),
    )
}

impl DenseQwenLayerwiseAdapter {}

impl DenseQwenLayerwiseAdapter {
    pub(crate) fn model_type(&self) -> &str {
        &self.args.model_type
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
        let mut units = Vec::new();
        if select(EMBEDDING_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    "model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    "model.norm",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            if let Some(head) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    build_module_binding_plan_with_recipes(
                        head,
                        "lm_head",
                        store,
                        BTreeMap::new(),
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no group {group}"
            )));
        }
        let index = i32::try_from(index).map_err(|_| {
            Error::UnsupportedArchitecture("dense-Qwen layer index exceeds i32".into())
        })?;
        let mut layer = TransformerBlock::new_for_layer(&self.args, index, stream)?;
        if self.sparse_expert_cache {
            replace_qwen_expert_bank(&mut layer, &self.args, index as usize, 0, None, stream)?;
        }
        Ok(layer)
    }

    pub(crate) fn register_parallel_parameters(
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
        if let Some(head) = &self.lm_head {
            planner.register(
                crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
                    head,
                    "lm_head",
                    self.args.hidden_size,
                    self.args.vocab_size as usize,
                    false,
                )?,
            )?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new_for_layer(&self.args, index as i32, stream)?;
            register_qwen_layer_parallel_plan(
                planner,
                &layer,
                &self.args,
                &format!("model.layers.{index}"),
            )?;
        }
        Ok(())
    }

    pub(crate) fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        if group != 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen has no group {group}"
            )));
        }
        let prefix = format!("model.layers.{index}");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let q = tensor("self_attn.q_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} query")))?;
        let k = tensor("self_attn.k_proj")
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} key")))?;
        let mut args = self.args.clone();
        args.num_attention_heads = q.local_shape()[0] as i32 / args.head_dim;
        args.num_key_value_heads = k.local_shape()[0] as i32 / args.head_dim;
        if args.is_moe() {
            let expert = layout
                .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                .ok_or_else(|| {
                    Error::Parallel(format!("missing TP layout for {prefix} experts"))
                })?;
            args.moe_intermediate_size = expert.local_shape()[1] as i32 / 2;
        } else {
            let gate = tensor("mlp.gate_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLP")))?;
            args.intermediate_size = gate.local_shape()[0] as i32;
        }
        let mut layer = TransformerBlock::new_for_layer(&args, index as i32, stream)?;
        if self.sparse_expert_cache {
            replace_qwen_expert_bank(
                &mut layer,
                &self.args,
                index,
                0,
                Some(args.moe_intermediate_size),
                stream,
            )?;
        }
        Ok(layer)
    }

    pub(crate) fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for expert-local layer construction".into(),
            ));
        }
        let mut layer = self.new_layer(group, index, stream)?;
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?
        };
        let FeedForward::Moe(moe) = &mut layer.mlp else {
            return Err(Error::Parallel(format!(
                "dense Qwen layer {index} is not an MoE layer"
            )));
        };
        moe.experts = Experts::new(
            local_experts,
            self.args.hidden_size,
            self.args.moe_intermediate_size,
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.gate_up_proj")),
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.down_proj")),
            stream,
        )?;
        Ok(layer)
    }

    pub(crate) fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for combined tensor/expert layer construction"
                    .into(),
            ));
        }
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen expert count exceeds i32".into()))?
        };
        let FeedForward::Moe(moe) = &mut layer.mlp else {
            return Err(Error::Parallel(format!(
                "dense Qwen layer {index} is not an MoE layer"
            )));
        };
        let local_intermediate = moe.experts.intermediate_dim;
        moe.experts = Experts::new(
            local_experts,
            self.args.hidden_size,
            local_intermediate,
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.gate_up_proj")),
            self.args
                .weight_quantization_for(&format!("model.layers.{index}.mlp.experts.down_proj")),
            stream,
        )?;
        Ok(layer)
    }

    pub(crate) fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    pub(crate) fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    pub(crate) fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &TransformerBlock,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        qwen_text_layer_bindings(
            layer,
            &self.args,
            &format!("model.layers.{index}"),
            store,
            self.sparse_expert_cache,
        )
    }

    pub(crate) fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &TransformerBlock,
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

    pub(crate) fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &TransformerBlock,
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
                if target.contains(".experts.") {
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

    pub(crate) fn quantizes_static_binding(&self, _binding: &WeightBinding) -> bool {
        true
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        match (layout, assignment) {
            (None, None) => self.new_layer(group, index, stream),
            (Some(layout), None) => self.new_parallel_layer(group, index, layout, stream),
            (None, Some(assignment)) => {
                self.new_expert_parallel_layer(group, index, assignment, stream)
            }
            (Some(layout), Some(assignment)) => {
                self.new_tensor_expert_parallel_layer(group, index, layout, assignment, stream)
            }
        }
    }

    pub(crate) fn tensor_expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &TransformerBlock,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings =
            self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)?;
        shard_layer_bindings(
            bindings,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
    }

    pub(crate) fn cartesian_layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &TransformerBlock,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        match (layout, assignment) {
            (None, None) => {
                // The execution layer can have transformed target geometry
                // (for example load-time affine quantization). Bindings must
                // continue to describe the adapter's source checkpoint
                // geometry and are transformed only during population.
                let source = self.new_layer(group, index, stream)?;
                self.layer_bindings(group, index, &source, store)
            }
            (Some(layout), None) => {
                self.parallel_layer_bindings(group, index, layer, store, layout, stream)
            }
            (None, Some(assignment)) => {
                self.expert_parallel_layer_bindings(group, index, layer, store, assignment, stream)
            }
            (Some(layout), Some(assignment)) => self.tensor_expert_parallel_layer_bindings(
                group, index, layer, store, layout, assignment, stream,
            ),
        }
    }
}

impl DenseQwenLayerwiseAdapter {}

fn validate_dense_qwen_cache<C: KeyValueCache>(
    caches: &[Option<C>],
    expected: usize,
) -> Result<(), Error> {
    if caches.len() != expected {
        return Err(Exception::custom(format!(
            "dense-Qwen cache has {} layers, expected {expected}",
            caches.len()
        ))
        .into());
    }
    for (index, cache) in caches.iter().enumerate() {
        cache.as_ref().ok_or_else(|| {
            Exception::custom(format!("dense-Qwen cache is missing layer {index}"))
        })?;
    }
    Ok(())
}

fn dense_qwen_attention_mask<C: KeyValueCache>(
    hidden: &Array,
    explicit: Option<&Array>,
    caches: &[Option<C>],
    stream: &Stream,
) -> Result<Option<Array>, Error> {
    if let Some(mask) = explicit {
        return Ok(Some(mask.clone()));
    }
    match create_attention_mask(hidden, caches, Some(true), stream)? {
        Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
        Some(AttentionMask::Causal) => Err(Error::UnsupportedArchitecture(
            "dense-Qwen layerwise execution requires an array attention mask".into(),
        )),
        None => Ok(None),
    }
}

fn retained_cache_arrays<C: KeyValueCache>(caches: &[Option<C>], index: usize) -> Vec<&Array> {
    caches[index]
        .as_ref()
        .map(KeyValueCache::retained_arrays)
        .unwrap_or_default()
}

pub(crate) fn qwen_text_layer_bindings(
    layer: &TransformerBlock,
    args: &DecoderConfig,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    Ok(
        qwen_text_layer_binding_plan(layer, args, prefix, store, external_experts)?
            .build_bindings(store)?,
    )
}

pub(crate) fn qwen_text_layer_binding_plan(
    layer: &TransformerBlock,
    args: &DecoderConfig,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
) -> Result<ModuleBindingPlan, Error> {
    if external_experts {
        return Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            prefix,
            store,
            BTreeMap::new(),
            |name| name.starts_with("mlp.experts."),
        )?);
    }
    let expert_prefix = format!("{prefix}.mlp.experts");
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut recipes = BTreeMap::new();
    if keys.contains(&format!("{expert_prefix}.gate_proj"))
        && keys.contains(&format!("{expert_prefix}.up_proj"))
    {
        recipes.insert(
            "mlp.experts.gate_up_proj".to_string(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(
                        format!("{expert_prefix}.gate_proj"),
                        TensorSelection::Full,
                    ),
                    DerivedWeightRecipe::source(
                        format!("{expert_prefix}.up_proj"),
                        TensorSelection::Full,
                    ),
                ],
            },
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{expert_prefix}.gate_proj{suffix}");
            let up = format!("{expert_prefix}.up_proj{suffix}");
            if keys.contains(&gate) && keys.contains(&up) {
                recipes.insert(
                    format!("mlp.experts.gate_up_proj{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, TensorSelection::Full),
                            DerivedWeightRecipe::source(up, TensorSelection::Full),
                        ],
                    },
                );
            }
        }
    } else if args.is_moe() && !keys.contains(&format!("{expert_prefix}.gate_up_proj")) {
        let mut gate_up = Vec::with_capacity(args.num_experts as usize);
        let mut down = Vec::with_capacity(args.num_experts as usize);
        for expert in 0..args.num_experts as usize {
            let gate = split_expert_key(&keys, &expert_prefix, expert, &["gate_proj", "w1"])?;
            let up = split_expert_key(&keys, &expert_prefix, expert, &["up_proj", "w3"])?;
            let down_key = split_expert_key(&keys, &expert_prefix, expert, &["down_proj", "w2"])?;
            gate_up.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, TensorSelection::Full),
                    DerivedWeightRecipe::source(up, TensorSelection::Full),
                ],
            });
            down.push(DerivedWeightRecipe::source(down_key, TensorSelection::Full));
        }
        recipes.insert(
            "mlp.experts.gate_up_proj".to_string(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: gate_up,
            },
        );
        recipes.insert(
            "mlp.experts.down_proj".to_string(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: down,
            },
        );
    }
    Ok(build_module_binding_plan_with_recipes(
        layer, prefix, store, recipes,
    )?)
}

pub(crate) fn qwen3_expert_catalog(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_cartesian(args, store, "model.layers", None)
}

pub(crate) fn qwen3_expert_catalog_at(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layer_root: &str,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_cartesian(args, store, layer_root, None)
}

/// Builds expert-granular bindings under an optional TP semantic layout.
///
/// Expert-axis selection is applied by the catalog recipe first; TP selection
/// is then composed over each expert's output geometry. This preserves atomic
/// expert caching while avoiding a full expert copy on every TP coordinate.
pub(crate) fn qwen3_expert_catalog_cartesian(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layer_root: &str,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut entries = Vec::new();
    for layer in 0..usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen3 layer count is negative".into()))?
    {
        let prefix = format!("{layer_root}.{layer}.mlp.experts");
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..usize::try_from(args.num_experts)
            .map_err(|_| Error::UnsupportedArchitecture("Qwen3 expert count is negative".into()))?
        {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", packed_gate_up.clone()),
                    ("down_proj", packed_down.clone()),
                ] {
                    bindings.push(recipe_binding(
                        name,
                        DerivedWeightRecipe::source(
                            key,
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        ),
                        store,
                    )?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        bindings.push(recipe_binding(
                            name,
                            DerivedWeightRecipe::source(
                                key,
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            ),
                            store,
                        )?);
                    }
                }
            } else if keys.contains(&format!("{prefix}.gate_proj"))
                && keys.contains(&format!("{prefix}.up_proj"))
                && keys.contains(&packed_down)
            {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                format!("{prefix}.gate_proj"),
                                selection.clone(),
                            ),
                            DerivedWeightRecipe::source(
                                format!("{prefix}.up_proj"),
                                selection.clone(),
                            ),
                        ],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    store,
                )?);
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        bindings.push(recipe_binding(
                            &format!("gate_up_proj{suffix}"),
                            DerivedWeightRecipe::Concatenate {
                                axis: 1,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate, selection.clone()),
                                    DerivedWeightRecipe::source(up, selection.clone()),
                                ],
                            },
                            store,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        bindings.push(recipe_binding(
                            &format!("down_proj{suffix}"),
                            DerivedWeightRecipe::source(down, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else {
                if args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj"))
                    .is_some()
                    || args
                        .weight_quantization_for(&format!("{prefix}.down_proj"))
                        .is_some()
                {
                    return Err(Error::Quantization(
                        "split Qwen3 experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                            .into(),
                    ));
                }
                let gate = split_expert_key(&keys, &prefix, expert, &["gate_proj", "w1"])?;
                let up = split_expert_key(&keys, &prefix, expert, &["up_proj", "w3"])?;
                let down = split_expert_key(&keys, &prefix, expert, &["down_proj", "w2"])?;
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![
                                DerivedWeightRecipe::source(gate, TensorSelection::Full),
                                DerivedWeightRecipe::source(up, TensorSelection::Full),
                            ],
                        }],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::source(down, TensorSelection::Full)],
                    },
                    store,
                )?);
            }
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
                    Error::UnsupportedArchitecture("Qwen3 expert byte total overflowed".into())
                })
            })?;
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = recipe.infer(store)?;
    let mut bindings = BindingPlan::new(vec![PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    }])
    .and_then(|plan| plan.build_bindings(store))
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(bindings.pop().expect("single planned expert binding"))
}

fn split_expert_key(
    keys: &BTreeSet<String>,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<String, Error> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|key| keys.contains(key))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen3 checkpoint is missing split expert {expert} projection {:?}",
                projections
            ))
        })
}

#[cfg(test)]
mod neutral_runtime_tests {
    #[test]
    fn production_model_and_loaders_use_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let wrapper_start = source
            .find("pub struct LayerwiseDecoder")
            .expect("dense-Qwen production wrapper");
        let adapter_start = source
            .find("/// Pipeline-staging legacy adapter retained")
            .expect("pipeline-only legacy adapter marker");
        let production = &source[wrapper_start..adapter_start];
        assert!(production.contains("DenseQwenExecution"));
        for legacy in ["LayerwiseModel<", ".adapter()", ".adapter_mut()"] {
            assert!(
                !production.contains(legacy),
                "production dense-Qwen wrapper still references {legacy}"
            );
        }
        let loaders_start = source
            .find("fn resolve_dense_qwen_store")
            .expect("neutral dense-Qwen loader");
        let loaders = &source[loaders_start..adapter_start];
        for legacy in [
            "load_layerwise_model(",
            "load_layerwise_model_with_quantization(",
            "load_tensor_parallel_layerwise_model(",
            "DenseQwenLayerwiseAdapter::new(",
        ] {
            assert!(
                !loaders.contains(legacy),
                "production dense-Qwen loaders still reference {legacy}"
            );
        }
    }
}
