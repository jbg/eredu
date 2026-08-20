//! Bounded layer execution for the shared Muse-Glimmer decoder.

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
    time::{Duration, Instant},
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::indexing::{NewAxis, TryIndexOp},
    ops::{concatenate_axis, zeros_dtype, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::{
    vision::{VisionBlock, VisionConfig, VisionState, VisionStatic},
    DecoderConfig, Experts, FeedForward, TransformerBlock,
};
use crate::composition::mlx_architectures::muse_glimmer as resident;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxParameterTree},
    backend::mlx::nn::{
        attention::AttentionInput,
        linear::{
            build_unloaded_maybe_quantized_lm_head_with_quantization,
            project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
        },
    },
    backend::mlx::nn::{
        parallel::{
            planned_kv_head_layout, register_gqa_projection_group, register_linear_parameter_group,
            register_swiglu_projection_group, GqaProjectionNames, LinearParallelism,
            SwiGluProjectionNames, VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes,
        build_module_binding_plan_with_recipes_excluding,
    },
    backend::mlx::runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    backend::mlx::runtime::checkpoint::store::{TensorSelection, WeightStoreBackend},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, recipe::DerivedWeightRecipe,
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
        validate_prompt_cache_model_identity, LayerCachePolicy, PromptCacheDescriptor,
        PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
    },
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "muse_glimmer.static.embedding";
const NORM_UNIT: &str = "muse_glimmer.static.norm";
const HEAD_UNIT: &str = "muse_glimmer.static.output";
const VISION_STATIC_UNIT: &str = "muse_glimmer.static.vision";
const VISION_PATCH_WEIGHT: &str = "vision_tower.patch_embedder.patch_embedding.weight";

fn vision_static_bindings(
    vision: &VisionStatic,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let checkpoint_key = format!("model.{VISION_PATCH_WEIGHT}");
    let mut recipes = BTreeMap::new();
    if store
        .source_metadata(&checkpoint_key)
        .is_ok_and(|metadata| metadata.logical_shape.len() == 4)
    {
        recipes.insert(
            VISION_PATCH_WEIGHT.into(),
            DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source(
                    checkpoint_key,
                    TensorSelection::Full,
                )),
                shape: vec![
                    vision.config.hidden_size as usize,
                    (vision.config.temporal_patch_size
                        * 3
                        * vision.config.patch_size
                        * vision.config.patch_size) as usize,
                ],
            },
        );
    }
    Ok(
        build_module_binding_plan_with_recipes(vision, "model", store, recipes)?
            .build_bindings(store)?,
    )
}

type MuseGlimmerUnit = MlxParameterTree<MuseGlimmerLayer>;
type MuseGlimmerStatic = MlxParameterTree<MuseGlimmerStaticModules>;
type MuseGlimmerResidentRuntime = LayerwiseRuntime<
    MuseGlimmerArchitecture,
    MlxBackend,
    MuseGlimmerLayerwiseCache,
    MlxResidentPolicy<MuseGlimmerUnit>,
>;
type MuseGlimmerBoundedRuntime = LayerwiseRuntime<
    MuseGlimmerArchitecture,
    MlxBackend,
    MuseGlimmerLayerwiseCache,
    MlxLayerwisePolicy<MuseGlimmerUnit, MuseGlimmerUnitFactory>,
>;

enum MuseGlimmerRuntime {
    Resident(MuseGlimmerResidentRuntime),
    Layerwise(MuseGlimmerBoundedRuntime),
}

#[derive(Debug, Clone, ModuleParameters)]
struct MuseGlimmerReplicatedStatic {
    #[param]
    vision: Option<VisionStatic>,
    #[param]
    embedding: MaybeQuantized<nn::Embedding>,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct MuseGlimmerParallelStatic {
    #[param]
    vision: Option<VisionStatic>,
    #[param]
    embedding: VocabParallelEmbedding,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: Option<VocabParallelLmHead>,
}

#[derive(Debug, Clone)]
enum MuseGlimmerStaticModules {
    Replicated(MuseGlimmerReplicatedStatic),
    Parallel(MuseGlimmerParallelStatic),
}

macro_rules! muse_glimmer_static_parameters {
    ($self:ident, $method:ident $(, $arg:expr)?) => {
        match $self {
            MuseGlimmerStaticModules::Replicated(module) => module.$method($($arg)?),
            MuseGlimmerStaticModules::Parallel(module) => module.$method($($arg)?),
        }
    };
}

impl ModuleParameters for MuseGlimmerStaticModules {
    fn num_parameters(&self) -> usize {
        muse_glimmer_static_parameters!(self, num_parameters)
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        muse_glimmer_static_parameters!(self, parameters)
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        muse_glimmer_static_parameters!(self, parameters_mut)
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        muse_glimmer_static_parameters!(self, trainable_parameters)
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        muse_glimmer_static_parameters!(self, freeze_parameters, recursive)
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        muse_glimmer_static_parameters!(self, unfreeze_parameters, recursive)
    }

    fn all_frozen(&self) -> Option<bool> {
        muse_glimmer_static_parameters!(self, all_frozen)
    }

    fn any_frozen(&self) -> Option<bool> {
        muse_glimmer_static_parameters!(self, any_frozen)
    }
}

struct MuseGlimmerUnitFactory {
    adapter: MuseGlimmerLayerwiseAdapter,
    vision_units: usize,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
}

impl MlxUnitFactory<MuseGlimmerUnit> for MuseGlimmerUnitFactory {
    fn build(&mut self, ordinal: usize, stream: &Stream) -> Result<MuseGlimmerUnit, Error> {
        let (group, index) = if ordinal < self.vision_units {
            (0, ordinal)
        } else {
            (1, ordinal - self.vision_units)
        };
        let layer = match self.parallel_layout.as_deref() {
            Some(layout) => self
                .adapter
                .new_parallel_layer(group, index, layout, stream)?,
            None => self.adapter.new_layer(group, index, stream)?,
        };
        let sparse = self.adapter.sparse_expert_cache && group == 1;
        MlxParameterTree::new_filtered(layer, "", |name| !sparse || !name.starts_with("experts."))
            .map_err(|error| Error::Parallel(error.to_string()))
    }
}

struct MuseGlimmerArchitecture {
    adapter: MuseGlimmerLayerwiseAdapter,
    static_modules: MuseGlimmerStatic,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl MuseGlimmerArchitecture {
    fn from_adapter(adapter: MuseGlimmerLayerwiseAdapter) -> Result<Self, Error> {
        let modules = match (&adapter.parallel_embedding, &adapter.parallel_lm_head) {
            (Some(embedding), head) => {
                MuseGlimmerStaticModules::Parallel(MuseGlimmerParallelStatic {
                    vision: adapter.vision.clone(),
                    embedding: embedding.clone(),
                    norm: adapter.norm.clone(),
                    lm_head: head.clone(),
                })
            }
            (None, _) => MuseGlimmerStaticModules::Replicated(MuseGlimmerReplicatedStatic {
                vision: adapter.vision.clone(),
                embedding: adapter.embedding.clone(),
                norm: adapter.norm.clone(),
                lm_head: adapter.lm_head.clone(),
            }),
        };
        Ok(Self {
            static_modules: MlxParameterTree::new(modules, "")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            adapter,
            parallel_topology: None,
        })
    }

    fn sync_adapter_static(&mut self) {
        match &*self.static_modules {
            MuseGlimmerStaticModules::Replicated(modules) => {
                self.adapter.vision = modules.vision.clone();
                self.adapter.embedding = modules.embedding.clone();
                self.adapter.norm = modules.norm.clone();
                self.adapter.lm_head = modules.lm_head.clone();
            }
            MuseGlimmerStaticModules::Parallel(modules) => {
                self.adapter.vision = modules.vision.clone();
                self.adapter.parallel_embedding = Some(modules.embedding.clone());
                self.adapter.norm = modules.norm.clone();
                self.adapter.parallel_lm_head = modules.lm_head.clone();
            }
        }
    }
}

impl LayeredArchitecture<MlxBackend, MuseGlimmerLayerwiseCache> for MuseGlimmerArchitecture {
    type Input<'a> = MuseGlimmerAdapterInput<'a>;
    type StaticModules = MuseGlimmerStatic;
    type Unit = MuseGlimmerUnit;
    type ForwardContext = MuseGlimmerForwardContext;
    type RetainedContextValues<'a> = std::vec::IntoIter<&'a Array>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.adapter.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        self.adapter.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.adapter.layer_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer group {group} has no unit {index}"
            )));
        }
        Ok(self.adapter.layer_checkpoint_prefix(group, index))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        let layer = self.adapter.new_layer(group, index, stream)?;
        let sparse = self.adapter.sparse_expert_cache && group == 1;
        MlxParameterTree::new_filtered(layer, "", |name| !sparse || !name.starts_with("experts."))
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut MuseGlimmerLayerwiseCache,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.sync_adapter_static();
        self.adapter.validate_cache(cache)?;
        self.adapter.begin_forward(input, cache, stream)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &Array,
        dependencies: &[&Array],
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let dependencies = dependencies
            .iter()
            .map(|value| (*value).clone())
            .collect::<Vec<_>>();
        self.adapter
            .begin_execution_group(group, initial, &dependencies, cache, forward, stream)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        self.adapter.should_execute_group(group, forward)
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.adapter
            .forward_layer(group, index, &mut **unit, hidden, cache, forward, stream)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.adapter
            .complete_execution_group(group, hidden, cache, forward, stream)
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.adapter.finish(hidden, cache, forward, stream)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward
            .mask
            .iter()
            .chain(forward.requested_mask.iter())
            .chain(forward.parts.iter().filter_map(|part| match part {
                MusePreparedPart::Text(value) => Some(value),
                MusePreparedPart::Visual(_) => None,
            }))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl ParallelLayeredArchitecture<MlxBackend, MuseGlimmerLayerwiseCache>
    for MuseGlimmerArchitecture
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut MuseGlimmerLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.sync_adapter_static();
        self.adapter.validate_cache(cache)?;
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("Muse-Glimmer parallel topology was not configured".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        self.adapter
            .begin_forward_with_execution(input, cache, &execution)
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("Muse-Glimmer parallel topology was not configured".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        self.adapter.forward_layer_with_execution(
            group_index,
            index,
            &mut **unit,
            hidden,
            cache,
            forward,
            &execution,
        )
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("Muse-Glimmer parallel topology was not configured".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        self.adapter
            .finish_with_execution(hidden, cache, forward, &execution)
    }
}

/// Architecture-owned KV cache accepted by the canonical Muse-Glimmer adapter.
pub enum MuseGlimmerLayerwiseCache {
    /// Append-only device KV caches.
    Concat {
        /// Layer-to-cache-slot assignment.
        layout: StateLayout,
        /// Cache storage for each assigned layer.
        caches: Vec<Option<ConcatKeyValueCache>>,
    },
    /// Sliding device KV caches used by expert-parallel execution.
    Sliding {
        /// Layer-to-cache-slot assignment.
        layout: StateLayout,
        /// Cache storage for each assigned layer.
        caches: Vec<Option<SlidingKeyValueCache>>,
    },
    /// Paged KV caches used by expert-parallel execution.
    Paged {
        /// Layer-to-cache-slot assignment.
        layout: StateLayout,
        /// Cache storage for each assigned layer.
        caches: Vec<Option<PagedKeyValueCache>>,
    },
}

impl MuseGlimmerLayerwiseCache {
    pub(crate) fn concat(layout: StateLayout, caches: Vec<Option<ConcatKeyValueCache>>) -> Self {
        Self::Concat { layout, caches }
    }

    #[cfg(test)]
    pub(crate) fn sliding(layout: StateLayout, caches: Vec<Option<SlidingKeyValueCache>>) -> Self {
        Self::Sliding { layout, caches }
    }

    pub(crate) fn paged(layout: StateLayout, caches: Vec<Option<PagedKeyValueCache>>) -> Self {
        Self::Paged { layout, caches }
    }
}

impl RuntimeState<MlxBackend> for MuseGlimmerLayerwiseCache {
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
        if address.group() == 0 {
            return Ok(Vec::new().into_iter());
        }
        if address.group() != 1 {
            return Err(StateError::UnknownLayer {
                layer: address.group(),
                count: 2,
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
        let caches = match self {
            Self::Concat { caches, .. } => retained_cache_arrays(caches, index),
            Self::Sliding { caches, .. } => retained_cache_arrays(caches, index),
            Self::Paged { caches, .. } => retained_cache_arrays(caches, index),
        };
        Ok(caches.into_iter())
    }
}

struct MuseGlimmerExecution {
    runtime: MuseGlimmerRuntime,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl MuseGlimmerExecution {
    fn architecture(&self) -> &MuseGlimmerArchitecture {
        match &self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime.architecture(),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut MuseGlimmerArchitecture {
        match &mut self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime.architecture_mut(),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.architecture_mut(),
        }
    }

    fn adapter(&self) -> &MuseGlimmerLayerwiseAdapter {
        &self.architecture().adapter
    }

    fn adapter_mut(&mut self) -> &mut MuseGlimmerLayerwiseAdapter {
        &mut self.architecture_mut().adapter
    }

    fn metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime.policy().checkpoint_store(),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store(),
        }
    }

    fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        match &self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime.policy().checkpoint_store_arc(),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.policy().checkpoint_store_arc(),
        }
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime.policy().residency_report(),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.policy().residency_report(),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.runtime {
            MuseGlimmerRuntime::Resident(_) => Ok(None),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.adapter().prompt_cache_model_identity(self.topology)
    }

    fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        Ok(self.prompt_cache_model_identity()?.layer_layout)
    }

    fn prompt_cache_rank_identity(&self) -> Option<crate::core::cache::CacheRankIdentity> {
        self.topology
            .map(crate::backend::mlx::cache::prompt_cache_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    fn prompt_cache_directory(&self, root: &Path) -> std::path::PathBuf {
        match self.topology {
            Some(topology) => root.join(format!("rank-{:05}", topology.global_rank)),
            None => root.to_path_buf(),
        }
    }

    fn save_prompt_cache(
        &self,
        cache: &mut MuseGlimmerLayerwiseCache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(&descriptor, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.adapter().save_prompt_cache(
            cache,
            &self.prompt_cache_directory(destination.as_ref()),
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MuseGlimmerLayerwiseCache, PromptCacheManifest), Error> {
        let identity = self.prompt_cache_model_identity()?;
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        self.adapter().load_prompt_cache(
            &self.prompt_cache_directory(directory.as_ref()),
            expected,
            &identity,
            prefix_token_ids,
            options,
            stream,
        )
    }

    fn forward(
        &mut self,
        input: MuseGlimmerAdapterInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime
                .forward(input, cache, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    fn forward_tensor_parallel(
        &mut self,
        input: MuseGlimmerAdapterInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime
                .forward_parallel(input, cache, group, stream)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    fn forward_with_layer_executor<E>(
        &mut self,
        input: MuseGlimmerAdapterInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        stream: &Stream,
        mut execute: E,
    ) -> Result<Array, Error>
    where
        E: FnMut(
            &mut MuseGlimmerLayerwiseAdapter,
            usize,
            usize,
            &mut MuseGlimmerLayer,
            &Array,
            &mut MuseGlimmerLayerwiseCache,
            &mut MuseGlimmerForwardContext,
            &Stream,
        ) -> Result<Array, Error>,
    {
        let execute = |architecture: &mut MuseGlimmerArchitecture,
                       group,
                       index,
                       unit: &mut MuseGlimmerUnit,
                       hidden: &Array,
                       cache: &mut MuseGlimmerLayerwiseCache,
                       forward: &mut MuseGlimmerForwardContext,
                       stream: &Stream| {
            execute(
                &mut architecture.adapter,
                group,
                index,
                &mut **unit,
                hidden,
                cache,
                forward,
                stream,
            )
        };
        match &mut self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, execute)
                .map_err(|error| Error::Parallel(error.to_string())),
            MuseGlimmerRuntime::Layerwise(runtime) => runtime
                .forward_with_unit_executor(input, cache, stream, execute)
                .map_err(|error| Error::Parallel(error.to_string())),
        }
    }

    fn forward_with_observer(
        &mut self,
        input: MuseGlimmerAdapterInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        let mut observer =
            crate::backend::mlx::runtime::execution::inspection::ActivationObserverProxy(observer);
        self.forward_with_layer_executor(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                adapter.forward_layer_with_observer(
                    group,
                    index,
                    layer,
                    hidden,
                    cache,
                    context,
                    stream,
                    &mut observer,
                )
            },
        )
    }

    fn clear_device_group(&self, name: &str) -> Result<(), Error> {
        let graph = self.architecture().execution_graph()?;
        let group = graph
            .groups()
            .iter()
            .find(|group| group.id() == name)
            .ok_or_else(|| Error::Parallel(format!("unknown Muse-Glimmer group {name}")))?;
        match &self.runtime {
            MuseGlimmerRuntime::Resident(runtime) => {
                runtime.policy().clear_device_group(group.id())
            }
            MuseGlimmerRuntime::Layerwise(runtime) => {
                runtime.policy().clear_device_group(group.id())
            }
        }
    }
}

/// Host-backed Muse-Glimmer causal LM.
pub struct LayerwiseDecoder {
    execution: MuseGlimmerExecution,
}

impl LayerwiseDecoder {
    pub(crate) fn state_layout(&self) -> Result<StateLayout, Error> {
        StateLayout::new(self.prompt_cache_layer_layout()?)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Returns the normalized decoder configuration.
    pub fn args(&self) -> &DecoderConfig {
        self.execution.adapter().args()
    }

    pub(crate) fn dflash_weight_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<(MaybeQuantized<nn::Embedding>, MaybeQuantized<nn::Linear>), Exception> {
        self.execution
            .adapter()
            .dflash_weight_snapshot(stream, copy)
    }

    pub(crate) fn prefill_dflash(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        target_layers: &[usize],
        stream: &Stream,
    ) -> Result<DFlashTargetOutput, Exception> {
        self.forward_dflash(
            MuseGlimmerAdapterInput::Prefill(input),
            cache,
            target_layers,
            false,
            stream,
        )
    }

    pub(crate) fn verify_dflash(
        &mut self,
        tokens: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        target_layers: &[usize],
        component_timing: bool,
        stream: &Stream,
    ) -> Result<DFlashTargetOutput, Exception> {
        self.forward_dflash(
            MuseGlimmerAdapterInput::Decode {
                inputs: tokens,
                mask: None,
            },
            cache,
            target_layers,
            component_timing,
            stream,
        )
    }

    fn forward_dflash(
        &mut self,
        input: MuseGlimmerAdapterInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        target_layers: &[usize],
        component_timing: bool,
        stream: &Stream,
    ) -> Result<DFlashTargetOutput, Exception> {
        let mut captured = BTreeMap::<usize, Array>::new();
        let mut observer = |name: &str, value: &Array| -> Result<(), Exception> {
            for &layer in target_layers {
                let suffix = format!(".layers.{layer}.output");
                if name.ends_with(&suffix) {
                    captured.insert(layer, value.clone());
                }
            }
            Ok(())
        };
        let mut device_time = Duration::ZERO;
        let logits = if component_timing {
            let mut observer =
                crate::backend::mlx::runtime::execution::inspection::ActivationObserverProxy(
                    &mut observer,
                );
            self.execution.forward_with_layer_executor(
                input,
                cache,
                stream,
                |adapter, group, index, layer, hidden, cache, context, stream| {
                    let output = adapter.forward_layer_with_observer(
                        group,
                        index,
                        layer,
                        hidden,
                        cache,
                        context,
                        stream,
                        &mut observer,
                    )?;
                    device_time +=
                        safemlx::transforms::async_eval_timed([&output], stream)?.elapsed()?;
                    Ok(output)
                },
            )
        } else {
            self.execution
                .forward_with_observer(input, cache, stream, &mut observer)
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(DFlashTargetOutput {
            logits,
            states: concatenate_dflash_target_states(&mut captured, target_layers, stream)?,
            device_time,
        })
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
                            .expect("validated Muse-Glimmer attention window fits i32"),
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
    ) -> Result<MuseGlimmerLayerwiseCache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(MuseGlimmerLayerwiseCache::concat(
                self.state_layout()?,
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
                    self.execution.prompt_cache_rank_identity(),
                )?;
                Ok(MuseGlimmerLayerwiseCache::paged(
                    self.state_layout()?,
                    caches,
                ))
            }
        }
    }

    /// Returns aggregate live KV paging observations, if paging is enabled.
    pub fn cache_residency_report(
        &self,
        cache: &MuseGlimmerLayerwiseCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            MuseGlimmerLayerwiseCache::Concat { .. }
            | MuseGlimmerLayerwiseCache::Sliding { .. } => Ok(None),
        }
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.execution.parallel_info()
    }

    /// Returns generalized parameter-residency and memory metadata.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        self.execution.metadata()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns the architecture identity used to validate persisted prompt caches.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
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
            MuseGlimmerLayerwiseCache::concat(self.state_layout()?, std::mem::take(cache));
        let result = self.execution.save_prompt_cache(
            &mut owned,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer prompt-cache wrapper changed variants")
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
        let (cache, manifest) = self.execution.load_prompt_cache(
            directory,
            expected,
            prefix_token_ids,
            options,
            stream,
        )?;
        let MuseGlimmerLayerwiseCache::Concat { caches: cache, .. } = cache else {
            return Err(Error::Parallel(
                "Muse-Glimmer prompt-cache restore returned a non-concat representation".into(),
            ));
        };
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
            .adapter()
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

    /// Runs a rank-local tensor-parallel forward pass through the generalized
    /// execution-group engine.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut MuseGlimmerLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            cache,
            group,
            stream,
        )
    }

    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MuseGlimmerLayerwiseCache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            MuseGlimmerAdapterInput::Prefill(input),
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
            MuseGlimmerLayerwiseCache::concat(self.state_layout()?, std::mem::take(cache));
        let result = self.execution.forward(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer concat cache wrapper changed variants")
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
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        let mut owned =
            MuseGlimmerLayerwiseCache::concat(self.state_layout()?, std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let MuseGlimmerLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer concat cache wrapper changed variants")
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
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    ) -> Result<Array, Error> {
        let mut owned =
            MuseGlimmerLayerwiseCache::paged(self.state_layout()?, std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let MuseGlimmerLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer paged cache wrapper changed variants")
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
        let mut owned =
            MuseGlimmerLayerwiseCache::paged(self.state_layout()?, std::mem::take(cache));
        let result = self.execution.forward(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer paged cache wrapper changed variants")
        };
        *cache = owned;
        result
    }

    /// Clears temporary device decoder copies.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("vision_encoder")?;
        self.execution.clear_device_group("text_decoder")
    }
}

/// Logits plus the five raw target-layer residual states consumed by DFlash.
pub(crate) struct DFlashTargetOutput {
    pub(crate) logits: Array,
    pub(crate) states: Array,
    pub(crate) device_time: Duration,
}

fn concatenate_dflash_target_states(
    captured: &mut BTreeMap<usize, Array>,
    target_layers: &[usize],
    stream: &Stream,
) -> Result<Array, Exception> {
    let mut states = Vec::with_capacity(target_layers.len());
    for &layer in target_layers {
        states.push(captured.remove(&layer).ok_or_else(|| {
            Exception::custom(format!(
                "Muse-Glimmer DFlash did not capture target layer {layer}"
            ))
        })?);
    }
    concatenate_axis(&states, -1, stream)
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
        let mut owned = MuseGlimmerLayerwiseCache::concat(
            self.state_layout()
                .map_err(|error| Exception::custom(error.to_string()))?,
            std::mem::take(cache),
        );
        let result = self
            .execution
            .forward(MuseGlimmerAdapterInput::Prefill(input), &mut owned, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream);
        let MuseGlimmerLayerwiseCache::Concat { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer concat cache wrapper changed variants")
        };
        *cache = owned;
        result
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
        let mut owned = MuseGlimmerLayerwiseCache::paged(
            self.state_layout()
                .map_err(|error| Exception::custom(error.to_string()))?,
            std::mem::take(cache),
        );
        let result = self
            .execution
            .forward(MuseGlimmerAdapterInput::Prefill(input), &mut owned, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream);
        let MuseGlimmerLayerwiseCache::Paged { caches: owned, .. } = owned else {
            unreachable!("Muse-Glimmer paged cache wrapper changed variants")
        };
        *cache = owned;
        result
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

fn muse_glimmer_execution_layout(
    adapter: &MuseGlimmerLayerwiseAdapter,
) -> Result<ExecutionUnitLayout, Error> {
    let graph = adapter.execution_graph()?;
    let counts = (0..graph.groups().len())
        .map(|group| adapter.layer_count(group))
        .collect::<Result<Vec<_>, _>>()?;
    ExecutionUnitLayout::new(&graph, counts).map_err(|error| Error::Parallel(error.to_string()))
}

fn muse_glimmer_static_bindings(
    adapter: &MuseGlimmerLayerwiseAdapter,
    modules: &MuseGlimmerStaticModules,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let qualify =
        |prefix: &str, bindings: Vec<WeightBinding>| -> Result<Vec<WeightBinding>, Error> {
            bindings
                .into_iter()
                .map(|binding| {
                    let name = format!("{prefix}.{}", binding.name());
                    binding.with_name(name).map_err(Into::into)
                })
                .collect()
        };
    let MuseGlimmerStaticModules::Replicated(modules) = modules else {
        return Err(Error::Parallel(
            "global Muse-Glimmer bindings require replicated static modules".into(),
        ));
    };
    let root = adapter.language_model_root();
    let mut bindings = Vec::new();
    if let Some(vision) = &modules.vision {
        bindings.extend(qualify("vision", vision_static_bindings(vision, store)?)?);
    }
    bindings.extend(qualify(
        "embedding",
        build_module_binding_plan_with_recipes(
            &modules.embedding,
            &format!("{root}.embed_tokens"),
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?,
    )?);
    bindings.extend(qualify(
        "norm",
        build_module_binding_plan_with_recipes(
            &modules.norm,
            &format!("{root}.norm"),
            store,
            BTreeMap::new(),
        )?
        .build_bindings(store)?,
    )?);
    if let Some(head) = &modules.lm_head {
        bindings.extend(qualify(
            "lm_head",
            build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
                .build_bindings(store)?,
        )?);
    }
    Ok(bindings)
}

fn muse_glimmer_ordinal(adapter: &MuseGlimmerLayerwiseAdapter, ordinal: usize) -> (usize, usize) {
    let vision_units = adapter
        .args
        .vision_config
        .as_ref()
        .map_or(0, VisionConfig::layer_count);
    if ordinal < vision_units {
        (0, ordinal)
    } else {
        (1, ordinal - vision_units)
    }
}

fn muse_glimmer_raw_unit(
    adapter: &MuseGlimmerLayerwiseAdapter,
    ordinal: usize,
    stream: &Stream,
) -> Result<MuseGlimmerLayer, Error> {
    let (group, index) = muse_glimmer_ordinal(adapter, ordinal);
    adapter.new_layer(group, index, stream)
}

fn muse_glimmer_raw_unit_bindings(
    adapter: &MuseGlimmerLayerwiseAdapter,
    ordinal: usize,
    unit: &MuseGlimmerLayer,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let (group, index) = muse_glimmer_ordinal(adapter, ordinal);
    adapter.layer_bindings(group, index, unit, store)
}

fn muse_glimmer_unit_bindings(
    adapter: &MuseGlimmerLayerwiseAdapter,
    ordinal: usize,
    unit: &MuseGlimmerUnit,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    muse_glimmer_raw_unit_bindings(adapter, ordinal, unit, store)
}

fn replicated_muse_glimmer_static(
    adapter: &MuseGlimmerLayerwiseAdapter,
) -> MuseGlimmerStaticModules {
    MuseGlimmerStaticModules::Replicated(MuseGlimmerReplicatedStatic {
        vision: adapter.vision.clone(),
        embedding: adapter.embedding.clone(),
        norm: adapter.norm.clone(),
        lm_head: adapter.lm_head.clone(),
    })
}

fn fresh_muse_glimmer_adapter(
    source: &MuseGlimmerLayerwiseAdapter,
    stream: &Stream,
) -> Result<MuseGlimmerLayerwiseAdapter, Error> {
    let mut adapter = if source.sparse_expert_cache {
        MuseGlimmerLayerwiseAdapter::new_external_experts(source.args.clone(), stream)?
    } else {
        MuseGlimmerLayerwiseAdapter::new(source.args.clone(), stream)?
    };
    adapter.parallel_kv_heads = source.parallel_kv_heads.clone();
    Ok(adapter)
}

fn resolve_muse_glimmer_store(
    store: SharedCheckpointSource,
    adapter: &MuseGlimmerLayerwiseAdapter,
) -> Result<SharedCheckpointSource, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend != WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan = adapter.safetensors_checkpoint_plan()?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_muse_glimmer_store(
    store: SharedCheckpointSource,
    source: &MuseGlimmerLayerwiseAdapter,
    quantization: WeightQuantization,
    stream: &Stream,
) -> Result<
    (
        SharedCheckpointSource,
        MuseGlimmerLayerwiseAdapter,
        eredu_runtime::WeightMaterializationReport,
    ),
    Error,
> {
    let target = source.load_time_quantized(quantization, stream)?;
    let source_static = replicated_muse_glimmer_static(source);
    let target_static = replicated_muse_glimmer_static(&target);
    let source_units = fresh_muse_glimmer_adapter(source, stream)?;
    let target_units = fresh_muse_glimmer_adapter(&target, stream)?;
    let static_binding_adapter = fresh_muse_glimmer_adapter(source, stream)?;
    let unit_binding_adapter = fresh_muse_glimmer_adapter(source, stream)?;
    let unit_count = muse_glimmer_execution_layout(source)?.len();
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |ordinal, stream| muse_glimmer_raw_unit(&source_units, ordinal, stream),
        move |ordinal, stream| muse_glimmer_raw_unit(&target_units, ordinal, stream),
        unit_count,
        quantization,
        stream,
        move |modules, store| muse_glimmer_static_bindings(&static_binding_adapter, modules, store),
        move |ordinal, unit, store| {
            muse_glimmer_raw_unit_bindings(&unit_binding_adapter, ordinal, unit, store)
        },
    )?;
    Ok((store, target, report))
}

fn load_muse_glimmer_with_store(
    store: SharedCheckpointSource,
    adapter: MuseGlimmerLayerwiseAdapter,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerExecution, Error> {
    let store = resolve_muse_glimmer_store(store, &adapter)?;
    let (store, adapter, materialization) = match quantization {
        Some(quantization) => {
            let (store, adapter, report) =
                quantize_muse_glimmer_store(store, &adapter, quantization, stream)?;
            (store, adapter, Some(report))
        }
        None => (store, adapter, None),
    };
    let layout = muse_glimmer_execution_layout(&adapter)?;
    let vision_units = adapter
        .args
        .vision_config
        .as_ref()
        .map_or(0, VisionConfig::layer_count);
    let factory = MuseGlimmerUnitFactory {
        adapter: fresh_muse_glimmer_adapter(&adapter, stream)?,
        vision_units,
        parallel_layout: None,
    };
    let static_binding_adapter = fresh_muse_glimmer_adapter(&adapter, stream)?;
    let unit_binding_adapter = fresh_muse_glimmer_adapter(&adapter, stream)?;
    let sparse = adapter.sparse_expert_cache;
    let model_type = adapter.args.model_type.clone();
    let quantization = adapter.args.weight_quantization();
    let mut architecture = MuseGlimmerArchitecture::from_adapter(adapter)?;
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        layout,
        options,
        stream,
        weights_stream,
        move |key| sparse && key.contains(".mlp.experts."),
        move |modules, store| {
            muse_glimmer_static_bindings(&static_binding_adapter, &**modules, store)
        },
        move |ordinal, unit, store, _| {
            muse_glimmer_unit_bindings(&unit_binding_adapter, ordinal, &unit, store)
        },
    )?;
    metadata.set_model_type(model_type);
    metadata.set_quantization(quantization);
    metadata.set_materialization(materialization);
    let runtime = if options.is_fully_resident() {
        MuseGlimmerRuntime::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        MuseGlimmerRuntime::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MuseGlimmerExecution {
        runtime,
        metadata,
        parallel_info: None,
        topology: None,
    })
}

fn load_muse_glimmer_parallel_with_store(
    store: SharedCheckpointSource,
    mut adapter: MuseGlimmerLayerwiseAdapter,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerExecution, Error> {
    let store = resolve_muse_glimmer_store(store, &adapter)?;
    let mut planner = build.planner();
    adapter.register_parallel_parameters(build, &mut planner, stream)?;
    let (_, local_layout) = planner.finish()?;
    adapter.configure_parallel_static(build, &local_layout, stream)?;

    let global_adapter = MuseGlimmerLayerwiseAdapter::new(adapter.args.clone(), stream)?;
    let global_static = replicated_muse_glimmer_static(&global_adapter);
    let static_bindings =
        muse_glimmer_static_bindings(&global_adapter, &global_static, store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    let unit_count = muse_glimmer_execution_layout(&global_adapter)?.len();
    for ordinal in 0..unit_count {
        let unit = muse_glimmer_raw_unit(&global_adapter, ordinal, stream)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&muse_glimmer_raw_unit_bindings(
                &global_adapter,
                ordinal,
                &unit,
                store.as_ref(),
            )?)?)
            .ok_or_else(|| {
                Error::Parallel("Muse-Glimmer global parameter bytes overflowed".into())
            })?;
    }

    let shared_layout = Arc::new(local_layout);
    let mut factory_adapter = fresh_muse_glimmer_adapter(&adapter, stream)?;
    factory_adapter.configure_parallel_static(build, &shared_layout, stream)?;
    let vision_units = adapter
        .args
        .vision_config
        .as_ref()
        .map_or(0, VisionConfig::layer_count);
    let factory = MuseGlimmerUnitFactory {
        adapter: factory_adapter,
        vision_units,
        parallel_layout: Some(Arc::clone(&shared_layout)),
    };
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let unit_binding_adapter = global_adapter;
    let sparse = adapter.sparse_expert_cache;
    let model_type = adapter.args.model_type.clone();
    let quantization = adapter.args.weight_quantization();
    let layout = muse_glimmer_execution_layout(&adapter)?;
    let mut architecture = MuseGlimmerArchitecture::from_adapter(adapter)?;
    architecture.parallel_topology = Some(build.topology());
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        layout,
        options,
        stream,
        weights_stream,
        move |key| sparse && key.contains(".mlp.experts."),
        move |_, store| shard_layer_bindings(static_bindings, "", store, &static_layout),
        move |ordinal, _local, store, stream| {
            let global = muse_glimmer_raw_unit(&unit_binding_adapter, ordinal, stream)?;
            let (group, index) = muse_glimmer_ordinal(&unit_binding_adapter, ordinal);
            shard_layer_bindings(
                unit_binding_adapter.layer_bindings(group, index, &global, store)?,
                &unit_binding_adapter.layer_checkpoint_prefix(group, index),
                store,
                &unit_layout,
            )
        },
    )?;
    metadata.set_model_type(model_type.clone());
    metadata.set_quantization(quantization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("Muse-Glimmer local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("Muse-Glimmer device parameter bytes overflowed".into()))?;
    let info = eredu_runtime::ParallelModelInfo::new(
        build.topology(),
        model_type,
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
        MuseGlimmerRuntime::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        MuseGlimmerRuntime::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(MuseGlimmerExecution {
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
            should_quantize_on_load("Muse-Glimmer", args.weight_quantization(), requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let source_adapter = MuseGlimmerLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_muse_glimmer_with_store(
            store,
            source_adapter,
            options,
            quantize_on_load,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Muse-Glimmer checkpoints through the generalized
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
    let adapter = MuseGlimmerLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_muse_glimmer_parallel_with_store(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
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
    if architecture != "muse-glimmer" {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer tensor-parallel loader cannot load GGUF architecture {architecture:?}"
        )));
    }
    let is_moe = false;
    let mmproj = open_mmproj_for_checkpoint(checkpoint)?;
    let (mut args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    apply_mmproj_config(checkpoint, metadata, &mut args, mmproj.as_ref())?;
    let store = muse_gguf_store(
        checkpoint,
        mmproj.as_ref(),
        &args,
        options.max_mapped_shards(),
    )?;
    let execution = load_muse_glimmer_parallel_with_store(
        store,
        MuseGlimmerLayerwiseAdapter::new(args, stream)?,
        options,
        build,
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
    if architecture != "muse-glimmer" {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer loader cannot load GGUF architecture {architecture:?}"
        )));
    }
    let is_moe = false;
    let mmproj = open_mmproj_for_checkpoint(checkpoint)?;
    let (mut args, eos_token_ids) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, architecture, is_moe)?;
    apply_mmproj_config(checkpoint, metadata, &mut args, mmproj.as_ref())?;
    let store = muse_gguf_store(
        checkpoint,
        mmproj.as_ref(),
        &args,
        residency.max_mapped_shards(),
    )?;

    if let Some(expert_options) = residency.expert_cache() {
        let _ = expert_options;
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer is dense and does not support sparse expert-cache residency".into(),
        ));
    }
    let execution = load_muse_glimmer_with_store(
        store,
        MuseGlimmerLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((LayerwiseDecoder { execution }, eos_token_ids))
}

fn open_mmproj_for_checkpoint(
    checkpoint: &GgufCheckpoint,
) -> Result<Option<resident::MuseGlimmerMmprojGguf>, Error> {
    let path = checkpoint
        .catalog()
        .shards()
        .first()
        .map(|shard| shard.path())
        .ok_or_else(|| Error::UnsupportedArchitecture("Muse-Glimmer GGUF has no shards".into()))?;
    resident::open_sibling_mmproj(path)
}

fn apply_mmproj_config(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    args: &mut DecoderConfig,
    mmproj: Option<&resident::MuseGlimmerMmprojGguf>,
) -> Result<(), Error> {
    let Some(mmproj) = mmproj else {
        return Ok(());
    };
    crate::composition::mlx::structural::validate_muse_glimmer_projector_gguf(
        checkpoint,
        metadata,
        &mmproj.checkpoint,
        &mmproj.metadata,
    )
    .into_loader_result()?;
    let mut vision = VisionConfig::from_gguf_metadata(&mmproj.metadata, args.hidden_size)?;
    let configs = crate::backend::mlx::runtime::checkpoint::load::gguf_quantization_configs(
        &mmproj.checkpoint,
        resident::translate_mmproj_weight_name,
    )?;
    if let Some(names) = args.quantized_weights.as_mut() {
        names.extend(configs.keys().cloned());
    }
    if let Some(existing) = args.quantized_weight_configs.as_mut() {
        existing.extend(configs.clone());
    }
    vision.quantized_weight_configs = configs;
    args.vision_config = Some(vision);
    Ok(())
}

fn muse_gguf_store(
    checkpoint: &GgufCheckpoint,
    mmproj: Option<&resident::MuseGlimmerMmprojGguf>,
    args: &DecoderConfig,
    max_mapped_shards: usize,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    let text_plan = super::checkpoint::gguf_plan(args).map_err(Error::UnsupportedArchitecture)?;
    let mut builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(checkpoint.catalog().clone(), &text_plan, |name| {
            resident::translate_gguf_weight_name(name, false)
        })?;
    if let Some(mmproj) = mmproj {
        let vision = args.vision_config.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Muse-Glimmer projector is present without validated vision geometry".into(),
            )
        })?;
        let projector_plan = super::checkpoint::projector_gguf_plan(args, vision)
            .map_err(Error::UnsupportedArchitecture)?;
        builder = builder.add_checkpoint(
            mmproj.checkpoint.catalog().clone(),
            &projector_plan,
            resident::translate_mmproj_store_weight_name,
        )?;
    }
    Ok(Arc::new(builder.build()?))
}

pub(crate) fn prepare_gguf_pipeline_source(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(DecoderConfig, SharedCheckpointSource), Error> {
    let mmproj = open_mmproj_for_checkpoint(checkpoint)?;
    let (mut args, _) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, "muse-glimmer", false)?;
    apply_mmproj_config(checkpoint, metadata, &mut args, mmproj.as_ref())?;
    let store = muse_gguf_store(checkpoint, mmproj.as_ref(), &args, max_mapped_shards)?;
    Ok((args, store))
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
    let source_adapter = MuseGlimmerLayerwiseAdapter::new_external_experts(args.clone(), stream)?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_muse_glimmer_with_store(
        store,
        source_adapter,
        non_expert.into(),
        quantize_on_load,
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
    execution.adapter_mut().expert_cache = Some(cache);
    Ok(LayerwiseDecoder { execution })
}

/// Dense-Qwen adapter sharing one complete-block execution path.
pub struct MuseGlimmerLayerwiseAdapter {
    args: DecoderConfig,
    vision: Option<VisionStatic>,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl MuseGlimmerLayerwiseAdapter {
    /// Creates metadata-only static Muse-Glimmer modules.
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
        let vision = args
            .vision_config
            .clone()
            .map(|config| VisionStatic::new(config, args.projector_hidden_size, stream))
            .transpose()?;
        Ok(Self {
            args,
            vision,
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

    pub(crate) fn vision_mut(&mut self) -> Option<&mut VisionStatic> {
        self.vision.as_mut()
    }

    pub(crate) fn embedding_mut(&mut self) -> &mut MaybeQuantized<nn::Embedding> {
        &mut self.embedding
    }

    pub(crate) fn parallel_embedding_mut(&mut self) -> Option<&mut VocabParallelEmbedding> {
        self.parallel_embedding.as_mut()
    }

    pub(crate) fn norm_mut(&mut self) -> &mut nn::RmsNorm {
        &mut self.norm
    }

    pub(crate) fn lm_head_mut(&mut self) -> Option<&mut MaybeQuantized<nn::Linear>> {
        self.lm_head.as_mut()
    }

    pub(crate) fn parallel_lm_head_mut(&mut self) -> Option<&mut VocabParallelLmHead> {
        self.parallel_lm_head.as_mut()
    }

    fn pipeline_cache(&self) -> MuseGlimmerLayerwiseCache {
        MuseGlimmerLayerwiseCache::concat(
            StateLayout::new(
                self.prompt_cache_model_identity(None)
                    .expect("validated Muse-Glimmer pipeline cache identity")
                    .layer_layout,
            )
            .expect("validated Muse-Glimmer pipeline state layout"),
            self.args
                .attention_schedule
                .iter()
                .map(|policy| {
                    Some(match policy.window() {
                        Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated Muse-Glimmer window fits i32"),
                        ),
                        None => ConcatKeyValueCache::new(),
                    })
                })
                .collect(),
        )
    }

    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineIngressState, Error> {
        let cache = self.pipeline_cache();
        let forward = self.prepare_multimodal_prefill(typed, execution, stream)?;
        Ok(MuseGlimmerPipelineIngressState { cache, forward })
    }

    pub(crate) fn begin_pipeline_continuation(
        &self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelineIngressState, Error> {
        input::validate(typed)?;
        let vision = self.vision.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer pipeline continuation has no vision projector".into())
        })?;
        let mut grids = Vec::new();
        for part in typed.parts {
            if matches!(
                part.modality,
                input::Modality::Image | input::Modality::Video
            ) {
                if !matches!(part.payload, input::InputPayload::Tensor(_)) {
                    return Err(Error::Parallel(
                        "Muse-Glimmer pipeline continuation requires tensor media payloads".into(),
                    ));
                }
                grids.push(
                    part.metadata
                        .vision_grid_thw
                        .ok_or_else(|| {
                            Error::Parallel(
                                "Muse-Glimmer pipeline continuation omitted vision_grid_thw".into(),
                            )
                        })?
                        .clone(),
                );
            }
        }
        if grids.is_empty() {
            return Err(Error::Parallel(
                "Muse-Glimmer pipeline continuation requires visual input".into(),
            ));
        }
        let refs = grids.iter().collect::<Vec<_>>();
        let grids = concatenate_axis(&refs, 0, stream)?;
        let entries = crate::composition::mlx_architectures::qwen::vl::vision::grid_thw_from_array(
            &grids, stream,
        )?;
        let patches = entries.iter().map(|(t, h, w)| t * h * w).sum::<i32>();
        let state = vision.continuation_state(&grids, stream)?;
        let hidden = zeros_dtype(
            &[patches, vision.config.hidden_size],
            Dtype::Float32,
            stream,
        )?;
        Ok(MuseGlimmerPipelineIngressState {
            cache: self.pipeline_cache(),
            forward: eredu_runtime::LayeredForwardState {
                hidden,
                context: MuseGlimmerForwardContext {
                    mask: None,
                    requested_mask: None,
                    parts: Vec::new(),
                    vision: Some(state),
                },
            },
        })
    }

    pub(crate) fn pipeline_ingress_active(&self, state: &MuseGlimmerPipelineIngressState) -> bool {
        state.forward.context.vision.is_some()
    }

    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &MuseGlimmerPipelineIngressState,
    ) -> Vec<Array> {
        vec![state.forward.hidden.clone()]
    }

    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut MuseGlimmerPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.forward.hidden = hidden;
        Ok(())
    }

    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut MuseGlimmerLayer,
        state: &mut MuseGlimmerPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        state.forward.hidden = match execution {
            Some(execution) => self.forward_layer_with_execution(
                0,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                execution,
            )?,
            None => self.forward_layer(
                0,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        Ok(self.pipeline_ingress_arrays(state))
    }

    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: MuseGlimmerPipelineIngressState,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelinePrepared, Error> {
        let hidden = self.begin_execution_group(
            1,
            &state.forward.hidden,
            &[state.forward.hidden.clone()],
            &mut state.cache,
            &mut state.forward.context,
            stream,
        )?;
        Ok(MuseGlimmerPipelinePrepared { hidden })
    }

    pub(crate) fn prepare_pipeline_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        vision_layers: &mut [MuseGlimmerLayer],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<MuseGlimmerPipelinePrepared, Error> {
        let mut state = self.begin_pipeline_ingress(typed, execution, stream)?;
        if self.pipeline_ingress_active(&state) {
            let expected = self
                .args
                .vision_config
                .as_ref()
                .map_or(0, VisionConfig::layer_count);
            if vision_layers.len() != expected {
                return Err(Error::Parallel(format!(
                    "Muse-Glimmer local pipeline ingress owns {} vision blocks, expected {expected}",
                    vision_layers.len()
                )));
            }
            for (index, layer) in vision_layers.iter_mut().enumerate() {
                self.forward_pipeline_vision_layer(index, layer, &mut state, execution, stream)?;
            }
        }
        self.finish_pipeline_ingress(state, stream)
    }

    pub(crate) fn prepare_pipeline_tokens(
        &mut self,
        tokens: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = match (&mut self.parallel_embedding, execution) {
            (Some(embedding), Some(execution)) => embedding.forward(tokens, execution)?,
            _ => self.embedding.forward(tokens, stream)?,
        };
        let stream = execution.map_or(stream, |execution| execution.stream());
        Ok(resident::rms_norm_without_scale(
            &hidden,
            self.args.rms_norm_eps,
            stream,
        )?)
    }

    pub(crate) fn finish_pipeline_text(
        &mut self,
        hidden: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        let logits = match (
            &mut self.parallel_lm_head,
            &mut self.parallel_embedding,
            execution,
        ) {
            (Some(head), _, Some(execution)) => {
                head.forward(&hidden, execution)?.all_gather(execution)?
            }
            (None, Some(embedding), Some(execution)) => embedding
                .project_logits(&hidden, execution)?
                .all_gather(execution)?,
            _ => project_logits_maybe_quantized(
                &mut self.lm_head,
                &mut self.embedding,
                &hidden,
                stream,
            )?,
        };
        Ok(resident::scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )?)
    }

    fn dflash_weight_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<(MaybeQuantized<nn::Embedding>, MaybeQuantized<nn::Linear>), Exception> {
        if self.args.tie_word_embeddings {
            return Err(Exception::custom(
                "Muse-Glimmer DFlash requires the target's untied raw output head",
            ));
        }
        if self.parallel_embedding.is_some() || self.parallel_lm_head.is_some() {
            return Err(Exception::custom(
                "Muse-Glimmer DFlash snapshots are unavailable from a TP-sharded target",
            ));
        }
        let mut embedding = self.embedding.clone();
        let mut head = self
            .lm_head
            .clone()
            .ok_or_else(|| Exception::custom("Muse-Glimmer target has no raw output head"))?;
        if copy {
            let embedding_parameters = embedding.parameters().flatten();
            let head_parameters = head.parameters().flatten();
            safemlx::transforms::async_eval_with_event(
                embedding_parameters
                    .values()
                    .copied()
                    .chain(head_parameters.values().copied()),
            )?
            .synchronize()?;
            embedding.copy_to_stream(stream)?;
            head.copy_to_stream(stream)?;
            let embedding_parameters = embedding.parameters().flatten();
            let head_parameters = head.parameters().flatten();
            safemlx::transforms::async_eval_with_event(
                embedding_parameters
                    .values()
                    .copied()
                    .chain(head_parameters.values().copied()),
            )?
            .synchronize()?;
        }
        Ok((embedding, head))
    }

    fn language_model_root(&self) -> &'static str {
        match self.args.weight_convention {
            resident::WeightConvention::HuggingFace => "model.language_model",
            resident::WeightConvention::Gguf => "model",
        }
    }
}

/// Attention mask shared by every temporary Muse-Glimmer decoder block.
pub struct MuseGlimmerForwardContext {
    mask: Option<Array>,
    requested_mask: Option<Array>,
    parts: Vec<MusePreparedPart>,
    vision: Option<VisionState>,
}

enum MusePreparedPart {
    Text(Array),
    Visual(i32),
}

/// Muse-Glimmer input consumed by the architecture-neutral layerwise engine.
pub enum MuseGlimmerAdapterInput<'a> {
    /// Ordered text/media prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Cached text decode input.
    Decode {
        /// Token ids shaped `[batch, sequence]`.
        inputs: &'a Array,
        /// Optional caller-provided attention mask.
        mask: Option<&'a Array>,
    },
}

/// Placement state transported while a Muse-Glimmer vision tower is split
/// across pipeline owners.
pub(crate) struct MuseGlimmerPipelineIngressState {
    cache: MuseGlimmerLayerwiseCache,
    forward: eredu_runtime::LayeredForwardState<Array, MuseGlimmerForwardContext>,
}

/// Decoder-facing result of Muse-Glimmer multimodal ingress.
pub(crate) struct MuseGlimmerPipelinePrepared {
    pub(crate) hidden: Array,
}

/// One temporary Muse-Glimmer execution unit.
pub enum MuseGlimmerLayer {
    /// One vision transformer block.
    Vision(Box<VisionBlock>),
    /// One language transformer block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for MuseGlimmerLayer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(layer) => layer.num_parameters(),
            Self::Text(layer) => layer.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.parameters(),
            Self::Text(layer) => layer.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(layer) => layer.parameters_mut(),
            Self::Text(layer) => layer.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(layer) => layer.trainable_parameters(),
            Self::Text(layer) => layer.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.freeze_parameters(recursive),
            Self::Text(layer) => layer.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(layer) => layer.unfreeze_parameters(recursive),
            Self::Text(layer) => layer.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.all_frozen(),
            Self::Text(layer) => layer.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(layer) => layer.any_frozen(),
            Self::Text(layer) => layer.any_frozen(),
        }
    }
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
    register_linear_parameter_group(
        planner,
        &attention.gate_proj,
        &format!("{prefix}.self_attn.gate_proj"),
        LinearParallelism::Column,
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

impl MuseGlimmerLayerwiseAdapter {
    pub(crate) fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weight_configs = None;
        if let Some(vision) = &mut args.vision_config {
            vision.apply_load_time_quantization(quantization);
        }
        if self.sparse_expert_cache {
            Self::new_external_experts(args, stream)
        } else {
            Self::new(args, stream)
        }
    }
}

impl MuseGlimmerLayerwiseAdapter {
    pub(crate) fn model_type(&self) -> &str {
        &self.args.model_type
    }

    pub(crate) fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        super::checkpoint::safetensors_plan(&self.args)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
    }

    pub(crate) fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        quantizes_static_target(self.args.vision_config.as_ref(), target)
    }

    pub(crate) fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Muse-Glimmer cache layer count"))?;
        let local_kv_heads = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_kv_heads.clone().ok_or_else(|| {
                    Error::Parallel(
                    "Muse-Glimmer parallel cache identity requested before local layout configuration"
                        .into(),
                )
                })?
            }
            _ => vec![self.args.num_key_value_heads; layer_count],
        };
        if local_kv_heads.len() != layer_count {
            return Err(Error::Parallel(format!(
                "Muse-Glimmer parallel cache geometry has {} layers, expected {layer_count}",
                local_kv_heads.len()
            )));
        }
        let layer_layout = crate::LayerSchedule::new(
            layer_count,
            self.args
                .attention_schedule
                .iter()
                .zip(local_kv_heads)
                .map(|(attention, kv_heads)| {
                    LayerCachePolicy::key_value(*attention, kv_heads, self.args.head_dim)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| Error::Parallel(error.to_string()))?,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(PromptCacheModelIdentity {
            model_family: "muse_glimmer".into(),
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
            layer_layout,
        })
    }

    pub(crate) fn save_prompt_cache(
        &self,
        cache: &mut MuseGlimmerLayerwiseCache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        match cache {
            MuseGlimmerLayerwiseCache::Concat { caches: cache, .. } => resident::save_prompt_cache(
                &self.args,
                cache,
                destination,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            )
            .map_err(Into::into),
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => {
                for cache in caches.iter_mut().flatten() {
                    cache.finalize()?;
                }
                caches
                    .iter()
                    .flatten()
                    .next()
                    .ok_or_else(|| Error::Parallel("cannot persist an empty Muse-Glimmer cache".into()))?
                    .manager()
                    .save_prompt_cache(destination, descriptor, prefix_token_ids, &[], options)
                    .map_err(|error| Error::Parallel(error.to_string()))
            }
            MuseGlimmerLayerwiseCache::Sliding { .. } => Err(Error::Parallel(
                "Muse-Glimmer sliding-cache persistence is unsupported; use concat or paged cache state".into(),
            )),
        }
    }

    pub(crate) fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(MuseGlimmerLayerwiseCache, PromptCacheManifest), Error> {
        let (cache, manifest) = resident::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )?;
        Ok((
            MuseGlimmerLayerwiseCache::concat(
                StateLayout::new(identity.layer_layout.clone())
                    .map_err(|error| Error::Parallel(error.to_string()))?,
                cache,
            ),
            manifest,
        ))
    }

    pub(crate) fn validate_cache(
        &self,
        cache: &mut MuseGlimmerLayerwiseCache,
    ) -> Result<(), Error> {
        let expected = usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer layer count {} is invalid",
                self.args.num_hidden_layers
            ))
        })?;
        match cache {
            MuseGlimmerLayerwiseCache::Concat { caches, .. } => {
                if caches.is_empty() {
                    *caches = self
                        .args
                        .attention_schedule
                        .iter()
                        .map(|policy| {
                            Some(match policy.window() {
                                Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                                    i32::try_from(window.get())
                                        .expect("validated Muse-Glimmer attention window fits i32"),
                                ),
                                None => ConcatKeyValueCache::new(),
                            })
                        })
                        .collect();
                }
                validate_muse_glimmer_cache(caches, expected)
            }
            MuseGlimmerLayerwiseCache::Sliding { caches, .. } => {
                validate_muse_glimmer_cache(caches, expected)
            }
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => {
                validate_muse_glimmer_cache(caches, expected)
            }
        }
    }

    pub(crate) fn begin_forward<'a>(
        &mut self,
        input: MuseGlimmerAdapterInput<'a>,
        _cache: &mut MuseGlimmerLayerwiseCache,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, MuseGlimmerForwardContext>, Error> {
        match input {
            MuseGlimmerAdapterInput::Decode { inputs, mask } => {
                let hidden = self.embedding.forward(inputs, stream)?;
                let hidden =
                    resident::rms_norm_without_scale(&hidden, self.args.rms_norm_eps, stream)?;
                Ok(eredu_runtime::LayeredForwardState {
                    hidden,
                    context: MuseGlimmerForwardContext {
                        mask: None,
                        requested_mask: mask.cloned(),
                        parts: Vec::new(),
                        vision: None,
                    },
                })
            }
            MuseGlimmerAdapterInput::Prefill(input) => {
                self.prepare_multimodal_prefill(input, None, stream)
            }
        }
    }

    pub(crate) fn begin_forward_with_execution<'a>(
        &mut self,
        input: MuseGlimmerAdapterInput<'a>,
        _cache: &mut MuseGlimmerLayerwiseCache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, MuseGlimmerForwardContext>, Error> {
        match input {
            MuseGlimmerAdapterInput::Prefill(input) => {
                self.prepare_multimodal_prefill(input, Some(execution), execution.stream())
            }
            MuseGlimmerAdapterInput::Decode { inputs, mask } => {
                let hidden = match &mut self.parallel_embedding {
                    Some(embedding) => embedding.forward(inputs, execution)?,
                    None => self.embedding.forward(inputs, execution.stream())?,
                };
                let hidden = resident::rms_norm_without_scale(
                    &hidden,
                    self.args.rms_norm_eps,
                    execution.stream(),
                )?;
                Ok(eredu_runtime::LayeredForwardState {
                    hidden,
                    context: MuseGlimmerForwardContext {
                        mask: None,
                        requested_mask: mask.cloned(),
                        parts: Vec::new(),
                        vision: None,
                    },
                })
            }
        }
    }

    pub(crate) fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["vision_encoder", "text_decoder"]).map_err(Into::into)
    }

    pub(crate) fn should_execute_group(
        &self,
        group: usize,
        context: &MuseGlimmerForwardContext,
    ) -> bool {
        group != 0 || context.vision.is_some()
    }

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |config| config.layer_count())),
            1 => usize::try_from(self.args.num_hidden_layers).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer layer count {} is invalid",
                    self.args.num_hidden_layers
                ))
            }),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer has no execution group {group}"
            ))),
        }
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
        if select(VISION_STATIC_UNIT) {
            if let Some(vision) = &self.vision {
                units.push(StaticUnitBindings::new(
                    VISION_STATIC_UNIT,
                    vision_static_bindings(vision, store)?,
                )?);
            }
        }
        if select(EMBEDDING_UNIT) {
            let root = self.language_model_root();
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    &format!("{root}.embed_tokens"),
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            let root = self.language_model_root();
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    &format!("{root}.norm"),
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
    ) -> Result<MuseGlimmerLayer, Error> {
        if group == 0 {
            let config = self.args.vision_config.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "text-only Muse-Glimmer artifact has no vision execution group".into(),
                )
            })?;
            return Ok(MuseGlimmerLayer::Vision(Box::new(VisionBlock::new(
                config, index, stream,
            )?)));
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer has no execution group {group}"
            )));
        }
        let layer_index = i32::try_from(index).map_err(|_| {
            Error::UnsupportedArchitecture("Muse-Glimmer layer index exceeds i32".into())
        })?;
        let mut layer = TransformerBlock::new_for_layer(&self.args, layer_index, stream)?;
        if self.sparse_expert_cache {
            replace_qwen_expert_bank(&mut layer, &self.args, index, 0, None, stream)?;
        }
        Ok(MuseGlimmerLayer::Text(Box::new(layer)))
    }

    pub(crate) fn register_parallel_parameters(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let root = self.language_model_root();
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
                &self.embedding,
                &format!("{root}.embed_tokens"),
                self.args.vocab_size as usize,
                self.args.hidden_size,
                false,
            )?,
        )?;
        crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            &format!("{root}.norm"),
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
        if let Some(vision) = &self.vision {
            register_replicated_module(planner, vision, "model")?;
            for index in 0..vision.config.layer_count() {
                let layer = VisionBlock::new(&vision.config, index, stream)?;
                let prefix = format!("model.vision_tower.layers.{index}");
                register_linear_parameter_group(
                    planner,
                    &layer.attn.q_proj,
                    &format!("{prefix}.attn.q_proj"),
                    LinearParallelism::Column,
                )?;
                register_linear_parameter_group(
                    planner,
                    &layer.attn.k_proj,
                    &format!("{prefix}.attn.k_proj"),
                    LinearParallelism::Column,
                )?;
                register_linear_parameter_group(
                    planner,
                    &layer.attn.v_proj,
                    &format!("{prefix}.attn.v_proj"),
                    LinearParallelism::Column,
                )?;
                register_linear_parameter_group(
                    planner,
                    &layer.attn.proj,
                    &format!("{prefix}.attn.proj"),
                    LinearParallelism::Row,
                )?;
                register_linear_parameter_group(
                    planner,
                    &layer.mlp.fc1,
                    &format!("{prefix}.mlp.fc1"),
                    LinearParallelism::Column,
                )?;
                register_linear_parameter_group(
                    planner,
                    &layer.mlp.fc2,
                    &format!("{prefix}.mlp.fc2"),
                    LinearParallelism::Row,
                )?;
                register_replicated_module(planner, &layer.norm1, &format!("{prefix}.norm1"))?;
                register_replicated_module(planner, &layer.norm2, &format!("{prefix}.norm2"))?;
            }
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new_for_layer(&self.args, index as i32, stream)?;
            register_qwen_layer_parallel_plan(
                planner,
                &layer,
                &self.args,
                &format!("{root}.layers.{index}"),
            )?;
        }
        Ok(())
    }

    pub(crate) fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            self.args.num_hidden_layers as usize,
            self.args.head_dim,
            &format!("{}.layers", self.language_model_root()),
        )?);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args
                .weight_quantization_for("model.embed_tokens.weight"),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.weight_quantization_for("lm_head.weight"),
                context,
                stream,
            )?);
        }
        Ok(())
    }

    pub(crate) fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<MuseGlimmerLayer, Error> {
        if group == 0 {
            let config = self.args.vision_config.as_ref().ok_or_else(|| {
                Error::Parallel("Muse-Glimmer TP vision layer has no vision config".into())
            })?;
            let prefix = format!("model.vision_tower.layers.{index}");
            let q = layout
                .tensor(&format!("{prefix}.attn.q_proj.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.attn.q_proj.inner.weight")))
                .ok_or_else(|| Error::Parallel(format!("missing TP vision query for {prefix}")))?;
            let fc1 = layout
                .tensor(&format!("{prefix}.mlp.fc1.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.mlp.fc1.inner.weight")))
                .ok_or_else(|| Error::Parallel(format!("missing TP vision MLP for {prefix}")))?;
            let head_dim = config.hidden_size / config.num_heads;
            let local_heads = i32::try_from(q.local_shape()[0])
                .map_err(|_| Error::Parallel("Muse vision local width exceeds i32".into()))?
                / head_dim;
            let local_intermediate = i32::try_from(fc1.local_shape()[0]).map_err(|_| {
                Error::Parallel("Muse vision local intermediate exceeds i32".into())
            })?;
            return Ok(MuseGlimmerLayer::Vision(Box::new(
                VisionBlock::new_tensor_parallel(
                    config,
                    index,
                    local_heads,
                    local_intermediate,
                    stream,
                )?,
            )));
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer has no execution group {group}"
            )));
        }
        let prefix = format!("{}.layers.{index}", self.language_model_root());
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
        Ok(MuseGlimmerLayer::Text(Box::new(layer)))
    }

    pub(crate) fn new_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<MuseGlimmerLayer, Error> {
        Err(Error::Parallel(
            "Muse-Glimmer is dense and does not support expert parallelism".into(),
        ))
    }

    pub(crate) fn new_tensor_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &eredu_runtime::LocalModelLayout,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<MuseGlimmerLayer, Error> {
        Err(Error::Parallel(
            "Muse-Glimmer is dense and does not support TP+EP".into(),
        ))
    }

    pub(crate) fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("model.vision_tower.layers.{index}")
        } else {
            format!("{}.layers.{index}", self.language_model_root())
        }
    }

    pub(crate) fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &MuseGlimmerLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        match layer {
            MuseGlimmerLayer::Vision(_) if group == 0 => {
                Ok(build_module_binding_plan_with_recipes(
                    layer,
                    &self.layer_checkpoint_prefix(group, index),
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?)
            }
            MuseGlimmerLayer::Text(layer) if group == 1 => qwen_text_layer_bindings(
                layer,
                &self.args,
                &self.layer_checkpoint_prefix(group, index),
                store,
                self.sparse_expert_cache,
            ),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer group {group} contains the wrong layer kind at {index}"
            ))),
        }
    }

    pub(crate) fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &MuseGlimmerLayer,
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
        _layer: &MuseGlimmerLayer,
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

    pub(crate) fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut MuseGlimmerLayer,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        context: &mut MuseGlimmerForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if group == 0 {
            let MuseGlimmerLayer::Vision(layer) = layer else {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer vision group contains a text layer at {index}"
                )));
            };
            let vision = self.vision.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer vision execution requires the projector sidecar".into(),
                )
            })?;
            let state = context.vision.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture("Muse-Glimmer vision state is missing".into())
            })?;
            return Ok(vision.forward_block(layer, index, hidden, state, stream)?);
        }
        let MuseGlimmerLayer::Text(layer) = layer else {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer text group contains a vision layer at {index}"
            )));
        };
        match cache {
            MuseGlimmerLayerwiseCache::Concat { caches, .. } => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index]
                    .as_mut()
                    .expect("validated Muse-Glimmer cache"),
                context,
                stream,
            ),
            MuseGlimmerLayerwiseCache::Sliding { caches, .. } => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index]
                    .as_mut()
                    .expect("validated Muse-Glimmer cache"),
                context,
                stream,
            ),
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index]
                    .as_mut()
                    .expect("validated Muse-Glimmer cache"),
                context,
                stream,
            ),
        }
    }

    pub(crate) fn forward_layer_with_observer<
        O: eredu_runtime::ActivationObserver<Array, safemlx::error::Exception>,
    >(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut MuseGlimmerLayer,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        context: &mut MuseGlimmerForwardContext,
        stream: &Stream,
        observer: &mut O,
    ) -> Result<Array, Error> {
        if self.sparse_expert_cache {
            let prefix = self.layer_checkpoint_prefix(group, index);
            observer.observe(&format!("{prefix}.input"), hidden)?;
            let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
            observer.observe(&format!("{prefix}.output"), &output)?;
            return Ok(observer
                .intervene(&format!("{prefix}.output"), &output)?
                .unwrap_or(output));
        }
        let prefix = self.layer_checkpoint_prefix(group, index);
        if group == 0 {
            observer.observe(&format!("{prefix}.input"), hidden)?;
            let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
            observer.observe(&format!("{prefix}.output"), &output)?;
            return Ok(observer
                .intervene(&format!("{prefix}.output"), &output)?
                .unwrap_or(output));
        }
        let MuseGlimmerLayer::Text(layer) = layer else {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer text group contains a vision layer at {index}"
            )));
        };
        Ok(match cache {
            MuseGlimmerLayerwiseCache::Concat { caches, .. } => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(
                        caches[index]
                            .as_mut()
                            .expect("validated Muse-Glimmer cache"),
                    ),
                },
                stream,
                &prefix,
                observer,
            )?,
            MuseGlimmerLayerwiseCache::Sliding { caches, .. } => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(
                        caches[index]
                            .as_mut()
                            .expect("validated Muse-Glimmer cache"),
                    ),
                },
                stream,
                &prefix,
                observer,
            )?,
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(
                        caches[index]
                            .as_mut()
                            .expect("validated Muse-Glimmer cache"),
                    ),
                },
                stream,
                &prefix,
                observer,
            )?,
        })
    }

    pub(crate) fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut MuseGlimmerLayer,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        context: &mut MuseGlimmerForwardContext,
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
        if group == 0 {
            let MuseGlimmerLayer::Vision(layer) = layer else {
                return Err(Error::Parallel(format!(
                    "Muse-Glimmer vision group contains a text layer at {index}"
                )));
            };
            let vision = self.vision.as_ref().ok_or_else(|| {
                Error::Parallel("Muse-Glimmer TP vision state has no projector".into())
            })?;
            let state = context
                .vision
                .as_ref()
                .ok_or_else(|| Error::Parallel("Muse-Glimmer TP vision state is missing".into()))?;
            return Ok(vision.forward_block_tensor_parallel(
                layer,
                index,
                hidden,
                state,
                tp_group,
                execution.stream(),
            )?);
        }
        let MuseGlimmerLayer::Text(layer) = layer else {
            return Err(Error::Parallel(format!(
                "Muse-Glimmer text group contains a vision layer at {index}"
            )));
        };
        match cache {
            MuseGlimmerLayerwiseCache::Concat { caches, .. } => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
            MuseGlimmerLayerwiseCache::Sliding { caches, .. } => Ok(layer
                .forward_tensor_parallel(
                    hidden,
                    context.mask.as_ref(),
                    caches[index].as_mut(),
                    tp_group,
                    execution.stream(),
                )?),
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
        }
    }

    pub(crate) fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut MuseGlimmerLayerwiseCache,
        context: &mut MuseGlimmerForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if group != 1 {
            return match dependency_outputs {
                [] => Ok(initial_hidden.clone()),
                [hidden] => Ok(hidden.clone()),
                _ => Err(Error::UnsupportedArchitecture(
                    "Muse-Glimmer vision group received multiple dependencies".into(),
                )),
            };
        }
        let mut visual = if let Some(state) = context.vision.as_ref() {
            let encoded = dependency_outputs.first().unwrap_or(initial_hidden);
            Some(
                self.vision
                    .as_mut()
                    .ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Muse-Glimmer media input requires a loaded vision projector".into(),
                        )
                    })?
                    .finish(encoded, state, stream)?,
            )
        } else {
            None
        };
        let mut visual_offset = 0;
        let mut assembled = Vec::new();
        for part in &context.parts {
            match part {
                MusePreparedPart::Text(hidden) => assembled.push(hidden.clone()),
                MusePreparedPart::Visual(length) => {
                    let features = visual.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Muse-Glimmer visual placeholder has no encoded features".into(),
                        )
                    })?;
                    let next = visual_offset + *length;
                    if next > features.dim(0) {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Muse-Glimmer media features ended at {next}, but only {} are available",
                            features.dim(0)
                        )));
                    }
                    assembled.push(
                        features
                            .try_index_device((visual_offset..next, ..), stream)?
                            .try_index_device((NewAxis, .., ..), stream)?,
                    );
                    visual_offset = next;
                }
            }
        }
        let hidden = if assembled.is_empty() {
            dependency_outputs
                .first()
                .cloned()
                .unwrap_or_else(|| initial_hidden.clone())
        } else {
            let refs = assembled.iter().collect::<Vec<_>>();
            concatenate_axis(&refs, 1, stream)?
        };
        if let Some(features) = visual.take() {
            if visual_offset != features.dim(0) {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer produced {} visual features but prompt consumed {visual_offset}",
                    features.dim(0)
                )));
            }
        }
        context.mask = match cache {
            MuseGlimmerLayerwiseCache::Concat { caches, .. } => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
            MuseGlimmerLayerwiseCache::Sliding { caches, .. } => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
            MuseGlimmerLayerwiseCache::Paged { caches, .. } => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
        };
        Ok(hidden)
    }

    pub(crate) fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut MuseGlimmerLayerwiseCache,
        _context: &MuseGlimmerForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embedding,
            &hidden,
            stream,
        )?;
        Ok(resident::scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )?)
    }

    pub(crate) fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut MuseGlimmerLayerwiseCache,
        context: &MuseGlimmerForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<Array, Error> {
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, execution)?,
            None => embedding.project_logits(&hidden, execution)?,
        };
        let logits = logits.all_gather(execution)?;
        Ok(resident::scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            execution.stream(),
        )?)
    }

    pub(crate) fn new_cartesian_layer(
        &self,
        group: usize,
        index: usize,
        layout: Option<&eredu_runtime::LocalModelLayout>,
        assignment: Option<&crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<MuseGlimmerLayer, Error> {
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
        layer: &MuseGlimmerLayer,
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
        layer: &MuseGlimmerLayer,
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

    pub(crate) fn complete_execution_group(
        &mut self,
        _group: usize,
        hidden: &Array,
        _cache: &mut MuseGlimmerLayerwiseCache,
        _context: &mut MuseGlimmerForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(hidden.clone())
    }
}

fn quantizes_static_target(vision: Option<&VisionConfig>, target: &str) -> bool {
    let target = target.strip_prefix("model.").unwrap_or(target);
    if target.starts_with("vision_tower.")
        || target.starts_with("vision_adapter.")
        || target.starts_with("vision_projection.")
    {
        return vision.is_some_and(|config| config.quantized_weight_configs.contains_key(target));
    }
    true
}

impl MuseGlimmerLayerwiseAdapter {
    fn prepare_multimodal_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, MuseGlimmerForwardContext>, Error> {
        input::validate(typed)?;
        let mut parts = Vec::new();
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let merge = self
            .args
            .vision_config
            .as_ref()
            .map_or(2, |config| config.merge_size);
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    let hidden = match (&mut self.parallel_embedding, execution) {
                        (Some(embedding), Some(execution)) => {
                            embedding.forward(tokens, execution)?
                        }
                        _ => self.embedding.forward(tokens, stream)?,
                    };
                    let hidden =
                        resident::rms_norm_without_scale(&hidden, self.args.rms_norm_eps, stream)?;
                    parts.push(MusePreparedPart::Text(hidden));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(tensor),
                ) => {
                    if self.vision.is_none() {
                        return Err(Error::UnsupportedArchitecture(
                            "Muse-Glimmer text-only GGUF loading requires mmproj-kquant.gguf before image input is admitted".into(),
                        ));
                    }
                    if part.modality == input::Modality::Video
                        && self.args.weight_convention == resident::WeightConvention::Gguf
                    {
                        return Err(Error::UnsupportedArchitecture(
                            "the released Muse-Glimmer GGUF projector is image-only because temporal patch weights were collapsed during conversion".into(),
                        ));
                    }
                    let grid = part.metadata.vision_grid_thw.ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Muse-Glimmer {} input requires vision_grid_thw metadata",
                            part.modality.as_str()
                        ))
                    })?;
                    let entries =
                        crate::composition::mlx_architectures::qwen::vl::vision::grid_thw_from_array(
                            grid, stream,
                        )?;
                    let output_tokens = entries
                        .iter()
                        .map(|(t, h, w)| t * (h / merge) * (w / merge))
                        .sum::<i32>();
                    parts.push(MusePreparedPart::Visual(output_tokens));
                    pixels.push(tensor.clone());
                    grids.push(grid.clone());
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Muse-Glimmer does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        if pixels.is_empty() {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    MusePreparedPart::Text(hidden) => Some(hidden),
                    MusePreparedPart::Visual(_) => None,
                })
                .collect::<Vec<_>>();
            let hidden = concatenate_axis(&text, 1, stream)?;
            return Ok(eredu_runtime::LayeredForwardState {
                hidden,
                context: MuseGlimmerForwardContext {
                    mask: None,
                    requested_mask: None,
                    parts,
                    vision: None,
                },
            });
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let grid_refs = grids.iter().collect::<Vec<_>>();
        let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
        let grids = concatenate_axis(&grid_refs, 0, stream)?;
        let (hidden, vision) = self
            .vision
            .as_mut()
            .expect("validated Muse-Glimmer vision modules")
            .begin(&pixels, &grids, stream)?;
        Ok(eredu_runtime::LayeredForwardState {
            hidden,
            context: MuseGlimmerForwardContext {
                mask: None,
                requested_mask: None,
                parts,
                vision: Some(vision),
            },
        })
    }

    fn forward_cached_layer<C: KeyValueCache>(
        &self,
        index: usize,
        layer: &mut TransformerBlock,
        hidden: &Array,
        cache: &mut C,
        context: &MuseGlimmerForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Qwen3 sparse expert cache was not initialized".into(),
                )
            })?;
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
                                    self.args.hidden_size,
                                    self.args.moe_intermediate_size,
                                    load_time.or_else(|| {
                                        self.args.weight_quantization_for(&format!(
                                            "{prefix}.gate_up_proj"
                                        ))
                                    }),
                                    load_time.or_else(|| {
                                        self.args
                                            .weight_quantization_for(&format!("{prefix}.down_proj"))
                                    }),
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
        Ok(layer.forward(
            AttentionInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(cache),
            },
            stream,
        )?)
    }
}

fn validate_muse_glimmer_cache<C: KeyValueCache>(
    caches: &[Option<C>],
    expected: usize,
) -> Result<(), Error> {
    if caches.len() != expected {
        return Err(Exception::custom(format!(
            "Muse-Glimmer cache has {} layers, expected {expected}",
            caches.len()
        ))
        .into());
    }
    for (index, cache) in caches.iter().enumerate() {
        cache.as_ref().ok_or_else(|| {
            Exception::custom(format!("Muse-Glimmer cache is missing layer {index}"))
        })?;
    }
    Ok(())
}

fn muse_glimmer_attention_mask<C: KeyValueCache>(
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
            "Muse-Glimmer layerwise execution requires an array attention mask".into(),
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
    if external_experts {
        return Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            prefix,
            store,
            BTreeMap::new(),
            |name| name.starts_with("mlp.experts."),
        )?
        .build_bindings(store)?);
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
    Ok(
        build_module_binding_plan_with_recipes(layer, prefix, store, recipes)?
            .build_bindings(store)?,
    )
}

pub(crate) fn qwen3_expert_catalog(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    qwen3_expert_catalog_cartesian(args, store, "model.layers", None)
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
    let plan = BindingPlan::new(vec![PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    }])
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    plan.build_bindings(store)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
        .pop()
        .ok_or_else(|| Error::UnsupportedArchitecture("empty expert binding plan".into()))
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
    use super::*;

    #[test]
    fn production_model_and_loaders_use_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let production = source
            .split("/// Host-backed Muse-Glimmer causal LM.")
            .nth(1)
            .expect("production wrapper marker");
        let production = production
            .split("/// Dense-Qwen adapter sharing one complete-block execution path.")
            .next()
            .expect("legacy adapter marker");
        assert!(production.contains("execution: MuseGlimmerExecution"));
        assert!(production.contains("load_muse_glimmer_with_store("));
        assert!(production.contains("load_muse_glimmer_parallel_with_store("));
        assert!(!production.contains("LayerwiseModel<"));
        assert!(!production.contains("load_layerwise_model_with_quantization("));
        assert!(!production.contains("load_tensor_parallel_layerwise_model("));
    }

    #[test]
    fn tiny_muse_glimmer_executes_resident_and_bounded_text_layers() {
        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let weights_context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = context.stream();
        let weights_stream = weights_context.stream();
        let config = serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 22,
            "video_token_id": 23,
            "out_hidden_size": 32,
            "projector_hidden_size": 16,
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 24,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "rms_norm_eps": 0.00001,
                "post_norm_eps": 0.00001,
                "vocab_size": 24,
                "max_position_embeddings": 64,
                "rope_theta": 10000.0,
                "layer_types": ["sliding_attention", "full_attention"],
                "layer_rope_theta": [10000.0, 0.0],
                "sliding_window": 8,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "qk_scale_factor": 1.0,
                "output_multiplier": 1.0,
                "final_logit_softcapping": 30.0
            },
            "vision_config": {
                "model_type": "muse_glimmer_vision",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "patch_size": 2,
                "patch_temporal": 1,
                "merge_size": 2,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "max_position_embeddings": 4,
                "layer_norm_eps": 0.00001,
                "hidden_act": "gelu",
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        });
        let args = resident::config_from_hf_value(&config).unwrap();
        let adapter = MuseGlimmerLayerwiseAdapter::new(args.clone(), stream).unwrap();
        let mut arrays = BTreeMap::<String, Array>::new();
        let mut insert_module = |prefix: &str, module: &dyn ModuleParameters| {
            for (name, parameter) in module.parameters().flatten() {
                arrays.insert(
                    format!("{prefix}.{name}"),
                    zeros_dtype(parameter.shape(), parameter.dtype(), stream).unwrap(),
                );
            }
        };
        if let Some(vision) = &adapter.vision {
            insert_module("model", vision);
        }
        insert_module("model.language_model.embed_tokens", &adapter.embedding);
        insert_module("model.language_model.norm", &adapter.norm);
        insert_module("lm_head", adapter.lm_head.as_ref().unwrap());
        for group in 0..2 {
            for index in 0..adapter.layer_count(group).unwrap() {
                let layer = adapter.new_layer(group, index, stream).unwrap();
                insert_module(&adapter.layer_checkpoint_prefix(group, index), &layer);
            }
        }
        let directory = tempfile::tempdir().unwrap();
        Array::save_safetensors(
            arrays.iter().map(|(name, array)| (name.as_str(), array)),
            None,
            directory.path().join("model.safetensors"),
        )
        .unwrap();
        let tokens = Array::from_slice(&[1u32, 2], &[1, 2]);

        for residency in [
            LayerWeightResidency::FullyResident,
            eredu_runtime::LayerwiseLoadOptions::default().into(),
        ] {
            let store =
                open_safetensors_weight_store(directory.path(), residency.max_mapped_shards())
                    .unwrap();
            let execution = load_muse_glimmer_with_store(
                store,
                MuseGlimmerLayerwiseAdapter::new(args.clone(), stream).unwrap(),
                residency,
                None,
                stream,
                weights_stream,
            )
            .unwrap();
            let mut model = LayerwiseDecoder { execution };
            let mut cache = model.new_cache();
            let logits = model.forward(&tokens, None, &mut cache, stream).unwrap();
            safemlx::transforms::eval([&logits]).unwrap();
            assert_eq!(logits.shape(), &[1, 2, 24]);
        }
    }
}
