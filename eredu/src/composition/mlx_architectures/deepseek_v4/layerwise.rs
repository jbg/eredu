//! Checkpoint-format-independent bounded-residency execution for DeepSeek V4.

use eredu_runtime::{
    ExecutionGraph, ExecutionUnitLayout, ExpertCacheLoadOptions, ExpertIdentity, ExpertPass,
    LayerWeightResidency, LayeredArchitecture, LayeredForwardState, LayerwiseRuntime,
    NonExpertWeightResidency, ParallelLayeredArchitecture, StaticUnitBindings, WeightResidency,
};

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::{OffloadUnit, WeightBinding};

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{
        broadcast_to, indexing::NewAxis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue,
    },
    Array, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::{MlxBackend, MlxParameterTree},
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::{
        checkpoint::{
            binding::{
                binding_bytes, build_module_binding_plan_with_recipes,
                build_module_binding_plan_with_recipes_excluding, populate_module_from_lease,
                populate_module_from_lease_excluding,
            },
            binding_plan::{BindingPlan, PlannedBinding},
            recipe::{DerivedWeightRecipe, RecipeDtype},
            store::{open_gguf_checkpoint_source, TensorSelection},
        },
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
                MlxUnitFactory,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_module_store_with_bindings,
                shard_layer_bindings, ArchitectureAdapter, LoadTimeQuantizableAdapter,
            },
        },
        residency::{
            expert_cache::{
                ExpertCache, ExpertCacheError, ExpertCacheReport, ExpertCatalogEntry,
                ExpertRouteBatch,
            },
            manager::ResidentUnitLease,
        },
    },
};

use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions, ResidencyReport};

use super::{
    attention::AttentionCache,
    model::{
        load_prompt_cache_with_identity, prompt_cache_model_identity, save_prompt_cache_with_rank,
        Cache, DecoderLayer, Model as ResidentModel, ModelArgs,
    },
};

const EMBEDDING_UNIT: &str = "deepseek_v4.static.embedding";
const NORM_UNIT: &str = "deepseek_v4.static.norm";
const HC_HEAD_UNIT: &str = "deepseek_v4.static.hc_head";
const HEAD_UNIT: &str = "deepseek_v4.static.output";
const DRAFT_UNIT: &str = "deepseek_v4.static.draft";

type DeepSeekV4Unit = MlxParameterTree<DecoderLayer>;
type DeepSeekV4Static = MlxParameterTree<ResidentModel>;
type DeepSeekV4ResidentRuntime =
    LayerwiseRuntime<DeepSeekV4Architecture, MlxBackend, Cache, MlxResidentPolicy<DeepSeekV4Unit>>;
type DeepSeekV4BoundedRuntime = LayerwiseRuntime<
    DeepSeekV4Architecture,
    MlxBackend,
    Cache,
    MlxLayerwisePolicy<DeepSeekV4Unit, DeepSeekV4UnitFactory>,
>;

enum DeepSeekV4Execution {
    Resident(DeepSeekV4ResidentRuntime),
    Layerwise(DeepSeekV4BoundedRuntime),
}

#[derive(Clone)]
struct DeepSeekV4UnitFactory {
    args: ModelArgs,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
    sparse_experts: bool,
}

impl MlxUnitFactory<DeepSeekV4Unit> for DeepSeekV4UnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<DeepSeekV4Unit, Error> {
        let layer = build_deepseek_v4_unit(
            &self.args,
            index,
            self.parallel_layout.as_deref(),
            self.parallel_topology,
            stream,
        )?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_experts || !name.starts_with("ffn.switch_mlp.")
        })
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

struct DeepSeekV4Architecture {
    args: ModelArgs,
    static_model: DeepSeekV4Static,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DeepSeekV4Architecture {
    fn new(args: ModelArgs, sparse_expert_cache: bool, stream: &Stream) -> Result<Self, Error> {
        let mut static_model = ResidentModel::new(args.clone(), stream)?;
        static_model.model.layers.clear();
        Ok(Self {
            args,
            static_model: MlxParameterTree::new(static_model, "")
                .map_err(|error| Error::Parallel(error.to_string()))?,
            sparse_expert_cache,
            expert_cache: None,
            parallel_topology: None,
        })
    }

    fn new_cache(&self) -> Result<Cache, Exception> {
        self.static_model.new_cache()
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache()?;
        }
        let expected = super::model::state_layout(&self.args)?;
        if eredu_runtime::RuntimeState::<MlxBackend>::layout(cache) != &expected {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek V4 cache layout does not match decoder geometry".into(),
            ));
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 cache has {} target layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        Ok(())
    }
}

type DeepSeekV4RetainedContext<'a> = std::iter::Chain<
    std::iter::Once<&'a Array>,
    std::iter::Chain<std::option::IntoIter<&'a Array>, std::option::Iter<'a, Array>>,
>;

impl LayeredArchitecture<MlxBackend, Cache> for DeepSeekV4Architecture {
    type Input<'a> = &'a Array;
    type StaticModules = DeepSeekV4Static;
    type Unit = DeepSeekV4Unit;
    type ForwardContext = DeepSeekV4ForwardContext;
    type RetainedContextValues<'a> = DeepSeekV4RetainedContext<'a>;
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
                "DeepSeek V4 has no execution group {group}"
            )))
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 has no decoder unit {index}"
            )));
        }
        Ok(format!("layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_model
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_model
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        self.group_unit_count(group)?;
        let layer = build_deepseek_v4_unit(&self.args, index, None, None, stream)?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_expert_cache || !name.starts_with("ffn.switch_mlp.")
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
        let embedded = self
            .static_model
            .model
            .embed_tokens
            .forward(input, stream)?;
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        let hidden = broadcast_to(
            &hidden,
            &[
                embedded.dim(0),
                embedded.dim(1),
                self.args.hc_mult,
                self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(LayeredForwardState {
            hidden,
            context: DeepSeekV4ForwardContext {
                input_ids: input.clone(),
                captures: Vec::new(),
                draft_hidden: None,
            },
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
                "DeepSeek V4 execution group {group} received {} dependencies",
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
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_experts(
                        &self.args,
                        expert_cache,
                        index,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                },
            )?
        } else {
            layer.forward(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
            )?
        };
        capture_draft_hidden(&self.args, index, &output, context, stream)?;
        Ok(output)
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.static_model.model.hc_head.forward(hidden, stream)?;
        let hidden = self.static_model.model.norm.forward(&hidden, stream)?;
        Ok(self.static_model.lm_head.forward(&hidden, stream)?)
    }

    fn retained_context_values<'a>(
        &'a self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        std::iter::once(&context.input_ids).chain(
            context
                .captures
                .last()
                .map(|(_, value)| value)
                .into_iter()
                .chain(context.draft_hidden.iter()),
        )
    }
}

impl ParallelLayeredArchitecture<MlxBackend, Cache> for DeepSeekV4Architecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.begin_forward(input, cache, stream)
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group_index)?;
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_tensor_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                group,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_experts(
                        &self.args,
                        expert_cache,
                        index,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                },
            )?
        } else {
            layer.forward_tensor_parallel(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                group,
                stream,
            )?
        };
        capture_draft_hidden(&self.args, index, &output, context, stream)?;
        Ok(output)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        cache: &mut Cache,
        context: &Self::ForwardContext,
        _group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.finish_forward(hidden, cache, context, stream)
    }
}

fn deepseek_v4_layerwise_error(error: impl std::fmt::Display) -> Error {
    Error::Parallel(error.to_string())
}

/// DeepSeek V4 decoder using the generalized residency executor.
pub struct DeepSeekV4LayerwiseModel {
    execution: DeepSeekV4Execution,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DeepSeekV4LayerwiseModel {
    fn architecture(&self) -> &DeepSeekV4Architecture {
        match &self.execution {
            DeepSeekV4Execution::Resident(execution) => execution.architecture(),
            DeepSeekV4Execution::Layerwise(execution) => execution.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut DeepSeekV4Architecture {
        match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution.architecture_mut(),
            DeepSeekV4Execution::Layerwise(execution) => execution.architecture_mut(),
        }
    }

    fn prompt_cache_rank_identity(&self) -> Option<crate::core::cache::CacheRankIdentity> {
        self.parallel_topology
            .map(crate::backend::mlx::cache::prompt_cache_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    /// Validated architecture arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.architecture().args
    }

    /// Creates target and embedded-draft cache state.
    pub fn new_cache(&self) -> Result<Cache, Exception> {
        self.architecture().new_cache()
    }

    /// Creates resident or explicitly bounded cache state independently of
    /// parameter residency.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        let rank = self.prompt_cache_rank_identity();
        match policy {
            CacheResidencyPolicy::Device => self.new_cache().map_err(Into::into),
            CacheResidencyPolicy::Paged(options) => {
                let manager =
                    crate::backend::mlx::runtime::cache::residency::CacheResidencyManager::new(
                        options,
                    )
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                self.architecture()
                    .static_model
                    .new_cache_with_manager(manager, rank)
                    .map_err(Into::into)
            }
        }
    }

    /// Runs target decoding.
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution
                .forward(tokens, cache, stream)
                .map_err(deepseek_v4_layerwise_error),
            DeepSeekV4Execution::Layerwise(execution) => execution
                .forward(tokens, cache, stream)
                .map_err(deepseek_v4_layerwise_error),
        }
    }

    /// Runs a rank-local tensor-parallel target pass through the generalized executor.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution
                .forward_parallel(tokens, cache, group, stream)
                .map_err(deepseek_v4_layerwise_error),
            DeepSeekV4Execution::Layerwise(execution) => execution
                .forward_parallel(tokens, cache, group, stream)
                .map_err(deepseek_v4_layerwise_error),
        }
    }

    /// Executes replicated layers while delegating routed experts to an EP owner.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |_architecture: &mut DeepSeekV4Architecture,
                            _group: usize,
                            index: usize,
                            layer: &mut DeepSeekV4Unit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut DeepSeekV4ForwardContext,
                            stream: &Stream| {
            let output = layer.forward_with_expert_executor(
                hidden,
                mask,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )?;
            capture_draft_hidden(&_architecture.args, index, &output, context, stream)?;
            Ok(output)
        };
        match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution
                .forward_with_unit_executor(tokens, cache, stream, execute_unit)
                .map_err(deepseek_v4_layerwise_error),
            DeepSeekV4Execution::Layerwise(execution) => execution
                .forward_with_unit_executor(tokens, cache, stream, execute_unit)
                .map_err(deepseek_v4_layerwise_error),
        }
    }

    /// Executes TP-sharded attention while delegating routed experts to EP.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |_architecture: &mut DeepSeekV4Architecture,
                            _group_index: usize,
                            index: usize,
                            layer: &mut DeepSeekV4Unit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut DeepSeekV4ForwardContext,
                            group: &safemlx::distributed::Group,
                            stream: &Stream| {
            let output = layer.forward_tensor_with_expert_executor(
                hidden,
                mask,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                group,
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )?;
            capture_draft_hidden(&_architecture.args, index, &output, context, stream)?;
            Ok(output)
        };
        match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution
                .forward_parallel_with_unit_executor(
                    tokens,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                )
                .map_err(deepseek_v4_layerwise_error),
            DeepSeekV4Execution::Layerwise(execution) => execution
                .forward_parallel_with_unit_executor(
                    tokens,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                )
                .map_err(deepseek_v4_layerwise_error),
        }
    }

    /// Returns the active Cartesian topology and rank-local parameter accounting.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Returns generalized parameter-residency and materialization metadata.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.parallel_topology = Some(topology);
        self.architecture_mut().parallel_topology = Some(topology);
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.architecture().static_model.mtp_len()
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = match &mut self.execution {
            DeepSeekV4Execution::Resident(execution) => execution
                .forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
                .map_err(|error| Exception::custom(error.to_string()))?,
            DeepSeekV4Execution::Layerwise(execution) => execution
                .forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("DeepSeek V4 layerwise pass did not retain draft hidden state")
        })?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    pub(crate) fn forward_mtp_target_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let (logits, context) = if let Some(tensor_group) = tensor_group {
            let execute_unit = |architecture: &mut DeepSeekV4Architecture,
                                _group_index: usize,
                                index: usize,
                                layer: &mut DeepSeekV4Unit,
                                hidden: &Array,
                                cache: &mut Cache,
                                context: &mut DeepSeekV4ForwardContext,
                                group: &safemlx::distributed::Group,
                                stream: &Stream| {
                let output = layer.forward_tensor_with_expert_executor(
                    hidden,
                    None,
                    Some(&mut cache.layers[index]),
                    &context.input_ids,
                    group,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?;
                capture_draft_hidden(&architecture.args, index, &output, context, stream)?;
                Ok(output)
            };
            match &mut self.execution {
                DeepSeekV4Execution::Resident(execution) => execution
                    .forward_parallel_with_unit_executor_and_context_hook(
                        tokens,
                        cache,
                        tensor_group,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
                DeepSeekV4Execution::Layerwise(execution) => execution
                    .forward_parallel_with_unit_executor_and_context_hook(
                        tokens,
                        cache,
                        tensor_group,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
            }
            .map_err(|error| Exception::custom(error.to_string()))?
        } else {
            let execute_unit = |architecture: &mut DeepSeekV4Architecture,
                                _group: usize,
                                index: usize,
                                layer: &mut DeepSeekV4Unit,
                                hidden: &Array,
                                cache: &mut Cache,
                                context: &mut DeepSeekV4ForwardContext,
                                stream: &Stream| {
                let output = layer.forward_with_expert_executor(
                    hidden,
                    None,
                    Some(&mut cache.layers[index]),
                    &context.input_ids,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?;
                capture_draft_hidden(&architecture.args, index, &output, context, stream)?;
                Ok(output)
            };
            match &mut self.execution {
                DeepSeekV4Execution::Resident(execution) => execution
                    .forward_with_unit_executor_and_context_hook(
                        tokens,
                        cache,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
                DeepSeekV4Execution::Layerwise(execution) => execution
                    .forward_with_unit_executor_and_context_hook(
                        tokens,
                        cache,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
            }
            .map_err(|error| Exception::custom(error.to_string()))?
        };
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("DeepSeek V4 EP pass did not retain MTP/DSpark hidden state")
                })?,
                tokens: tokens.clone(),
            },
        )
    }

    /// Current bounded-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            DeepSeekV4Execution::Resident(execution) => execution.policy().residency_report(),
            DeepSeekV4Execution::Layerwise(execution) => execution.policy().residency_report(),
        }
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            DeepSeekV4Execution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            DeepSeekV4Execution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
        }
    }

    /// Dense disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            DeepSeekV4Execution::Resident(_) => Ok(None),
            DeepSeekV4Execution::Layerwise(execution) => execution.policy().dense_stream_report(),
        }
    }

    /// Independent expert-cache telemetry when configured.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        Ok(self
            .architecture()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()?)
    }

    /// Returns the exact generic prompt-cache layout including pooling state.
    pub(crate) fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.architecture().static_model.prompt_cache_layer_layout()
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        prompt_cache_model_identity(
            self.args(),
            self.parallel_topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
        )
    }

    /// Atomically persists target KV plus compressed pooling/indexer state.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            self.prompt_cache_rank_identity(),
            stream,
        )
    }

    /// Restores target KV plus compressed pooling/indexer state exactly.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        load_prompt_cache_with_identity(
            self.args(),
            directory,
            expected,
            prefix_token_ids,
            self.prompt_cache_model_identity()?,
            options,
            stream,
        )
    }
}

impl CausalModel<Cache> for DeepSeekV4LayerwiseModel {
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
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.forward(tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for DeepSeekV4LayerwiseModel
{
    type Cache = Cache;
    type DraftCache = super::model::DraftCache;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.forward_mtp_target(&tokens, cache, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.forward_mtp_target(tokens, cache, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::prefill_draft_cache(
            &mut self.architecture_mut().static_model,
            output,
            tokens,
            cache,
            stream,
        )
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn restore_target_checkpoint(
        cache: &mut Self::Cache,
        checkpoint: &Self::Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        cache.restore_target_checkpoint(checkpoint, stream)
    }

    fn draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        draft_index: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::draft_logits(
            &mut self.architecture_mut().static_model,
            hidden,
            last_token,
            draft_index,
            cache,
            stream,
        )
    }

    fn fused_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::fused_draft_logits(
            &mut self.architecture_mut().static_model,
            hidden,
            last_token,
            proposal_capacity,
            cache,
            stream,
        )
    }

    fn adjust_fused_draft_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::adjust_fused_draft_logits(
            &mut self.architecture_mut().static_model,
            logits,
            last_token,
            stream,
        )
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::advance_draft_cache(
            &mut self.architecture_mut().static_model,
            hidden,
            tokens,
            cache,
            stream,
        )
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

/// Token generation over a bounded V4 model.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    crate::backend::mlx::nn::generation::Generate<'a, DeepSeekV4LayerwiseModel, Cache, S>;

/// Per-forward token state retained across streamed decoder units.
pub struct DeepSeekV4ForwardContext {
    input_ids: Array,
    captures: Vec<(usize, Array)>,
    draft_hidden: Option<Array>,
}

/// Architecture adapter used by the generalized layerwise executor.
pub struct DeepSeekV4LayerwiseAdapter {
    args: ModelArgs,
    static_model: ResidentModel,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DeepSeekV4LayerwiseAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut resident = ResidentModel::new(args.clone(), stream)?;
        // Decoder layers are residency units owned by `LayerwiseModel`. Keep only
        // the shared static target and draft modules in this holder.
        resident.model.layers.clear();
        Ok(Self {
            static_model: resident,
            sparse_expert_cache: false,
            expert_cache: None,
            parallel_topology: None,
            args,
        })
    }

    fn new_sparse(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    fn new_cache(&self) -> Result<Cache, Exception> {
        let layers = self.args.compress_ratios[..self.args.num_hidden_layers as usize]
            .iter()
            .map(|ratio| AttentionCache::new_for_ratio(*ratio, self.args.sliding_window))
            .collect::<Result<_, _>>()?;
        let mut cache = self.static_model.new_cache()?;
        cache.layers = layers;
        Ok(cache)
    }

    fn layer_recipes(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let root = format!("layers.{index}");
        let mut recipes = BTreeMap::new();
        for (local, target) in layer.parameters().flatten() {
            let recipe = if let Some(recipe) =
                expert_bank_recipe(local.as_ref(), target.shape(), &root, &self.args, store)?
            {
                recipe
            } else if let Some(rest) = local.strip_prefix("attn.wo_a.projections.") {
                grouped_output_recipe(rest, &root, &self.args, store)?
            } else {
                let raw = raw_layer_key(&root, local.as_ref());
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            };
            recipes.insert(local.to_string(), recipe);
        }
        Ok(recipes)
    }

    fn draft_static_unit(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Option<StaticUnitBindings>, Error> {
        if let Some(mtp) = &self.static_model.mtp {
            return Ok(Some(StaticUnitBindings::new(
                DRAFT_UNIT,
                build_module_binding_plan_with_recipes(
                    mtp,
                    "mtp",
                    store,
                    draft_recipes(mtp, &self.args, store, false)?,
                )?
                .build_bindings(store)?,
            )?));
        }
        if let Some(dspark) = &self.static_model.dspark {
            return Ok(Some(StaticUnitBindings::new(
                DRAFT_UNIT,
                build_module_binding_plan_with_recipes(
                    dspark,
                    "mtp",
                    store,
                    draft_recipes(dspark, &self.args, store, true)?,
                )?
                .build_bindings(store)?,
            )?));
        }
        Ok(None)
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Self::new_sparse(args, stream)
    }

    pub(crate) fn configure_cartesian_layout(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.configure_parallel_static(build, layout, stream)
    }

    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => Some(&mut self.static_model.model.embed_tokens),
            "norm" => Some(&mut self.static_model.model.norm),
            "hc_head" => Some(&mut self.static_model.model.hc_head),
            "output" => Some(&mut self.static_model.lm_head),
            "draft" => self
                .static_model
                .mtp
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters)
                .or_else(|| {
                    self.static_model
                        .dspark
                        .as_mut()
                        .map(|module| module as &mut dyn ModuleParameters)
                }),
            _ => None,
        }
    }

    pub(crate) fn pipeline_embed(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embedded = self
            .static_model
            .model
            .embed_tokens
            .forward(tokens, stream)?;
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        broadcast_to(
            &hidden,
            &[
                embedded.dim(0),
                embedded.dim(1),
                self.args.hc_mult,
                self.args.hidden_size,
            ],
            stream,
        )
    }

    pub(crate) fn pipeline_finish(
        &mut self,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let hidden = self.static_model.model.hc_head.forward(hidden, stream)?;
        let hidden = self.static_model.model.norm.forward(&hidden, stream)?;
        self.static_model.lm_head.forward(&hidden, stream)
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.static_model.mtp_len()
    }

    pub(crate) fn embedded_mtp_cache(&self) -> super::model::DraftCache {
        self.static_model
            .new_cache()
            .expect("validated DeepSeek V4 draft cache geometry")
            .mtp_layers
    }

    pub(crate) fn embedded_mtp_cache_with_manager(
        &self,
        manager: crate::backend::mlx::runtime::cache::residency::CacheResidencyManager,
        rank: Option<crate::CacheRankIdentity>,
    ) -> Result<super::model::DraftCache, Error> {
        Ok(self
            .static_model
            .new_cache_with_manager(manager, rank)?
            .mtp_layers)
    }

    pub(crate) fn prefill_pipeline_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let mut target_cache = self.static_model.new_cache()?;
        target_cache.mtp_layers.clone_from(cache);
        let pipeline_output;
        let output = if self.args.dspark.is_none() && output.hidden.ndim() == 3 {
            pipeline_output = crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits: output.logits.clone(),
                hidden: output.hidden.reshape(
                    &[
                        output.hidden.dim(0),
                        output.hidden.dim(1),
                        self.args.hc_mult,
                        self.args.hidden_size,
                    ],
                    stream,
                )?,
                tokens: output.tokens.clone(),
            };
            &pipeline_output
        } else {
            output
        };
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::prefill_draft_cache(
            &mut self.static_model,
            output,
            tokens,
            &mut target_cache,
            stream,
        )?;
        cache.clone_from(&target_cache.mtp_layers);
        Ok(())
    }

    pub(crate) fn forward_pipeline_mtp(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let hidden = if hidden.ndim() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            hidden.clone()
        };
        let (logits, hidden) = self
            .static_model
            .forward_mtp_draft(&hidden, tokens, depth, cache, stream)?;
        let hidden = hidden.reshape(
            &[
                hidden.dim(0),
                hidden.dim(1),
                self.args.hc_mult * self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    pub(crate) fn fused_pipeline_draft_logits(
        &mut self,
        hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<Option<Array>, Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::fused_draft_logits(
            &mut self.static_model,
            hidden,
            last_token,
            proposal_capacity,
            cache,
            stream,
        )
    }

    pub(crate) fn adjust_pipeline_fused_logits(
        &mut self,
        logits: Array,
        last_token: u32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::adjust_fused_draft_logits(
            &mut self.static_model,
            logits,
            last_token,
            stream,
        )
    }

    pub(crate) fn advance_pipeline_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut super::model::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let hidden = if self.args.dspark.is_none() && hidden.ndim() == 3 {
            hidden.reshape(
                &[
                    hidden.dim(0),
                    hidden.dim(1),
                    self.args.hc_mult,
                    self.args.hidden_size,
                ],
                stream,
            )?
        } else {
            hidden.clone()
        };
        <ResidentModel as crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget>::advance_draft_cache(
            &mut self.static_model,
            &hidden,
            tokens,
            cache,
            stream,
        )
    }
}

impl LoadTimeQuantizableAdapter for DeepSeekV4LayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let args = self.args.with_load_time_quantization(quantization)?;
        if self.sparse_expert_cache {
            Self::new_sparse(args, stream)
        } else {
            Self::new(args, stream)
        }
    }
}

impl ArchitectureAdapter for DeepSeekV4LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = DeepSeekV4ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, Error> {
        super::checkpoint::safetensors_plan(&self.args).map_err(Error::UnsupportedArchitecture)
    }

    fn quantization(&self) -> Option<WeightQuantization> {
        self.args.quantization
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        if target == "head.weight" {
            return true;
        }
        if !target.starts_with("mtp.") {
            return false;
        }
        if target.ends_with(".gate_up_proj") || target.ends_with(".down_proj") {
            return true;
        }
        target.ends_with(".weight")
            && !target.ends_with("norm.weight")
            && !target.contains(".markov_w1.")
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        prompt_cache_model_identity(
            &self.args,
            topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
        )
    }

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor.clone(),
            prefix_token_ids,
            options,
            descriptor.topology.cache_rank_identity(),
            stream,
        )
    }

    fn load_prompt_cache(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity.clone(),
            options,
            stream,
        )
    }

    fn static_units(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let source = |key: &str| DerivedWeightRecipe::source(key, TensorSelection::Full);
        Ok(vec![
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.static_model.model.embed_tokens,
                    "embed",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.static_model.model.norm,
                    "norm",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                HC_HEAD_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.static_model.model.hc_head,
                    "hc_head",
                    store,
                    BTreeMap::from([
                        ("function".into(), source("hc_head_fn")),
                        ("base".into(), source("hc_head_base")),
                        ("scale".into(), source("hc_head_scale")),
                    ]),
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.static_model.lm_head,
                    "head",
                    store,
                    qwen_linear_recipes("head", &self.static_model.lm_head),
                )?
                .build_bindings(store)?,
            )?,
        ]
        .into_iter()
        .chain(self.draft_static_unit(store)?)
        .collect())
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected =
            4 + usize::from(self.static_model.mtp.is_some() || self.static_model.dspark.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 adapter received {} static leases, expected {expected}",
                leases.len(),
            )));
        }
        populate_module_from_lease(&mut self.static_model.model.embed_tokens, &leases[0])?;
        populate_module_from_lease(&mut self.static_model.model.norm, &leases[1])?;
        populate_module_from_lease(&mut self.static_model.model.hc_head, &leases[2])?;
        populate_module_from_lease(&mut self.static_model.lm_head, &leases[3])?;
        if let Some(lease) = leases.get(4) {
            if let Some(mtp) = &mut self.static_model.mtp {
                populate_module_from_lease(mtp, lease)?;
            } else if let Some(dspark) = &mut self.static_model.dspark {
                populate_module_from_lease(dspark, lease)?;
            }
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache()?;
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(
                "DeepSeek V4 cache does not match decoder depth".into(),
            ));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Cache,
        stream: &Stream,
    ) -> Result<eredu_runtime::LayeredForwardState<Array, Self::ForwardContext>, Error> {
        let embedded = self
            .static_model
            .model
            .embed_tokens
            .forward(input, stream)?;
        let hidden = embedded.try_index_device((.., .., NewAxis, ..), stream)?;
        let hidden = broadcast_to(
            &hidden,
            &[
                embedded.dim(0),
                embedded.dim(1),
                self.args.hc_mult,
                self.args.hidden_size,
            ],
            stream,
        )?;
        Ok(eredu_runtime::LayeredForwardState {
            hidden,
            context: DeepSeekV4ForwardContext {
                input_ids: input.clone(),
                captures: Vec::new(),
                draft_hidden: None,
            },
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.num_hidden_layers as usize)
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek V4 decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new(&self.args, index, stream)?)
    }

    fn register_parallel_parameters(
        &self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let _ = context;
        use crate::backend::mlx::runtime::distributed::parallel::register_replicated_module;

        register_replicated_module(planner, &self.static_model.model.embed_tokens, "embed")?;
        register_replicated_module(planner, &self.static_model.model.norm, "norm")?;
        register_replicated_module(planner, &self.static_model.model.hc_head, "hc_head")?;
        register_replicated_module(planner, &self.static_model.lm_head, "head")?;
        if let Some(mtp) = &self.static_model.mtp {
            register_replicated_module(planner, mtp, "mtp")?;
        }
        if let Some(dspark) = &self.static_model.dspark {
            register_replicated_module(planner, dspark, "mtp")?;
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = DecoderLayer::new(&self.args, index, stream)?;
            register_v4_layer_parallel_plan(planner, &layer, index)?;
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        _layout: &eredu_runtime::LocalModelLayout,
        _stream: &Stream,
    ) -> Result<(), Error> {
        self.parallel_topology = Some(context.topology());
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        let target = format!("layers.{index}.attn.wq_b.weight");
        let query = layout.tensor(&target).ok_or_else(|| {
            Error::Parallel(format!("missing DeepSeek V4 TP layout for {target}"))
        })?;
        if query.fell_back_to_replication() {
            return Ok(DecoderLayer::new(&self.args, index, stream)?);
        }
        let units = query.logical_units().ok_or_else(|| {
            Error::Parallel(format!("DeepSeek V4 TP query {target} has no head domain"))
        })?;
        let global_heads = usize::try_from(self.args.num_attention_heads)
            .map_err(|_| Error::Parallel("DeepSeek V4 head count exceeds usize".into()))?;
        if units == 0 || !global_heads.is_multiple_of(units) {
            return Err(Error::Parallel(format!(
                "DeepSeek V4 query head domain {units} does not divide {global_heads} heads"
            )));
        }
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("DeepSeek V4 TP layer was built before topology binding".into())
        })?;
        let heads_per_unit = global_heads / units;
        let widths = (0..topology.tensor_parallel_size)
            .map(|rank| {
                crate::core::balanced_contiguous_range(
                    units,
                    topology.tensor_parallel_size,
                    rank,
                    false,
                )
                .map(|range| range.len() * heads_per_unit)
                .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let local_heads = i32::try_from(widths[topology.tensor_parallel_rank])
            .map_err(|_| Error::Parallel("DeepSeek V4 local head count exceeds i32".into()))?;
        Ok(DecoderLayer::new_parallel(
            &self.args,
            index,
            local_heads,
            widths,
            stream,
        )?)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.new_layer(group, index, stream)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        _assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.new_parallel_layer(group, index, layout, stream)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::backend::mlx::runtime::distributed::expert::ExpertAssignment>, Error>
    {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if self.args.n_routed_experts <= 0 {
            return Err(Error::Parallel(
                "DeepSeek V4 PP+EP requires routed experts".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.n_routed_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("deepseek_v4.layer.{index:05}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &DecoderLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let bindings = build_module_binding_plan_with_recipes_excluding(
            layer,
            &format!("layers.{index}"),
            store,
            self.layer_recipes(layer, index, store)?,
            |name| self.sparse_expert_cache && name.starts_with("ffn.switch_mlp."),
        )?;
        Ok(bindings.build_bindings(store)?)
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &DecoderLayer,
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
                if target.starts_with("ffn.switch_mlp.") {
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

    fn populate_layer(
        &self,
        _group: usize,
        _index: usize,
        layer: &mut DecoderLayer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.sparse_expert_cache {
            populate_module_from_lease_excluding(layer, lease, |name| {
                name.starts_with("ffn.switch_mlp.")
            })?;
        } else {
            populate_module_from_lease(layer, lease)?;
        }
        Ok(())
    }

    fn additional_consumed_checkpoint_keys(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .source_keys()
                .into_iter()
                .filter(|key| is_routed_expert_source(key))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn forward_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut DecoderLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut DeepSeekV4ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_experts(
                        &self.args,
                        expert_cache,
                        index,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                },
            )?
        } else {
            layer.forward(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                stream,
            )?
        };
        capture_draft_hidden(&self.args, index, &output, context, stream)?;
        Ok(output)
    }

    fn forward_layer_with_execution(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut DecoderLayer,
        hidden: &Array,
        cache: &mut Cache,
        context: &mut DeepSeekV4ForwardContext,
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
        let output = if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            layer.forward_tensor_with_expert_executor(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                tp_group,
                execution.stream(),
                |flat, indices, weights, stream| {
                    execute_cached_experts(
                        &self.args,
                        expert_cache,
                        index,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                },
            )?
        } else {
            layer.forward_tensor_parallel(
                hidden,
                None,
                Some(&mut cache.layers[index]),
                &context.input_ids,
                tp_group,
                execution.stream(),
            )?
        };
        capture_draft_hidden(&self.args, index, &output, context, execution.stream())?;
        Ok(output)
    }

    fn retained_arrays<'a>(&self, cache: &'a Cache, _group: usize, index: usize) -> Vec<&'a Array> {
        cache.layers[index].retained_arrays()
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _context: &DeepSeekV4ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.static_model.model.hc_head.forward(hidden, stream)?;
        let hidden = self.static_model.model.norm.forward(&hidden, stream)?;
        Ok(self.static_model.lm_head.forward(&hidden, stream)?)
    }

    fn ignores_checkpoint_key(&self, _key: &str) -> bool {
        false
    }
}

fn is_routed_expert_source(key: &str) -> bool {
    key.contains(".ffn.experts.") || key.contains(".ffn.expert_banks.")
}

fn deepseek_v4_execution_layout(args: &ModelArgs) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["text_decoder"])?;
    ExecutionUnitLayout::new(&graph, [args.num_hidden_layers as usize])
        .map_err(|error| Error::Parallel(error.to_string()))
}

fn deepseek_v4_static_model(args: &ModelArgs, stream: &Stream) -> Result<ResidentModel, Error> {
    let mut model = ResidentModel::new(args.clone(), stream)?;
    model.model.layers.clear();
    Ok(model)
}

fn deepseek_v4_static_bindings(
    model: &ResidentModel,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let source = |key: &str| DerivedWeightRecipe::source(key, TensorSelection::Full);
    let mut bindings = Vec::new();
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &model.model.embed_tokens,
            "model.embed_tokens",
            store,
            BTreeMap::from([("weight".into(), source("embed.weight"))]),
        )?
        .build_bindings(store)?,
    );
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &model.model.norm,
            "model.norm",
            store,
            BTreeMap::from([("weight".into(), source("norm.weight"))]),
        )?
        .build_bindings(store)?,
    );
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &model.model.hc_head,
            "model.hc_head",
            store,
            BTreeMap::from([
                ("function".into(), source("hc_head_fn")),
                ("base".into(), source("hc_head_base")),
                ("scale".into(), source("hc_head_scale")),
            ]),
        )?
        .build_bindings(store)?,
    );
    bindings.extend(
        build_module_binding_plan_with_recipes(
            &model.lm_head,
            "lm_head",
            store,
            qwen_linear_recipes("head", &model.lm_head),
        )?
        .build_bindings(store)?,
    );
    if let Some(mtp) = &model.mtp {
        bindings.extend(
            build_module_binding_plan_with_recipes(
                mtp,
                "mtp",
                store,
                draft_recipes(mtp, &model.args, store, false)?,
            )?
            .build_bindings(store)?,
        );
    }
    if let Some(dspark) = &model.dspark {
        bindings.extend(
            build_module_binding_plan_with_recipes(
                dspark,
                "dspark",
                store,
                draft_recipes(dspark, &model.args, store, true)?,
            )?
            .build_bindings(store)?,
        );
    }
    Ok(bindings)
}

fn deepseek_v4_runtime_static_bindings(
    model: &ResidentModel,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    deepseek_v4_static_bindings(model, store)?
        .into_iter()
        .map(|binding| {
            let name = binding
                .logical_target()
                .unwrap_or(binding.name())
                .to_string();
            binding.with_name(name).map_err(Into::into)
        })
        .collect()
}

fn deepseek_v4_layer_recipes(
    args: &ModelArgs,
    layer: &DecoderLayer,
    index: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let root = format!("layers.{index}");
    let mut recipes = BTreeMap::new();
    for (local, target) in layer.parameters().flatten() {
        let recipe = if let Some(recipe) =
            expert_bank_recipe(local.as_ref(), target.shape(), &root, args, store)?
        {
            recipe
        } else if let Some(rest) = local.strip_prefix("attn.wo_a.projections.") {
            grouped_output_recipe(rest, &root, args, store)?
        } else {
            DerivedWeightRecipe::source(raw_layer_key(&root, local.as_ref()), TensorSelection::Full)
        };
        recipes.insert(local.to_string(), recipe);
    }
    Ok(recipes)
}

fn deepseek_v4_unit_bindings(
    args: &ModelArgs,
    index: usize,
    layer: &DecoderLayer,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    sparse_experts: bool,
) -> Result<Vec<WeightBinding>, Error> {
    Ok(build_module_binding_plan_with_recipes_excluding(
        layer,
        &format!("layers.{index}"),
        store,
        deepseek_v4_layer_recipes(args, layer, index, store)?,
        |name| sparse_experts && name.starts_with("ffn.switch_mlp."),
    )?
    .build_bindings(store)?)
}

fn build_deepseek_v4_unit(
    args: &ModelArgs,
    index: usize,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    topology: Option<crate::backend::mlx::MlxParallelContext>,
    stream: &Stream,
) -> Result<DecoderLayer, Error> {
    let Some(layout) = layout else {
        return Ok(DecoderLayer::new(args, index, stream)?);
    };
    let target = format!("layers.{index}.attn.wq_b.weight");
    let query = layout
        .tensor(&target)
        .ok_or_else(|| Error::Parallel(format!("missing DeepSeek V4 TP layout for {target}")))?;
    if query.fell_back_to_replication() {
        return Ok(DecoderLayer::new(args, index, stream)?);
    }
    let units = query.logical_units().ok_or_else(|| {
        Error::Parallel(format!("DeepSeek V4 TP query {target} has no head domain"))
    })?;
    let global_heads = usize::try_from(args.num_attention_heads)
        .map_err(|_| Error::Parallel("DeepSeek V4 head count exceeds usize".into()))?;
    if units == 0 || !global_heads.is_multiple_of(units) {
        return Err(Error::Parallel(format!(
            "DeepSeek V4 query head domain {units} does not divide {global_heads} heads"
        )));
    }
    let topology = topology.ok_or_else(|| {
        Error::Parallel("DeepSeek V4 TP unit was built without a topology".into())
    })?;
    let heads_per_unit = global_heads / units;
    let widths = (0..topology.tensor_parallel_size)
        .map(|rank| {
            crate::core::balanced_contiguous_range(
                units,
                topology.tensor_parallel_size,
                rank,
                false,
            )
            .map(|range| range.len() * heads_per_unit)
            .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let local_heads = i32::try_from(widths[topology.tensor_parallel_rank])
        .map_err(|_| Error::Parallel("DeepSeek V4 local head count exceeds i32".into()))?;
    Ok(DecoderLayer::new_parallel(
        args,
        index,
        local_heads,
        widths,
        stream,
    )?)
}

fn resolve_deepseek_v4_store(
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
                "DeepSeek V4 checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn quantize_deepseek_v4_store(
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
    let target_args = source_args.with_load_time_quantization(quantization)?;
    let source_static = deepseek_v4_static_model(source_args, stream)?;
    let target_static = deepseek_v4_static_model(&target_args, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &source_static,
        &target_static,
        move |index, stream| Ok(DecoderLayer::new(&source_units, index, stream)?),
        move |index, stream| Ok(DecoderLayer::new(&target_units, index, stream)?),
        source_args.num_hidden_layers as usize,
        quantization,
        stream,
        deepseek_v4_static_bindings,
        move |index, layer, store| {
            deepseek_v4_unit_bindings(&binding_args, index, layer, store, sparse_experts)
        },
    )?;
    Ok((store, target_args, report))
}

fn load_deepseek_v4_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let store = resolve_deepseek_v4_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_deepseek_v4_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut architecture = DeepSeekV4Architecture::new(args.clone(), sparse_experts, stream)?;
    let factory = DeepSeekV4UnitFactory {
        args: args.clone(),
        parallel_layout: None,
        parallel_topology: None,
        sparse_experts,
    };
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        deepseek_v4_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && is_routed_expert_source(key),
        |modules, store| deepseek_v4_runtime_static_bindings(&**modules, store),
        move |index, unit, store, _| {
            deepseek_v4_unit_bindings(&binding_args, index, &unit, store, sparse_experts)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        DeepSeekV4Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DeepSeekV4Execution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(DeepSeekV4LayerwiseModel {
        execution,
        metadata,
        parallel_info: None,
        parallel_topology: None,
    })
}

fn register_deepseek_v4_parallel_parameters(
    planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    use crate::backend::mlx::runtime::distributed::parallel::register_replicated_module;

    let model = deepseek_v4_static_model(args, stream)?;
    register_replicated_module(planner, &model.model.embed_tokens, "model.embed_tokens")?;
    register_replicated_module(planner, &model.model.norm, "model.norm")?;
    register_replicated_module(planner, &model.model.hc_head, "model.hc_head")?;
    register_replicated_module(planner, &model.lm_head, "lm_head")?;
    if let Some(mtp) = &model.mtp {
        register_replicated_module(planner, mtp, "mtp")?;
    }
    if let Some(dspark) = &model.dspark {
        register_replicated_module(planner, dspark, "dspark")?;
    }
    for index in 0..args.num_hidden_layers as usize {
        let layer = DecoderLayer::new(args, index, stream)?;
        register_v4_layer_parallel_plan(planner, &layer, index)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_deepseek_v4_parallel_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let store = resolve_deepseek_v4_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_deepseek_v4_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut planner = build.planner();
    register_deepseek_v4_parallel_parameters(&mut planner, &args, stream)?;
    let (_, local_layout) = planner.finish()?;
    if local_layout.is_empty() {
        return Err(Error::Parallel(
            "DeepSeek V4 declared no tensor-parallel parameters".into(),
        ));
    }

    let mut architecture = DeepSeekV4Architecture::new(args.clone(), sparse_experts, stream)?;
    architecture.parallel_topology = Some(build.topology());
    let global_static = deepseek_v4_static_model(&args, stream)?;
    let static_bindings = deepseek_v4_static_bindings(&global_static, store.as_ref())?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    for index in 0..args.num_hidden_layers as usize {
        let layer = DecoderLayer::new(&args, index, stream)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&deepseek_v4_unit_bindings(
                &args,
                index,
                &layer,
                store.as_ref(),
                sparse_experts,
            )?)?)
            .ok_or_else(|| {
                Error::Parallel("DeepSeek V4 global parameter bytes overflowed".into())
            })?;
    }

    let shared_layout = Arc::new(local_layout);
    let factory = DeepSeekV4UnitFactory {
        args: args.clone(),
        parallel_layout: Some(Arc::clone(&shared_layout)),
        parallel_topology: Some(build.topology()),
        sparse_experts,
    };
    let static_layout = Arc::clone(&shared_layout);
    let unit_layout = Arc::clone(&shared_layout);
    let binding_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        deepseek_v4_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && is_routed_expert_source(key),
        move |_, store| {
            shard_layer_bindings(static_bindings, "", store, &static_layout).and_then(|bindings| {
                bindings
                    .into_iter()
                    .map(|binding| {
                        let name = binding
                            .logical_target()
                            .unwrap_or(binding.name())
                            .to_string();
                        binding.with_name(name).map_err(Into::into)
                    })
                    .collect()
            })
        },
        move |index, _local, store, stream| {
            let global = DecoderLayer::new(&binding_args, index, stream)?;
            let bindings =
                deepseek_v4_unit_bindings(&binding_args, index, &global, store, sparse_experts)?;
            shard_layer_bindings(bindings, &format!("layers.{index}"), store, &unit_layout)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    metadata.set_materialization(materialization);
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("DeepSeek V4 local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("DeepSeek V4 device parameter bytes overflowed".into()))?;
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
        DeepSeekV4Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DeepSeekV4Execution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(DeepSeekV4LayerwiseModel {
        execution,
        metadata,
        parallel_info: Some(info),
        parallel_topology: Some(build.topology()),
    })
}

/// Loads V4 with resident, host-windowed, or dense disk-streamed layers.
pub fn load_deepseek_v4_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = super::model::get_model_args(model_dir)?;
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4", requested_quantization)?;
    load_deepseek_v4_with_store(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        args,
        options,
        quantization,
        false,
        stream,
        weights_stream,
    )
}

/// Loads a canonical llama.cpp `deepseek4` GGUF through the generalized
/// resident/layerwise/dense-stream and independent-expert-cache engine.
pub(crate) fn load_deepseek_v4_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV4LayerwiseModel, Vec<u32>), Error> {
    crate::composition::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek4,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = super::model::prepare_gguf_checkpoint(checkpoint, metadata)?;
    let quantization = prepared
        .args
        .resolve_load_time_quantization("DeepSeek V4 GGUF", requested_quantization)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            super::model::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    let sparse_experts = residency.expert_cache().is_some();
    let mut model = load_deepseek_v4_with_store(
        Arc::clone(&store),
        prepared.args.clone(),
        residency.layers(),
        quantization,
        sparse_experts,
        stream,
        weights_stream,
    )?;
    if let Some(options) = residency.expert_cache() {
        let checkpoint_store = model.checkpoint_store_arc();
        let entries = expert_catalog(&prepared.args, checkpoint_store.as_ref())?;
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
    }
    Ok((model, prepared.eos_token_ids))
}

/// Loads a canonical `deepseek4` GGUF with TP composed with any layer residency.
pub(crate) fn load_deepseek_v4_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV4LayerwiseModel, Vec<u32>), Error> {
    let residency = WeightResidency::with_layers(options);
    crate::composition::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek4,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = super::model::prepare_gguf_checkpoint(checkpoint, metadata)?;
    let quantization = prepared.args.resolve_load_time_quantization(
        "DeepSeek V4 GGUF tensor parallel",
        requested_quantization,
    )?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            super::model::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let model = load_deepseek_v4_parallel_with_store(
        store,
        prepared.args,
        options,
        quantization,
        build,
        false,
        stream,
        weights_stream,
    )?;
    Ok((model, prepared.eos_token_ids))
}

/// Loads V4 through the generalized tensor-parallel and residency engine.
pub(crate) fn load_deepseek_v4_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_deepseek_v4_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            requested_quantization,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = super::model::get_model_args(model_dir)?;
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4 tensor parallel", requested_quantization)?;
    load_deepseek_v4_parallel_with_store(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        args,
        options,
        quantization,
        build,
        false,
        stream,
        weights_stream,
    )
}

/// Loads V4 with routed experts in independent cache units.
pub fn load_deepseek_v4_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = super::model::get_model_args(model_dir)?;
    let quantization = args.resolve_load_time_quantization(
        "DeepSeek V4 independent expert cache",
        requested_quantization,
    )?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut model = load_deepseek_v4_with_store(
        Arc::clone(&store),
        args.clone(),
        non_expert.into(),
        quantization,
        true,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = model.checkpoint_store_arc();
    let entries = expert_catalog(&args, checkpoint_store.as_ref())?;
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

/// Builds the non-expert V4 base used by pure EP execution.
pub(crate) fn load_deepseek_v4_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let quantization =
        args.resolve_load_time_quantization("DeepSeek V4 expert parallel", requested_quantization)?;
    load_deepseek_v4_with_store(
        store,
        args,
        non_expert.into(),
        quantization,
        true,
        stream,
        weights_stream,
    )
}

/// Builds the TP-sharded non-expert V4 base used by TP+EP execution.
pub(crate) fn load_deepseek_v4_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    requested_quantization: Option<WeightQuantization>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV4LayerwiseModel, Error> {
    let quantization = args.resolve_load_time_quantization(
        "DeepSeek V4 tensor/expert parallel",
        requested_quantization,
    )?;
    load_deepseek_v4_parallel_with_store(
        store,
        args,
        non_expert.into(),
        quantization,
        build,
        true,
        stream,
        weights_stream,
    )
}

fn capture_draft_hidden(
    args: &ModelArgs,
    index: usize,
    output: &Array,
    context: &mut DeepSeekV4ForwardContext,
    stream: &Stream,
) -> Result<(), Error> {
    if let Some(dspark) = &args.dspark {
        if let Some(position) = dspark
            .target_layer_ids
            .iter()
            .position(|wanted| *wanted == index as i32)
        {
            context
                .captures
                .push((position, safemlx::ops::mean_axis(output, 2, false, stream)?));
        }
    }
    if index + 1 == args.num_hidden_layers as usize {
        context.draft_hidden = if args.dspark.is_some() {
            context.captures.sort_by_key(|(position, _)| *position);
            Some(safemlx::ops::concatenate_axis(
                &context
                    .captures
                    .iter()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>(),
                -1,
                stream,
            )?)
        } else {
            Some(output.clone())
        };
    }
    Ok(())
}

fn register_v4_layer_parallel_plan(
    planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
    layer: &DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    use crate::backend::mlx::runtime::distributed::parallel::{
        array_parameter_member, partitioned_projection_members, register_replicated_module,
        ProjectionSharding,
    };
    use eredu_runtime::{MemberSharding, ParameterGroupSpec, ParameterRole};

    let prefix = format!("layers.{index}");
    let attention = &layer.attn;
    let query_prefix = format!("{prefix}.attn.wq_b");
    let preferred_heads = usize::try_from(attention.heads)
        .map_err(|_| Error::Parallel("DeepSeek V4 query head count exceeds usize".into()))?;
    let (units, mut members) = partitioned_projection_members(
        &[(
            &attention.wq_b,
            query_prefix.as_str(),
            ProjectionSharding::Column,
        )],
        preferred_heads,
    )?;
    members.push(array_parameter_member(
        format!("{prefix}.attn.attn_sink"),
        attention.attn_sink.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.attn.query_heads"),
        ParameterRole::AttentionHeads,
        units,
        members,
    )?)?;

    register_replicated_module(planner, &attention.wq_a, &format!("{prefix}.attn.wq_a"))?;
    register_replicated_module(planner, &attention.q_norm, &format!("{prefix}.attn.q_norm"))?;
    register_replicated_module(planner, &attention.wkv, &format!("{prefix}.attn.wkv"))?;
    register_replicated_module(
        planner,
        &attention.kv_norm,
        &format!("{prefix}.attn.kv_norm"),
    )?;
    register_replicated_module(planner, &attention.wo_a, &format!("{prefix}.attn.wo_a"))?;
    register_replicated_module(planner, &attention.wo_b, &format!("{prefix}.attn.wo_b"))?;
    register_replicated_module(planner, &layer.ffn, &format!("{prefix}.ffn"))?;
    register_replicated_module(planner, &layer.attn_norm, &format!("{prefix}.attn_norm"))?;
    register_replicated_module(planner, &layer.ffn_norm, &format!("{prefix}.ffn_norm"))?;
    register_replicated_module(planner, &layer.attn_hc, &format!("{prefix}.attn_hc"))?;
    register_replicated_module(planner, &layer.ffn_hc, &format!("{prefix}.ffn_hc"))?;
    if let Some(compressor) = &attention.compressor {
        register_replicated_module(planner, compressor, &format!("{prefix}.attn.compressor"))?;
    }
    if let Some(indexer) = &attention.indexer {
        register_replicated_module(planner, indexer, &format!("{prefix}.attn.indexer"))?;
    }
    Ok(())
}

fn raw_layer_key(root: &str, local: &str) -> String {
    let mut key = format!("{root}.{local}");
    for sublayer in ["attn", "ffn"] {
        for (runtime, raw) in [("function", "fn"), ("base", "base"), ("scale", "scale")] {
            key = key.replace(
                &format!(".{sublayer}_hc.{runtime}"),
                &format!(".hc_{sublayer}_{raw}"),
            );
        }
    }
    for (runtime, raw) in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")] {
        key = key.replace(
            &format!(".ffn.shared_experts.{runtime}."),
            &format!(".ffn.shared_experts.{raw}."),
        );
    }
    key = key
        .replace(".ffn.gate.router.weight", ".ffn.gate.weight")
        .replace(".ffn.gate.router.e_score_correction_bias", ".ffn.gate.bias");
    if key.ends_with(".weight_scale_inv") {
        key.truncate(key.len() - ".weight_scale_inv".len());
        key.push_str(".scale");
    }
    key
}

fn grouped_output_recipe(
    rest: &str,
    root: &str,
    args: &ModelArgs,
    _store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<DerivedWeightRecipe, Error> {
    let (group, component) = rest.split_once('.').ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("invalid V4 grouped output parameter {rest:?}"))
    })?;
    let group: usize = group.parse().map_err(|_| {
        Error::UnsupportedArchitecture(format!("invalid V4 output group {group:?}"))
    })?;
    let (source, rows) = match component {
        "weight" => (
            format!("{root}.attn.wo_a.weight"),
            args.o_lora_rank as usize,
        ),
        "weight_scale_inv" => (
            format!("{root}.attn.wo_a.scale"),
            ((args.o_lora_rank + 127) / 128) as usize,
        ),
        "scales" => (
            format!("{root}.attn.wo_a.scales"),
            args.o_lora_rank as usize,
        ),
        "biases" => (
            format!("{root}.attn.wo_a.biases"),
            args.o_lora_rank as usize,
        ),
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported V4 grouped output component {other:?}"
            )))
        }
    };
    Ok(DerivedWeightRecipe::source(
        source,
        TensorSelection::Range {
            axis: 0,
            start: group * rows,
            end: (group + 1) * rows,
        },
    ))
}

fn expert_bank_recipe(
    local: &str,
    target_shape: &[i32],
    root: &str,
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let component = match local {
        "ffn.switch_mlp.gate_up_proj" => Some((true, "weight")),
        "ffn.switch_mlp.gate_up_proj_scales" => Some((true, "scale")),
        "ffn.switch_mlp.down_proj" => Some((false, "weight")),
        "ffn.switch_mlp.down_proj_scales" => Some((false, "scale")),
        _ => None,
    };
    let Some((gate_up, component)) = component else {
        return Ok(None);
    };
    let bank = |projection: &str| format!("{root}.ffn.expert_banks.{projection}.{component}");
    if store.source_metadata(&bank("w1")).is_ok() {
        let source =
            |projection: &str| DerivedWeightRecipe::source(bank(projection), TensorSelection::Full);
        let recipe = if gate_up {
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![source("w1"), source("w3")],
            }
        } else {
            source("w2")
        };
        recipe.infer(store)?;
        return Ok(Some(recipe));
    }
    let source = |expert: i32, projection: &str| {
        DerivedWeightRecipe::source(
            format!("{root}.ffn.experts.{expert}.{projection}.{component}"),
            TensorSelection::Full,
        )
    };
    let mut experts = Vec::with_capacity(args.n_routed_experts as usize);
    for expert in 0..args.n_routed_experts {
        experts.push(if gate_up {
            DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source(expert, "w1"), source(expert, "w3")],
            }
        } else {
            source(expert, "w2")
        });
    }
    let mut recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: experts,
    };
    if component == "weight" && args.expert_dtype.as_deref() == Some("fp4") {
        recipe = DerivedWeightRecipe::View {
            input: Box::new(recipe),
            dtype: RecipeDtype::U32,
            shape: target_shape.iter().map(|value| *value as usize).collect(),
        };
    }
    recipe.infer(store)?;
    Ok(Some(recipe))
}

fn qwen_linear_recipes(
    raw_prefix: &str,
    linear: &crate::composition::mlx_architectures::qwen::hybrid::qwen3_5::QwenLinear,
) -> BTreeMap<String, DerivedWeightRecipe> {
    let mut recipes = BTreeMap::from([(
        "weight".into(),
        DerivedWeightRecipe::source(format!("{raw_prefix}.weight"), TensorSelection::Full),
    )]);
    if linear.weight_scale_inv.as_ref().is_some() {
        recipes.insert(
            "weight_scale_inv".into(),
            DerivedWeightRecipe::source(format!("{raw_prefix}.scale"), TensorSelection::Full),
        );
    }
    recipes
}

fn draft_recipes<M: ModuleParameters>(
    module: &M,
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    dspark: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let last = args.num_nextn_predict_layers as usize - 1;
    let mut recipes = BTreeMap::new();
    for (local, target) in module.parameters().flatten() {
        let recipe = if dspark {
            if let Some(rest) = local.strip_prefix("layers.") {
                let (depth, field) = rest.split_once('.').ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!("invalid DSpark parameter {local:?}"))
                })?;
                draft_decoder_recipe(field, target.shape(), &format!("mtp.{depth}"), args, store)?
            } else {
                let raw = match local.as_ref() {
                    "main_proj.weight" => "mtp.0.main_proj.weight".into(),
                    "main_proj.weight_scale_inv" => "mtp.0.main_proj.scale".into(),
                    "main_norm.weight" => "mtp.0.main_norm.weight".into(),
                    "norm.weight" => format!("mtp.{last}.norm.weight"),
                    "hc_head.function" => format!("mtp.{last}.hc_head_fn"),
                    "hc_head.base" => format!("mtp.{last}.hc_head_base"),
                    "hc_head.scale" => format!("mtp.{last}.hc_head_scale"),
                    "markov_w1.weight" => {
                        format!("mtp.{last}.markov_head.markov_w1.weight")
                    }
                    "markov_w2.weight" => {
                        format!("mtp.{last}.markov_head.markov_w2.weight")
                    }
                    "confidence_head.weight" => {
                        format!("mtp.{last}.confidence_head.proj.weight")
                    }
                    other => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "unsupported DSpark static parameter {other:?}"
                        )))
                    }
                };
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            }
        } else {
            let rest = local.strip_prefix("layers.").ok_or_else(|| {
                Error::UnsupportedArchitecture(format!("invalid V4 MTP parameter {local:?}"))
            })?;
            let (depth, field) = rest.split_once('.').ok_or_else(|| {
                Error::UnsupportedArchitecture(format!("invalid V4 MTP parameter {local:?}"))
            })?;
            let root = format!("mtp.{depth}");
            if let Some(field) = field.strip_prefix("decoder.") {
                draft_decoder_recipe(field, target.shape(), &root, args, store)?
            } else {
                let raw = match field {
                    "hc_head.function" => format!("{root}.hc_head_fn"),
                    "hc_head.base" => format!("{root}.hc_head_base"),
                    "hc_head.scale" => format!("{root}.hc_head_scale"),
                    other if other.ends_with(".weight_scale_inv") => {
                        format!(
                            "{root}.{}.scale",
                            other.trim_end_matches(".weight_scale_inv")
                        )
                    }
                    other => format!("{root}.{other}"),
                };
                DerivedWeightRecipe::source(raw, TensorSelection::Full)
            }
        };
        recipes.insert(local.to_string(), recipe);
    }
    Ok(recipes)
}

fn draft_decoder_recipe(
    field: &str,
    target_shape: &[i32],
    root: &str,
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<DerivedWeightRecipe, Error> {
    if let Some(recipe) = expert_bank_recipe(field, target_shape, root, args, store)? {
        return Ok(recipe);
    }
    if let Some(rest) = field.strip_prefix("attn.wo_a.projections.") {
        return grouped_output_recipe(rest, root, args, store);
    }
    Ok(DerivedWeightRecipe::source(
        raw_layer_key(root, field),
        TensorSelection::Full,
    ))
}

pub(crate) fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("layers.{layer}.ffn.experts");
        let bank_root = format!("layers.{layer}.ffn.expert_banks");
        let fused_banks = store
            .source_metadata(&format!("{bank_root}.w1.weight"))
            .is_ok();
        for expert in 0..args.n_routed_experts as usize {
            let mut bindings = Vec::new();
            for (name, gate_up, component) in [
                ("gate_up_proj", true, "weight"),
                ("down_proj", false, "weight"),
                ("gate_up_proj_scales", true, "scale"),
                ("down_proj_scales", false, "scale"),
            ] {
                let scale_probe = if fused_banks {
                    format!("{bank_root}.w1.scale")
                } else {
                    format!("{root}.{expert}.w1.scale")
                };
                if component == "scale"
                    && args.expert_dtype.as_deref().is_none()
                    && store.source_metadata(&scale_probe).is_err()
                {
                    continue;
                }
                let source = |projection: &str| {
                    DerivedWeightRecipe::source(
                        if fused_banks {
                            format!("{bank_root}.{projection}.{component}")
                        } else {
                            format!("{root}.{expert}.{projection}.{component}")
                        },
                        if fused_banks {
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            }
                        } else {
                            TensorSelection::Full
                        },
                    )
                };
                let mut recipe = if gate_up {
                    let recipe = DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![source("w1"), source("w3")],
                    };
                    if fused_banks {
                        recipe
                    } else {
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![source("w1"), source("w3")],
                            }],
                        }
                    }
                } else if fused_banks {
                    source("w2")
                } else {
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![source("w2")],
                    }
                };
                if component == "weight" && args.expert_dtype.as_deref() == Some("fp4") {
                    let output = if gate_up {
                        2 * args.moe_intermediate_size
                    } else {
                        args.hidden_size
                    };
                    let input = if gate_up {
                        args.hidden_size
                    } else {
                        args.moe_intermediate_size
                    };
                    recipe = DerivedWeightRecipe::View {
                        input: Box::new(recipe),
                        dtype: RecipeDtype::U32,
                        shape: vec![1, output as usize, (input / 8) as usize],
                    };
                }
                bindings.push(deepseek_v4_recipe_binding(name, recipe, store)?);
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("V4 expert byte total overflowed".into())
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

fn deepseek_v4_recipe_binding(
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

#[allow(clippy::too_many_arguments)]
fn execute_cached_experts(
    args: &ModelArgs,
    cache: &ExpertCache,
    layer: usize,
    flat: &Array,
    indices: &Array,
    weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Exception> {
    cache
        .execute_routes_bounded(
            ExpertRouteBatch::new(layer, flat, indices, weights, pass),
            stream,
            |flat, acquired, weights, stream| {
                if acquired.is_empty() {
                    return Err(ExpertCacheError::EmptyRoutedBank {
                        architecture: "DeepSeek-V4",
                    });
                }
                let started = Instant::now();
                let quantization = match args.expert_dtype.as_deref() {
                    Some("fp4") => Some(WeightQuantization::MxFp4),
                    Some("fp8") => None,
                    None => args.quantization,
                    Some(_) => unreachable!("validated expert dtype"),
                };
                let bank = crate::backend::mlx::nn::moe::PackedSwiGluExperts::new(
                    acquired.identities().len() as i32,
                    args.hidden_size,
                    args.moe_intermediate_size,
                    quantization,
                    quantization,
                    stream,
                )?;
                let mut bank = if args.expert_dtype.as_deref() == Some("fp8") {
                    bank.with_native_fp8_e8m0(stream)?
                } else {
                    bank
                }
                .with_swiglu_limit(args.swiglu_limit)?;
                bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
                bank.gate_up_proj_scales =
                    Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
                bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
                bank.down_proj_scales =
                    Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
                cache.record_compact_bank(pass, acquired.scratch_bytes(), started.elapsed())?;
                Ok(bank.forward(flat, acquired.compact_routes(), weights, stream)?)
            },
        )
        .map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
mod neutral_runtime_tests {
    #[test]
    fn production_model_and_loaders_use_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let wrapper_start = source
            .find("pub struct DeepSeekV4LayerwiseModel")
            .expect("DeepSeek V4 production wrapper");
        let adapter_start = source
            .find("/// Architecture adapter used by the generalized layerwise executor.")
            .expect("pipeline-only legacy adapter marker");
        let wrapper = &source[wrapper_start..adapter_start];
        assert!(wrapper.contains("DeepSeekV4Execution"));
        for legacy in ["LayerwiseModel<", ".adapter()", ".adapter_mut()"] {
            assert!(
                !wrapper.contains(legacy),
                "production DeepSeek V4 wrapper still references {legacy}"
            );
        }

        let loaders_start = source
            .find("fn load_deepseek_v4_with_store")
            .expect("neutral DeepSeek V4 loader");
        let tests_start = source
            .find("#[cfg(test)]\nmod neutral_runtime_tests")
            .expect("DeepSeek V4 source-boundary tests");
        let loaders = &source[loaders_start..tests_start];
        for legacy in [
            "load_layerwise_model_with_quantization(",
            "load_tensor_parallel_layerwise_model_with_quantization(",
            "DeepSeekV4LayerwiseAdapter::new(",
            "DeepSeekV4LayerwiseAdapter::new_sparse(",
        ] {
            assert!(
                !loaders.contains(legacy),
                "production DeepSeek V4 loaders still reference {legacy}"
            );
        }
    }
}
