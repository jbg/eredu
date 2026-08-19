//! Bounded layer execution for DeepSeek-V3 and DeepSeek-R1 checkpoints.

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
    module::{Module, ModuleParamMut, ModuleParamRef, ModuleParameters, Param},
    nn,
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        shared::{MlxBackend, MlxParameterTree},
        tensor::create_causal_mask,
    },
    backend::mlx::runtime::checkpoint::binding::{
        binding_bytes, build_module_binding_plan_with_recipes,
        build_module_binding_plan_with_recipes_excluding, canonical_checkpoint_name, ModuleBindingPlan,
    },
    backend::mlx::runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    backend::mlx::runtime::checkpoint::store::{open_gguf_checkpoint_source, TensorSelection},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, recipe::DerivedWeightRecipe,
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, partitioned_projection_members,
        register_partitioned_projection_group, register_projection_module,
        register_replicated_module, ParallelPlanBuilder, ProjectionSharding,
    },
    backend::mlx::runtime::execution::generic::{
        prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        MlxUnitFactory,
    },
    backend::mlx::runtime::execution::layerwise::{
        open_safetensors_weight_store, quantize_module_store_with_bindings, shard_layer_bindings,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheReport, ExpertCatalogEntry, ExpertRouteBatch,
    },
    composition::mlx_architectures::deepseek_v3::model::{
        self as resident, Cache, DecoderLayer, LayerPolicy, ModelArgs,
    },
};
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "deepseek_v3.static.embedding";
const NORM_UNIT: &str = "deepseek_v3.static.norm";
const HEAD_UNIT: &str = "deepseek_v3.static.output";
const MTP_UNIT: &str = "deepseek_v3.static.mtp";

type DeepSeekMtpExpertExecutor<'a> =
    dyn FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception> + 'a;

#[derive(Debug, Clone, ModuleParameters)]
struct DeepSeekMtpLayer {
    #[param]
    enorm: nn::RmsNorm,
    #[param]
    hnorm: nn::RmsNorm,
    #[param]
    eh_proj: MaybeQuantized<nn::Linear>,
    #[param]
    decoder: DecoderLayer,
    #[param]
    shared_norm: nn::RmsNorm,
    #[param]
    shared_head: crate::composition::mlx::speculative::embedded::EmbeddedMtpVocabHead,
}

#[derive(Debug, Clone, ModuleParameters)]
struct DeepSeekMtpModule {
    #[param]
    layers: Vec<DeepSeekMtpLayer>,
}

impl DeepSeekMtpModule {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Option<Self>, Error> {
        let count = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
            Error::UnsupportedArchitecture("DeepSeek MTP layer count is negative".into())
        })?;
        if count == 0 {
            return Ok(None);
        }
        let mut policies = args.layer_schedule.iter().cloned().collect::<Vec<_>>();
        policies.extend(std::iter::repeat_n(LayerPolicy::SparseMoe, count));
        let mut mtp_args = args.clone();
        mtp_args.layer_schedule = crate::LayerSchedule::new(policies.len(), policies)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let layers = (0..count)
            .map(|index| {
                let global = args.num_hidden_layers + index as i32;
                Ok(DeepSeekMtpLayer {
                    enorm: nn::RmsNorm::unloaded(
                        args.hidden_size,
                        args.rms_norm_eps,
                        Dtype::Float32,
                        stream,
                    )?,
                    hnorm: nn::RmsNorm::unloaded(
                        args.hidden_size,
                        args.rms_norm_eps,
                        Dtype::Float32,
                        stream,
                    )?,
                    eh_proj: common::linear::unloaded_maybe_quantized_linear(
                        args.hidden_size * 2,
                        args.hidden_size,
                        false,
                        args.weight_quantization_for(&format!(
                            "model.layers.{global}.eh_proj.weight"
                        )),
                        stream,
                    )?,
                    decoder: DecoderLayer::new_layerwise(&mtp_args, global, stream)?,
                    shared_norm: nn::RmsNorm::unloaded(
                        args.hidden_size,
                        args.rms_norm_eps,
                        Dtype::Float32,
                        stream,
                    )?,
                    shared_head:
                        crate::composition::mlx::speculative::embedded::EmbeddedMtpVocabHead::new(
                            args.hidden_size,
                            args.vocab_size as usize,
                            args.weight_quantization_for(&format!(
                                "model.layers.{global}.shared_head.head.weight"
                            )),
                            stream,
                        )?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Some(Self { layers }))
    }

    fn len(&self) -> usize {
        self.layers.len()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [crate::backend::mlx::runtime::cache::CompressedLatentCache],
        expert_cache: Option<&ExpertCache>,
        mut external_expert: Option<&mut DeepSeekMtpExpertExecutor<'_>>,
        args: &ModelArgs,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let layer_count = self.layers.len();
        let cache_count = cache.len();
        let layer = self
            .layers
            .get_mut(depth % layer_count)
            .ok_or_else(|| Exception::custom("DeepSeek checkpoint does not contain MTP layers"))?;
        let cache = cache.get_mut(depth % cache_count).ok_or_else(|| {
            Exception::custom("DeepSeek MTP cache does not match prediction layers")
        })?;
        let embeddings = layer.enorm.forward(embeddings, stream)?;
        let hidden = layer.hnorm.forward(hidden, stream)?;
        let fused = concatenate_axis(&[&embeddings, &hidden], -1, stream)?;
        let fused = layer.eh_proj.forward(&fused, stream)?;
        let mask = (fused.dim(1) > 1)
            .then(|| create_causal_mask(fused.dim(1), Some(cache.offset()), None, None, stream))
            .transpose()?;
        let hidden = if expert_cache.is_some() || external_expert.is_some() {
            let global = args.num_hidden_layers as usize + depth % layer_count;
            let pass = if fused.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            let execute = |flat: &Array, indices: &Array, weights: &Array, stream: &Stream| {
                if let Some(execute) = external_expert.as_deref_mut() {
                    execute(global, flat, indices, weights, stream)
                } else {
                    execute_cached_deepseek_experts(
                        args,
                        expert_cache.expect("checked DeepSeek MTP expert source"),
                        global,
                        flat,
                        indices,
                        weights,
                        pass,
                        stream,
                    )
                }
            };
            match execution.filter(|execution| execution.is_tensor_parallel()) {
                Some(execution) => layer.decoder.forward_tensor_with_expert_executor(
                    &fused,
                    mask.as_ref(),
                    Some(cache),
                    execution.group().ok_or_else(|| {
                        Exception::custom("DeepSeek MTP TP execution is missing its group")
                    })?,
                    stream,
                    execute,
                )?,
                None => layer.decoder.forward_sparse_experts(
                    &fused,
                    mask.as_ref(),
                    Some(cache),
                    stream,
                    execute,
                )?,
            }
        } else {
            match execution.filter(|execution| execution.is_tensor_parallel()) {
                Some(execution) => layer.decoder.forward_tensor_parallel(
                    &fused,
                    mask.as_ref(),
                    Some(cache),
                    execution.group().ok_or_else(|| {
                        Exception::custom("DeepSeek MTP TP execution is missing its group")
                    })?,
                    stream,
                )?,
                None => layer
                    .decoder
                    .forward_stage(&fused, mask.as_ref(), Some(cache), stream)?,
            }
        };
        let normalized = layer.shared_norm.forward(&hidden, stream)?;
        let logits = layer.shared_head.forward(&normalized, execution, stream)?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }
}

#[derive(Debug, Clone)]
enum DeepSeekV3Embedding {
    Replicated(MaybeQuantized<nn::Embedding>),
    Parallel(VocabParallelEmbedding),
}

impl ModuleParameters for DeepSeekV3Embedding {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Replicated(module) => module.num_parameters(),
            Self::Parallel(module) => module.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Replicated(module) => module.parameters(),
            Self::Parallel(module) => module.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Replicated(module) => module.parameters_mut(),
            Self::Parallel(module) => module.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Replicated(module) => module.trainable_parameters(),
            Self::Parallel(module) => module.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(module) => module.freeze_parameters(recursive),
            Self::Parallel(module) => module.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(module) => module.unfreeze_parameters(recursive),
            Self::Parallel(module) => module.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(module) => module.all_frozen(),
            Self::Parallel(module) => module.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(module) => module.any_frozen(),
            Self::Parallel(module) => module.any_frozen(),
        }
    }
}

#[derive(Debug, Clone)]
enum DeepSeekV3Head {
    Replicated(MaybeQuantized<nn::Linear>),
    Parallel(VocabParallelLmHead),
}

impl ModuleParameters for DeepSeekV3Head {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Replicated(module) => module.num_parameters(),
            Self::Parallel(module) => module.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Replicated(module) => module.parameters(),
            Self::Parallel(module) => module.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Replicated(module) => module.parameters_mut(),
            Self::Parallel(module) => module.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Replicated(module) => module.trainable_parameters(),
            Self::Parallel(module) => module.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(module) => module.freeze_parameters(recursive),
            Self::Parallel(module) => module.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Replicated(module) => module.unfreeze_parameters(recursive),
            Self::Parallel(module) => module.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(module) => module.all_frozen(),
            Self::Parallel(module) => module.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Replicated(module) => module.any_frozen(),
            Self::Parallel(module) => module.any_frozen(),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DeepSeekV3StaticModel {
    #[param]
    embedding: DeepSeekV3Embedding,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: DeepSeekV3Head,
    #[param]
    mtp: Option<DeepSeekMtpModule>,
}

type DeepSeekV3Unit = MlxParameterTree<DecoderLayer>;
type DeepSeekV3Static = MlxParameterTree<DeepSeekV3StaticModel>;
type DeepSeekV3ResidentRuntime =
    LayerwiseRuntime<DeepSeekV3Architecture, MlxBackend, Cache, MlxResidentPolicy<DeepSeekV3Unit>>;
type DeepSeekV3BoundedRuntime = LayerwiseRuntime<
    DeepSeekV3Architecture,
    MlxBackend,
    Cache,
    MlxLayerwisePolicy<DeepSeekV3Unit, DeepSeekV3UnitFactory>,
>;

enum DeepSeekV3Execution {
    Resident(DeepSeekV3ResidentRuntime),
    Layerwise(DeepSeekV3BoundedRuntime),
}

#[derive(Clone)]
struct DeepSeekV3UnitFactory {
    args: ModelArgs,
    parallel_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    sparse_experts: bool,
}

impl MlxUnitFactory<DeepSeekV3Unit> for DeepSeekV3UnitFactory {
    fn build(&mut self, index: usize, stream: &Stream) -> Result<DeepSeekV3Unit, Error> {
        let layer =
            build_deepseek_v3_unit(&self.args, index, self.parallel_layout.as_deref(), stream)?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_experts || !name.starts_with("mlp.experts.")
        })
        .map_err(|error| Error::Parallel(error.to_string()))
    }
}

struct DeepSeekV3Architecture {
    args: ModelArgs,
    static_model: DeepSeekV3Static,
    sparse_experts: bool,
    expert_cache: Option<ExpertCache>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

impl DeepSeekV3Architecture {
    fn new(
        args: ModelArgs,
        sparse_experts: bool,
        parallel: Option<(
            crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
            &eredu_runtime::LocalModelLayout,
        )>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut mtp = DeepSeekMtpModule::new(&args, stream)?;
        let (embedding, lm_head, parallel_topology) = match parallel {
            Some((build, layout)) => {
                if let Some(mtp) = &mut mtp {
                    configure_deepseek_v3_mtp_parallel(mtp, &args, build, layout, stream)?;
                }
                (
                    DeepSeekV3Embedding::Parallel(VocabParallelEmbedding::unloaded(
                        args.vocab_size as usize,
                        args.hidden_size,
                        args.weight_quantization_for("model.embed_tokens.weight"),
                        build,
                        stream,
                    )?),
                    DeepSeekV3Head::Parallel(VocabParallelLmHead::unloaded(
                        args.hidden_size,
                        args.vocab_size as usize,
                        args.weight_quantization_for("lm_head.weight"),
                        build,
                        stream,
                    )?),
                    Some(build.topology()),
                )
            }
            None => (
                DeepSeekV3Embedding::Replicated(
                    common::linear::unloaded_maybe_quantized_embedding(
                        args.vocab_size,
                        args.hidden_size,
                        args.weight_quantization_for("model.embed_tokens.weight"),
                        stream,
                    )?,
                ),
                DeepSeekV3Head::Replicated(common::linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )?),
                None,
            ),
        };
        let static_model = DeepSeekV3StaticModel {
            embedding,
            norm: nn::RmsNorm::unloaded(
                args.hidden_size,
                args.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            lm_head,
            mtp,
        };
        Ok(Self {
            args,
            static_model: MlxParameterTree::new_filtered(static_model, "", |name| {
                !sparse_experts
                    || !(name.starts_with("mtp.layers.") && name.contains(".decoder.mlp.experts."))
            })
            .map_err(|error| Error::Parallel(error.to_string()))?,
            sparse_experts,
            expert_cache: None,
            parallel_topology,
        })
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = Cache::new(&self.args);
        }
        if cache.layers.len() != self.args.layer_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.layer_schedule.len()
            )));
        }
        Ok(())
    }
}

type DeepSeekV3RetainedContext<'a> =
    std::iter::Chain<std::option::Iter<'a, Array>, std::option::Iter<'a, Array>>;

impl LayeredArchitecture<MlxBackend, Cache> for DeepSeekV3Architecture {
    type Input<'a> = &'a Array;
    type StaticModules = DeepSeekV3Static;
    type Unit = DeepSeekV3Unit;
    type ForwardContext = DeepSeekV3ForwardContext;
    type RetainedContextValues<'a> = DeepSeekV3RetainedContext<'a>;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.layer_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 has no execution group {group}"
            )))
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= self.group_unit_count(group)? {
            return Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 has no decoder unit {index}"
            )));
        }
        Ok(format!("model.layers.{index}"))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_model
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_model
    }

    fn build_unit(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Unit, Error> {
        self.group_unit_count(group)?;
        let layer = build_deepseek_v3_unit(&self.args, index, None, stream)?;
        MlxParameterTree::new_filtered(layer, "", |name| {
            !self.sparse_experts || !name.starts_with("mlp.experts.")
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
        let DeepSeekV3Embedding::Replicated(embedding) = &mut self.static_model.embedding else {
            return Err(Error::Parallel(
                "DeepSeek-V3 parallel embedding requires a parallel forward".into(),
            ));
        };
        let hidden = embedding.forward(input, stream)?;
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                stream,
            )?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: DeepSeekV3ForwardContext {
                mask,
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
        _forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 execution group {group} received {} dependencies",
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
        forward: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group)?;
        if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
                hidden,
                forward.mask.as_ref(),
                Some(&mut cache.layers[index]),
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_deepseek_experts(
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
            )?);
        }
        Ok(layer.forward_stage(
            hidden,
            forward.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
        )?)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group)?;
        forward.draft_hidden = Some(hidden.clone());
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _forward: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.static_model.norm.forward(hidden, stream)?;
        let DeepSeekV3Head::Replicated(head) = &mut self.static_model.lm_head else {
            return Err(Error::Parallel(
                "DeepSeek-V3 parallel head requires a parallel forward".into(),
            ));
        };
        Ok(head.forward(&hidden, stream)?)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter().chain(forward.draft_hidden.iter())
    }
}

impl ParallelLayeredArchitecture<MlxBackend, Cache> for DeepSeekV3Architecture {
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<LayeredForwardState<Array, Self::ForwardContext>, Error> {
        self.validate_cache(cache)?;
        let topology = self.parallel_topology.ok_or_else(|| {
            Error::Parallel("DeepSeek-V3 parallel forward has no topology".into())
        })?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        let hidden = match &mut self.static_model.embedding {
            DeepSeekV3Embedding::Parallel(embedding) => embedding.forward(input, &execution)?,
            DeepSeekV3Embedding::Replicated(embedding) => embedding.forward(input, stream)?,
        };
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                stream,
            )?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: DeepSeekV3ForwardContext {
                mask,
                draft_hidden: None,
            },
        })
    }

    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        layer: &mut Self::Unit,
        hidden: &Array,
        cache: &mut Cache,
        forward: &mut Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.group_unit_count(group_index)?;
        if let Some(expert_cache) = &self.expert_cache {
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_tensor_with_expert_executor(
                hidden,
                forward.mask.as_ref(),
                Some(&mut cache.layers[index]),
                group,
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_deepseek_experts(
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
            )?);
        }
        Ok(layer.forward_tensor_parallel(
            hidden,
            forward.mask.as_ref(),
            Some(&mut cache.layers[index]),
            group,
            stream,
        )?)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &Array,
        _cache: &mut Cache,
        _forward: &Self::ForwardContext,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.static_model.norm.forward(hidden, stream)?;
        let topology = self
            .parallel_topology
            .ok_or_else(|| Error::Parallel("DeepSeek-V3 parallel output has no topology".into()))?;
        let execution = crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(topology, group, stream)?;
        match &mut self.static_model.lm_head {
            DeepSeekV3Head::Parallel(head) => {
                Ok(head.forward(&hidden, &execution)?.all_gather(&execution)?)
            }
            DeepSeekV3Head::Replicated(head) => Ok(head.forward(&hidden, stream)?),
        }
    }
}

fn build_deepseek_v3_unit(
    args: &ModelArgs,
    index: usize,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    stream: &Stream,
) -> Result<DecoderLayer, Error> {
    let Some(layout) = layout else {
        return Ok(DecoderLayer::new_layerwise(args, index as i32, stream)?);
    };
    let prefix = format!("model.layers.{index}");
    let tensor = |suffix: &str| {
        layout
            .tensor(&format!("{prefix}.{suffix}.weight"))
            .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
    };
    let attention = tensor("self_attn.q_proj")
        .or_else(|| tensor("self_attn.q_b_proj"))
        .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLA query")))?;
    let local_heads = i32::try_from(attention.local_shape()[0])
        .map_err(|_| Error::Parallel("DeepSeek local query width exceeds i32".into()))?
        / (args.qk_nope_head_dim + args.qk_rope_head_dim);
    let local_width = |suffix: &str, axis: usize, fallback: i32| -> Result<i32, Error> {
        tensor(suffix)
            .map(|value| {
                value
                    .local_shape()
                    .get(axis)
                    .copied()
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "DeepSeek TP layout for {prefix}.{suffix} has no axis {axis}"
                        ))
                    })
                    .and_then(|width| {
                        i32::try_from(width).map_err(|_| {
                            Error::Parallel(format!(
                                "DeepSeek local width for {prefix}.{suffix} exceeds i32"
                            ))
                        })
                    })
            })
            .transpose()
            .map(|value| value.unwrap_or(fallback))
    };
    let dense_intermediate = local_width("mlp.gate_proj", 0, args.intermediate_size)?;
    let routed_intermediate = layout
        .tensor(&format!("{prefix}.mlp.experts.gate_proj"))
        .map(|value| {
            i32::try_from(value.local_shape()[1]).map_err(|_| {
                Error::Parallel("DeepSeek local routed-expert width exceeds i32".into())
            })
        })
        .transpose()?
        .unwrap_or(args.moe_intermediate_size);
    let shared_intermediate = local_width(
        "mlp.shared_experts.gate_proj",
        0,
        args.moe_intermediate_size * args.n_shared_experts,
    )?;
    Ok(DecoderLayer::new_parallel_layerwise(
        args,
        index as i32,
        local_heads,
        dense_intermediate,
        routed_intermediate,
        shared_intermediate,
        stream,
    )?)
}

fn configure_deepseek_v3_mtp_parallel(
    mtp: &mut DeepSeekMtpModule,
    args: &ModelArgs,
    context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    layout: &eredu_runtime::LocalModelLayout,
    stream: &Stream,
) -> Result<(), Error> {
    let count = mtp.layers.len();
    let mut policies = args.layer_schedule.iter().cloned().collect::<Vec<_>>();
    policies.extend(std::iter::repeat_n(LayerPolicy::SparseMoe, count));
    let mut mtp_args = args.clone();
    mtp_args.layer_schedule = crate::LayerSchedule::new(policies.len(), policies)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    for (index, layer) in mtp.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{index}.decoder");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let attention = tensor("self_attn.q_proj")
            .or_else(|| tensor("self_attn.q_b_proj"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLA query")))?;
        let local_heads = i32::try_from(attention.local_shape()[0])
            .map_err(|_| Error::Parallel("DeepSeek MTP query width exceeds i32".into()))?
            / (args.qk_nope_head_dim + args.qk_rope_head_dim);
        let local_width = |suffix: &str, axis: usize, fallback: i32| -> Result<i32, Error> {
            tensor(suffix)
                .map(|value| {
                    value
                        .local_shape()
                        .get(axis)
                        .copied()
                        .ok_or_else(|| {
                            Error::Parallel(format!(
                                "DeepSeek TP layout for {prefix}.{suffix} has no axis {axis}"
                            ))
                        })
                        .and_then(|width| {
                            i32::try_from(width).map_err(|_| {
                                Error::Parallel(format!(
                                    "DeepSeek local width for {prefix}.{suffix} exceeds i32"
                                ))
                            })
                        })
                })
                .transpose()
                .map(|value| value.unwrap_or(fallback))
        };
        let dense_intermediate = local_width("mlp.gate_proj", 0, args.intermediate_size)?;
        let routed_intermediate = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_proj"))
            .map(|value| {
                i32::try_from(value.local_shape()[1])
                    .map_err(|_| Error::Parallel("DeepSeek MTP routed width exceeds i32".into()))
            })
            .transpose()?
            .unwrap_or(args.moe_intermediate_size);
        let shared_intermediate = local_width(
            "mlp.shared_experts.gate_proj",
            0,
            args.moe_intermediate_size * args.n_shared_experts,
        )?;
        let global = args.num_hidden_layers as usize + index;
        layer.decoder = DecoderLayer::new_parallel_layerwise(
            &mtp_args,
            global as i32,
            local_heads,
            dense_intermediate,
            routed_intermediate,
            shared_intermediate,
            stream,
        )?;
        layer.shared_head.configure_parallel(context, stream)?;
    }
    Ok(())
}

fn deepseek_v3_layerwise_error(error: impl std::fmt::Display) -> Error {
    Error::Parallel(error.to_string())
}

fn deepseek_v3_layer_recipes(
    args: &ModelArgs,
    sparse_experts: bool,
    layer: &DecoderLayer,
    index: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let prefix = format!("model.layers.{index}");
    let normalized = normalized_checkpoint_keys(store);
    let keys = store.source_keys();
    let mut recipes = BTreeMap::new();
    for local_name in layer.parameters().flatten().keys() {
        if sparse_experts && expert_destination(local_name.as_ref()).is_some() {
            continue;
        }
        let destination = format!("{prefix}.{local_name}");
        let canonical = canonical_checkpoint_name(&destination);
        if keys.contains(&destination) || keys.contains(&canonical) {
            continue;
        }
        if let Some((projection, component)) = expert_destination(local_name.as_ref()) {
            let mut inputs = Vec::with_capacity(args.n_routed_experts as usize);
            for expert in 0..args.n_routed_experts {
                let runtime = format!("{prefix}.mlp.experts.{expert}.{projection}.{component}");
                let raw = normalized.get(&runtime).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek-V3 checkpoint is missing split expert tensor {runtime}"
                    ))
                })?;
                inputs.push(DerivedWeightRecipe::source(
                    raw.clone(),
                    TensorSelection::Full,
                ));
            }
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::Stack { axis: 0, inputs },
            );
            continue;
        }
        let raw = normalized
            .get(&destination)
            .or_else(|| normalized.get(&canonical))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "DeepSeek-V3 checkpoint is missing runtime parameter {canonical}"
                ))
            })?;
        recipes.insert(
            local_name.to_string(),
            DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
        );
    }
    Ok(recipes)
}

fn deepseek_v3_unit_bindings(
    args: &ModelArgs,
    sparse_experts: bool,
    index: usize,
    layer: &DecoderLayer,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    Ok(build_module_binding_plan_with_recipes_excluding(
        layer,
        &format!("model.layers.{index}"),
        store,
        deepseek_v3_layer_recipes(args, sparse_experts, layer, index, store)?,
        |name| sparse_experts && name.starts_with("mlp.experts."),
    )?
    .build_bindings(store)?)
}

fn deepseek_v3_mtp_recipes(
    mtp: &DeepSeekMtpModule,
    args: &ModelArgs,
    sparse_experts: bool,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut recipes = BTreeMap::new();
    for (index, layer) in mtp.layers.iter().enumerate() {
        let global = args.num_hidden_layers as usize + index;
        let decoder_recipes =
            deepseek_v3_layer_recipes(args, sparse_experts, &layer.decoder, global, store)?;
        for (name, recipe) in decoder_recipes {
            recipes.insert(format!("layers.{index}.decoder.{name}"), recipe);
        }
        for local_name in layer.decoder.parameters().flatten().keys() {
            if sparse_experts && expert_destination(local_name.as_ref()).is_some() {
                continue;
            }
            let target = format!("layers.{index}.decoder.{local_name}");
            if recipes.contains_key(&target) {
                continue;
            }
            let destination = format!("model.layers.{global}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek MTP checkpoint is missing {canonical}"
                    ))
                })?;
            recipes.insert(
                target,
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        for (local, remote) in [
            ("enorm.weight", "enorm.weight"),
            ("hnorm.weight", "hnorm.weight"),
            ("eh_proj.weight", "eh_proj.weight"),
            ("shared_norm.weight", "shared_head.norm.weight"),
            ("shared_head.weight", "shared_head.head.weight"),
        ] {
            let destination = format!("model.layers.{global}.{remote}");
            let raw = normalized.get(&destination).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "DeepSeek MTP checkpoint is missing {destination}"
                ))
            })?;
            recipes.insert(
                format!("layers.{index}.{local}"),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
    }
    Ok(recipes)
}

fn name_runtime_binding(binding: WeightBinding, name: String) -> Result<WeightBinding, Error> {
    binding.with_name(name).map_err(Into::into)
}

fn deepseek_v3_static_bindings(
    model: &DeepSeekV3StaticModel,
    args: &ModelArgs,
    sparse_experts: bool,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, Error> {
    let DeepSeekV3Embedding::Replicated(embedding) = &model.embedding else {
        return Err(Error::Parallel(
            "DeepSeek-V3 global static binding model has a sharded embedding".into(),
        ));
    };
    let DeepSeekV3Head::Replicated(head) = &model.lm_head else {
        return Err(Error::Parallel(
            "DeepSeek-V3 global static binding model has a sharded head".into(),
        ));
    };
    let mut bindings = build_module_binding_plan_with_recipes(
        embedding,
        "model.embed_tokens",
        store,
        BTreeMap::new(),
    )?
    .build_bindings(store)?
    .into_iter()
    .map(|binding| {
        let name = format!("embedding.{}", binding.name());
        name_runtime_binding(binding, name)
    })
    .collect::<Result<Vec<_>, _>>()?;
    bindings.extend(
        build_module_binding_plan_with_recipes(&model.norm, "model.norm", store, BTreeMap::new())?
            .build_bindings(store)?
            .into_iter()
            .map(|binding| {
                let name = format!("norm.{}", binding.name());
                name_runtime_binding(binding, name)
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    bindings.extend(
        build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
            .build_bindings(store)?
            .into_iter()
            .map(|binding| {
                let name = format!("lm_head.{}", binding.name());
                name_runtime_binding(binding, name)
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    if let Some(mtp) = &model.mtp {
        bindings.extend(
            build_module_binding_plan_with_recipes_excluding(
                mtp,
                "",
                store,
                deepseek_v3_mtp_recipes(mtp, args, sparse_experts, store)?,
                |name| sparse_experts && name.contains(".decoder.mlp.experts."),
            )?
            .build_bindings(store)?
            .into_iter()
            .map(|binding| {
                let name = format!("mtp.{}", binding.name());
                name_runtime_binding(binding, name)
            })
            .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(bindings)
}

fn deepseek_v3_execution_layout(args: &ModelArgs) -> Result<ExecutionUnitLayout, Error> {
    let graph = ExecutionGraph::chain(["text_decoder"])?;
    ExecutionUnitLayout::new(&graph, [args.layer_schedule.len()])
        .map_err(|error| Error::Parallel(error.to_string()))
}

/// DeepSeek-V3/R1 causal LM using bounded residency for decoder blocks.
pub struct DeepSeekV3LayerwiseModel {
    execution: DeepSeekV3Execution,
    metadata: eredu_runtime::LayerwiseModelMetadata,
    parallel_info:
        Option<eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>>,
    parallel_topology: Option<crate::backend::mlx::MlxParallelContext>,
}

pub(crate) struct DeepSeekTensorMtpTarget<'a> {
    model: &'a mut DeepSeekV3LayerwiseModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> DeepSeekTensorMtpTarget<'a> {
    pub(crate) fn new(
        model: &'a mut DeepSeekV3LayerwiseModel,
        group: &'a safemlx::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl DeepSeekV3LayerwiseModel {
    fn architecture(&self) -> &DeepSeekV3Architecture {
        match &self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.architecture(),
            DeepSeekV3Execution::Layerwise(execution) => execution.architecture(),
        }
    }

    fn architecture_mut(&mut self) -> &mut DeepSeekV3Architecture {
        match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.architecture_mut(),
            DeepSeekV3Execution::Layerwise(execution) => execution.architecture_mut(),
        }
    }

    fn prompt_cache_rank_identity(&self) -> Option<crate::core::cache::CacheRankIdentity> {
        self.parallel_topology
            .map(crate::backend::mlx::cache::prompt_cache_topology)
            .and_then(|topology| topology.cache_rank_identity())
    }

    /// Returns the validated architecture arguments.
    pub fn args(&self) -> &ModelArgs {
        &self.architecture().args
    }

    /// Returns immutable metadata describing the selected residency policy.
    pub fn residency_metadata(&self) -> &eredu_runtime::LayerwiseModelMetadata {
        &self.metadata
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args().layer_schedule.len();
        Ok(PromptCacheModelIdentity {
            model_family: "deepseek_v3".into(),
            effective_model_type: self.args().model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(self.args()),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: self.parallel_topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
            layer_layout: PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                self.args().kv_lora_rank,
                self.args().qk_rope_head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        })
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.parallel_topology = Some(topology);
        self.architecture_mut().parallel_topology = Some(topology);
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

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.parallel_info.as_ref()
    }

    /// Creates one compressed MLA cache per decoder block.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.args()).with_mtp_layers(self.mtp_len())
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.architecture()
            .static_model
            .mtp
            .as_ref()
            .map_or(0, DeepSeekMtpModule::len)
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => {
                execution.forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
            }
            DeepSeekV3Execution::Layerwise(execution) => {
                execution.forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("DeepSeek layerwise pass did not retain MTP hidden state")
        })?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn forward_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [crate::backend::mlx::runtime::cache::CompressedLatentCache],
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let architecture = self.architecture_mut();
        let embeddings = match &mut architecture.static_model.embedding {
            DeepSeekV3Embedding::Replicated(embedding) => embedding.forward(tokens, stream)?,
            DeepSeekV3Embedding::Parallel(_) => {
                return Err(Exception::custom(
                    "DeepSeek replicated MTP cannot use a parallel embedding",
                ))
            }
        };
        let expert_cache = architecture.expert_cache.as_ref();
        let args = &architecture.args;
        architecture
            .static_model
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                expert_cache,
                None,
                args,
                None,
                stream,
            )
    }

    fn forward_mtp_target_tensor(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => execution
                .forward_parallel_with_context_hook(tokens, cache, group, stream, |_, _, _| Ok(())),
            DeepSeekV3Execution::Layerwise(execution) => execution
                .forward_parallel_with_context_hook(tokens, cache, group, stream, |_, _, _| Ok(())),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("DeepSeek tensor pass did not retain MTP hidden state")
        })?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn forward_mtp_draft_tensor(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [crate::backend::mlx::runtime::cache::CompressedLatentCache],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("DeepSeek MTP target has no parallel topology"))?
            .topology();
        let execution =
            crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                topology, group, stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let architecture = self.architecture_mut();
        let embeddings = match &mut architecture.static_model.embedding {
            DeepSeekV3Embedding::Parallel(embedding) => embedding
                .forward(tokens, &execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            DeepSeekV3Embedding::Replicated(embedding) => embedding.forward(tokens, stream)?,
        };
        let expert_cache = architecture.expert_cache.as_ref();
        let args = &architecture.args;
        architecture
            .static_model
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                expert_cache,
                None,
                args,
                Some(&execution),
                stream,
            )
    }

    /// Creates ordinary or paged compressed attention state independently of weight residency.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        let rank = self.prompt_cache_rank_identity();
        Cache::new_with_options_and_rank(self.args(), policy.clone(), rank)
            .and_then(|cache| match policy {
                CacheResidencyPolicy::Device => Ok(cache.with_mtp_layers(self.mtp_len())),
                CacheResidencyPolicy::Paged(_) => cache.with_paged_mtp_layers(self.mtp_len(), rank),
            })
            .map_err(Into::into)
    }

    /// Lazily catalogs a compatible persisted compressed prefix.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        let args = self.args();
        let layer_count = args.layer_schedule.len();
        let identity = PromptCacheModelIdentity {
            model_family: "deepseek_v3".into(),
            effective_model_type: args.model_type.clone(),
            architecture_fingerprint: resident::prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: self.parallel_topology.map_or_else(
                PromptCacheTopology::default,
                crate::backend::mlx::cache::prompt_cache_topology,
            ),
            layer_layout: PromptCacheModelIdentity::compressed_layouts(
                layer_count,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
            )
            .map_err(|error| Exception::custom(error.to_string()))?,
        };
        validate_prompt_cache_model_identity(expected, &identity)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Cache::load_prompt_cache(
            self.args(),
            directory,
            expected,
            &identity,
            prefix_token_ids,
            options,
        )
        .map_err(Into::into)
    }

    /// Persists a prefix through the generalized execution contract.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let _ = stream;
        cache
            .save_prompt_cache(destination, descriptor, prefix_token_ids, options)
            .map_err(Into::into)
    }

    /// Returns current logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match &self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.policy().residency_report(),
            DeepSeekV3Execution::Layerwise(execution) => execution.policy().residency_report(),
        }
    }
    /// Returns dense-stream observations when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.execution {
            DeepSeekV3Execution::Resident(_) => Ok(None),
            DeepSeekV3Execution::Layerwise(execution) => execution.policy().dense_stream_report(),
        }
    }

    /// Returns sparse expert-cache telemetry when that residency mode is active.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.architecture()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        match &self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.policy().checkpoint_store(),
            DeepSeekV3Execution::Layerwise(execution) => execution.policy().checkpoint_store(),
        }
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        match &self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.policy().checkpoint_store_arc(),
            DeepSeekV3Execution::Layerwise(execution) => execution.policy().checkpoint_store_arc(),
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
            DeepSeekV3Execution::Resident(execution) => {
                execution.forward_parallel(inputs, cache, group, stream)
            }
            DeepSeekV3Execution::Layerwise(execution) => {
                execution.forward_parallel(inputs, cache, group, stream)
            }
        }
        .map_err(deepseek_v3_layerwise_error)
    }

    /// Runs MLA and dense/MoE decoder blocks while preserving compressed state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => execution.forward(inputs, cache, stream),
            DeepSeekV3Execution::Layerwise(execution) => execution.forward(inputs, cache, stream),
        }
        .map_err(deepseek_v3_layerwise_error)
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let execute = |architecture: &mut DeepSeekV3Architecture,
                       _group: usize,
                       index: usize,
                       layer: &mut DeepSeekV3Unit,
                       hidden: &Array,
                       cache: &mut Cache,
                       context: &mut DeepSeekV3ForwardContext,
                       stream: &Stream| {
            let prefix = format!("model.layers.{index}");
            if architecture.sparse_experts {
                observer.observe(&format!("{prefix}.input"), hidden)?;
                let expert_cache = architecture.expert_cache.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "DeepSeek-V3 sparse expert cache was not initialized".into(),
                    )
                })?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let output = layer.forward_sparse_experts(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    stream,
                    |flat, indices, weights, stream| {
                        execute_cached_deepseek_experts(
                            &architecture.args,
                            expert_cache,
                            index,
                            flat,
                            indices,
                            weights,
                            pass,
                            stream,
                        )
                    },
                )?;
                observer.observe(&format!("{prefix}.output"), &output)?;
                return Ok(observer
                    .intervene(&format!("{prefix}.output"), &output)?
                    .unwrap_or(output));
            }
            Ok(layer.forward_stage_with_observer(
                hidden,
                context.mask.as_ref(),
                Some(&mut cache.layers[index]),
                stream,
                &prefix,
                observer,
            )?)
        };
        match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => {
                execution.forward_with_unit_executor(inputs, cache, stream, execute)
            }
            DeepSeekV3Execution::Layerwise(execution) => {
                execution.forward_with_unit_executor(inputs, cache, stream, execute)
            }
        }
        .map_err(deepseek_v3_layerwise_error)
    }

    /// Runs streamed layers while delegating routed experts to a caller.
    pub(crate) fn forward_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |_architecture: &mut DeepSeekV3Architecture,
                            _group: usize,
                            index: usize,
                            layer: &mut DeepSeekV3Unit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut DeepSeekV3ForwardContext,
                            stream: &Stream| {
            Ok(layer.forward_sparse_experts(
                hidden,
                mask.or(context.mask.as_ref()),
                Some(&mut cache.layers[index]),
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )?)
        };
        match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => {
                execution.forward_with_unit_executor(inputs, cache, stream, execute_unit)
            }
            DeepSeekV3Execution::Layerwise(execution) => {
                execution.forward_with_unit_executor(inputs, cache, stream, execute_unit)
            }
        }
        .map_err(deepseek_v3_layerwise_error)
    }

    /// Runs TP-sharded MLA and dense/shared projections while delegating
    /// routed experts to the matching EP subgroup.
    pub(crate) fn forward_tensor_expert_parallel<F>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let execute_unit = |_architecture: &mut DeepSeekV3Architecture,
                            _group: usize,
                            index: usize,
                            layer: &mut DeepSeekV3Unit,
                            hidden: &Array,
                            cache: &mut Cache,
                            context: &mut DeepSeekV3ForwardContext,
                            group: &safemlx::distributed::Group,
                            stream: &Stream| {
            Ok(layer.forward_tensor_with_expert_executor(
                hidden,
                mask.or(context.mask.as_ref()),
                Some(&mut cache.layers[index]),
                group,
                stream,
                |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
            )?)
        };
        match &mut self.execution {
            DeepSeekV3Execution::Resident(execution) => execution
                .forward_parallel_with_unit_executor(
                    inputs,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                ),
            DeepSeekV3Execution::Layerwise(execution) => execution
                .forward_parallel_with_unit_executor(
                    inputs,
                    cache,
                    tensor_group,
                    stream,
                    execute_unit,
                ),
        }
        .map_err(deepseek_v3_layerwise_error)
    }

    pub(crate) fn forward_mtp_target_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let (logits, context) = if let Some(tensor_group) = tensor_group {
            let execute_unit = |_architecture: &mut DeepSeekV3Architecture,
                                _group: usize,
                                index: usize,
                                layer: &mut DeepSeekV3Unit,
                                hidden: &Array,
                                cache: &mut Cache,
                                context: &mut DeepSeekV3ForwardContext,
                                group: &safemlx::distributed::Group,
                                stream: &Stream| {
                Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    group,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            };
            match &mut self.execution {
                DeepSeekV3Execution::Resident(execution) => execution
                    .forward_parallel_with_unit_executor_and_context_hook(
                        inputs,
                        cache,
                        tensor_group,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
                DeepSeekV3Execution::Layerwise(execution) => execution
                    .forward_parallel_with_unit_executor_and_context_hook(
                        inputs,
                        cache,
                        tensor_group,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
            }
        } else {
            let execute_unit = |_architecture: &mut DeepSeekV3Architecture,
                                _group: usize,
                                index: usize,
                                layer: &mut DeepSeekV3Unit,
                                hidden: &Array,
                                cache: &mut Cache,
                                context: &mut DeepSeekV3ForwardContext,
                                stream: &Stream| {
                Ok(layer.forward_sparse_experts(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            };
            match &mut self.execution {
                DeepSeekV3Execution::Resident(execution) => execution
                    .forward_with_unit_executor_and_context_hook(
                        inputs,
                        cache,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
                DeepSeekV3Execution::Layerwise(execution) => execution
                    .forward_with_unit_executor_and_context_hook(
                        inputs,
                        cache,
                        stream,
                        execute_unit,
                        |_, _, _| Ok(()),
                    ),
            }
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("DeepSeek EP pass did not retain MTP hidden state")
                })?,
                tokens: inputs.clone(),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_mtp_draft_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [crate::backend::mlx::runtime::cache::CompressedLatentCache],
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("DeepSeek EP MTP target has no topology"))?
            .topology();
        let execution = tensor_group
            .map(|group| {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )
            })
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let architecture = self.architecture_mut();
        let embeddings = match (execution.as_ref(), &mut architecture.static_model.embedding) {
            (Some(execution), DeepSeekV3Embedding::Parallel(embedding)) => embedding
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            (_, DeepSeekV3Embedding::Replicated(embedding)) => embedding.forward(tokens, stream)?,
            (None, DeepSeekV3Embedding::Parallel(_)) => {
                return Err(Exception::custom(
                    "DeepSeek MTP parallel embedding requires a TP group",
                ))
            }
        };
        let args = &architecture.args;
        architecture
            .static_model
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                None,
                Some(&mut execute),
                args,
                execution.as_ref(),
                stream,
            )
    }

    /// Clears temporary decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        match &self.execution {
            DeepSeekV3Execution::Resident(execution) => {
                execution.policy().clear_device_group("text_decoder")
            }
            DeepSeekV3Execution::Layerwise(execution) => {
                execution.policy().clear_device_group("text_decoder")
            }
        }
    }
}

impl CausalModel<Cache> for DeepSeekV3LayerwiseModel {
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

fn resolve_deepseek_v3_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: &ModelArgs,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    if store.is_checkpoint_contract_resolved()
        || store.source_diagnostics()?.backend
            != eredu_checkpoint::store::WeightStoreBackend::Safetensors
    {
        return Ok(store);
    }
    let plan =
        super::checkpoint::safetensors_plan(args, true).map_err(Error::UnsupportedArchitecture)?;
    let resolved = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), &plan)
        .map_err(|validation| {
            Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 checkpoint contract did not resolve: {validation:?}"
            ))
        })?;
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolved),
    ))
}

fn deepseek_v3_quantized_args(args: &ModelArgs, quantization: WeightQuantization) -> ModelArgs {
    let mut target = args.clone();
    target.quantization_config = None;
    target.quantization = Some(quantization);
    target.quantized_weight_configs = None;
    target
}

fn quantize_deepseek_v3_store(
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
    let target_args = deepseek_v3_quantized_args(source_args, quantization);
    let source_static =
        DeepSeekV3Architecture::new(source_args.clone(), sparse_experts, None, stream)?;
    let target_static =
        DeepSeekV3Architecture::new(target_args.clone(), sparse_experts, None, stream)?;
    let source_units = source_args.clone();
    let target_units = target_args.clone();
    let binding_args = source_args.clone();
    let static_args = source_args.clone();
    let (store, report) = quantize_module_store_with_bindings(
        store,
        &*source_static.static_model,
        &*target_static.static_model,
        move |index, stream| {
            Ok(DecoderLayer::new_layerwise(
                &source_units,
                index as i32,
                stream,
            )?)
        },
        move |index, stream| {
            Ok(DecoderLayer::new_layerwise(
                &target_units,
                index as i32,
                stream,
            )?)
        },
        source_args.layer_schedule.len(),
        quantization,
        stream,
        move |model, store| deepseek_v3_static_bindings(model, &static_args, sparse_experts, store),
        move |index, layer, store| {
            deepseek_v3_unit_bindings(&binding_args, sparse_experts, index, layer, store)
        },
    )?;
    Ok((store, target_args, report))
}

fn load_deepseek_v3_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let store = resolve_deepseek_v3_store(store, &args)?;
    let (store, args, materialization) = match quantization {
        Some(quantization) => {
            let (store, args, report) =
                quantize_deepseek_v3_store(store, &args, sparse_experts, quantization, stream)?;
            (store, args, Some(report))
        }
        None => (store, args, None),
    };
    let mut architecture = DeepSeekV3Architecture::new(args.clone(), sparse_experts, None, stream)?;
    let factory = DeepSeekV3UnitFactory {
        args: args.clone(),
        parallel_layout: None,
        sparse_experts,
    };
    let binding_args = args.clone();
    let static_args = args.clone();
    let (policy, mut metadata) = prepare_layerwise_policy_with_bindings(
        Arc::clone(&store),
        architecture.static_modules_mut(),
        factory,
        deepseek_v3_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        move |modules, store| {
            deepseek_v3_static_bindings(&**modules, &static_args, sparse_experts, store)
        },
        move |index, unit, store, _| {
            deepseek_v3_unit_bindings(&binding_args, sparse_experts, index, &unit, store)
        },
    )?;
    metadata.set_model_type(args.model_type.clone());
    metadata.set_quantization(args.quantization);
    metadata.set_materialization(materialization);
    let execution = if options.is_fully_resident() {
        DeepSeekV3Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DeepSeekV3Execution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    Ok(DeepSeekV3LayerwiseModel {
        execution,
        metadata,
        parallel_info: None,
        parallel_topology: None,
    })
}

fn register_deepseek_v3_parallel_parameters(
    planner: &mut ParallelPlanBuilder,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<(), Error> {
    let static_model = DeepSeekV3Architecture::new(args.clone(), false, None, stream)?;
    let DeepSeekV3Embedding::Replicated(embedding) = &static_model.static_model.embedding else {
        unreachable!("global DeepSeek static model is replicated")
    };
    let DeepSeekV3Head::Replicated(head) = &static_model.static_model.lm_head else {
        unreachable!("global DeepSeek static model is replicated")
    };
    planner.register(
        crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
            embedding,
            "model.embed_tokens",
            args.vocab_size as usize,
            args.hidden_size,
            false,
        )?,
    )?;
    crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
        planner,
        &static_model.static_model.norm,
        "model.norm",
    )?;
    planner.register(
        crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
            head,
            "lm_head",
            args.hidden_size,
            args.vocab_size as usize,
            false,
        )?,
    )?;
    for index in 0..args.layer_schedule.len() {
        let layer = DecoderLayer::new_layerwise(args, index as i32, stream)?;
        register_deepseek_layer_parallel_plan(planner, &layer, index)?;
    }
    if let Some(mtp) = &static_model.static_model.mtp {
        for (index, layer) in mtp.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            register_replicated_module(planner, &layer.enorm, &format!("{prefix}.enorm"))?;
            register_replicated_module(planner, &layer.hnorm, &format!("{prefix}.hnorm"))?;
            register_projection_module(
                planner,
                &layer.eh_proj,
                &format!("{prefix}.eh_proj"),
                ProjectionSharding::Replicated,
            )?;
            register_deepseek_layer_parallel_plan_at(
                planner,
                &layer.decoder,
                args.num_hidden_layers as usize + index,
                &format!("{prefix}.decoder"),
            )?;
            register_replicated_module(
                planner,
                &layer.shared_norm,
                &format!("{prefix}.shared_norm"),
            )?;
            layer
                .shared_head
                .register(planner, &format!("{prefix}.shared_head"))?;
        }
    }
    Ok(())
}

fn load_deepseek_v3_parallel_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    sparse_experts: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let store = resolve_deepseek_v3_store(store, &args)?;
    let mut planner = build.planner();
    register_deepseek_v3_parallel_parameters(&mut planner, &args, stream)?;
    let (_, local_layout) = planner.finish()?;
    if local_layout.is_empty() {
        return Err(Error::Parallel(
            "DeepSeek-V3 declared no tensor-parallel parameters".into(),
        ));
    }
    let shared_layout = Arc::new(local_layout);
    let mut architecture = DeepSeekV3Architecture::new(
        args.clone(),
        sparse_experts,
        Some((build, shared_layout.as_ref())),
        stream,
    )?;
    architecture.parallel_topology = Some(build.topology());
    let global_static = DeepSeekV3Architecture::new(args.clone(), sparse_experts, None, stream)?;
    let static_bindings = deepseek_v3_static_bindings(
        &global_static.static_model,
        &args,
        sparse_experts,
        store.as_ref(),
    )?;
    let mut global_parameter_bytes = binding_bytes(&static_bindings)?;
    for index in 0..args.layer_schedule.len() {
        let layer = DecoderLayer::new_layerwise(&args, index as i32, stream)?;
        global_parameter_bytes = global_parameter_bytes
            .checked_add(binding_bytes(&deepseek_v3_unit_bindings(
                &args,
                sparse_experts,
                index,
                &layer,
                store.as_ref(),
            )?)?)
            .ok_or_else(|| {
                Error::Parallel("DeepSeek-V3 global parameter bytes overflowed".into())
            })?;
    }
    let factory = DeepSeekV3UnitFactory {
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
        deepseek_v3_execution_layout(&args)?,
        options,
        stream,
        weights_stream,
        move |key| sparse_experts && key.contains(".mlp.experts."),
        move |_, store| shard_layer_bindings(static_bindings, "", store, &static_layout),
        move |index, _local, store, stream| {
            let global = DecoderLayer::new_layerwise(&binding_args, index as i32, stream)?;
            let bindings =
                deepseek_v3_unit_bindings(&binding_args, sparse_experts, index, &global, store)?;
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
    let execution = if options.is_fully_resident() {
        DeepSeekV3Execution::Resident(LayerwiseRuntime::new(
            architecture,
            policy.into_resident(stream)?,
        ))
    } else {
        DeepSeekV3Execution::Layerwise(LayerwiseRuntime::new(architecture, policy))
    };
    let local_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.layer_parameter_bytes())
        .ok_or_else(|| Error::Parallel("DeepSeek-V3 local parameter bytes overflowed".into()))?;
    let maximum_device_parameter_bytes = metadata
        .static_device_bytes()
        .checked_add(metadata.maximum_device_layer_bytes())
        .ok_or_else(|| Error::Parallel("DeepSeek-V3 device parameter bytes overflowed".into()))?;
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
    Ok(DeepSeekV3LayerwiseModel {
        execution,
        metadata,
        parallel_info: Some(info),
        parallel_topology: Some(build.topology()),
    })
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for DeepSeekV3LayerwiseModel
{
    type Cache = Cache;
    type DraftCache = Vec<crate::backend::mlx::runtime::cache::CompressedLatentCache>;

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
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        let count = cache.mtp_layers.len();
        for depth in 0..count {
            let _ = self.forward_mtp_draft(&hidden, &next, depth, &mut cache.mtp_layers, stream)?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn restore_target_checkpoint(
        cache: &mut Cache,
        checkpoint: &Cache,
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
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.forward_mtp_draft(hidden, &token, draft_index, cache, stream)?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..cache.len() {
            let _ = self.forward_mtp_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for DeepSeekTensorMtpTarget<'_>
{
    type Cache = Cache;
    type DraftCache = Vec<crate::backend::mlx::runtime::cache::CompressedLatentCache>;

    fn prefill_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let tokens = input::text_token_ids(input, stream)?;
        cache.reset()?;
        self.model
            .forward_mtp_target_tensor(&tokens, cache, self.group, stream)
    }

    fn verify_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        self.model
            .forward_mtp_target_tensor(tokens, cache, self.group, stream)
    }

    fn prefill_draft_cache(
        &mut self,
        output: &crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let sequence = tokens.dim(1);
        if sequence <= 1 {
            return Ok(());
        }
        let hidden = output
            .hidden
            .try_index_device((.., ..sequence - 1, ..), stream)?;
        let next = tokens.try_index_device((.., 1..), stream)?;
        for depth in 0..cache.mtp_layers.len() {
            let _ = self.model.forward_mtp_draft_tensor(
                &hidden,
                &next,
                depth,
                &mut cache.mtp_layers,
                self.group,
                stream,
            )?;
        }
        Ok(())
    }

    fn draft_cache(cache: &Cache) -> Self::DraftCache {
        cache.mtp_layers.clone()
    }

    fn commit_draft_cache(cache: &mut Cache, draft: &Self::DraftCache) {
        cache.mtp_layers.clone_from(draft);
    }

    fn restore_target_checkpoint(
        cache: &mut Cache,
        checkpoint: &Cache,
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
        let token = Array::from_slice(&[last_token], &[1, 1]);
        let output = self.model.forward_mtp_draft_tensor(
            hidden,
            &token,
            draft_index,
            cache,
            self.group,
            stream,
        )?;
        Ok((output.logits, output.hidden))
    }

    fn advance_draft_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut Self::DraftCache,
        stream: &Stream,
    ) -> Result<(), Exception> {
        for depth in 0..cache.len() {
            let _ = self
                .model
                .forward_mtp_draft_tensor(hidden, tokens, depth, cache, self.group, stream)?;
        }
        Ok(())
    }

    fn max_draft_tokens(&self) -> usize {
        self.model.mtp_len()
    }
}

/// Loads DeepSeek-V3/R1 through the generalized execution engine.
pub fn load_deepseek_v3_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    if quantization.is_some() && args.native_fp8_config().is_some() {
        return Err(Error::Quantization(
            "native DeepSeek block-FP8 weights cannot be implicitly transcoded".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("DeepSeek-V3", args.affine_quantization()?, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    load_deepseek_v3_with_store(
        store,
        args,
        options,
        quantize_on_load,
        false,
        stream,
        weights_stream,
    )
}

/// Loads DeepSeek-V3/R1 through the generalized tensor-parallel engine.
pub(crate) fn load_deepseek_v3_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_deepseek_v3_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    load_deepseek_v3_parallel_with_store(
        open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
        args,
        options,
        build,
        false,
        stream,
        weights_stream,
    )
}

pub(crate) fn load_deepseek_v3_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV3LayerwiseModel, Vec<u32>), Error> {
    let residency = WeightResidency::with_layers(options);
    crate::composition::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek2,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let model = load_deepseek_v3_parallel_with_store(
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

pub(crate) fn load_deepseek_v3_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(DeepSeekV3LayerwiseModel, Vec<u32>), Error> {
    crate::composition::mlx::structural::validate_gguf(
        crate::core::GgufArchitecture::DeepSeek2,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, None, weights_stream)?;
    let args = prepared.args;
    let gguf_plan = super::checkpoint::gguf_plan(&args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            residency.max_mapped_shards(),
        )?);
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_deepseek_gguf_sparse_with_store(
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
    let model = load_deepseek_v3_with_store(
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

/// Loads replicated DeepSeek GGUF parameters for sparse expert-parallel
/// execution without materializing any routed-expert bank.
fn load_deepseek_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let mut model = load_deepseek_v3_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantization,
        true,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = model.checkpoint_store_arc();
    let entries = deepseek_expert_catalog(&args, checkpoint_store.as_ref())?;
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

/// Builds the streamed nonexpert DeepSeek execution base used by distributed EP.
pub(crate) fn load_deepseek_v3_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    args.validate()?;
    load_deepseek_v3_with_store(
        store,
        args,
        non_expert.into(),
        None,
        true,
        stream,
        weights_stream,
    )
}

pub(crate) fn load_deepseek_v3_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    load_deepseek_v3_parallel_with_store(
        store,
        args,
        non_expert.into(),
        build,
        true,
        stream,
        weights_stream,
    )
}

/// Loads DeepSeek-V3/R1 with independently cached experts and bounded non-expert units.
pub fn load_deepseek_v3_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<DeepSeekV3LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "DeepSeek independent expert cache",
                args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut model = load_deepseek_v3_with_store(
        store,
        args.clone(),
        non_expert.into(),
        quantize_on_load,
        true,
        stream,
        weights_stream,
    )?;
    let store = model.checkpoint_store_arc();
    let entries = deepseek_expert_catalog(&args, store.as_ref())?;
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
    model.architecture_mut().expert_cache = Some(cache);
    Ok(model)
}

/// Adapter for compressed MLA and mixed dense/MoE DeepSeek decoder blocks.
pub struct DeepSeekV3LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: MaybeQuantized<nn::Linear>,
    mtp: Option<DeepSeekMtpModule>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl DeepSeekV3LayerwiseAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mtp = DeepSeekMtpModule::new(&args, stream)?;
        Ok(Self {
            embedding: common::linear::unloaded_maybe_quantized_embedding(
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
            lm_head: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?,
            mtp,
            parallel_embedding: None,
            parallel_lm_head: None,
            sparse_expert_cache: false,
            expert_cache: None,
            args,
        })
    }

    fn new_sparse(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Self::new_sparse(args, stream)
    }

    /// Returns the validated architecture arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
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
            "embedding" => self
                .parallel_embedding
                .as_mut()
                .map(|module| module.inner_mut() as &mut dyn ModuleParameters)
                .or(Some(&mut self.embedding)),
            "mtp" => self
                .mtp
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            _ => None,
        }
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, DeepSeekMtpModule::len)
    }

    pub(crate) fn embedded_mtp_cache(
        &self,
    ) -> Vec<crate::backend::mlx::runtime::cache::CompressedLatentCache> {
        (0..self.embedded_mtp_len())
            .map(|_| crate::backend::mlx::runtime::cache::CompressedLatentCache::new())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_pipeline_mtp<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [crate::backend::mlx::runtime::cache::CompressedLatentCache],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        external_expert: Option<&mut F>,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let embeddings = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| {
                    Exception::custom("DeepSeek pipeline MTP has no TP embedding shard")
                })?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => self.embedding.forward(tokens, stream)?,
        };
        let expert_cache = self.expert_cache.as_ref();
        self.mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("DeepSeek checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                tokens,
                depth,
                cache,
                expert_cache,
                external_expert.map(|execute| execute as &mut DeepSeekMtpExpertExecutor<'_>),
                &self.args,
                execution,
                stream,
            )
    }

    fn recipes_for_layer(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let prefix = format!("model.layers.{index}");
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.source_keys();
        let mut recipes = BTreeMap::new();

        for local_name in layer.parameters().flatten().keys() {
            if self.sparse_expert_cache && expert_destination(local_name.as_ref()).is_some() {
                continue;
            }
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            if let Some((projection, component)) = expert_destination(local_name.as_ref()) {
                let mut inputs = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    let runtime = format!("{prefix}.mlp.experts.{expert}.{projection}.{component}");
                    let raw = normalized.get(&runtime).ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "DeepSeek-V3 checkpoint is missing split expert tensor {runtime}"
                        ))
                    })?;
                    inputs.push(DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Full,
                    ));
                }
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::Stack { axis: 0, inputs },
                );
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek-V3 checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }

    fn binding_plan_for_layer(
        &self,
        layer: &DecoderLayer,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<ModuleBindingPlan, Error> {
        let prefix = format!("model.layers.{index}");
        Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            &prefix,
            store,
            self.recipes_for_layer(layer, index, store)?,
            |name| self.sparse_expert_cache && name.starts_with("mlp.experts."),
        )?)
    }

    fn mtp_recipes(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture("DeepSeek model has no MTP module".into())
        })?;
        let normalized = normalized_checkpoint_keys(store);
        let mut recipes = BTreeMap::new();
        for (index, layer) in mtp.layers.iter().enumerate() {
            let global = self.args.num_hidden_layers as usize + index;
            for (name, recipe) in self.recipes_for_layer(&layer.decoder, global, store)? {
                recipes.insert(format!("layers.{index}.decoder.{name}"), recipe);
            }
            for (local, remote) in [
                ("enorm.weight", "enorm.weight"),
                ("hnorm.weight", "hnorm.weight"),
                ("eh_proj.weight", "eh_proj.weight"),
                ("shared_norm.weight", "shared_head.norm.weight"),
                ("shared_head.weight", "shared_head.head.weight"),
            ] {
                let destination = format!("model.layers.{global}.{remote}");
                let raw = normalized.get(&destination).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DeepSeek MTP checkpoint is missing {destination}"
                    ))
                })?;
                recipes.insert(
                    format!("layers.{index}.{local}"),
                    DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
                );
            }
        }
        Ok(recipes)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_deepseek_experts(
    args: &ModelArgs,
    expert_cache: &ExpertCache,
    layer: usize,
    flat: &Array,
    indices: &Array,
    weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Exception> {
    expert_cache
        .execute_routes_bounded(
            ExpertRouteBatch::new(layer, flat, indices, weights, pass),
            stream,
            |flat, acquired, weights, stream| {
                if acquired.is_empty() {
                    return Err(ExpertCacheError::EmptyRoutedBank {
                        architecture: "DeepSeek-V3",
                    });
                }
                let started = Instant::now();
                let mut bank = resident::RoutedExperts::new_compact(
                    args,
                    layer as i32,
                    acquired.identities().len() as i32,
                    stream,
                )?;
                if let Some(quantization) = expert_cache.weight_quantization() {
                    bank.use_fp8 = false;
                    bank.gate_affine = Some(quantization);
                    bank.up_affine = Some(quantization);
                    bank.down_affine = Some(quantization);
                    bank.gate_iquant = None;
                    bank.up_iquant = None;
                    bank.down_iquant = None;
                }
                bank.gate_proj = Param::new(Some(acquired.compact_binding("gate_proj", stream)?));
                bank.gate_proj_scale_inv =
                    Param::new(acquired.optional_compact_binding("gate_proj_scale_inv", stream)?);
                bank.gate_proj_scales =
                    Param::new(acquired.optional_compact_binding("gate_proj_scales", stream)?);
                bank.gate_proj_biases =
                    Param::new(acquired.optional_compact_binding("gate_proj_biases", stream)?);
                bank.up_proj = Param::new(Some(acquired.compact_binding("up_proj", stream)?));
                bank.up_proj_scale_inv =
                    Param::new(acquired.optional_compact_binding("up_proj_scale_inv", stream)?);
                bank.up_proj_scales =
                    Param::new(acquired.optional_compact_binding("up_proj_scales", stream)?);
                bank.up_proj_biases =
                    Param::new(acquired.optional_compact_binding("up_proj_biases", stream)?);
                bank.down_proj = Param::new(Some(acquired.compact_binding("down_proj", stream)?));
                bank.down_proj_scale_inv =
                    Param::new(acquired.optional_compact_binding("down_proj_scale_inv", stream)?);
                bank.down_proj_scales =
                    Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
                bank.down_proj_biases =
                    Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
                expert_cache.record_compact_bank(
                    pass,
                    acquired.scratch_bytes(),
                    started.elapsed(),
                )?;
                Ok(bank.forward_local(flat, acquired.compact_routes(), weights, stream)?)
            },
        )
        .map_err(|error| Exception::custom(error.to_string()))
}

fn normalized_checkpoint_keys(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| (canonical_checkpoint_name(&raw), raw))
        .collect()
}

fn expert_destination(local_name: &str) -> Option<(&'static str, &'static str)> {
    ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .find_map(|projection| {
            [
                ("", "weight"),
                ("_scale_inv", "weight_scale_inv"),
                ("_scales", "scales"),
                ("_biases", "biases"),
            ]
            .into_iter()
            .find_map(|(runtime_suffix, checkpoint_component)| {
                (local_name == format!("mlp.experts.{projection}{runtime_suffix}"))
                    .then_some((projection, checkpoint_component))
            })
        })
}

/// Per-forward causal mask shared by all MLA blocks.
pub struct DeepSeekV3ForwardContext {
    mask: Option<Array>,
    draft_hidden: Option<Array>,
}

fn register_deepseek_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &DecoderLayer,
    index: usize,
) -> Result<(), Error> {
    register_deepseek_layer_parallel_plan_at(
        planner,
        layer,
        index,
        &format!("model.layers.{index}"),
    )
}

fn register_deepseek_layer_parallel_plan_at(
    planner: &mut ParallelPlanBuilder,
    layer: &DecoderLayer,
    _index: usize,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    let attention_prefix = format!("{prefix}.self_attn");
    let mut projection_names = Vec::new();
    for (name, projection) in [
        ("q_proj", attention.q_proj.as_ref()),
        ("q_b_proj", attention.q_b_proj.as_ref()),
        ("kv_b_proj", attention.kv_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            projection_names.push((
                projection,
                format!("{attention_prefix}.{name}"),
                ProjectionSharding::Column,
            ));
        }
    }
    projection_names.push((
        &attention.o_proj,
        format!("{attention_prefix}.o_proj"),
        ProjectionSharding::Row,
    ));
    let projections = projection_names
        .iter()
        .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
        .collect::<Vec<_>>();
    let preferred_heads = usize::try_from(attention.num_heads)
        .map_err(|_| Error::Parallel("DeepSeek MLA head count exceeds usize".into()))?;
    let (mut head_units, mut head_members) =
        partitioned_projection_members(&projections, preferred_heads)?;
    let mut packed_names = Vec::new();
    for (name, projection) in [
        ("k_b_proj", attention.k_b_proj.as_ref()),
        ("v_b_proj", attention.v_b_proj.as_ref()),
    ] {
        if let Some(projection) = projection {
            packed_names.push((
                projection,
                format!("{attention_prefix}.{name}"),
                ProjectionSharding::Column,
            ));
        }
    }
    if !packed_names.is_empty() {
        let packed = packed_names
            .iter()
            .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
            .collect::<Vec<_>>();
        let (packed_units, packed_members) = partitioned_projection_members(&packed, head_units)?;
        head_units = packed_units;
        head_members.extend(packed_members);
    }
    planner.register(ParameterGroupSpec::partitioned(
        format!("{attention_prefix}.heads"),
        ParameterRole::AttentionHeads,
        head_units,
        head_members,
    )?)?;
    for (name, projection) in [
        ("q_a_proj", attention.q_a_proj.as_ref()),
        ("kv_a_proj_with_mqa", Some(&attention.kv_a_proj_with_mqa)),
    ] {
        if let Some(projection) = projection {
            register_projection_module(
                planner,
                projection,
                &format!("{prefix}.self_attn.{name}"),
                ProjectionSharding::Replicated,
            )?;
        }
    }
    for (name, module) in [
        ("q_a_layernorm", attention.q_a_layernorm.as_ref()),
        ("kv_a_layernorm", Some(&attention.kv_a_layernorm)),
    ] {
        if let Some(module) = module {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let register_mlp = |planner: &mut ParallelPlanBuilder,
                        mlp: &resident::Mlp,
                        prefix: &str,
                        intermediate: i32|
     -> Result<(), Error> {
        let intermediate = usize::try_from(intermediate)
            .map_err(|_| Error::Parallel("DeepSeek feed-forward width exceeds usize".into()))?;
        let gate = format!("{prefix}.gate_proj");
        let up = format!("{prefix}.up_proj");
        let down = format!("{prefix}.down_proj");
        register_partitioned_projection_group(
            planner,
            &format!("{prefix}.intermediate"),
            ParameterRole::FeedForwardIntermediate,
            &[
                (&mlp.gate_proj, gate.as_str(), ProjectionSharding::Column),
                (&mlp.up_proj, up.as_str(), ProjectionSharding::Column),
                (&mlp.down_proj, down.as_str(), ProjectionSharding::Row),
            ],
            intermediate,
        )
    };
    match &layer.mlp {
        resident::FeedForward::Dense(mlp) => register_mlp(
            planner,
            mlp,
            &format!("{prefix}.mlp"),
            mlp.gate_proj.output_dims,
        )?,
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            register_mlp(
                planner,
                &moe.shared_experts,
                &format!("{prefix}.mlp.shared_experts"),
                moe.shared_experts.gate_proj.output_dims,
            )?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_size).map_err(|_| {
                Error::Parallel("DeepSeek routed-expert width exceeds usize".into())
            })?;
            let down_alignment = if experts.use_fp8 {
                128
            } else {
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel(
                                "DeepSeek expert quantization group exceeds usize".into(),
                            )
                        })
                    })?
            };
            let expert_units = aligned_partition_units(
                &format!("{prefix}.mlp.experts.intermediate"),
                intermediate,
                1,
                down_alignment,
            )?;
            let mut members = Vec::new();
            for (name, value) in [
                ("gate_proj", experts.gate_proj.as_ref().as_ref()),
                (
                    "gate_proj_scale_inv",
                    experts.gate_proj_scale_inv.as_ref().as_ref(),
                ),
                (
                    "gate_proj_scales",
                    experts.gate_proj_scales.as_ref().as_ref(),
                ),
                (
                    "gate_proj_biases",
                    experts.gate_proj_biases.as_ref().as_ref(),
                ),
                ("up_proj", experts.up_proj.as_ref().as_ref()),
                (
                    "up_proj_scale_inv",
                    experts.up_proj_scale_inv.as_ref().as_ref(),
                ),
                ("up_proj_scales", experts.up_proj_scales.as_ref().as_ref()),
                ("up_proj_biases", experts.up_proj_biases.as_ref().as_ref()),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{prefix}.mlp.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 1 },
                    )?);
                }
            }
            for (name, value) in [
                ("down_proj", experts.down_proj.as_ref().as_ref()),
                (
                    "down_proj_scale_inv",
                    experts.down_proj_scale_inv.as_ref().as_ref(),
                ),
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
    Ok(())
}

impl DeepSeekV3LayerwiseAdapter {}

impl DeepSeekV3LayerwiseAdapter {
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
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.lm_head,
                    "lm_head",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(MTP_UNIT) {
            if let Some(mtp) = &self.mtp {
                units.push(StaticUnitBindings::new(
                    MTP_UNIT,
                    build_module_binding_plan_with_recipes_excluding(
                        mtp,
                        "",
                        store,
                        self.mtp_recipes(store)?,
                        |name| self.sparse_expert_cache && name.contains(".decoder.mlp.experts."),
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }

    pub(crate) fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.layer_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "DeepSeek-V3 decoder has no execution group {group}"
            )))
        }
    }

    pub(crate) fn new_layer(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new_layerwise(
            &self.args,
            index as i32,
            stream,
        )?)
    }

    pub(crate) fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if let Some(moe) = layer.mlp.moe_mut() {
            moe.experts = resident::RoutedExperts::new_compact(
                &self.args,
                index as i32,
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local DeepSeek expert count exceeds i32".into())
                    })?
                },
                stream,
            )?;
        }
        Ok(layer)
    }

    pub(crate) fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        if let Some(moe) = layer.mlp.moe_mut() {
            let intermediate = moe.experts.intermediate_size;
            moe.experts = resident::RoutedExperts::new_compact_with_width(
                &self.args,
                index as i32,
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local DeepSeek expert count exceeds i32".into())
                    })?
                },
                intermediate,
                stream,
            )?;
        }
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
        if self.args.n_routed_experts <= 0
            || !self
                .args
                .layer_schedule
                .iter()
                .any(|policy| *policy == LayerPolicy::SparseMoe)
        {
            return Err(Error::Parallel(
                "DeepSeek PP+EP requires a checkpoint with sparse MoE layers".into(),
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
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_lm_head_parameter_group(
                &self.lm_head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?,
        )?;
        for index in 0..self.args.layer_schedule.len() {
            let layer = DecoderLayer::new_layerwise(&self.args, index as i32, stream)?;
            register_deepseek_layer_parallel_plan(planner, &layer, index)?;
        }
        if let Some(mtp) = &self.mtp {
            for (index, layer) in mtp.layers.iter().enumerate() {
                let prefix = format!("layers.{index}");
                register_replicated_module(planner, &layer.enorm, &format!("{prefix}.enorm"))?;
                register_replicated_module(planner, &layer.hnorm, &format!("{prefix}.hnorm"))?;
                register_projection_module(
                    planner,
                    &layer.eh_proj,
                    &format!("{prefix}.eh_proj"),
                    ProjectionSharding::Replicated,
                )?;
                register_deepseek_layer_parallel_plan_at(
                    planner,
                    &layer.decoder,
                    self.args.num_hidden_layers as usize + index,
                    &format!("{prefix}.decoder"),
                )?;
                register_replicated_module(
                    planner,
                    &layer.shared_norm,
                    &format!("{prefix}.shared_norm"),
                )?;
                layer
                    .shared_head
                    .register(planner, &format!("{prefix}.shared_head"))?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_parallel_static(
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
        if let Some(mtp) = &mut self.mtp {
            let count = mtp.layers.len();
            let mut policies = self.args.layer_schedule.iter().cloned().collect::<Vec<_>>();
            policies.extend(std::iter::repeat_n(LayerPolicy::SparseMoe, count));
            let mut mtp_args = self.args.clone();
            mtp_args.layer_schedule = crate::LayerSchedule::new(policies.len(), policies)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            for (index, layer) in mtp.layers.iter_mut().enumerate() {
                let prefix = format!("layers.{index}.decoder");
                let tensor = |suffix: &str| {
                    layout
                        .tensor(&format!("{prefix}.{suffix}.weight"))
                        .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
                };
                let attention = tensor("self_attn.q_proj")
                    .or_else(|| tensor("self_attn.q_b_proj"))
                    .ok_or_else(|| {
                        Error::Parallel(format!("missing TP layout for {prefix} MLA query"))
                    })?;
                let local_heads = i32::try_from(attention.local_shape()[0])
                    .map_err(|_| Error::Parallel("DeepSeek MTP query width exceeds i32".into()))?
                    / (self.args.qk_nope_head_dim + self.args.qk_rope_head_dim);
                let local_width = |suffix: &str,
                                   axis: usize,
                                   fallback: i32|
                 -> Result<i32, Error> {
                    tensor(suffix)
                            .map(|value| {
                                value
                                    .local_shape()
                                    .get(axis)
                                    .copied()
                                    .ok_or_else(|| {
                                        Error::Parallel(format!(
                                            "DeepSeek TP layout for {prefix}.{suffix} has no axis {axis}"
                                        ))
                                    })
                                    .and_then(|width| {
                                        i32::try_from(width).map_err(|_| {
                                            Error::Parallel(format!(
                                                "DeepSeek local width for {prefix}.{suffix} exceeds i32"
                                            ))
                                        })
                                    })
                            })
                            .transpose()
                            .map(|value| value.unwrap_or(fallback))
                };
                let dense_intermediate =
                    local_width("mlp.gate_proj", 0, self.args.intermediate_size)?;
                let routed_intermediate = layout
                    .tensor(&format!("{prefix}.mlp.experts.gate_proj"))
                    .map(|value| {
                        i32::try_from(value.local_shape()[1]).map_err(|_| {
                            Error::Parallel("DeepSeek MTP routed width exceeds i32".into())
                        })
                    })
                    .transpose()?
                    .unwrap_or(self.args.moe_intermediate_size);
                let shared_intermediate = local_width(
                    "mlp.shared_experts.gate_proj",
                    0,
                    self.args.moe_intermediate_size * self.args.n_shared_experts,
                )?;
                let global = self.args.num_hidden_layers as usize + index;
                layer.decoder = DecoderLayer::new_parallel_layerwise(
                    &mtp_args,
                    global as i32,
                    local_heads,
                    dense_intermediate,
                    routed_intermediate,
                    shared_intermediate,
                    stream,
                )?;
                layer.shared_head.configure_parallel(context, stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<DecoderLayer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let tensor = |suffix: &str| {
            layout
                .tensor(&format!("{prefix}.{suffix}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{suffix}.inner.weight")))
        };
        let attention = tensor("self_attn.q_proj")
            .or_else(|| tensor("self_attn.q_b_proj"))
            .ok_or_else(|| Error::Parallel(format!("missing TP layout for {prefix} MLA query")))?;
        let local_heads = i32::try_from(attention.local_shape()[0])
            .map_err(|_| Error::Parallel("DeepSeek local query width exceeds i32".into()))?
            / (self.args.qk_nope_head_dim + self.args.qk_rope_head_dim);
        let local_width = |suffix: &str, axis: usize, fallback: i32| -> Result<i32, Error> {
            tensor(suffix)
                .map(|value| {
                    value
                        .local_shape()
                        .get(axis)
                        .copied()
                        .ok_or_else(|| {
                            Error::Parallel(format!(
                                "DeepSeek TP layout for {prefix}.{suffix} has no axis {axis}"
                            ))
                        })
                        .and_then(|width| {
                            i32::try_from(width).map_err(|_| {
                                Error::Parallel(format!(
                                    "DeepSeek local width for {prefix}.{suffix} exceeds i32"
                                ))
                            })
                        })
                })
                .transpose()
                .map(|value| value.unwrap_or(fallback))
        };
        let dense_intermediate = local_width("mlp.gate_proj", 0, self.args.intermediate_size)?;
        let routed_intermediate = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_proj"))
            .map(|value| {
                i32::try_from(value.local_shape()[1]).map_err(|_| {
                    Error::Parallel("DeepSeek local routed-expert width exceeds i32".into())
                })
            })
            .transpose()?
            .unwrap_or(self.args.moe_intermediate_size);
        let shared_intermediate = local_width(
            "mlp.shared_experts.gate_proj",
            0,
            self.args.moe_intermediate_size * self.args.n_shared_experts,
        )?;
        Ok(DecoderLayer::new_parallel_layerwise(
            &self.args,
            index as i32,
            local_heads,
            dense_intermediate,
            routed_intermediate,
            shared_intermediate,
            stream,
        )?)
    }

    pub(crate) fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    pub(crate) fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &DecoderLayer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        Ok(self
            .binding_plan_for_layer(layer, index, store)?
            .build_bindings(store)?)
    }

    pub(crate) fn parallel_layer_bindings(
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

    pub(crate) fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &DecoderLayer,
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
    ) -> Result<DecoderLayer, Error> {
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
        layer: &DecoderLayer,
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
        layer: &DecoderLayer,
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

pub(crate) fn deepseek_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut expert_layers = args
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| (*policy == LayerPolicy::SparseMoe).then_some(layer))
        .collect::<Vec<_>>();
    expert_layers.extend(
        (0..args.num_nextn_predict_layers as usize)
            .map(|index| args.num_hidden_layers as usize + index),
    );
    deepseek_expert_catalog_for_layers(args, store, expert_layers)
}

pub(crate) fn deepseek_pipeline_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layers: impl IntoIterator<Item = usize>,
    include_mtp: bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut expert_layers = layers
        .into_iter()
        .filter(|layer| args.layer_schedule.get(*layer) == Some(&LayerPolicy::SparseMoe))
        .collect::<Vec<_>>();
    if include_mtp {
        expert_layers.extend(
            (0..args.num_nextn_predict_layers as usize)
                .map(|index| args.num_hidden_layers as usize + index),
        );
    }
    deepseek_expert_catalog_for_layers(args, store, expert_layers)
}

fn deepseek_expert_catalog_for_layers(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    expert_layers: impl IntoIterator<Item = usize>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for layer in expert_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for expert in 0..usize::try_from(args.n_routed_experts).map_err(|_| {
            Error::UnsupportedArchitecture("DeepSeek-V3 expert count is negative".into())
        })? {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let packed = normalized.get(&format!("{prefix}.{projection}"));
                for (runtime_suffix, checkpoint_component, required) in [
                    ("", "weight", true),
                    ("_scale_inv", "weight_scale_inv", false),
                    ("_scales", "scales", false),
                    ("_biases", "biases", false),
                ] {
                    let binding_name = format!("{projection}{runtime_suffix}");
                    let recipe = if let Some(packed_key) = packed {
                        let runtime = format!("{prefix}.{projection}{runtime_suffix}");
                        match normalized.get(&runtime) {
                            Some(raw) => Some(DerivedWeightRecipe::source(
                                raw.clone(),
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            )),
                            None if required => Some(DerivedWeightRecipe::source(
                                packed_key.clone(),
                                TensorSelection::Range {
                                    axis: 0,
                                    start: expert,
                                    end: expert + 1,
                                },
                            )),
                            None => None,
                        }
                    } else {
                        let runtime =
                            format!("{prefix}.{expert}.{projection}.{checkpoint_component}");
                        match normalized.get(&runtime) {
                            Some(raw) => Some(DerivedWeightRecipe::Stack {
                                axis: 0,
                                inputs: vec![DerivedWeightRecipe::source(
                                    raw.clone(),
                                    TensorSelection::Full,
                                )],
                            }),
                            None if required => {
                                return Err(Error::UnsupportedArchitecture(format!(
                                    "DeepSeek-V3 checkpoint is missing expert tensor {runtime}"
                                )));
                            }
                            None => None,
                        }
                    };
                    if let Some(recipe) = recipe {
                        bindings.push(deepseek_recipe_binding(&binding_name, recipe, store)?);
                    }
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "DeepSeek-V3 expert byte total overflowed".into(),
                    )
                })
            })?;
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn deepseek_recipe_binding(
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

/// DeepSeek token generation using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, DeepSeekV3LayerwiseModel, Cache, S>;

#[cfg(test)]
mod neutral_runtime_tests {
    #[test]
    fn production_model_and_loaders_use_the_neutral_layerwise_runtime() {
        let source = include_str!("layerwise.rs");
        let wrapper_start = source
            .find("pub struct DeepSeekV3LayerwiseModel")
            .expect("DeepSeek V3 production wrapper");
        let adapter_start = source
            .find("/// Adapter for compressed MLA and mixed dense/MoE DeepSeek decoder blocks.")
            .expect("pipeline-only legacy adapter marker");
        let production = &source[wrapper_start..adapter_start];
        assert!(production.contains("DeepSeekV3Execution"));
        for legacy in ["LayerwiseModel<", ".adapter()", ".adapter_mut()"] {
            assert!(
                !production.contains(legacy),
                "production DeepSeek V3 wrapper still references {legacy}"
            );
        }

        let loaders_start = source
            .find("fn resolve_deepseek_v3_store")
            .expect("neutral DeepSeek V3 loader");
        let loaders = &source[loaders_start..adapter_start];
        for legacy in [
            "load_layerwise_model(",
            "load_layerwise_model_with_quantization(",
            "load_tensor_parallel_layerwise_model(",
            "DeepSeekV3LayerwiseAdapter::new(",
            "DeepSeekV3LayerwiseAdapter::new_sparse(",
        ] {
            assert!(
                !loaders.contains(legacy),
                "production DeepSeek V3 loaders still reference {legacy}"
            );
        }
    }
}
