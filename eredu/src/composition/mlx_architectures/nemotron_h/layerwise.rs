//! Unified fully resident and bounded layer execution for Nemotron-H.

use eredu_runtime::{
    ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, LayerWeightResidency,
    NonExpertWeightResidency, WeightResidency,
};

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::{
    MemberSharding, OffloadUnit, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
    WeightBinding,
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
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        linear::project_logits_maybe_quantized,
        parallel::{
            register_gqa_projection_group, GqaProjectionNames, VocabParallelEmbedding,
            VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::KeyValueCache,
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, build_module_binding_plan_with_recipes_excluding,
        canonical_checkpoint_name, packed_companion_checkpoint_name, populate_module_from_lease,
        populate_module_from_lease_excluding,
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
    backend::mlx::runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter, StaticUnitBindings,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheReport, ExpertCatalogEntry, ExpertRouteBatch,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    composition::mlx_architectures::nemotron_h::model::{
        self as resident, BlockInput, Cache, Experts, LayerCache, LayerPolicy, ModelArgs,
        TransformerBlock,
    },
};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "nemotron_h.static.embedding";
const NORM_UNIT: &str = "nemotron_h.static.norm";
const HEAD_UNIT: &str = "nemotron_h.static.output";
const MTP_UNIT: &str = "nemotron_h.static.mtp";

type NemotronMtpExpertExecutor<'a> =
    dyn FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception> + 'a;

#[derive(Debug, Clone, ModuleParameters)]
struct NemotronMtpModule {
    #[param]
    enorm: Vec<nn::RmsNorm>,
    #[param]
    hnorm: Vec<nn::RmsNorm>,
    #[param]
    eh_proj: Vec<MaybeQuantized<nn::Linear>>,
    #[param]
    blocks: Vec<TransformerBlock>,
    #[param]
    final_norms: Vec<nn::RmsNorm>,
    pattern_len: usize,
    steps: usize,
    policies: Vec<LayerPolicy>,
}

impl NemotronMtpModule {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Option<Self>, Error> {
        let policies = args.mtp_policies()?;
        if policies.is_empty() {
            return Ok(None);
        }
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(|_| {
            Error::UnsupportedArchitecture("Nemotron-H MTP layer count is negative".into())
        })?;
        let pattern_len = policies.len() / steps;
        let mut block_args = args.clone();
        let mut complete = args.layer_schedule.iter().copied().collect::<Vec<_>>();
        complete.extend(policies.iter().copied());
        block_args.layer_schedule = crate::LayerSchedule::new(complete.len(), complete)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let blocks = policies
            .iter()
            .enumerate()
            .map(|(index, _)| {
                TransformerBlock::new(&block_args, args.num_hidden_layers as usize + index, stream)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let final_norms = (0..steps)
            .map(|_| {
                nn::RmsNorm::unloaded(
                    args.hidden_size,
                    args.layer_norm_epsilon,
                    args.weight_dtype(),
                    stream,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            enorm: (0..steps)
                .map(|_| {
                    nn::RmsNorm::unloaded(
                        args.hidden_size,
                        args.layer_norm_epsilon,
                        args.weight_dtype(),
                        stream,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            hnorm: (0..steps)
                .map(|_| {
                    nn::RmsNorm::unloaded(
                        args.hidden_size,
                        args.layer_norm_epsilon,
                        args.weight_dtype(),
                        stream,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            eh_proj: (0..steps)
                .map(|step| {
                    common::linear::unloaded_maybe_quantized_linear_with_dtype(
                        args.hidden_size * 2,
                        args.hidden_size,
                        false,
                        args.weight_quantization_for(&format!(
                            "mtp.layers.{}.eh_proj.weight",
                            step * pattern_len
                        )),
                        args.weight_dtype(),
                        stream,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            blocks,
            final_norms,
            pattern_len,
            steps,
            policies,
        }))
    }

    fn len(&self) -> usize {
        self.steps
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &mut self,
        hidden: &Array,
        embeddings: &Array,
        depth: usize,
        cache: &mut [LayerCache],
        expert_cache: Option<&ExpertCache>,
        mut external_expert: Option<&mut NemotronMtpExpertExecutor<'_>>,
        args: &ModelArgs,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if cache.len() != self.blocks.len() || depth >= self.steps {
            return Err(Exception::custom(
                "Nemotron-H MTP cache does not match prediction geometry",
            ));
        }
        let embeddings = self.enorm[depth].forward(embeddings, stream)?;
        let hidden = self.hnorm[depth].forward(hidden, stream)?;
        let fused = concatenate_axis(&[&embeddings, &hidden], -1, stream)?;
        let mut hidden = self.eh_proj[depth].forward(&fused, stream)?;
        let start = depth * self.pattern_len;
        let end = start + self.pattern_len;
        let mask = if hidden.dim(1) > 1 {
            let offset = cache[start..end]
                .iter()
                .find_map(LayerCache::offset)
                .unwrap_or(0);
            let offset_cache = vec![Some(OffsetOnlyCache(offset))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom(
                        "Nemotron-H MTP requires an array causal mask",
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        for (relative, (block, layer_cache)) in self.blocks[start..end]
            .iter_mut()
            .zip(&mut cache[start..end])
            .enumerate()
        {
            let index = start + relative;
            let input = BlockInput {
                x: &hidden,
                mask: mask.as_ref(),
                cache: Some(layer_cache),
            };
            hidden = if self.policies[index] == LayerPolicy::SparseMoe {
                if expert_cache.is_some() || external_expert.is_some() {
                    let global = args.num_hidden_layers as usize + index;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    let execute =
                        |flat: &Array, indices: &Array, weights: &Array, stream: &Stream| {
                            if let Some(execute) = external_expert.as_deref_mut() {
                                execute(global, flat, indices, weights, stream)
                            } else {
                                execute_cached_nemotron_experts(
                                    args,
                                    expert_cache.expect("checked Nemotron-H MTP expert source"),
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
                        Some(execution) => block.forward_tensor_with_expert_executor(
                            input,
                            execution.group().ok_or_else(|| {
                                Exception::custom(
                                    "Nemotron-H MTP TP execution is missing its group",
                                )
                            })?,
                            stream,
                            execute,
                        )?,
                        None => block.forward_sparse_experts(input, stream, execute)?,
                    }
                } else {
                    match execution.filter(|execution| execution.is_tensor_parallel()) {
                        Some(execution) => block.forward_tensor_parallel(
                            input,
                            execution.group().ok_or_else(|| {
                                Exception::custom(
                                    "Nemotron-H MTP TP execution is missing its group",
                                )
                            })?,
                            stream,
                        )?,
                        None => block.forward(input, stream)?,
                    }
                }
            } else {
                match execution.filter(|execution| execution.is_tensor_parallel()) {
                    Some(execution) => block.forward_tensor_parallel(
                        input,
                        execution.group().ok_or_else(|| {
                            Exception::custom("Nemotron-H MTP TP execution is missing its group")
                        })?,
                        stream,
                    )?,
                    None => block.forward(input, stream)?,
                }
            };
        }
        self.final_norms[depth].forward(&hidden, stream)
    }
}

fn register_nemotron_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &ModelArgs,
    index: usize,
) -> Result<(), Error> {
    register_nemotron_layer_parallel_plan_at(
        planner,
        layer,
        args,
        index,
        &format!("model.layers.{index}"),
    )
}

fn register_nemotron_layer_parallel_plan_at(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    args: &ModelArgs,
    index: usize,
    prefix: &str,
) -> Result<(), Error> {
    register_replicated_module(planner, &layer.norm, &format!("{prefix}.norm"))?;
    match layer.policy {
        LayerPolicy::Mamba => {
            let mamba = layer.mamba.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its Mamba mixer"
                ))
            })?;
            let intermediate = usize::try_from(mamba.intermediate_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba width exceeds usize".into()))?;
            let grouped = usize::try_from(mamba.n_groups * mamba.ssm_state_size)
                .map_err(|_| Error::Parallel("Nemotron Mamba group width exceeds usize".into()))?;
            let heads = usize::try_from(mamba.num_heads)
                .map_err(|_| Error::Parallel("Nemotron Mamba heads exceed usize".into()))?;
            let groups = usize::try_from(mamba.n_groups)
                .map_err(|_| Error::Parallel("Nemotron Mamba groups exceed usize".into()))?;
            if groups == 0 || !heads.is_multiple_of(groups) {
                return Err(Error::Parallel(format!(
                    "Nemotron Mamba geometry has {heads} heads and {groups} groups"
                )));
            }
            let out_prefix = format!("{prefix}.mamba.out_proj");
            let (units, mut members) = partitioned_projection_members(
                &[(
                    &mamba.out_proj,
                    out_prefix.as_str(),
                    ProjectionSharding::Row,
                )],
                groups,
            )?;
            let in_segments = vec![
                0..intermediate,
                intermediate..2 * intermediate,
                2 * intermediate..2 * intermediate + grouped,
                2 * intermediate + grouped..2 * intermediate + 2 * grouped,
                2 * intermediate + 2 * grouped..2 * intermediate + 2 * grouped + heads,
            ];
            let projection_size = in_segments.last().map_or(0, |segment| segment.end);
            for (name, parameter) in mamba.in_proj.parameters().flatten() {
                let shape = parameter
                    .shape()
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            Error::Parallel(format!(
                                "parameter {prefix}.mamba.in_proj.{name} has negative dimension {dimension}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if shape.first().copied() != Some(projection_size) {
                    return Err(Error::Parallel(format!(
                        "Nemotron Mamba input member {prefix}.mamba.in_proj.{name} has shape {shape:?}, expected fused output width {projection_size}"
                    )));
                }
                members.push(ParameterMemberSpec::new(
                    format!("{prefix}.mamba.in_proj.{name}"),
                    shape,
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: in_segments.clone(),
                    },
                ));
            }
            let conv_segments = vec![
                0..intermediate,
                intermediate..intermediate + grouped,
                intermediate + grouped..intermediate + 2 * grouped,
            ];
            members.push(array_parameter_member(
                format!("{prefix}.mamba.conv1d.weight"),
                mamba.conv1d.weight.as_ref(),
                MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: conv_segments.clone(),
                },
            )?);
            if let Some(bias) = mamba.conv1d.bias.as_ref().as_ref() {
                members.push(array_parameter_member(
                    format!("{prefix}.mamba.conv1d.bias"),
                    bias,
                    MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: conv_segments,
                    },
                )?);
            }
            for (name, value) in [
                ("dt_bias", mamba.dt_bias.as_ref()),
                ("A_log", mamba.A_log.as_ref()),
                ("D", mamba.D.as_ref()),
            ] {
                members.push(array_parameter_member(
                    format!("{prefix}.mamba.{name}"),
                    value,
                    MemberSharding::Partitioned { axis: 0 },
                )?);
            }
            members.push(array_parameter_member(
                format!("{prefix}.mamba.norm.weight"),
                mamba.norm.weight.as_ref(),
                MemberSharding::Partitioned { axis: 0 },
            )?);
            planner.register(ParameterGroupSpec::partitioned(
                format!("{prefix}.mamba.groups"),
                ParameterRole::Channels,
                units,
                members,
            )?)?;
        }
        LayerPolicy::SelfAttention(_) => {
            let attention = layer.attention.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its attention mixer"
                ))
            })?;
            register_gqa_projection_group(
                planner,
                &format!("{prefix}.attention"),
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
        }
        LayerPolicy::DenseMlp => {
            let mlp = layer.mlp.as_ref().ok_or_else(|| {
                Error::Parallel(format!("Nemotron-H layer {index} is missing its dense MLP"))
            })?;
            let intermediate = usize::try_from(args.intermediate_size)
                .map_err(|_| Error::Parallel("Nemotron dense MLP width exceeds usize".into()))?;
            let up = format!("{prefix}.mlp.up_proj");
            let down = format!("{prefix}.mlp.down_proj");
            register_partitioned_projection_group(
                planner,
                &format!("{prefix}.mlp.intermediate"),
                ParameterRole::FeedForwardIntermediate,
                &[
                    (&mlp.up_proj, up.as_str(), ProjectionSharding::Column),
                    (&mlp.down_proj, down.as_str(), ProjectionSharding::Row),
                ],
                intermediate,
            )?;
        }
        LayerPolicy::SparseMoe => {
            let moe = layer.moe.as_ref().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H layer {index} is missing its sparse MoE"
                ))
            })?;
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.moe.gate"))?;
            let shared =
                usize::try_from(args.moe_shared_expert_intermediate_size).map_err(|_| {
                    Error::Parallel("Nemotron shared-expert width exceeds usize".into())
                })?;
            let shared_up = format!("{prefix}.moe.shared_experts.up_proj");
            let shared_down = format!("{prefix}.moe.shared_experts.down_proj");
            register_partitioned_projection_group(
                planner,
                &format!("{prefix}.moe.shared_experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                &[
                    (
                        &moe.shared_experts.up_proj,
                        shared_up.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &moe.shared_experts.down_proj,
                        shared_down.as_str(),
                        ProjectionSharding::Row,
                    ),
                ],
                shared,
            )?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_size).map_err(|_| {
                Error::Parallel("Nemotron routed-expert width exceeds usize".into())
            })?;
            let alignment = experts
                .down_quantization
                .map(WeightQuantization::group_size)
                .or_else(|| {
                    experts
                        .down_iquant
                        .map(|quantization| quantization.group_size())
                })
                .map_or(Ok(1usize), |alignment| {
                    usize::try_from(alignment).map_err(|_| {
                        Error::Parallel("Nemotron expert alignment exceeds usize".into())
                    })
                })?;
            let units = aligned_partition_units(
                &format!("{prefix}.moe.experts.intermediate"),
                intermediate,
                1,
                alignment,
            )?;
            let mut members = vec![array_parameter_member(
                format!("{prefix}.moe.experts.up_proj"),
                experts.up_proj.as_ref(),
                MemberSharding::Partitioned { axis: 1 },
            )?];
            for (name, value) in [
                ("up_proj_scales", experts.up_proj_scales.as_ref().as_ref()),
                ("up_proj_biases", experts.up_proj_biases.as_ref().as_ref()),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{prefix}.moe.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 1 },
                    )?);
                }
            }
            members.push(array_parameter_member(
                format!("{prefix}.moe.experts.down_proj"),
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
                        format!("{prefix}.moe.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::partitioned(
                format!("{prefix}.moe.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                units,
                members,
            )?)?;
        }
    }
    Ok(())
}

/// Nemotron-H causal LM using bounded residency for hybrid blocks.
pub struct NemotronHLayerwiseModel {
    execution: LayerwiseModel<NemotronHLayerwiseAdapter>,
}

pub(crate) struct NemotronHTensorMtpTarget<'a> {
    model: &'a mut NemotronHLayerwiseModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> NemotronHTensorMtpTarget<'a> {
    pub(crate) fn new(
        model: &'a mut NemotronHLayerwiseModel,
        group: &'a safemlx::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl NemotronHLayerwiseModel {
    /// Returns validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Creates cache/state matching the hybrid block pattern.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution
            .adapter()
            .mtp
            .as_ref()
            .map_or(0, NemotronMtpModule::len)
    }

    fn forward_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(tokens, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Nemotron-H layerwise pass did not retain MTP hidden state")
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
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let adapter = self.execution.adapter_mut();
        let embeddings = adapter.embeddings.forward(tokens, stream)?;
        let expert_cache = adapter.expert_cache.as_ref();
        let args = &adapter.args;
        let hidden = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Nemotron-H checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                depth,
                cache,
                expert_cache,
                None,
                args,
                None,
                stream,
            )?;
        let logits = project_logits_maybe_quantized(
            &mut adapter.lm_head,
            &mut adapter.embeddings,
            &hidden,
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

    fn forward_mtp_target_tensor(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_tensor_parallel_with_context(tokens, cache, group, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("Nemotron-H tensor pass did not retain MTP hidden state")
                })?,
                tokens: tokens.clone(),
            },
        )
    }

    fn forward_mtp_draft_tensor(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [LayerCache],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception> {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("Nemotron-H MTP target has no parallel topology"))?
            .topology();
        let execution =
            crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                topology, group, stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        let adapter = self.execution.adapter_mut();
        let embeddings = adapter
            .parallel_embedding
            .as_mut()
            .ok_or_else(|| Exception::custom("Nemotron-H MTP has no TP embedding shard"))?
            .forward(tokens, &execution)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let expert_cache = adapter.expert_cache.as_ref();
        let args = &adapter.args;
        let hidden = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Nemotron-H checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                depth,
                cache,
                expert_cache,
                None,
                args,
                Some(&execution),
                stream,
            )?;
        let logits = match (
            adapter.parallel_lm_head.as_mut(),
            adapter.parallel_embedding.as_mut(),
        ) {
            (Some(head), _) => head
                .forward(&hidden, &execution)
                .and_then(|output| output.all_gather(&execution)),
            (None, Some(embedding)) => embedding
                .project_logits(&hidden, &execution)
                .and_then(|output| output.all_gather(&execution)),
            (None, None) => Err(Error::Parallel(
                "Nemotron-H MTP has no TP output projection".into(),
            )),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    /// Creates resident state or pages attention blocks while bounded Mamba
    /// convolution and recurrent tensors remain device resident.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        let rank = self.execution.prompt_cache_rank_identity();
        let cache = Cache::new_with_options_and_rank(self.args(), policy.clone(), rank)?;
        match (&self.execution.adapter().mtp, policy) {
            (Some(mtp), CacheResidencyPolicy::Device) => Ok(cache.with_mtp_policies(&mtp.policies)),
            (Some(mtp), CacheResidencyPolicy::Paged(_)) => cache
                .with_paged_mtp_policies(&mtp.policies, rank)
                .map_err(Into::into),
            (None, _) => Ok(cache),
        }
    }

    /// Returns aggregate live attention paging observations, if enabled.
    pub fn cache_residency_report(
        &self,
        cache: &Cache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        cache.residency_report().map_err(Into::into)
    }

    /// Returns the stable global architecture fingerprint used by prompt caches.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&eredu_runtime::ParallelModelInfo<crate::backend::mlx::MlxParallelContext>> {
        self.execution.parallel_info()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Persists a compatible prefix cache.
    pub fn save_prompt_cache(
        &self,
        cache: &mut Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.execution.save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible prefix cache.
    pub fn load_prompt_cache(
        &self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        self.execution
            .load_prompt_cache(directory, expected, prefix_token_ids, options, stream)
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

    /// Returns sparse expert-cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Into::into)
    }

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn eredu_checkpoint::store::CheckpointSource) {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Runs the hybrid decoder while preserving KV and Mamba state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }
    /// Runs a rank-local tensor-parallel hybrid forward pass.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(inputs, cache, group, stream)
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
        self.execution.forward_with_layer_executor(
            inputs,
            cache,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, stream| {
                Ok(layer.forward_sparse_experts(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Runs TP-sharded Mamba, attention, dense, and shared-expert projections
    /// while delegating routed experts to the matching EP subgroup.
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
        self.execution.forward_tensor_parallel_with_layer_executor(
            inputs,
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, execution| {
                let tp_group = execution.group().ok_or_else(|| {
                    Error::Parallel("Nemotron-H TP+EP execution requires an active TP group".into())
                })?;
                Ok(layer.forward_tensor_with_expert_executor(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    tp_group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
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
        let (logits, context) = match tensor_group {
            Some(tensor_group) => self
                .execution
                .forward_tensor_parallel_with_layer_executor_and_context(
                    inputs,
                    cache,
                    tensor_group,
                    stream,
                    |_adapter, _group, index, layer, hidden, cache, context, execution| {
                        Ok(layer.forward_tensor_with_expert_executor(
                            BlockInput {
                                x: hidden,
                                mask: context.mask.as_ref(),
                                cache: Some(&mut cache.layers[index]),
                            },
                            execution.group().ok_or_else(|| {
                                Error::Parallel(
                                    "Nemotron-H TP+EP MTP target is missing its TP group".into(),
                                )
                            })?,
                            execution.stream(),
                            |hidden, ids, weights, stream| {
                                execute(index, hidden, ids, weights, stream)
                            },
                        )?)
                    },
                ),
            None => self.execution.forward_with_layer_executor_and_context(
                inputs,
                cache,
                stream,
                |_adapter, _group, index, layer, hidden, cache, context, stream| {
                    Ok(layer.forward_sparse_experts(
                        BlockInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: Some(&mut cache.layers[index]),
                        },
                        stream,
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?)
                },
            ),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden: context.draft_hidden.ok_or_else(|| {
                    Exception::custom("Nemotron-H EP pass did not retain MTP hidden state")
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
        cache: &mut [LayerCache],
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("Nemotron-H EP MTP target has no topology"))?
            .topology();
        let execution = tensor_group
            .map(|group| {
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )
            })
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let adapter = self.execution.adapter_mut();
        let embeddings = match execution.as_ref() {
            Some(execution) => adapter
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Exception::custom("Nemotron-H MTP has no TP embedding shard"))?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => adapter.embeddings.forward(tokens, stream)?,
        };
        let args = &adapter.args;
        let hidden = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Nemotron-H checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                depth,
                cache,
                None,
                Some(&mut execute),
                args,
                execution.as_ref(),
                stream,
            )?;
        let logits = match execution.as_ref() {
            Some(execution) => match (
                adapter.parallel_lm_head.as_mut(),
                adapter.parallel_embedding.as_mut(),
            ) {
                (Some(head), _) => head
                    .forward(&hidden, execution)
                    .and_then(|output| output.all_gather(execution)),
                (None, Some(embedding)) => embedding
                    .project_logits(&hidden, execution)
                    .and_then(|output| output.all_gather(execution)),
                (None, None) => Err(Error::Parallel(
                    "Nemotron-H MTP has no TP output projection".into(),
                )),
            }
            .map_err(|error| Exception::custom(error.to_string()))?,
            None => project_logits_maybe_quantized(
                &mut adapter.lm_head,
                &mut adapter.embeddings,
                &hidden,
                stream,
            )?,
        };
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    /// Clears temporary hybrid blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalModel<Cache> for NemotronHLayerwiseModel {
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

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget for NemotronHLayerwiseModel {
    type Cache = Cache;
    type DraftCache = Vec<LayerCache>;

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
        for depth in 0..self.mtp_len() {
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
        for depth in 0..self.mtp_len() {
            let _ = self.forward_mtp_draft(hidden, tokens, depth, cache, stream)?;
        }
        Ok(())
    }
    fn max_draft_tokens(&self) -> usize {
        self.mtp_len()
    }
}

impl crate::composition::mlx::speculative::embedded::EmbeddedMtpTarget
    for NemotronHTensorMtpTarget<'_>
{
    type Cache = Cache;
    type DraftCache = Vec<LayerCache>;

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
        for depth in 0..self.model.mtp_len() {
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
        for depth in 0..self.model.mtp_len() {
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

/// Loads Nemotron-H through the generalized execution engine.
pub fn load_nemotron_h_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_nemotron_h_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Nemotron-H layerwise model", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(NemotronHLayerwiseModel {
        execution: load_layerwise_model_with_quantization(
            store,
            adapter,
            options,
            quantize_on_load,
            stream,
            weights_stream,
        )?,
    })
}
/// Loads Nemotron-H through the generalized tensor-parallel engine.
pub(crate) fn load_nemotron_h_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_nemotron_h_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    Ok(NemotronHLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            NemotronHLayerwiseAdapter::new(
                resident::get_nemotron_h_model_args(model_dir)?,
                stream,
            )?,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_nemotron_h_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(NemotronHLayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared =
        resident::prepare_nemotron_h_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            resident::translate_gguf_weight_name,
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        NemotronHLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        NemotronHLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

pub(crate) fn load_nemotron_h_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(NemotronHLayerwiseModel, Vec<u32>), Error> {
    let architecture = match metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(name)) if name == "nemotron_h" => {
            crate::core::GgufArchitecture::NemotronH
        }
        Some(GgufMetadataValue::String(name)) if name == "nemotron_h_moe" => {
            crate::core::GgufArchitecture::NemotronHMoe
        }
        Some(GgufMetadataValue::String(name)) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF architecture {name:?}; this loader supports nemotron_h and nemotron_h_moe"
            )));
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"general.architecture\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata is missing required key \"general.architecture\"".into(),
            ));
        }
    };
    let load_options = quantization
        .map(crate::backend::mlx::ModelLoadOptions::with_quantization)
        .unwrap_or_default()
        .with_weight_residency(residency);
    crate::composition::mlx::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        load_options,
    )
    .into_loader_result()?;
    let prepared =
        resident::prepare_nemotron_h_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
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
            load_nemotron_h_gguf_sparse_with_store(
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
    let execution = load_layerwise_model_with_quantization(
        store,
        NemotronHLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((
        NemotronHLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

fn load_nemotron_h_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    if !args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::SparseMoe)
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Nemotron-H MoE GGUF checkpoint".into(),
        ));
    }
    let mut adapter = NemotronHLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let checkpoint_store = execution.checkpoint_store_arc();
    let entries = nemotron_h_expert_catalog(&args, checkpoint_store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(match quantization {
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
    Ok(NemotronHLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Nemotron-H execution base used by distributed EP.
pub(crate) fn load_nemotron_h_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let mut adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(NemotronHLayerwiseModel { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_nemotron_h_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let mut adapter = NemotronHLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(NemotronHLayerwiseModel { execution })
}

/// Loads Nemotron-H with independently cached experts and bounded non-expert units.
pub fn load_nemotron_h_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<NemotronHLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_nemotron_h_model_args(model_dir)?;
    if !args
        .layer_schedule
        .iter()
        .any(|policy| *policy == LayerPolicy::SparseMoe)
    {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Nemotron-H MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Nemotron-H independent expert cache",
                args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut adapter = NemotronHLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantize_on_load,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = nemotron_h_expert_catalog(&args, store.as_ref())?;
    execution.adapter_mut().expert_cache = Some(match quantize_on_load {
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
    Ok(NemotronHLayerwiseModel { execution })
}

/// Adapter shared by Nemotron-H Mamba, attention, dense, and MoE blocks.
pub struct NemotronHLayerwiseAdapter {
    args: ModelArgs,
    embeddings: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    mtp: Option<NemotronMtpModule>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_geometry: Option<Vec<resident::ParallelLayerGeometry>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl NemotronHLayerwiseAdapter {
    /// Creates metadata-only pinned modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        resident::validate_model_args(&args)?;
        let mtp = NemotronMtpModule::new(&args, stream)?;
        let embeddings = common::linear::unloaded_maybe_quantized_embedding_with_dtype(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embeddings.weight"),
            args.weight_dtype(),
            stream,
        )?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, args.weight_dtype(), stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear_with_dtype(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                args.weight_dtype(),
                stream,
            )?)
        };
        Ok(Self {
            args,
            embeddings,
            norm,
            lm_head,
            mtp,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_geometry: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Creates a metadata-only adapter whose routed experts are supplied by
    /// the independent expert residency manager.
    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns validated model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Configures rank-local operator geometry for a Cartesian pipeline stage.
    pub(crate) fn configure_cartesian_layout(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.configure_parallel_static(build, layout, stream)
    }

    /// Returns the configured rank-local hybrid operator geometry.
    pub(crate) fn parallel_geometry(&self) -> Option<&[resident::ParallelLayerGeometry]> {
        self.parallel_geometry.as_deref()
    }

    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => {
                if let Some(module) = &mut self.parallel_embedding {
                    Some(module.inner_mut())
                } else {
                    Some(&mut self.embeddings)
                }
            }
            "output" => self
                .parallel_lm_head
                .as_mut()
                .map(|module| module.inner_mut() as &mut dyn ModuleParameters)
                .or_else(|| {
                    self.lm_head
                        .as_mut()
                        .map(|module| module as &mut dyn ModuleParameters)
                }),
            "mtp" => self
                .mtp
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            _ => None,
        }
    }

    pub(crate) fn embedded_mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, NemotronMtpModule::len)
    }

    pub(crate) fn embedded_mtp_cache(&self) -> Vec<LayerCache> {
        self.new_cache().mtp_layers
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_pipeline_mtp<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut [LayerCache],
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
                    Exception::custom("Nemotron-H pipeline MTP has no TP embedding shard")
                })?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => self.embeddings.forward(tokens, stream)?,
        };
        let expert_cache = self.expert_cache.as_ref();
        let hidden = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Nemotron-H checkpoint does not contain MTP layers"))?
            .forward_step(
                hidden,
                &embeddings,
                depth,
                cache,
                expert_cache,
                external_expert.map(|execute| execute as &mut NemotronMtpExpertExecutor<'_>),
                &self.args,
                execution,
                stream,
            )?;
        let logits = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => match (
                self.parallel_lm_head.as_mut(),
                self.parallel_embedding.as_mut(),
            ) {
                (Some(head), _) => head
                    .forward(&hidden, execution)
                    .and_then(|output| output.all_gather(execution)),
                (None, Some(embedding)) => embedding
                    .project_logits(&hidden, execution)
                    .and_then(|output| output.all_gather(execution)),
                (None, None) => Err(Error::Parallel(
                    "Nemotron-H pipeline MTP has no TP output projection".into(),
                )),
            }
            .map_err(|error| Exception::custom(error.to_string()))?,
            None => project_logits_maybe_quantized(
                &mut self.lm_head,
                &mut self.embeddings,
                &hidden,
                stream,
            )?,
        };
        Ok(
            crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput {
                logits,
                hidden,
                tokens: tokens.clone(),
            },
        )
    }

    fn new_cache(&self) -> Cache {
        let cache = Cache::new(&self.args);
        match &self.mtp {
            Some(mtp) => cache.with_mtp_policies(&mtp.policies),
            None => cache,
        }
    }

    fn recipes_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layer_index: Option<usize>,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let normalized = normalized_checkpoint_keys(store, &self.args)?;
        let keys = store.source_keys();
        let mut recipes = BTreeMap::new();

        let mtp_index = prefix
            .strip_prefix("mtp.layers.")
            .and_then(|tail| tail.split('.').next())
            .and_then(|index| index.parse::<usize>().ok());
        let sparse_prefix = if let Some(index) = layer_index
            .filter(|index| self.args.layer_schedule.get(*index) == Some(&LayerPolicy::SparseMoe))
        {
            Some(format!("model.layers.{index}.moe.experts"))
        } else {
            mtp_index
                .filter(|index| {
                    self.args
                        .mtp_policies()
                        .ok()
                        .and_then(|p| p.get(*index).copied())
                        == Some(LayerPolicy::SparseMoe)
                })
                .map(|index| format!("mtp.layers.{index}.moe.experts"))
        };
        if let Some(packed_prefix) = sparse_prefix {
            let complete_up = format!("{packed_prefix}.up_proj");
            let packed_up = format!("{complete_up}.weight");
            if !keys.contains(&complete_up)
                && !normalized.contains_key(&complete_up)
                && !keys.contains(&packed_up)
                && !normalized.contains_key(&packed_up)
            {
                let mut up = Vec::with_capacity(self.args.n_routed_experts as usize);
                let mut down = Vec::with_capacity(self.args.n_routed_experts as usize);
                for expert in 0..self.args.n_routed_experts {
                    up.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.up_proj.weight"),
                    )?);
                    down.push(source_for_normalized(
                        &normalized,
                        &format!("{packed_prefix}.{expert}.down_proj.weight"),
                    )?);
                }
                recipes.insert(
                    "moe.experts.up_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: up,
                    },
                );
                recipes.insert(
                    "moe.experts.down_proj".into(),
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: down,
                    },
                );
            }
        }

        if layer_index
            .is_some_and(|index| self.args.layer_schedule.get(index) == Some(&LayerPolicy::Mamba))
        {
            let parameters = module.parameters().flatten();
            for local_name in [
                "mamba.conv1d.weight",
                "mamba.A_log",
                "mamba.D",
                "mamba.norm.weight",
            ] {
                let source = format!("{prefix}.{local_name}");
                let Some(parameter) = parameters.get(local_name) else {
                    continue;
                };
                if !keys.contains(&source) {
                    continue;
                }
                let mut recipe = DerivedWeightRecipe::source(source, TensorSelection::Full);
                if local_name == "mamba.A_log" {
                    recipe = DerivedWeightRecipe::NegLog {
                        input: Box::new(recipe),
                    };
                }
                let expected = parameter
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as usize)
                    .collect::<Vec<_>>();
                if recipe.infer(store)?.shape() != expected {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: expected,
                    };
                }
                recipes.insert(local_name.to_string(), recipe);
            }
        }

        let parameters = module.parameters().flatten();
        for (local_name, parameter) in &parameters {
            if self.sparse_expert_cache && local_name.starts_with("moe.experts.") {
                continue;
            }
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = if prefix.is_empty() {
                local_name.to_string()
            } else {
                format!("{prefix}.{local_name}")
            };
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            let packed_weight = format!("{canonical}.weight");
            if parameter.dtype() == Dtype::Uint32 && keys.contains(&packed_weight) {
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::source(packed_weight, TensorSelection::Full),
                );
                continue;
            }
            if let Some(companion) = packed_companion_checkpoint_name(&canonical)
                .filter(|companion| keys.contains(companion))
            {
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::source(companion, TensorSelection::Full),
                );
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Nemotron-H checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(recipes)
    }

    fn mtp_recipes(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture("Nemotron-H model has no MTP module".into())
        })?;
        let normalized = normalized_checkpoint_keys(store, &self.args)?;
        let mut recipes = BTreeMap::new();
        for (index, block) in mtp.blocks.iter().enumerate() {
            for (name, recipe) in
                self.recipes_for_module(block, &format!("mtp.layers.{index}"), store, None)?
            {
                recipes.insert(format!("blocks.{index}.{name}"), recipe);
            }
        }
        for step in 0..mtp.steps {
            let start = step * mtp.pattern_len;
            let end = start + mtp.pattern_len - 1;
            for (local, remote) in [
                (
                    format!("enorm.{step}.weight"),
                    format!("mtp.layers.{start}.enorm.weight"),
                ),
                (
                    format!("hnorm.{step}.weight"),
                    format!("mtp.layers.{start}.hnorm.weight"),
                ),
                (
                    format!("eh_proj.{step}.weight"),
                    format!("mtp.layers.{start}.eh_proj.weight"),
                ),
                (
                    format!("final_norms.{step}.weight"),
                    format!("mtp.layers.{end}.final_layernorm.weight"),
                ),
            ] {
                let raw = normalized.get(&remote).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Nemotron-H MTP checkpoint is missing {remote}"
                    ))
                })?;
                recipes.insert(
                    local,
                    DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
                );
            }
        }
        Ok(recipes)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_nemotron_experts(
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
                let started = Instant::now();
                let prefix = if layer < args.num_hidden_layers as usize {
                    format!("model.layers.{layer}.moe.experts")
                } else {
                    format!(
                        "mtp.layers.{}.moe.experts",
                        layer - args.num_hidden_layers as usize
                    )
                };
                let mut bank = Experts::new_with_dtype(
                    acquired.identities().len() as i32,
                    args.hidden_size,
                    args.moe_intermediate_size,
                    [
                        args.weight_quantization_for(&format!("{prefix}.up_proj")),
                        args.weight_quantization_for(&format!("{prefix}.down_proj")),
                    ],
                    args.weight_dtype(),
                    stream,
                )?;
                bank.up_proj = Param::new(
                    acquired
                        .compact_binding("up_proj", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.up_proj_scales = Param::new(
                    acquired
                        .optional_compact_binding("up_proj_scales", stream)
                        .map_err(|error| Exception::custom(error.to_string()))?,
                );
                bank.up_proj_biases = Param::new(
                    acquired
                        .optional_compact_binding("up_proj_biases", stream)
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

fn normalized_checkpoint_keys(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &ModelArgs,
) -> Result<BTreeMap<String, String>, Error> {
    let mut normalized = BTreeMap::new();
    let mtp_policies = args.mtp_policies()?;
    for raw in store.source_keys() {
        if let Some(rest) = raw
            .strip_prefix("mtp.layers.")
            .or_else(|| raw.strip_prefix("model.mtp.layers."))
        {
            if let Some((index, suffix)) = rest.split_once('.') {
                let index = index.parse::<usize>().map_err(|error| {
                    Error::UnsupportedArchitecture(format!(
                        "invalid Nemotron-H MTP layer index in {raw:?}: {error}"
                    ))
                })?;
                let policy = mtp_policies.get(index).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Nemotron-H checkpoint MTP layer {index} is outside configured geometry"
                    ))
                })?;
                let field = match policy {
                    LayerPolicy::SelfAttention(_) => "attention",
                    LayerPolicy::SparseMoe => "moe",
                    _ => unreachable!("validated Nemotron-H MTP policies"),
                };
                let suffix = suffix
                    .strip_prefix("mixer.")
                    .map_or_else(|| suffix.to_string(), |suffix| format!("{field}.{suffix}"));
                normalized.insert(format!("mtp.layers.{index}.{suffix}"), raw);
                continue;
            }
        }
        let rewritten = resident::rewrite_nemotron_h_weight_key(&raw, args)?;
        let runtime = if let Some(rest) = rewritten.strip_prefix("model.backbone.") {
            format!("model.{rest}")
        } else if let Some(rest) = rewritten.strip_prefix("backbone.") {
            format!("model.{rest}")
        } else {
            rewritten
        };
        normalized.insert(runtime, raw);
    }
    Ok(normalized)
}

fn source_for_normalized(
    normalized: &BTreeMap<String, String>,
    runtime: &str,
) -> Result<DerivedWeightRecipe, Error> {
    normalized
        .get(runtime)
        .cloned()
        .map(|key| DerivedWeightRecipe::source(key, TensorSelection::Full))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Nemotron-H checkpoint is missing split expert tensor {runtime}"
            ))
        })
}

/// Per-forward causal mask shared by attention blocks.
pub struct NemotronHForwardContext {
    mask: Option<Array>,
    draft_hidden: Option<Array>,
}

struct OffsetOnlyCache(i32);

impl KeyValueCache for OffsetOnlyCache {
    fn offset(&self) -> i32 {
        self.0
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        _stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        Ok((keys, values))
    }
}

impl LoadTimeQuantizableAdapter for NemotronHLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for NemotronHLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = TransformerBlock;
    type ForwardContext = NemotronHForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error>
    {
        super::checkpoint::safetensors_plan(&self.args)
            .map_err(Error::UnsupportedArchitecture)
            .map(Into::into)
    }

    fn quantization(&self) -> Option<WeightQuantization> {
        self.args.quantization
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid Nemotron-H cache layer count"))?;
        let layer_layout = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                let geometry = self.parallel_geometry.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "Nemotron-H parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?;
                resident::prompt_cache_layer_layout_with_geometry(&self.args, geometry)
            }
            _ => resident::prompt_cache_layer_layout(&self.args),
        }?;
        Ok(PromptCacheModelIdentity {
            model_family: "nemotron_h".into(),
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

    fn save_prompt_cache(
        &self,
        cache: &mut Self::Cache,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        let rank = descriptor.topology.cache_rank_identity();
        resident::Model::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            rank,
            stream,
        )
        .map_err(Into::into)
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
        resident::Model::load_paged_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            options,
            stream,
        )
        .map_err(Into::into)
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
                build_module_binding_plan_with_recipes(
                    &self.embeddings,
                    "model.embeddings",
                    store,
                    self.recipes_for_module(&self.embeddings, "model.embeddings", store, None)?,
                )?
                .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    "model.norm_f",
                    store,
                    self.recipes_for_module(&self.norm, "model.norm_f", store, None)?,
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
                        self.recipes_for_module(head, "lm_head", store, None)?,
                    )?
                    .build_bindings(store)?,
                )?);
            }
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
                        |name| self.sparse_expert_cache && name.contains(".moe.experts."),
                    )?
                    .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = 2 + usize::from(self.lm_head.is_some()) + usize::from(self.mtp.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        if let Some(v) = &mut self.parallel_embedding {
            populate_module_from_lease(v.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embeddings, &leases[0])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[1])?;
        let mut index = 2;
        if let Some(v) = &mut self.parallel_lm_head {
            populate_module_from_lease(v.inner_mut(), &leases[index])?;
            index += 1;
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[index])?;
            index += 1;
        }
        if let Some(mtp) = &mut self.mtp {
            if self.sparse_expert_cache {
                populate_module_from_lease_excluding(mtp, &leases[index], |name| {
                    name.contains(".moe.experts.")
                })?;
            } else {
                populate_module_from_lease(mtp, &leases[index])?;
            }
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.args.layer_schedule.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Nemotron-H cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.layer_schedule.len()
            )));
        }
        for (index, (policy, cache)) in self
            .args
            .layer_schedule
            .iter()
            .zip(&cache.layers)
            .enumerate()
        {
            let matches = match (policy, cache) {
                (LayerPolicy::Mamba, LayerCache::Mamba(_))
                | (LayerPolicy::DenseMlp, LayerCache::Mlp)
                | (LayerPolicy::SparseMoe, LayerCache::Moe) => true,
                (LayerPolicy::SelfAttention(expected), LayerCache::Attention(cache)) => {
                    *expected == cache.policy()
                }
                _ => false,
            };
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Nemotron-H cache kind does not match layer schedule at layer {index}"
                )));
            }
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embeddings.forward(input, stream)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Nemotron-H requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: NemotronHForwardContext {
                mask,
                draft_hidden: None,
            },
        })
    }
    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(v) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let hidden = v.forward(input, execution)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), execution.stream())? {
                Some(AttentionMask::Array(v)) => Some(v),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Nemotron-H requires an array causal mask".into(),
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: NemotronHForwardContext {
                mask,
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
                "Nemotron-H has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        TransformerBlock::new(&self.args, index, stream)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if let Some(moe) = &mut layer.moe {
            let prefix = format!("model.layers.{index}.moe.experts");
            moe.experts = Experts::new_with_dtype(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local Nemotron-H expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                self.args.moe_intermediate_size,
                [
                    self.args
                        .weight_quantization_for(&format!("{prefix}.up_proj")),
                    self.args
                        .weight_quantization_for(&format!("{prefix}.down_proj")),
                ],
                self.args.weight_dtype(),
                stream,
            )?;
        }
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
        if let Some(moe) = &mut layer.moe {
            let prefix = format!("model.layers.{index}.moe.experts");
            let intermediate = moe.experts.intermediate_size;
            moe.experts = Experts::new_with_dtype(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local Nemotron-H expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                intermediate,
                [
                    self.args
                        .weight_quantization_for(&format!("{prefix}.up_proj")),
                    self.args
                        .weight_quantization_for(&format!("{prefix}.down_proj")),
                ],
                self.args.weight_dtype(),
                stream,
            )?;
        }
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
        if !self
            .args
            .layer_schedule
            .iter()
            .any(|policy| *policy == LayerPolicy::SparseMoe)
        {
            return Err(Error::Parallel(
                "Nemotron-H PP+EP requires a checkpoint with sparse MoE layers".into(),
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
    fn register_parallel_parameters(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        planner.register(
            crate::backend::mlx::nn::parallel::vocab_embedding_parameter_group(
                &self.embeddings,
                "model.embeddings",
                self.args.vocab_size as usize,
                self.args.hidden_size,
                false,
            )?,
        )?;
        crate::backend::mlx::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.norm_f",
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
            let layer = TransformerBlock::new(&self.args, index, stream)?;
            register_nemotron_layer_parallel_plan(planner, &layer, &self.args, index)?;
        }
        if let Some(mtp) = &self.mtp {
            for step in 0..mtp.steps {
                for (name, module) in [
                    (format!("enorm.{step}"), &mtp.enorm[step]),
                    (format!("hnorm.{step}"), &mtp.hnorm[step]),
                    (format!("final_norms.{step}"), &mtp.final_norms[step]),
                ] {
                    register_replicated_module(planner, module, &name)?;
                }
                register_projection_module(
                    planner,
                    &mtp.eh_proj[step],
                    &format!("eh_proj.{step}"),
                    ProjectionSharding::Replicated,
                )?;
            }
            let mut extended = self.args.clone();
            let mut policies = self.args.layer_schedule.iter().copied().collect::<Vec<_>>();
            policies.extend(mtp.policies.iter().copied());
            extended.layer_schedule = crate::LayerSchedule::new(policies.len(), policies)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            for (index, block) in mtp.blocks.iter().enumerate() {
                register_nemotron_layer_parallel_plan_at(
                    planner,
                    block,
                    &extended,
                    self.args.num_hidden_layers as usize + index,
                    &format!("blocks.{index}"),
                )?;
            }
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let local_dimension = |target: &str, axis: usize| -> Result<i32, Error> {
            let tensor = layout.tensor(target).ok_or_else(|| {
                Error::Parallel(format!("missing Nemotron-H TP layout for {target}"))
            })?;
            let dimension = tensor.local_shape().get(axis).copied().ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H TP layout for {target} has no axis {axis}"
                ))
            })?;
            i32::try_from(dimension)
                .map_err(|_| Error::Parallel(format!("Nemotron-H local {target} exceeds i32")))
        };
        let projection_dimension = |prefix: &str, axis: usize| -> Result<i32, Error> {
            for target in [format!("{prefix}.weight"), format!("{prefix}.inner.weight")] {
                if layout.tensor(&target).is_some() {
                    return local_dimension(&target, axis);
                }
            }
            Err(Error::Parallel(format!(
                "missing Nemotron-H TP projection layout for {prefix}"
            )))
        };
        let geometry_for = |policy: LayerPolicy,
                            prefix: &str|
         -> Result<resident::ParallelLayerGeometry, Error> {
            Ok(match policy {
                LayerPolicy::Mamba => {
                    let heads = local_dimension(&format!("{prefix}.mamba.dt_bias"), 0)?;
                    let conv = local_dimension(&format!("{prefix}.mamba.conv1d.weight"), 0)?;
                    let intermediate =
                        heads.checked_mul(self.args.mamba_head_dim).ok_or_else(|| {
                            Error::Parallel("Nemotron-H local Mamba width overflowed".into())
                        })?;
                    let grouped = conv.checked_sub(intermediate).ok_or_else(|| {
                        Error::Parallel("Nemotron-H local convolution width is inconsistent".into())
                    })?;
                    let divisor = self.args.ssm_state_size.checked_mul(2).ok_or_else(|| {
                        Error::Parallel("Nemotron-H Mamba state divisor overflowed".into())
                    })?;
                    if divisor <= 0 || grouped % divisor != 0 {
                        return Err(Error::Parallel(format!(
                            "Nemotron-H local convolution width {conv} does not encode integral Mamba groups"
                        )));
                    }
                    resident::ParallelLayerGeometry::Mamba {
                        heads,
                        groups: grouped / divisor,
                    }
                }
                LayerPolicy::SelfAttention(_) => resident::ParallelLayerGeometry::Attention {
                    query_heads: projection_dimension(&format!("{prefix}.attention.q_proj"), 0)?
                        / self.args.head_dim,
                    kv_heads: projection_dimension(&format!("{prefix}.attention.k_proj"), 0)?
                        / self.args.head_dim,
                },
                LayerPolicy::DenseMlp => resident::ParallelLayerGeometry::DenseMlp {
                    intermediate: projection_dimension(&format!("{prefix}.mlp.up_proj"), 0)?,
                },
                LayerPolicy::SparseMoe => resident::ParallelLayerGeometry::SparseMoe {
                    routed_intermediate: local_dimension(
                        &format!("{prefix}.moe.experts.up_proj"),
                        1,
                    )?,
                    shared_intermediate: projection_dimension(
                        &format!("{prefix}.moe.shared_experts.up_proj"),
                        0,
                    )?,
                },
            })
        };
        let mut geometry = Vec::with_capacity(self.args.layer_schedule.len());
        for (index, policy) in self.args.layer_schedule.iter().enumerate() {
            let prefix = format!("model.layers.{index}");
            geometry.push(geometry_for(*policy, &prefix)?);
        }
        self.parallel_geometry = Some(geometry);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args.weight_quantization_for("model.embeddings.weight"),
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
        if let Some(mtp) = &mut self.mtp {
            let mut extended = self.args.clone();
            let mut policies = self.args.layer_schedule.iter().copied().collect::<Vec<_>>();
            policies.extend(mtp.policies.iter().copied());
            extended.layer_schedule = crate::LayerSchedule::new(policies.len(), policies)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            for (index, block) in mtp.blocks.iter_mut().enumerate() {
                let prefix = format!("blocks.{index}");
                let geometry = geometry_for(mtp.policies[index], &prefix)?;
                *block = TransformerBlock::new_parallel_layerwise(
                    &extended,
                    self.args.num_hidden_layers as usize + index,
                    geometry,
                    stream,
                )?;
            }
        }
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
        let _ = layout;
        let geometry = self
            .parallel_geometry
            .as_ref()
            .and_then(|geometry| geometry.get(index))
            .copied()
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "Nemotron-H local geometry is unavailable for layer {index}"
                ))
            })?;
        TransformerBlock::new_parallel_layerwise(&self.args, index, geometry, stream)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("nemotron_h.layer.{index:05}")
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
                |name| name.starts_with("moe.experts."),
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
        let recipes = self.recipes_for_module(layer, &prefix, store, Some(index))?;
        let bindings = if self.sparse_expert_cache {
            build_module_binding_plan_with_recipes_excluding(
                layer,
                &prefix,
                store,
                recipes,
                |name| name.starts_with("moe.experts."),
            )?
            .build_bindings(store)?
        } else {
            build_module_binding_plan_with_recipes(layer, &prefix, store, recipes)?
                .build_bindings(store)?
        };
        bindings
            .into_iter()
            .map(|binding| {
                if matches!(
                    binding.name(),
                    "moe.experts.up_proj" | "moe.experts.down_proj"
                ) {
                    let target = format!("{prefix}.{}.weight", binding.name());
                    binding.with_logical_target(target).map_err(Error::from)
                } else {
                    Ok(binding)
                }
            })
            .collect()
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
                if target.contains(".moe.experts.") {
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
                .filter(|key| key.contains(".experts."))
                .collect()
        } else {
            Vec::new()
        }
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
        if self.sparse_expert_cache {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Nemotron-H sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                stream,
                |flat, indices, weights, stream| {
                    execute_cached_nemotron_experts(
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
        Ok(layer.forward(
            BlockInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(&mut cache.layers[index]),
            },
            stream,
        )?)
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
        Ok(layer.forward_tensor_parallel(
            BlockInput {
                x: hidden,
                mask: context.mask.as_ref(),
                cache: Some(&mut cache.layers[index]),
            },
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

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        self.layer_count(group)?;
        context.draft_hidden = Some(hidden.clone());
        Ok(hidden.clone())
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.embeddings,
            &hidden,
            stream,
        )?)
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
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let logits = match &mut self.parallel_lm_head {
            Some(head) => head.forward(&hidden, execution)?,
            None => embedding.project_logits(&hidden, execution)?,
        };
        logits.all_gather(execution)
    }
}

pub(crate) fn nemotron_h_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries =
        nemotron_h_expert_catalog_for_layers(args, store, 0..args.num_hidden_layers as usize)?;
    let normalized = normalized_checkpoint_keys(store, args)?;
    for (physical, policy) in args.mtp_policies()?.into_iter().enumerate() {
        if policy != LayerPolicy::SparseMoe {
            continue;
        }
        let global = args.num_hidden_layers as usize + physical;
        entries.extend(nemotron_expert_entries(
            args,
            &normalized,
            store,
            global,
            &format!("mtp.layers.{physical}.moe.experts"),
        )?);
    }
    Ok(entries)
}

pub(crate) fn nemotron_h_pipeline_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layers: impl IntoIterator<Item = usize>,
    include_mtp: bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = nemotron_h_expert_catalog_for_layers(args, store, layers)?;
    if include_mtp {
        let normalized = normalized_checkpoint_keys(store, args)?;
        for (physical, policy) in args.mtp_policies()?.into_iter().enumerate() {
            if policy != LayerPolicy::SparseMoe {
                continue;
            }
            entries.extend(nemotron_expert_entries(
                args,
                &normalized,
                store,
                args.num_hidden_layers as usize + physical,
                &format!("mtp.layers.{physical}.moe.experts"),
            )?);
        }
    }
    Ok(entries)
}

pub(crate) fn nemotron_h_expert_catalog_for_layers(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layers: impl IntoIterator<Item = usize>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store, args)?;
    let mut entries = Vec::new();
    for layer in layers {
        if args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        entries.extend(nemotron_expert_entries(
            args,
            &normalized,
            store,
            layer,
            &format!("model.layers.{layer}.moe.experts"),
        )?);
    }
    Ok(entries)
}

fn nemotron_expert_entries(
    args: &ModelArgs,
    normalized: &BTreeMap<String, String>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    identity_layer: usize,
    prefix: &str,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let packed = normalized.contains_key(&format!("{prefix}.up_proj"));
    (0..args.n_routed_experts as usize)
        .map(|expert| -> Result<ExpertCatalogEntry, Error> {
            let identity = ExpertIdentity::new(identity_layer, expert);
            let mut planned = Vec::new();
            for projection in ["up_proj", "down_proj"] {
                let weight_recipe = if packed {
                    DerivedWeightRecipe::source(
                        normalized[&format!("{prefix}.{projection}")].clone(),
                        TensorSelection::Range {
                            axis: 0,
                            start: expert,
                            end: expert + 1,
                        },
                    )
                } else {
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![source_for_normalized(
                            normalized,
                            &format!("{prefix}.{expert}.{projection}.weight"),
                        )?],
                    }
                };
                planned.push(nemotron_planned_binding(projection, weight_recipe, store)?);
                if packed {
                    for suffix in ["scales", "biases"] {
                        let runtime = format!("{prefix}.{projection}_{suffix}");
                        if let Some(raw) = normalized.get(&runtime) {
                            planned.push(nemotron_planned_binding(
                                &format!("{projection}_{suffix}"),
                                DerivedWeightRecipe::source(
                                    raw.clone(),
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
                }
            }
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Nemotron-H expert byte total overflowed".into())
                })
            })?;
            Ok(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?)
        })
        .collect()
}

fn nemotron_planned_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<PlannedBinding, Error> {
    let metadata = recipe.infer(store)?;
    Ok(PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    })
}

/// Nemotron-H token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, NemotronHLayerwiseModel, Cache, S>;
