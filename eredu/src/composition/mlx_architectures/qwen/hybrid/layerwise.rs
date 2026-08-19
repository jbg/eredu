//! Shared bounded layer execution for Qwen3-Next and Qwen3.5 text models.

use eredu_runtime::LayerWeightResidency;

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
    module::{Module, ModuleParameters, Param},
    ops::{concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Stream,
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
        parallel::{VocabParallelEmbedding, VocabParallelLmHead},
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::KeyValueCache,
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, build_module_binding_plan_with_recipes_excluding,
        canonical_checkpoint_name, is_materialized_module_parameter, populate_module_from_lease,
        populate_module_from_lease_excluding, ModuleBindingPlan,
    },
    backend::mlx::runtime::checkpoint::store::{TensorSelection, WeightStoreBackend},
    backend::mlx::runtime::checkpoint::{
        binding_plan::{BindingPlan, PlannedBinding},
        quantization::should_quantize_on_load,
        recipe::{DerivedWeightRecipe, RecipeDtype},
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, partitioned_projection_members,
        register_partitioned_projection_group, register_replicated_module, ParallelPlanBuilder,
        ProjectionSharding,
    },
    backend::mlx::runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter, StaticUnitBindings,
        WeightResidency,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry, ExpertIdentity,
        ExpertPass, ExpertRouteBatch,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    composition::mlx_architectures::qwen::{
        hybrid::{
            qwen3_5::{
                self as resident, BlockInput, Cache, Experts, LayerCache, LayerPolicy, ModelArgs,
                MtpModule, Qwen3NextRmsNorm, QwenMtpStepOutput, QwenWeightFormat, TransformerBlock,
            },
            qwen3_next,
        },
        vl::vision::{
            configure_vision_parallel_static, grid_thw_from_array, new_parallel_vision_block,
            vision_parallel_parameter_groups, QwenVisionBlock, QwenVisionLayerwiseState,
            QwenVisionLayerwiseStatic, QwenVisionTransformer, VisionConfig,
        },
    },
    core::attention::AttentionPolicy,
};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "qwen_hybrid.static.embedding";
const NORM_UNIT: &str = "qwen_hybrid.static.norm";
const HEAD_UNIT: &str = "qwen_hybrid.static.output";
const VISION_STATIC_UNIT: &str = "qwen_hybrid.static.vision";
const MTP_STATIC_UNIT: &str = "qwen_hybrid.static.mtp";

fn common_partition_units(preferred: usize, widths: impl IntoIterator<Item = usize>) -> usize {
    const fn gcd(mut left: usize, mut right: usize) -> usize {
        while right != 0 {
            let remainder = left % right;
            left = right;
            right = remainder;
        }
        left
    }
    widths.into_iter().fold(preferred, gcd)
}

fn parameter_shape(name: &str, parameter: &Array) -> Result<Vec<usize>, Error> {
    parameter
        .shape()
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Error::Parallel(format!(
                    "Qwen hybrid parameter {name} has negative dimension {dimension}"
                ))
            })
        })
        .collect()
}

fn register_qwen_mlp_partition(
    planner: &mut ParallelPlanBuilder,
    mlp: &resident::Mlp,
    prefix: &str,
    intermediate: i32,
    role: ParameterRole,
) -> Result<(), Error> {
    let intermediate = usize::try_from(intermediate)
        .map_err(|_| Error::Parallel("Qwen hybrid intermediate width exceeds usize".into()))?;
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let down = format!("{prefix}.down_proj");
    register_partitioned_projection_group(
        planner,
        &format!("{prefix}.intermediate"),
        role,
        &[
            (&mlp.gate_proj, gate.as_str(), ProjectionSharding::Column),
            (&mlp.up_proj, up.as_str(), ProjectionSharding::Column),
            (&mlp.down_proj, down.as_str(), ProjectionSharding::Row),
        ],
        intermediate,
    )
}

fn register_qwen_hybrid_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    index: usize,
    args: &ModelArgs,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    register_qwen_hybrid_layer_parallel_plan_at(planner, layer, &prefix, args)
}

fn register_qwen_hybrid_layer_parallel_plan_at(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    prefix: &str,
    args: &ModelArgs,
) -> Result<(), Error> {
    if let Some(attention) = &layer.self_attn {
        let query_width = usize::try_from(attention.n_heads * attention.head_dim)
            .map_err(|_| Error::Parallel("Qwen hybrid query width exceeds usize".into()))?;
        let kv_heads = usize::try_from(attention.n_kv_heads)
            .map_err(|_| Error::Parallel("Qwen hybrid KV heads exceed usize".into()))?;
        let k = format!("{prefix}.self_attn.k_proj");
        let v = format!("{prefix}.self_attn.v_proj");
        let o = format!("{prefix}.self_attn.o_proj");
        let (mut units, mut members) = partitioned_projection_members(
            &[
                (&attention.k_proj, k.as_str(), ProjectionSharding::Column),
                (&attention.v_proj, v.as_str(), ProjectionSharding::Column),
                (&attention.o_proj, o.as_str(), ProjectionSharding::Row),
            ],
            kv_heads,
        )?;
        for (name, parameter) in attention.q_proj.parameters().flatten() {
            let target = format!("{prefix}.self_attn.q_proj.{name}");
            let shape = parameter_shape(&target, parameter)?;
            let output = shape.first().copied().ok_or_else(|| {
                Error::Parallel(format!("Qwen hybrid query member {target} is scalar"))
            })?;
            let width = if name.as_ref() == "weight_scale_inv" {
                let blocks = query_width.div_ceil(128);
                if output != 2 * blocks {
                    return Err(Error::Parallel(format!(
                        "Qwen hybrid FP8 query/gate boundary at width {query_width} is not block-aligned"
                    )));
                }
                blocks
            } else {
                if output != 2 * query_width {
                    return Err(Error::Parallel(format!(
                        "Qwen hybrid query member {target} has output width {output}, expected {}",
                        2 * query_width
                    )));
                }
                query_width
            };
            units = common_partition_units(units, [width]);
            members.push(ParameterMemberSpec::new(
                target,
                shape,
                MemberSharding::PartitionedSegments {
                    axis: 0,
                    segments: vec![0..width, width..2 * width],
                },
            ));
        }
        planner.register(ParameterGroupSpec::partitioned(
            format!("{prefix}.self_attn.heads"),
            ParameterRole::AttentionHeads,
            units,
            members,
        )?)?;
        for (name, module) in [("q_norm", &attention.q_norm), ("k_norm", &attention.k_norm)] {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
        register_replicated_module(
            planner,
            &attention.rope,
            &format!("{prefix}.self_attn.rope"),
        )?;
    }
    if let Some(linear_attention) = &layer.linear_attn {
        let key_heads = usize::try_from(linear_attention.num_k_heads)
            .map_err(|_| Error::Parallel("Qwen recurrent key heads exceed usize".into()))?;
        let key_width = usize::try_from(linear_attention.key_dim)
            .map_err(|_| Error::Parallel("Qwen recurrent key width exceeds usize".into()))?;
        let value_width = usize::try_from(linear_attention.value_dim)
            .map_err(|_| Error::Parallel("Qwen recurrent value width exceeds usize".into()))?;
        let out = format!("{prefix}.linear_attn.out_proj");
        let (mut units, mut members) = partitioned_projection_members(
            &[(
                &linear_attention.out_proj,
                out.as_str(),
                ProjectionSharding::Row,
            )],
            key_heads,
        )?;
        for (name, projection) in [
            ("in_proj_z", &linear_attention.in_proj_z),
            ("in_proj_b", &linear_attention.in_proj_b),
            ("in_proj_a", &linear_attention.in_proj_a),
        ] {
            let projection_prefix = format!("{prefix}.linear_attn.{name}");
            let (projection_units, projection_members) = partitioned_projection_members(
                &[(
                    projection,
                    projection_prefix.as_str(),
                    ProjectionSharding::Column,
                )],
                units,
            )?;
            units = projection_units;
            members.extend(projection_members);
        }
        for (name, parameter) in linear_attention.in_proj_qkv.parameters().flatten() {
            let target = format!("{prefix}.linear_attn.in_proj_qkv.{name}");
            let shape = parameter_shape(&target, parameter)?;
            let (key_segment, value_segment) = if name.as_ref() == "weight_scale_inv" {
                (key_width.div_ceil(128), value_width.div_ceil(128))
            } else {
                (key_width, value_width)
            };
            let segments = vec![
                0..key_segment,
                key_segment..2 * key_segment,
                2 * key_segment..2 * key_segment + value_segment,
            ];
            if shape.first().copied() != segments.last().map(|segment| segment.end) {
                return Err(Error::Parallel(format!(
                    "Qwen recurrent projection {target} cannot preserve its Q/K/V boundaries"
                )));
            }
            units = common_partition_units(units, [key_segment, value_segment]);
            members.push(ParameterMemberSpec::new(
                target,
                shape,
                MemberSharding::PartitionedSegments { axis: 0, segments },
            ));
        }
        members.push(array_parameter_member(
            format!("{prefix}.linear_attn.conv1d.weight"),
            linear_attention.conv1d.weight.as_ref(),
            MemberSharding::PartitionedSegments {
                axis: 0,
                segments: vec![
                    0..key_width,
                    key_width..2 * key_width,
                    2 * key_width..2 * key_width + value_width,
                ],
            },
        )?);
        for (name, value) in [
            ("dt_bias", linear_attention.dt_bias.as_ref()),
            ("A_log", linear_attention.A_log.as_ref()),
        ] {
            members.push(array_parameter_member(
                format!("{prefix}.linear_attn.{name}"),
                value,
                MemberSharding::Partitioned { axis: 0 },
            )?);
        }
        planner.register(ParameterGroupSpec::partitioned(
            format!("{prefix}.linear_attn.groups"),
            ParameterRole::Channels,
            units,
            members,
        )?)?;
        register_replicated_module(
            planner,
            &linear_attention.norm,
            &format!("{prefix}.linear_attn.norm"),
        )?;
    }
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    match &layer.mlp {
        resident::FeedForward::Dense(mlp) => register_qwen_mlp_partition(
            planner,
            mlp,
            &format!("{prefix}.mlp"),
            args.intermediate_size,
            ParameterRole::FeedForwardIntermediate,
        )?,
        resident::FeedForward::Moe(moe) => {
            register_replicated_module(planner, &moe.gate, &format!("{prefix}.mlp.gate"))?;
            register_replicated_module(
                planner,
                &moe.shared_expert_gate,
                &format!("{prefix}.mlp.shared_expert_gate"),
            )?;
            register_qwen_mlp_partition(
                planner,
                &moe.shared_expert,
                &format!("{prefix}.mlp.shared_expert"),
                args.shared_expert_intermediate_size,
                ParameterRole::ExpertIntermediate,
            )?;
            let experts = &moe.experts;
            let expert_prefix = format!("{prefix}.mlp.experts");
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Qwen hybrid expert width exceeds usize".into()))?;
            let alignment = if experts.use_fp8 {
                128
            } else {
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel("Qwen expert quantization group exceeds usize".into())
                        })
                    })?
            };
            let mut units = aligned_partition_units(
                &format!("{expert_prefix}.intermediate"),
                intermediate,
                1,
                alignment,
            )?;
            let mut members = Vec::new();
            for (name, value) in [
                ("gate_up_proj", Some(experts.gate_up_proj.as_ref())),
                (
                    "gate_up_proj_scale_inv",
                    experts.gate_up_proj_scale_inv.as_ref().as_ref(),
                ),
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
                    let target = format!("{expert_prefix}.{name}");
                    let shape = parameter_shape(&target, value)?;
                    let fused = shape.get(1).copied().ok_or_else(|| {
                        Error::Parallel(format!("Qwen expert member {target} has no axis 1"))
                    })?;
                    if !fused.is_multiple_of(2) {
                        return Err(Error::Parallel(format!(
                            "Qwen expert member {target} cannot preserve gate/up boundaries"
                        )));
                    }
                    let width = fused / 2;
                    units = common_partition_units(units, [width]);
                    members.push(ParameterMemberSpec::new(
                        target,
                        shape,
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: vec![0..width, width..2 * width],
                        },
                    ));
                }
            }
            for (name, value) in [
                ("down_proj", Some(experts.down_proj.as_ref())),
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
                    let target = format!("{expert_prefix}.{name}");
                    let shape = parameter_shape(&target, value)?;
                    let width = shape.get(2).copied().ok_or_else(|| {
                        Error::Parallel(format!("Qwen expert member {target} has no axis 2"))
                    })?;
                    units = common_partition_units(units, [width]);
                    members.push(ParameterMemberSpec::new(
                        target,
                        shape,
                        MemberSharding::Partitioned { axis: 2 },
                    ));
                }
            }
            planner.register(ParameterGroupSpec::partitioned(
                format!("{expert_prefix}.intermediate"),
                ParameterRole::ExpertIntermediate,
                units,
                members,
            )?)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum QwenHybridFamily {
    Qwen3Next,
    Qwen35,
}

/// Qwen3-Next or Qwen3.5 text model using host-backed hybrid blocks.
pub struct QwenHybridLayerwiseModel {
    execution: LayerwiseModel<QwenHybridLayerwiseAdapter>,
}

pub(crate) struct QwenHybridTensorMtpTarget<'a> {
    model: &'a mut QwenHybridLayerwiseModel,
    group: &'a safemlx::distributed::Group,
}

impl<'a> QwenHybridTensorMtpTarget<'a> {
    pub(crate) fn new(
        model: &'a mut QwenHybridLayerwiseModel,
        group: &'a safemlx::distributed::Group,
    ) -> Self {
        Self { model, group }
    }
}

impl QwenHybridLayerwiseModel {
    /// Returns normalized text-model arguments.
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

    pub(crate) fn vision_spatial_merge_size(&self) -> Option<i32> {
        self.vision_config().map(|vision| vision.spatial_merge_size)
    }

    pub(crate) fn vision_config(
        &self,
    ) -> Option<&crate::composition::mlx_architectures::qwen::vl::vision::VisionConfig> {
        self.execution
            .adapter()
            .vision
            .as_ref()
            .map(|vision| &vision.config)
    }

    /// Creates heterogeneous recurrent/full-attention cache state.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Creates resident hybrid state or pages full-attention blocks while
    /// recurrent convolution and transition state remains rank-local on device.
    pub fn new_cache_with_options(&self, policy: CacheResidencyPolicy) -> Result<Cache, Error> {
        match policy {
            CacheResidencyPolicy::Device => Ok(self.new_cache()),
            CacheResidencyPolicy::Paged(options) => Cache::new_paged(
                self.args(),
                options,
                self.execution.prompt_cache_rank_identity(),
            )
            .map_err(Into::into),
        }
    }

    /// Returns aggregate live full-attention paging telemetry, if enabled.
    pub fn cache_residency_report(
        &self,
        cache: &Cache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        cache.residency_report().map_err(Into::into)
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns generalized parameter-residency and encoding metadata.
    pub fn residency_metadata(
        &self,
    ) -> &crate::backend::mlx::runtime::execution::layerwise::LayerwiseModelMetadata {
        self.execution.metadata()
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
    ) -> Result<
        Option<crate::backend::mlx::runtime::execution::layerwise::DenseDiskStreamReport>,
        Error,
    > {
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

    /// Runs the shared hybrid decoder while preserving recurrent and KV state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(QwenHybridInput::Decode(inputs), cache, stream)
    }

    /// Runs the canonical execution path with stable per-unit observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution.forward_with_observer(
            QwenHybridInput::Decode(inputs),
            cache,
            stream,
            observer,
        )
    }

    /// Runs typed text or multimodal prefill through the canonical observer contract.
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution.forward_with_observer(
            QwenHybridInput::Prefill(input),
            cache,
            stream,
            observer,
        )
    }
    /// Runs a rank-local tensor-parallel text forward pass.
    pub(crate) fn forward_tensor_parallel(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            QwenHybridInput::Decode(inputs),
            cache,
            group,
            stream,
        )
    }

    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward_tensor_parallel(
            QwenHybridInput::Prefill(input),
            cache,
            group,
            stream,
        )
    }

    /// Runs TP-sharded hybrid operators while routed experts execute in the
    /// matching EP subgroup.
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
            QwenHybridInput::Decode(inputs),
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, execution| match layer {
                QwenHybridLayer::Text(block) => {
                    let tp_group = execution.group().ok_or_else(|| {
                        Error::Parallel(
                            "Qwen hybrid TP+EP execution requires an active TP group".into(),
                        )
                    })?;
                    Ok(block.forward_tensor_with_expert_executor(
                        BlockInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: Some(&mut cache.layers[index]),
                        },
                        tp_group,
                        execution.stream(),
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?)
                }
                QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                    "token-only Qwen hybrid TP+EP received a vision unit".into(),
                )),
            },
        )
    }

    /// Runs streamed text layers while delegating routed experts to a caller.
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
            QwenHybridInput::Decode(inputs),
            cache,
            stream,
            |adapter, _group, index, layer, hidden, cache, context, stream| match layer {
                QwenHybridLayer::Vision(block) => {
                    let vision = adapter.vision.as_mut().expect("vision group");
                    for job in &mut context.vision_jobs {
                        job.hidden = vision.forward_block(
                            block,
                            index,
                            job.hidden.clone(),
                            &job.state,
                            stream,
                        )?;
                        vision.capture_deepstack(index, &job.hidden, &mut job.state, stream)?;
                    }
                    Ok(context.vision_jobs[0].hidden.clone())
                }
                QwenHybridLayer::Text(block) => Ok(block.forward_sparse_experts(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?),
            },
        )
    }

    pub(crate) fn forward_mtp_with_expert_executor<F>(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        tensor_group: Option<&safemlx::distributed::Group>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let (logits, context) = match tensor_group {
            Some(tensor_group) => self
                .execution
                .forward_tensor_parallel_with_layer_executor_and_context(
                    QwenHybridInput::Decode(inputs),
                    cache,
                    tensor_group,
                    stream,
                    |_adapter, _group, index, layer, hidden, cache, context, execution| match layer
                    {
                        QwenHybridLayer::Vision(_) => Err(Error::Parallel(
                            "Qwen TP+EP MTP text target received a vision unit".into(),
                        )),
                        QwenHybridLayer::Text(block) => Ok(block
                            .forward_tensor_with_expert_executor(
                                BlockInput {
                                    x: hidden,
                                    mask: context.mask.as_ref(),
                                    cache: Some(&mut cache.layers[index]),
                                },
                                execution.group().ok_or_else(|| {
                                    Error::Parallel(
                                        "Qwen TP+EP MTP target is missing its TP group".into(),
                                    )
                                })?,
                                execution.stream(),
                                |hidden, ids, weights, stream| {
                                    execute(index, hidden, ids, weights, stream)
                                },
                            )?),
                    },
                ),
            None => self.execution.forward_with_layer_executor_and_context(
                QwenHybridInput::Decode(inputs),
                cache,
                stream,
                |adapter, _group, index, layer, hidden, cache, context, stream| match layer {
                    QwenHybridLayer::Vision(block) => {
                        let vision = adapter.vision.as_mut().expect("vision group");
                        for job in &mut context.vision_jobs {
                            job.hidden = vision.forward_block(
                                block,
                                index,
                                job.hidden.clone(),
                                &job.state,
                                stream,
                            )?;
                            vision.capture_deepstack(index, &job.hidden, &mut job.state, stream)?;
                        }
                        Ok(context.vision_jobs[0].hidden.clone())
                    }
                    QwenHybridLayer::Text(block) => Ok(block.forward_sparse_experts(
                        BlockInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: Some(&mut cache.layers[index]),
                        },
                        stream,
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?),
                },
            ),
        }
        .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Qwen layerwise pass did not retain MTP hidden state")
        })?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        cache.reset()?;
        self.forward_mtp(QwenHybridInput::Prefill(input), cache, stream)
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.forward_mtp(QwenHybridInput::Decode(tokens), cache, stream)
    }

    fn forward_mtp(
        &mut self,
        input: QwenHybridInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Qwen layerwise pass did not retain MTP hidden state")
        })?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    pub(crate) fn forward_mtp_head(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .adapter_mut()
            .forward_mtp_head(hidden, tokens, cache, stream)
    }

    fn forward_mtp_tensor(
        &mut self,
        input: QwenHybridInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_tensor_parallel_with_context(input, cache, group, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context
            .draft_hidden
            .ok_or_else(|| Exception::custom("Qwen tensor pass did not retain MTP hidden state"))?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    pub(crate) fn forward_mtp_head_tensor(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let topology = self
            .parallel_info()
            .ok_or_else(|| Exception::custom("Qwen MTP target has no parallel topology"))?
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
            .ok_or_else(|| Exception::custom("Qwen MTP has no TP embedding shard"))?
            .forward(tokens, &execution)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = adapter
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?
            .forward_tensor_parallel(hidden, &embeddings, cache, group, stream)?;
        match (
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
                "Qwen MTP has no TP output projection".into(),
            )),
        }
        .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn forward_mtp_head_with_expert_executor<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        tensor_group: Option<&safemlx::distributed::Group>,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let topology = tensor_group
            .map(|_| {
                self.parallel_info()
                    .map(|info| info.topology())
                    .ok_or_else(|| Exception::custom("Qwen MTP target has no parallel topology"))
            })
            .transpose()?;
        let execution = match (topology, tensor_group) {
            (Some(topology), Some(group)) => Some(
                crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::tensor_parallel(
                    topology, group, stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))?,
            ),
            _ => None,
        };
        self.execution
            .adapter_mut()
            .forward_pipeline_mtp(
                hidden,
                tokens,
                cache,
                execution.as_ref(),
                Some(execute),
                stream,
            )
            .map(|output| output.logits)
    }

    pub(crate) fn mtp_len(&self) -> usize {
        self.execution
            .adapter()
            .mtp
            .as_ref()
            .map_or(0, MtpModule::len)
    }

    /// Clears temporary vision and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalModel<Cache> for QwenHybridLayerwiseModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(QwenHybridInput::Prefill(input), cache, stream)
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

impl crate::composition::mlx_architectures::qwen::hybrid::mtp::QwenMtpTarget
    for QwenHybridTensorMtpTarget<'_>
{
    fn prefill_mtp_target(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        cache.reset()?;
        self.model
            .forward_mtp_tensor(QwenHybridInput::Prefill(input), cache, self.group, stream)
    }

    fn verify_mtp_target(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception> {
        self.model
            .forward_mtp_tensor(QwenHybridInput::Decode(tokens), cache, self.group, stream)
    }

    fn forward_mtp_drafter(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.model
            .forward_mtp_head_tensor(hidden, tokens, cache, self.group, stream)
    }

    fn mtp_layer_count(&self) -> usize {
        self.model.mtp_len()
    }
}

/// Loads a text-only Qwen3-Next model through generalized parameter residency.
pub fn load_qwen3_next_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    load_qwen_hybrid_layerwise_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads text-only Qwen3-Next through the generalized tensor-parallel engine.
pub(crate) fn load_qwen3_next_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let (model, _, is_next) = load_qwen_hybrid_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            None,
            options,
            build,
            stream,
            weights_stream,
        )?;
        if !is_next {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-Next loader received a non-Qwen3-Next GGUF checkpoint".into(),
            ));
        }
        return Ok(model);
    }
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    load_qwen_hybrid_tensor_parallel_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        options,
        build,
        stream,
        weights_stream,
    )
}

/// Loads a text-only or multimodal dense/MoE Qwen3.5 model through generalized residency.
pub fn load_qwen35_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let (args, image_token_id, video_token_id, vision) =
        resident::get_qwen3_5_model_args(model_dir)?;
    load_qwen_hybrid_layerwise_model_with_vision(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        image_token_id,
        video_token_id,
        vision,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

/// Loads a Qwen3.5 dense or MoE checkpoint through the generalized tensor-parallel engine.
pub(crate) fn load_qwen35_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let mmproj = resident::open_sibling_mmproj(model_dir)?;
        let (model, _, is_next) = load_qwen_hybrid_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            mmproj.as_ref(),
            options,
            build,
            stream,
            weights_stream,
        )?;
        if is_next {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3.5 loader received a Qwen3-Next GGUF checkpoint".into(),
            ));
        }
        return Ok(model);
    }
    let (args, _, _, vision) = resident::get_qwen3_5_model_args(model_dir)?;
    if vision.is_some() {
        return Err(Error::Parallel("the token-only Qwen3.5 TP loader does not accept a vision tower; use the multimodal execution-group loader".into()));
    }
    load_qwen_hybrid_tensor_parallel_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        options,
        build,
        stream,
        weights_stream,
    )
}

fn load_qwen_hybrid_tensor_parallel_model(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let adapter = QwenHybridLayerwiseAdapter::new(args, family, None, None, None, stream)?;
    Ok(QwenHybridLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            adapter,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_qwen_hybrid_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::Qwen35MmprojGguf>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenHybridLayerwiseModel, Vec<u32>, bool), Error> {
    let prepared =
        resident::prepare_qwen35_gguf_checkpoint(checkpoint, metadata, mmproj, weights_stream)?;
    let architecture = crate::core::GgufArchitecture::resolve(&prepared.architecture)?;
    let residency = WeightResidency::with_layers(options);
    crate::composition::mlx::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let is_next = prepared.architecture == "qwen3next";
    let family = if is_next {
        QwenHybridFamily::Qwen3Next
    } else {
        QwenHybridFamily::Qwen35
    };
    let gguf_variant = qwen_hybrid_gguf_variant(&prepared.args, is_next);
    let store = qwen_hybrid_gguf_store(
        checkpoint,
        mmproj,
        &prepared.args,
        gguf_variant,
        prepared.modalities.vision_config.as_ref(),
        options.max_mapped_shards(),
    )?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        QwenHybridLayerwiseAdapter::new(
            prepared.args,
            family,
            prepared.modalities.image_token_id,
            prepared.modalities.video_token_id,
            prepared.modalities.vision_config,
            stream,
        )?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        QwenHybridLayerwiseModel { execution },
        prepared.eos_token_ids,
        is_next,
    ))
}

pub(crate) fn load_qwen_hybrid_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::Qwen35MmprojGguf>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(QwenHybridLayerwiseModel, Vec<u32>, bool), Error> {
    let prepared =
        resident::prepare_qwen35_gguf_checkpoint(checkpoint, metadata, mmproj, weights_stream)?;
    let architecture = crate::core::GgufArchitecture::resolve(&prepared.architecture)?;
    crate::composition::mlx::structural::validate_gguf(
        architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let args = prepared.args;
    let modalities = prepared.modalities;
    let is_next = prepared.architecture == "qwen3next";
    let family = if is_next {
        QwenHybridFamily::Qwen3Next
    } else {
        QwenHybridFamily::Qwen35
    };
    let gguf_variant = qwen_hybrid_gguf_variant(&args, is_next);
    let store = qwen_hybrid_gguf_store(
        checkpoint,
        mmproj,
        &args,
        gguf_variant,
        modalities.vision_config.as_ref(),
        residency.max_mapped_shards(),
    )?;
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_qwen_hybrid_gguf_sparse_with_store(
                store,
                args,
                family,
                modalities,
                expert_options,
                residency.layers(),
                quantization,
                stream,
                weights_stream,
            )?,
            prepared.eos_token_ids,
            is_next,
        ));
    }
    let execution = load_layerwise_model_with_quantization(
        store,
        QwenHybridLayerwiseAdapter::new(
            args,
            family,
            modalities.image_token_id,
            modalities.video_token_id,
            modalities.vision_config,
            stream,
        )?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((
        QwenHybridLayerwiseModel { execution },
        prepared.eos_token_ids,
        is_next,
    ))
}

pub(crate) fn qwen_hybrid_gguf_store(
    checkpoint: &GgufCheckpoint,
    mmproj: Option<&resident::Qwen35MmprojGguf>,
    args: &resident::ModelArgs,
    variant: super::checkpoint::GgufVariant,
    vision_config: Option<&VisionConfig>,
    max_mapped_shards: usize,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    let text_plan = super::checkpoint::gguf_plan(args, checkpoint, variant)
        .map_err(Error::UnsupportedArchitecture)?;
    let mut builder = eredu_checkpoint::gguf_store::GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(
            checkpoint.catalog().clone(),
            &text_plan,
            resident::qwen35_translate_gguf_weight_name,
        )?;
    if let Some(mmproj) = mmproj {
        let deepstack = vision_config
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Qwen3.5 GGUF projector is present without validated vision geometry".into(),
                )
            })?
            .deepstack_layers();
        let vision = vision_config.expect("vision geometry checked above");
        let projector_plan = super::checkpoint::projector_gguf_plan(vision, args.hidden_size)
            .map_err(Error::UnsupportedArchitecture)?;
        builder =
            builder.add_checkpoint(mmproj.checkpoint.catalog().clone(), &projector_plan, move |name| {
                let translated =
                    crate::composition::mlx_architectures::qwen::vl::model::translate_qwen3_vl_mmproj_name(
                        name, &deepstack,
                    );
                translated
                    .strip_prefix("model.")
                    .unwrap_or(&translated)
                    .to_string()
            })?;
    }
    Ok(Arc::new(builder.build()?))
}

fn qwen_hybrid_gguf_variant(
    args: &resident::ModelArgs,
    is_next: bool,
) -> super::checkpoint::GgufVariant {
    if is_next {
        super::checkpoint::GgufVariant::Qwen3Next
    } else if args.num_experts > 0 {
        super::checkpoint::GgufVariant::Qwen35Moe
    } else {
        super::checkpoint::GgufVariant::Qwen35
    }
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    family: QwenHybridFamily,
    modalities: resident::Qwen35Modalities,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen hybrid MoE GGUF checkpoint".into(),
        ));
    }
    let mut adapter = QwenHybridLayerwiseAdapter::new(
        args.clone(),
        family,
        modalities.image_token_id,
        modalities.video_token_id,
        modalities.vision_config,
        stream,
    )?;
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
    let entries = qwen_hybrid_expert_catalog(&args, checkpoint_store.as_ref())?;
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
    Ok(QwenHybridLayerwiseModel { execution })
}

/// Loads Qwen3-Next with independently cached experts and bounded non-expert units.
pub fn load_qwen3_next_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = qwen3_next::get_qwen3_next_model_args(model_dir)?;
    if let Some(config) = &args.quantization_config {
        config.validate_supported()?;
    }
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-Next MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen3Next,
        None,
        None,
        None,
        options,
        quantization,
        non_expert,
        stream,
        weights_stream,
    )
}

/// Loads Qwen3.5 MoE with independently cached experts and bounded non-expert units.
pub fn load_qwen35_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let (args, image_token_id, video_token_id, vision) =
        resident::get_qwen3_5_model_args(model_dir)?;
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3.5 MoE checkpoint".into(),
        ));
    }
    load_qwen_hybrid_sparse_model(
        model_dir,
        args,
        QwenHybridFamily::Qwen35,
        image_token_id,
        video_token_id,
        vision,
        options,
        quantization,
        non_expert,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_sparse_model(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    vision_config: Option<VisionConfig>,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let non_expert = non_expert.into();
    let mut source_adapter = QwenHybridLayerwiseAdapter::new(
        args.clone(),
        family,
        image_token_id,
        video_token_id,
        vision_config.clone(),
        stream,
    )?;
    source_adapter.sparse_expert_cache = true;
    let quantize_on_load =
        resolve_on_load_quantization(&args, quantization, "Qwen hybrid independent expert cache")?;
    let store = open_safetensors_weight_store(model_dir, non_expert.max_mapped_shards())?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        source_adapter,
        non_expert,
        quantize_on_load,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = qwen_hybrid_expert_catalog(&args, store.as_ref())?;
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
    Ok(QwenHybridLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Qwen hybrid execution base used by distributed EP.
pub(crate) fn load_qwen_hybrid_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    is_qwen3_next: bool,
    modalities: resident::Qwen35Modalities,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let family = if is_qwen3_next {
        QwenHybridFamily::Qwen3Next
    } else {
        QwenHybridFamily::Qwen35
    };
    let mut adapter = QwenHybridLayerwiseAdapter::new(
        args,
        family,
        modalities.image_token_id,
        modalities.video_token_id,
        modalities.vision_config,
        stream,
    )?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(QwenHybridLayerwiseModel { execution })
}

/// Builds the TP-sharded nonexpert base used by combined Qwen hybrid TP+EP.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_qwen_hybrid_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    is_qwen3_next: bool,
    modalities: resident::Qwen35Modalities,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let family = if is_qwen3_next {
        QwenHybridFamily::Qwen3Next
    } else {
        QwenHybridFamily::Qwen35
    };
    let mut adapter = QwenHybridLayerwiseAdapter::new(
        args,
        family,
        modalities.image_token_id,
        modalities.video_token_id,
        modalities.vision_config,
        stream,
    )?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(QwenHybridLayerwiseModel { execution })
}

fn load_qwen_hybrid_layerwise_model(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    load_qwen_hybrid_layerwise_model_with_vision(
        model_dir,
        args,
        family,
        None,
        None,
        None,
        options,
        quantization,
        stream,
        weights_stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_qwen_hybrid_layerwise_model_with_vision(
    model_dir: &Path,
    args: ModelArgs,
    family: QwenHybridFamily,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    vision_config: Option<VisionConfig>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<QwenHybridLayerwiseModel, Error> {
    let options = options.into();
    let quantize_on_load = resolve_on_load_quantization(&args, quantization, "Qwen hybrid")?;
    let adapter = QwenHybridLayerwiseAdapter::new(
        args,
        family,
        image_token_id,
        video_token_id,
        vision_config,
        stream,
    )?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(QwenHybridLayerwiseModel {
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

fn resolve_on_load_quantization(
    args: &ModelArgs,
    requested: Option<WeightQuantization>,
    architecture: &str,
) -> Result<Option<WeightQuantization>, Error> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    if args.uses_fp8() {
        return Err(Error::Quantization(format!(
            "{architecture} on-load quantization requires floating-point weights; the checkpoint uses native FP8 weights"
        )));
    }
    should_quantize_on_load(architecture, args.quantization, requested)
        .map(|required| required.then_some(requested))
}

/// Shared adapter for recurrent linear-attention and full-attention Qwen blocks.
pub struct QwenHybridLayerwiseAdapter {
    args: ModelArgs,
    family: QwenHybridFamily,
    embedding: MaybeQuantized<safemlx::nn::Embedding>,
    norm: Qwen3NextRmsNorm,
    lm_head: Option<MaybeQuantized<safemlx::nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    mtp: Option<MtpModule>,
    vision: Option<QwenVisionLayerwiseStatic>,
    vision_config: Option<VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
    parallel_geometry: Option<Vec<resident::ParallelLayerGeometry>>,
}

impl QwenHybridLayerwiseAdapter {
    /// Exports the complete variable-size state needed by a later PP owner.
    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &QwenHybridPipelineIngressState,
    ) -> Vec<Array> {
        state
            .vision_jobs
            .iter()
            .flat_map(|job| {
                std::iter::once(job.hidden.clone())
                    .chain(job.state.retained_arrays().into_iter().cloned())
            })
            .collect()
    }

    /// Imports state produced by an earlier PP owner.
    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut QwenHybridPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        if state.vision_jobs.is_empty() {
            return if arrays.is_empty() {
                Ok(())
            } else {
                Err(Error::Parallel(
                    "Qwen3.5 text-only ingress received visual state".into(),
                ))
            };
        }
        if !arrays.len().is_multiple_of(state.vision_jobs.len()) {
            return Err(Error::Parallel(
                "Qwen3.5 distributed vision payload has ambiguous job ownership".into(),
            ));
        }
        let stride = arrays.len() / state.vision_jobs.len();
        if stride == 0 {
            return Err(Error::Parallel(
                "Qwen3.5 distributed vision payload omitted job hidden state".into(),
            ));
        }
        for (job, chunk) in state.vision_jobs.iter_mut().zip(arrays.chunks(stride)) {
            job.hidden = chunk[0].clone();
            job.state.replace_deepstack_features(chunk[1..].to_vec());
        }
        Ok(())
    }

    /// Creates the canonical text-only binding adapter used by bounded and
    /// pipeline execution.
    pub(crate) fn new_text(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let family = if args.model_type == "qwen3_next" {
            QwenHybridFamily::Qwen3Next
        } else {
            QwenHybridFamily::Qwen35
        };
        Self::new(args, family, None, None, None, stream)
    }

    pub(crate) fn new_text_external_experts(
        args: ModelArgs,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new_text(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Creates the shared stage-zero adapter for Qwen3.5 typed visual ingress.
    pub(crate) fn new_pipeline(
        args: ModelArgs,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_config: Option<VisionConfig>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new(
            args,
            QwenHybridFamily::Qwen35,
            image_token_id,
            video_token_id,
            vision_config,
            stream,
        )?;
        adapter.sparse_expert_cache = external_experts;
        Ok(adapter)
    }

    fn new(
        args: ModelArgs,
        family: QwenHybridFamily,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        vision_config: Option<VisionConfig>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.quantization_for("model.embed_tokens.weight")
                .or(args.quantization),
            stream,
        )?;
        let norm = Qwen3NextRmsNorm::new(args.hidden_size, args.rms_norm_eps, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.quantization_for("lm_head.weight")
                        .or(args.quantization),
                    stream,
                )?,
            )
        };
        let mtp = (args.mtp_num_hidden_layers > 0)
            .then(|| {
                MtpModule::new_with_format(&args, QwenWeightFormat::for_text(&args, None), stream)
            })
            .transpose()?;
        let vision = vision_config
            .clone()
            .map(|config| QwenVisionTransformer::new(config, stream))
            .transpose()?
            .map(QwenVisionLayerwiseStatic::from_transformer);
        Ok(Self {
            args,
            family,
            embedding,
            norm,
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            mtp,
            vision,
            vision_config,
            image_token_id,
            video_token_id,
            sparse_expert_cache: false,
            expert_cache: None,
            parallel_geometry: None,
        })
    }

    /// Returns normalized text-model arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Configures rank-local hybrid operator geometry for a Cartesian stage.
    pub(crate) fn configure_cartesian_layout(
        &mut self,
        build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.configure_parallel_static(build, layout, stream)
    }

    /// Returns configured rank-local attention, recurrent, and MLP geometry.
    pub(crate) fn parallel_geometry(&self) -> Option<&[resident::ParallelLayerGeometry]> {
        self.parallel_geometry.as_deref()
    }

    /// Embeds a decoder token step through the static module owned by stage zero.
    pub(crate) fn embed_pipeline_tokens(
        &mut self,
        tokens: &Array,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| {
                    Error::Parallel("Qwen3.5 pipeline adapter has no TP embedding shard".into())
                })?
                .forward(tokens, execution),
            _ => self.embedding.forward(tokens, stream).map_err(Into::into),
        }
    }

    /// Returns the configured media execution groups and their depths.
    pub(crate) fn pipeline_media_groups(&self) -> Vec<(usize, usize)> {
        self.vision
            .as_ref()
            .map(|vision| vec![(0, vision.config.layer_count())])
            .unwrap_or_default()
    }

    /// Returns the text execution group after any configured vision root.
    pub(crate) fn pipeline_text_group(&self) -> usize {
        usize::from(self.vision.is_some())
    }

    /// Selects a stage-owned static module without exposing its concrete type.
    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => {
                if let Some(module) = &mut self.parallel_embedding {
                    Some(module.inner_mut())
                } else {
                    Some(&mut self.embedding)
                }
            }
            "vision" => self
                .vision
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
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
        self.mtp.as_ref().map_or(0, MtpModule::len)
    }

    pub(crate) fn embedded_mtp_cache(&self) -> Vec<LayerCache> {
        self.new_cache().mtp_layers
    }

    pub(crate) fn forward_pipeline_mtp<F>(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        execute: Option<&mut F>,
        stream: &Stream,
    ) -> Result<QwenMtpStepOutput, Exception>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let embeddings = match execution.filter(|execution| execution.is_tensor_parallel()) {
            Some(execution) => self
                .parallel_embedding
                .as_mut()
                .ok_or_else(|| Exception::custom("Qwen pipeline MTP has no TP embedding shard"))?
                .forward(tokens, execution)
                .map_err(|error| Exception::custom(error.to_string()))?,
            None => self.embedding.forward(tokens, stream)?,
        };
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext::group);
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?;
        let hidden = match execute {
            Some(execute) => {
                let layer = self.args.num_hidden_layers as usize;
                mtp.forward_with_expert_executor(
                    hidden,
                    &embeddings,
                    cache,
                    tensor_group,
                    |hidden, ids, weights, stream| execute(layer, hidden, ids, weights, stream),
                    stream,
                )?
            }
            None => match tensor_group {
                Some(group) => {
                    mtp.forward_tensor_parallel(hidden, &embeddings, cache, group, stream)?
                }
                None => mtp.forward(hidden, &embeddings, cache, stream)?,
            },
        };
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
                    "Qwen pipeline MTP has no TP output projection".into(),
                )),
            }
            .map_err(|error| Exception::custom(error.to_string())),
            None => project_logits_maybe_quantized(
                &mut self.lm_head,
                &mut self.embedding,
                &hidden,
                stream,
            ),
        }?;
        Ok(QwenMtpStepOutput { logits, hidden })
    }

    fn execution_group_name(&self, group: usize) -> Result<&'static str, Error> {
        match (self.vision.is_some(), group) {
            (true, 0) => Ok("vision_encoder"),
            (true, 1) | (false, 0) => Ok("text_decoder"),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid model has no execution group {group}"
            ))),
        }
    }

    fn new_cache(&self) -> Cache {
        Cache::new(&self.args).expect("validated Qwen hybrid layer schedule")
    }

    fn forward_mtp_head(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut [LayerCache],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embeddings = self.embedding.forward(tokens, stream)?;
        let hidden = self
            .mtp
            .as_mut()
            .ok_or_else(|| Exception::custom("Qwen checkpoint does not contain MTP layers"))?
            .forward(hidden, &embeddings, cache, stream)?;
        project_logits_maybe_quantized(&mut self.lm_head, &mut self.embedding, &hidden, stream)
    }

    fn binding_plan_for_module(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layer_index: Option<usize>,
    ) -> Result<ModuleBindingPlan, Error> {
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.source_keys();
        let mut recipes = BTreeMap::new();

        if let Some(index) = layer_index {
            if self.family == QwenHybridFamily::Qwen3Next
                && self.args.layer_schedule.get(index) == Some(&LayerPolicy::LinearAttention)
            {
                add_fused_projection_recipes(&mut recipes, &normalized, index, &self.args)?;
            }
            if self.args.is_moe() {
                add_expert_recipes(
                    &mut recipes,
                    &normalized,
                    index,
                    &self.args,
                    self.args.uses_fp8(),
                )?;
            }
        }
        if store.source_diagnostics()?.backend == WeightStoreBackend::Gguf {
            add_qwen_gguf_transform_recipes(
                &mut recipes,
                module,
                prefix,
                store,
                &self.args,
                self.family,
            )?;
        }

        let parameters = module.parameters().flatten();
        for (local_name, parameter) in &parameters {
            if !is_materialized_module_parameter(local_name, parameter, &parameters) {
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
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen hybrid checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(build_module_binding_plan_with_recipes(
            module, prefix, store, recipes,
        )?)
    }

    fn mtp_binding_plan(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<ModuleBindingPlan, Error> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid model has no MTP module".into())
        })?;
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.source_keys();
        let mut recipes = BTreeMap::new();
        if self.args.is_moe() {
            for index in 0..self.args.mtp_num_hidden_layers as usize {
                add_expert_recipes_for_prefix(
                    &mut recipes,
                    &normalized,
                    &format!("mtp.layers.{index}.mlp.experts"),
                    &format!("layers.{index}.mlp"),
                    &self.args,
                    self.args.uses_fp8(),
                )?;
            }
        }
        let parameters = mtp.parameters().flatten();
        for (local_name, parameter) in &parameters {
            if !is_materialized_module_parameter(local_name, parameter, &parameters)
                || (self.sparse_expert_cache && local_name.contains(".mlp.experts."))
            {
                continue;
            }
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = format!("mtp.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            let raw = normalized
                .get(&destination)
                .or_else(|| normalized.get(&canonical))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen hybrid checkpoint is missing runtime parameter {canonical}"
                    ))
                })?;
            recipes.insert(
                local_name.to_string(),
                DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full),
            );
        }
        Ok(build_module_binding_plan_with_recipes_excluding(
            mtp,
            "mtp",
            store,
            recipes,
            |name| self.sparse_expert_cache && name.contains(".mlp.experts."),
        )?)
    }
}

fn add_qwen_gguf_transform_recipes(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    args: &ModelArgs,
    family: QwenHybridFamily,
) -> Result<(), Error> {
    let parameters = module.parameters().flatten();
    for local_name in parameters.keys() {
        let source = format!("{prefix}.{local_name}");
        let Ok(metadata) = store.source_metadata(&source) else {
            continue;
        };
        let expected = parameters[local_name.as_ref()]
            .shape()
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>();
        let mut recipe = DerivedWeightRecipe::source(source, TensorSelection::Full);
        if local_name.ends_with("linear_attn.A_log") {
            recipe = DerivedWeightRecipe::NegLog {
                input: Box::new(recipe),
            };
        }
        let offset_norm = (prefix == "model.norm" && local_name.as_ref() == "weight")
            || (local_name.ends_with("_layernorm.weight")
                || local_name.ends_with(".q_norm.weight")
                || local_name.ends_with(".k_norm.weight"));
        if offset_norm {
            recipe = DerivedWeightRecipe::SubtractOne {
                input: Box::new(recipe),
            };
        }
        if family == QwenHybridFamily::Qwen35 {
            recipe = qwen35_value_head_recipe(local_name, recipe, &metadata.logical_shape, args)?;
        }
        if recipe.infer(store)?.shape() != expected {
            recipe = DerivedWeightRecipe::Reshape {
                input: Box::new(recipe),
                shape: expected,
            };
        }
        if !matches!(
            recipe,
            DerivedWeightRecipe::Source {
                selection: TensorSelection::Full,
                ..
            }
        ) {
            recipes.insert(local_name.to_string(), recipe);
        }
    }
    Ok(())
}

fn qwen35_value_head_recipe(
    local_name: &str,
    recipe: DerivedWeightRecipe,
    shape: &[usize],
    args: &ModelArgs,
) -> Result<DerivedWeightRecipe, Error> {
    let num_k = usize_from_i32(args.linear_num_key_heads)?;
    let num_v = usize_from_i32(args.linear_num_value_heads)?;
    if num_k == 0 || num_v % num_k != 0 {
        return Err(Error::UnsupportedArchitecture(
            "invalid Qwen3.5 value-head grouping".into(),
        ));
    }
    let repeats = num_v / num_k;
    let value_head = usize_from_i32(args.linear_value_head_dim)?;
    let reorder = |input: DerivedWeightRecipe,
                   axis: usize,
                   head_width: usize,
                   original: Vec<usize>|
     -> DerivedWeightRecipe {
        let mut expanded = original.clone();
        expanded.splice(axis..=axis, [repeats, num_k, head_width]);
        let mut axes = (0..expanded.len()).collect::<Vec<_>>();
        axes.swap(axis, axis + 1);
        DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::Transpose {
                input: Box::new(DerivedWeightRecipe::Reshape {
                    input: Box::new(input),
                    shape: expanded,
                }),
                axes,
            }),
            shape: original,
        }
    };
    if local_name.ends_with("linear_attn.in_proj_qkv.weight")
        || local_name.ends_with("linear_attn.conv1d.weight")
    {
        if shape.len() != 2 {
            return Ok(recipe);
        }
        let prefix = 2usize
            .checked_mul(num_k)
            .and_then(|value| value.checked_mul(usize_from_i32(args.linear_key_head_dim).ok()?))
            .ok_or_else(|| {
                Error::UnsupportedArchitecture("Qwen3.5 value-tail width overflow".into())
            })?;
        let leading = DerivedWeightRecipe::Select {
            input: Box::new(recipe.clone()),
            selection: TensorSelection::Range {
                axis: 0,
                start: 0,
                end: prefix,
            },
        };
        let tail_shape = vec![shape[0] - prefix, shape[1]];
        let tail = DerivedWeightRecipe::Select {
            input: Box::new(recipe),
            selection: TensorSelection::Range {
                axis: 0,
                start: prefix,
                end: shape[0],
            },
        };
        return Ok(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![leading, reorder(tail, 0, value_head, tail_shape)],
        });
    }
    let (axis, head_width) = if local_name.ends_with("linear_attn.in_proj_z.weight") {
        (0, value_head)
    } else if local_name.ends_with("linear_attn.in_proj_a.weight")
        || local_name.ends_with("linear_attn.in_proj_b.weight")
        || local_name.ends_with("linear_attn.dt_bias")
        || local_name.ends_with("linear_attn.A_log")
    {
        (0, 1)
    } else if local_name.ends_with("linear_attn.out_proj.weight") {
        (1, value_head)
    } else {
        return Ok(recipe);
    };
    if shape.get(axis).copied() != Some(num_v * head_width) {
        return Ok(recipe);
    }
    Ok(reorder(recipe, axis, head_width, shape.to_vec()))
}

fn normalized_checkpoint_keys(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| {
            let runtime = ["model.language_model.", "language_model.", "model.model."]
                .into_iter()
                .find_map(|prefix| raw.strip_prefix(prefix).map(|rest| format!("model.{rest}")))
                .or_else(|| {
                    ["model.vision_tower.", "model.visual.", "vision_tower."]
                        .into_iter()
                        .find_map(|prefix| {
                            raw.strip_prefix(prefix)
                                .map(|rest| format!("visual.{rest}"))
                        })
                })
                .unwrap_or_else(|| raw.clone())
                .replace("visual.merger.mlp.0.", "visual.merger.mlp.fc1.")
                .replace("visual.merger.mlp.2.", "visual.merger.mlp.fc2.");
            (runtime, raw)
        })
        .collect()
}

fn add_fused_projection_recipes(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    index: usize,
    args: &ModelArgs,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}.linear_attn");
    let (qkvz_widths, ba_width) = qwen3_next::fused_projection_widths(args)?;
    for suffix in ["weight", "scales", "biases"] {
        let qkvz_runtime = format!("{prefix}.in_proj_qkvz.{suffix}");
        if let Some(raw) = normalized.get(&qkvz_runtime) {
            for (local, components) in [
                (format!("linear_attn.in_proj_qkv.{suffix}"), vec![0, 1, 2]),
                (format!("linear_attn.in_proj_z.{suffix}"), vec![3]),
            ] {
                let selected = DerivedWeightRecipe::source(
                    raw.clone(),
                    TensorSelection::Indices {
                        axis: 0,
                        indices: grouped_component_indices(
                            self::usize_from_i32(args.linear_num_key_heads)?,
                            &qkvz_widths,
                            &components,
                        )?,
                    },
                );
                recipes.insert(
                    local,
                    if suffix == "weight" {
                        selected
                    } else {
                        DerivedWeightRecipe::Cast {
                            input: Box::new(selected),
                            dtype: RecipeDtype::F32,
                        }
                    },
                );
            }
        }
        let ba_runtime = format!("{prefix}.in_proj_ba.{suffix}");
        if let Some(raw) = normalized.get(&ba_runtime) {
            for (local, component) in [
                (format!("linear_attn.in_proj_b.{suffix}"), 0),
                (format!("linear_attn.in_proj_a.{suffix}"), 1),
            ] {
                let selected = DerivedWeightRecipe::source(
                    raw.clone(),
                    TensorSelection::Indices {
                        axis: 0,
                        indices: grouped_component_indices(
                            usize_from_i32(args.linear_num_key_heads)?,
                            &[ba_width, ba_width],
                            &[component],
                        )?,
                    },
                );
                recipes.insert(
                    local,
                    if suffix == "weight" {
                        selected
                    } else {
                        DerivedWeightRecipe::Cast {
                            input: Box::new(selected),
                            dtype: RecipeDtype::F32,
                        }
                    },
                );
            }
        }
    }
    if args.uses_fp8() {
        let block_widths = qwen3_next::fp8_block_row_widths(&qkvz_widths)?;
        let qkvz_runtime = format!("{prefix}.in_proj_qkvz.weight_scale_inv");
        if let Some(raw) = normalized.get(&qkvz_runtime) {
            for (local, components) in [
                (
                    "linear_attn.in_proj_qkv.weight_scale_inv".to_string(),
                    vec![0, 1, 2],
                ),
                (
                    "linear_attn.in_proj_z.weight_scale_inv".to_string(),
                    vec![3],
                ),
            ] {
                recipes.insert(
                    local,
                    DerivedWeightRecipe::source(
                        raw.clone(),
                        TensorSelection::Indices {
                            axis: 0,
                            indices: grouped_component_indices(
                                usize_from_i32(args.linear_num_key_heads)?,
                                &block_widths,
                                &components,
                            )?,
                        },
                    ),
                );
            }
        }
        if normalized.contains_key(&format!("{prefix}.in_proj_ba.weight_scale_inv")) {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3-Next in_proj_ba must remain dense BF16 and cannot carry FP8 inverse scales"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn usize_from_i32(value: i32) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture("Qwen hybrid projection dimension is invalid".into())
    })
}

fn grouped_component_indices(
    groups: usize,
    widths: &[i32],
    components: &[usize],
) -> Result<Vec<usize>, Error> {
    let widths = widths
        .iter()
        .map(|width| usize_from_i32(*width))
        .collect::<Result<Vec<_>, _>>()?;
    let group_width = widths.iter().sum::<usize>();
    let mut starts = Vec::with_capacity(widths.len());
    let mut start = 0usize;
    for width in &widths {
        starts.push(start);
        start = start.checked_add(*width).ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid projection index overflow".into())
        })?;
    }
    let mut indices = Vec::new();
    for component in components {
        let width = *widths.get(*component).ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen hybrid projection component is invalid".into())
        })?;
        for group in 0..groups {
            let base = group
                .checked_mul(group_width)
                .and_then(|base| base.checked_add(starts[*component]))
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen hybrid projection index overflow".into())
                })?;
            indices.extend(base..base + width);
        }
    }
    Ok(indices)
}

fn add_expert_recipes(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    index: usize,
    args: &ModelArgs,
    fp8: bool,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}.mlp.experts");
    add_expert_recipes_for_prefix(recipes, normalized, &prefix, "mlp", args, fp8)
}

fn add_expert_recipes_for_prefix(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    local_prefix: &str,
    args: &ModelArgs,
    fp8: bool,
) -> Result<(), Error> {
    if normalized.contains_key(&format!("{prefix}.gate_up_proj")) {
        return Ok(());
    }
    if let (Some(gate), Some(up), Some(down)) = (
        normalized.get(&format!("{prefix}.gate_proj")),
        normalized.get(&format!("{prefix}.up_proj")),
        normalized.get(&format!("{prefix}.down_proj")),
    ) {
        recipes.insert(
            format!("{local_prefix}.experts.gate_up_proj"),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                    DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                ],
            },
        );
        recipes.insert(
            format!("{local_prefix}.experts.down_proj"),
            DerivedWeightRecipe::source(down.clone(), TensorSelection::Full),
        );
        for suffix in ["_scales", "_biases"] {
            if let (Some(gate), Some(up)) = (
                normalized.get(&format!("{prefix}.gate_proj{suffix}")),
                normalized.get(&format!("{prefix}.up_proj{suffix}")),
            ) {
                recipes.insert(
                    format!("{local_prefix}.experts.gate_up_proj{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                            DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                        ],
                    },
                );
            }
            if let Some(down) = normalized.get(&format!("{prefix}.down_proj{suffix}")) {
                recipes.insert(
                    format!("{local_prefix}.experts.down_proj{suffix}"),
                    DerivedWeightRecipe::source(down.clone(), TensorSelection::Full),
                );
            }
        }
        return Ok(());
    }
    let mut gate_up = Vec::with_capacity(args.num_experts as usize);
    let mut down = Vec::with_capacity(args.num_experts as usize);
    let mut gate_up_scale = Vec::new();
    let mut down_scale = Vec::new();
    for expert in 0..args.num_experts {
        let gate = expert_source(normalized, prefix, expert, &["gate_proj", "w1"], "weight")?;
        let up = expert_source(normalized, prefix, expert, &["up_proj", "w3"], "weight")?;
        down.push(expert_source(
            normalized,
            prefix,
            expert,
            &["down_proj", "w2"],
            "weight",
        )?);
        gate_up.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![gate, up],
        });
        if fp8 {
            let gate_scale = expert_source(
                normalized,
                prefix,
                expert,
                &["gate_proj"],
                "weight_scale_inv",
            )?;
            let up_scale =
                expert_source(normalized, prefix, expert, &["up_proj"], "weight_scale_inv")?;
            gate_up_scale.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate_scale, up_scale],
            });
            down_scale.push(expert_source(
                normalized,
                prefix,
                expert,
                &["down_proj"],
                "weight_scale_inv",
            )?);
        }
    }
    recipes.insert(
        format!("{local_prefix}.experts.gate_up_proj"),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: gate_up,
        },
    );
    recipes.insert(
        format!("{local_prefix}.experts.down_proj"),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: down,
        },
    );
    if fp8 {
        recipes.insert(
            format!("{local_prefix}.experts.gate_up_proj_scale_inv"),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: gate_up_scale,
            },
        );
        recipes.insert(
            format!("{local_prefix}.experts.down_proj_scale_inv"),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: down_scale,
            },
        );
    }
    Ok(())
}

fn expert_source(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: i32,
    projections: &[&str],
    suffix: &str,
) -> Result<DerivedWeightRecipe, Error> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.{suffix}"))
        .find_map(|runtime| normalized.get(&runtime).cloned())
        .map(|raw| DerivedWeightRecipe::source(raw, TensorSelection::Full))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen hybrid checkpoint is missing expert {expert} tensor under {prefix}"
            ))
        })
}

pub(crate) fn qwen_hybrid_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut prefixes = (0..args.num_hidden_layers as usize)
        .map(|layer| (layer, format!("model.layers.{layer}.mlp.experts")))
        .collect::<Vec<_>>();
    prefixes.extend((0..args.mtp_num_hidden_layers as usize).map(|index| {
        (
            args.num_hidden_layers as usize + index,
            format!("mtp.layers.{index}.mlp.experts"),
        )
    }));
    qwen_hybrid_expert_catalog_for_prefixes(args, store, prefixes)
}

pub(crate) fn qwen_hybrid_pipeline_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layers: impl IntoIterator<Item = usize>,
    include_mtp: bool,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut prefixes = layers
        .into_iter()
        .map(|layer| (layer, format!("model.layers.{layer}.mlp.experts")))
        .collect::<Vec<_>>();
    if include_mtp {
        prefixes.extend((0..args.mtp_num_hidden_layers as usize).map(|index| {
            (
                args.num_hidden_layers as usize + index,
                format!("mtp.layers.{index}.mlp.experts"),
            )
        }));
    }
    qwen_hybrid_expert_catalog_for_prefixes(args, store, prefixes)
}

fn qwen_hybrid_expert_catalog_for_prefixes(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    prefixes: impl IntoIterator<Item = (usize, String)>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for (layer, prefix) in prefixes {
        let packed = normalized.contains_key(&format!("{prefix}.gate_up_proj"));
        let split_banks = normalized.contains_key(&format!("{prefix}.gate_proj"))
            && normalized.contains_key(&format!("{prefix}.up_proj"))
            && normalized.contains_key(&format!("{prefix}.down_proj"));
        for expert in 0..args.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = Vec::new();
            if packed {
                for (name, required) in [
                    ("gate_up_proj", true),
                    ("gate_up_proj_scale_inv", false),
                    ("gate_up_proj_scales", false),
                    ("gate_up_proj_biases", false),
                    ("down_proj", true),
                    ("down_proj_scale_inv", false),
                    ("down_proj_scales", false),
                    ("down_proj_biases", false),
                ] {
                    let runtime = format!("{prefix}.{name}");
                    let Some(raw) = normalized.get(&runtime) else {
                        if required {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "Qwen hybrid checkpoint is missing packed expert tensor {runtime}"
                            )));
                        }
                        continue;
                    };
                    bindings.push(qwen_hybrid_recipe_binding(
                        name,
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
            } else if split_banks {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                bindings.push(qwen_hybrid_recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                normalized[&format!("{prefix}.gate_proj")].clone(),
                                selection.clone(),
                            ),
                            DerivedWeightRecipe::source(
                                normalized[&format!("{prefix}.up_proj")].clone(),
                                selection.clone(),
                            ),
                        ],
                    },
                    store,
                )?);
                bindings.push(qwen_hybrid_recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(
                        normalized[&format!("{prefix}.down_proj")].clone(),
                        selection.clone(),
                    ),
                    store,
                )?);
                for suffix in ["_scales", "_biases"] {
                    if let (Some(gate), Some(up)) = (
                        normalized.get(&format!("{prefix}.gate_proj{suffix}")),
                        normalized.get(&format!("{prefix}.up_proj{suffix}")),
                    ) {
                        bindings.push(qwen_hybrid_recipe_binding(
                            &format!("gate_up_proj{suffix}"),
                            DerivedWeightRecipe::Concatenate {
                                axis: 1,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate.clone(), selection.clone()),
                                    DerivedWeightRecipe::source(up.clone(), selection.clone()),
                                ],
                            },
                            store,
                        )?);
                    }
                    if let Some(down) = normalized.get(&format!("{prefix}.down_proj{suffix}")) {
                        bindings.push(qwen_hybrid_recipe_binding(
                            &format!("down_proj{suffix}"),
                            DerivedWeightRecipe::source(down.clone(), selection.clone()),
                            store,
                        )?);
                    }
                }
            } else {
                let gate = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["gate_proj", "w1"],
                    "weight",
                )?;
                let up = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["up_proj", "w3"],
                    "weight",
                )?;
                let down = expert_source(
                    &normalized,
                    &prefix,
                    expert as i32,
                    &["down_proj", "w2"],
                    "weight",
                )?;
                bindings.push(qwen_hybrid_recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![gate, up],
                        }],
                    },
                    store,
                )?);
                bindings.push(qwen_hybrid_recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![down],
                    },
                    store,
                )?);
                if args.uses_fp8() {
                    let gate = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["gate_proj"],
                        "weight_scale_inv",
                    )?;
                    let up = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["up_proj"],
                        "weight_scale_inv",
                    )?;
                    let down = expert_source(
                        &normalized,
                        &prefix,
                        expert as i32,
                        &["down_proj"],
                        "weight_scale_inv",
                    )?;
                    bindings.push(qwen_hybrid_recipe_binding(
                        "gate_up_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                        store,
                    )?);
                    bindings.push(qwen_hybrid_recipe_binding(
                        "down_proj_scale_inv",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                        store,
                    )?);
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid expert byte total overflowed".into(),
                    )
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

fn qwen_hybrid_recipe_binding(
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

/// Input mode for typed prefill and cached text decode.
pub enum QwenHybridInput<'a> {
    /// Ordered text and visual prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for cached decode.
    Decode(&'a Array),
}

enum QwenHybridPreparedPart {
    Ready(Array),
    Vision(usize),
}

struct QwenHybridVisionJob {
    hidden: Array,
    state: QwenVisionLayerwiseState,
}

/// Per-forward vision assembly and causal mask state.
pub struct QwenHybridForwardContext {
    mask: Option<Array>,
    parts: Vec<QwenHybridPreparedPart>,
    vision_jobs: Vec<QwenHybridVisionJob>,
    needs_assembly: bool,
    draft_hidden: Option<Array>,
}

/// Opaque state routed while placed owners execute the Qwen3.5 vision root.
pub(crate) struct QwenHybridPipelineIngressState {
    parts: Vec<QwenHybridPreparedPart>,
    vision_jobs: Vec<QwenHybridVisionJob>,
}

impl QwenHybridLayerwiseAdapter {
    /// Starts typed visual ingress using the same semantic preparation as the
    /// bounded execution-group path.
    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        input: input::ModelInput<'_>,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<QwenHybridPipelineIngressState, Error> {
        input::validate(input)?;
        let mut parts = Vec::with_capacity(input.parts.len());
        let mut vision_jobs = Vec::new();
        for part in input.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    parts.push(QwenHybridPreparedPart::Ready(
                        self.embed_pipeline_tokens(tokens, execution, stream)?,
                    ));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(pixels),
                ) => {
                    let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Qwen3.5 {} input requires qwen_grid_thw metadata",
                            part.modality.as_str()
                        ))
                    })?;
                    let vision = self.vision.as_mut().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen3.5 visual tensor input requires vision_config and visual weights"
                                .into(),
                        )
                    })?;
                    let token_id = if part.modality == input::Modality::Image {
                        self.image_token_id
                    } else {
                        self.video_token_id
                    };
                    if token_id.is_none() {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Qwen3.5 config does not define a {} token ID",
                            part.modality.as_str()
                        )));
                    }
                    let merge = vision.config.spatial_merge_size;
                    let merged = grid_thw_from_array(grid, stream)?
                        .into_iter()
                        .map(|(t, h, w)| t * (h / merge) * (w / merge))
                        .sum::<i32>();
                    if merged <= 0 {
                        return Err(Error::UnsupportedArchitecture(
                            "Qwen3.5 visual grid produced no merged tokens".into(),
                        ));
                    }
                    let (hidden, state) = vision.begin(pixels, grid, stream)?;
                    let job = vision_jobs.len();
                    vision_jobs.push(QwenHybridVisionJob { hidden, state });
                    parts.push(QwenHybridPreparedPart::Vision(job));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Embeddings(embeddings),
                ) => {
                    input::ensure_hidden_size(
                        embeddings,
                        self.args.hidden_size,
                        "Qwen3.5 visual embeddings",
                    )?;
                    parts.push(QwenHybridPreparedPart::Ready(embeddings.clone()));
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen3.5 pipeline input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        Ok(QwenHybridPipelineIngressState { parts, vision_jobs })
    }

    /// Reconstructs parameter-free vision scheduling state on a downstream
    /// PP owner. Patch/text embeddings and final projection remain on the
    /// placement-declared ingress owner; transported arrays replace the dummy
    /// hidden values before any local vision block executes.
    pub(crate) fn begin_pipeline_continuation(
        &self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<QwenHybridPipelineIngressState, Error> {
        input::validate(input)?;
        let vision = self.vision.as_ref().ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "Qwen3.5 visual continuation requires vision_config".into(),
            )
        })?;
        let mut vision_jobs = Vec::new();
        for part in input.parts {
            if !matches!(
                part.modality,
                input::Modality::Image | input::Modality::Video
            ) {
                continue;
            }
            let input::InputPayload::Tensor(pixels) = part.payload else {
                continue;
            };
            let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                Error::Parallel("Qwen3.5 continuation omitted qwen_grid_thw".into())
            })?;
            vision_jobs.push(QwenHybridVisionJob {
                hidden: pixels.clone(),
                state: vision.continuation_state(pixels, grid, stream)?,
            });
        }
        Ok(QwenHybridPipelineIngressState {
            parts: Vec::new(),
            vision_jobs,
        })
    }

    /// Returns whether one configured media group has work for this input.
    pub(crate) fn should_execute_pipeline_group(
        &self,
        group: usize,
        state: &QwenHybridPipelineIngressState,
    ) -> bool {
        self.execution_group_name(group)
            .is_ok_and(|name| name != "vision_encoder" || !state.vision_jobs.is_empty())
    }

    /// Executes one resident or leased vision block.
    pub(crate) fn forward_pipeline_media_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut QwenHybridLayer,
        state: &mut QwenHybridPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        if self.execution_group_name(group)? != "vision_encoder" {
            return Err(Error::Parallel(format!(
                "Qwen3.5 pipeline media unit {group}:{index} is not a vision block"
            )));
        }
        let QwenHybridLayer::Vision(block) = layer else {
            return Err(Error::Parallel(format!(
                "Qwen3.5 pipeline media unit {group}:{index} has text storage"
            )));
        };
        let vision = self.vision.as_mut().expect("configured vision group");
        for job in &mut state.vision_jobs {
            job.hidden = match execution.and_then(|execution| execution.group()) {
                Some(tp_group) => vision.forward_block_tensor_parallel(
                    block,
                    index,
                    job.hidden.clone(),
                    &job.state,
                    tp_group,
                    execution.expect("TP execution").stream(),
                )?,
                None => {
                    vision.forward_block(block, index, job.hidden.clone(), &job.state, stream)?
                }
            };
            match execution.and_then(|execution| execution.group()) {
                Some(tp_group) => vision.capture_deepstack_tensor_parallel(
                    index,
                    &job.hidden,
                    &mut job.state,
                    tp_group,
                    execution.expect("TP execution").stream(),
                )?,
                None => vision.capture_deepstack(index, &job.hidden, &mut job.state, stream)?,
            }
        }
        Ok(state
            .vision_jobs
            .iter()
            .flat_map(|job| {
                std::iter::once(job.hidden.clone())
                    .chain(job.state.retained_arrays().into_iter().cloned())
            })
            .collect())
    }

    /// Finishes all media roots and assembles exact decoder ingress embeddings.
    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: QwenHybridPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if let Some(vision) = &mut self.vision {
            for job in &mut state.vision_jobs {
                job.hidden = match execution.and_then(|execution| execution.group()) {
                    Some(group) => {
                        vision.finish_tensor_parallel(&job.hidden, &mut job.state, group, stream)?
                    }
                    None => vision.finish(&job.hidden, &mut job.state, stream)?,
                }
                .embeddings;
            }
        }
        let assembled = state
            .parts
            .iter()
            .map(|part| match part {
                QwenHybridPreparedPart::Ready(value) => value,
                QwenHybridPreparedPart::Vision(job) => &state.vision_jobs[*job].hidden,
            })
            .collect::<Vec<_>>();
        concatenate_axis(&assembled, 1, stream).map_err(Into::into)
    }
}

/// One leased vision or hybrid text block.
pub enum QwenHybridLayer {
    /// Qwen vision transformer block.
    Vision(Box<QwenVisionBlock>),
    /// Qwen recurrent or full-attention text block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for QwenHybridLayer {
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

impl LoadTimeQuantizableAdapter for QwenHybridLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weight_configs = None;
        let mut adapter = Self::new(
            args,
            self.family,
            self.image_token_id,
            self.video_token_id,
            self.vision_config.clone(),
            stream,
        )?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for QwenHybridLayerwiseAdapter {
    type Input<'a> = QwenHybridInput<'a>;
    type Cache = Cache;
    type Layer = QwenHybridLayer;
    type ForwardContext = QwenHybridForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error>
    {
        let variant = match self.family {
            QwenHybridFamily::Qwen3Next => super::checkpoint::SafetensorsVariant::Qwen3Next,
            QwenHybridFamily::Qwen35 => super::checkpoint::SafetensorsVariant::Qwen35,
        };
        super::checkpoint::safetensors_plan(&self.args, self.vision_config.as_ref(), variant)
            .map_err(Error::UnsupportedArchitecture)
            .map(Into::into)
    }

    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.args.quantization
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        !target.starts_with("visual.") && !target.contains(".visual.")
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.num_hidden_layers as usize;
        let layer_layout = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                let geometry = self.parallel_geometry.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "Qwen hybrid parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?;
                resident::prompt_cache_layer_layout_with_geometry(&self.args, geometry)
            }
            _ => resident::prompt_cache_layer_layout(&self.args),
        }?;
        Ok(PromptCacheModelIdentity {
            model_family: "qwen_hybrid".into(),
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
        resident::Model::save_prompt_cache(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
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
        _options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        resident::Model::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
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
                self.binding_plan_for_module(&self.embedding, "model.embed_tokens", store, None)?
                    .build_bindings(store)?,
            )?);
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                self.binding_plan_for_module(&self.norm, "model.norm", store, None)?
                    .build_bindings(store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            if let Some(head) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    self.binding_plan_for_module(head, "lm_head", store, None)?
                        .build_bindings(store)?,
                )?);
            }
        }
        if select(MTP_STATIC_UNIT) {
            if self.mtp.is_some() {
                units.push(StaticUnitBindings::new(
                    MTP_STATIC_UNIT,
                    self.mtp_binding_plan(store)?.build_bindings(store)?,
                )?);
            }
        }
        if select(VISION_STATIC_UNIT) {
            if let Some(vision) = &self.vision {
                units.push(StaticUnitBindings::new(
                    VISION_STATIC_UNIT,
                    self.binding_plan_for_module(vision, "visual", store, None)?
                        .build_bindings(store)?,
                )?);
            }
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = 2
            + usize::from(self.lm_head.is_some())
            + usize::from(self.mtp.is_some())
            + usize::from(self.vision.is_some());
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        if let Some(v) = &mut self.parallel_embedding {
            populate_module_from_lease(v.inner_mut(), &leases[0])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[0])?;
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
                    name.contains(".mlp.experts.")
                })?;
            } else {
                populate_module_from_lease(mtp, &leases[index])?;
            }
            index += 1;
        }
        if let Some(vision) = &mut self.vision {
            populate_module_from_lease(vision, &leases[index])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = self.new_cache();
            return Ok(());
        }
        if cache.layers.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        for (index, cache) in cache.layers.iter().enumerate() {
            let policy = self.args.layer_schedule.get(index).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen hybrid layer schedule is missing decoder layer {index}"
                ))
            })?;
            let matches = matches!(
                (policy, cache),
                (
                    LayerPolicy::SelfAttention(AttentionPolicy::Full),
                    LayerCache::FullAttention(_)
                ) | (LayerPolicy::LinearAttention, LayerCache::LinearAttention(_))
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen hybrid cache kind does not match layer schedule at layer {index}"
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
        let (hidden, parts, vision_jobs, needs_assembly) = match input {
            QwenHybridInput::Decode(tokens) => (
                self.embedding.forward(tokens, stream)?,
                Vec::new(),
                Vec::new(),
                false,
            ),
            QwenHybridInput::Prefill(typed) => {
                input::validate(typed)?;
                let mut parts = Vec::with_capacity(typed.parts.len());
                let mut jobs = Vec::new();
                for part in typed.parts {
                    match (part.modality, part.payload) {
                        (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                            parts.push(QwenHybridPreparedPart::Ready(
                                self.embedding.forward(tokens, stream)?,
                            ));
                        }
                        (
                            input::Modality::Image | input::Modality::Video,
                            input::InputPayload::Tensor(pixels),
                        ) => {
                            let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                                Error::UnsupportedArchitecture(format!(
                                    "Qwen3.5 {} input requires qwen_grid_thw metadata",
                                    part.modality.as_str()
                                ))
                            })?;
                            let vision = self.vision.as_mut().ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                    "Qwen3.5 visual tensor input requires vision_config and visual weights".into(),
                                )
                            })?;
                            let token_id = if part.modality == input::Modality::Image {
                                self.image_token_id
                            } else {
                                self.video_token_id
                            };
                            if token_id.is_none() {
                                return Err(Error::UnsupportedArchitecture(format!(
                                    "Qwen3.5 config does not define a {} token ID",
                                    part.modality.as_str()
                                )));
                            }
                            let merge = vision.config.spatial_merge_size;
                            let merged = grid_thw_from_array(grid, stream)?
                                .into_iter()
                                .map(|(t, h, w)| t * (h / merge) * (w / merge))
                                .sum::<i32>();
                            if merged <= 0 {
                                return Err(Error::UnsupportedArchitecture(
                                    "Qwen3.5 visual grid produced no merged tokens".into(),
                                ));
                            }
                            let (hidden, state) = vision.begin(pixels, grid, stream)?;
                            let job = jobs.len();
                            jobs.push(QwenHybridVisionJob { hidden, state });
                            parts.push(QwenHybridPreparedPart::Vision(job));
                        }
                        (
                            input::Modality::Image | input::Modality::Video,
                            input::InputPayload::Embeddings(embeddings),
                        ) => {
                            input::ensure_hidden_size(
                                embeddings,
                                self.args.hidden_size,
                                "Qwen3.5 visual embeddings",
                            )?;
                            parts.push(QwenHybridPreparedPart::Ready(embeddings.clone()));
                        }
                        (modality, _) => {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "Qwen3.5 layerwise input does not support {} payloads of this kind",
                                modality.as_str()
                            )));
                        }
                    }
                }
                if jobs.is_empty() {
                    let ready = parts
                        .iter()
                        .map(|part| match part {
                            QwenHybridPreparedPart::Ready(value) => value,
                            QwenHybridPreparedPart::Vision(_) => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    (concatenate_axis(&ready, 1, stream)?, parts, jobs, false)
                } else {
                    (jobs[0].hidden.clone(), parts, jobs, true)
                }
            }
        };
        let mask = if !needs_assembly && hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen hybrid decoder requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: QwenHybridForwardContext {
                mask,
                parts,
                vision_jobs,
                needs_assembly,
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
        let Some(embedding) = &mut self.parallel_embedding else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let stream = execution.stream();
        let hidden = match input {
            QwenHybridInput::Decode(tokens) => embedding.forward(tokens, execution)?,
            QwenHybridInput::Prefill(typed) => {
                input::validate(typed)?;
                let mut values = Vec::with_capacity(typed.parts.len());
                for part in typed.parts {
                    match (part.modality, part.payload) {
                        (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                            values.push(embedding.forward(tokens, execution)?)
                        }
                        (_, input::InputPayload::Embeddings(value)) => {
                            input::ensure_hidden_size(
                                value,
                                self.args.hidden_size,
                                "Qwen hybrid embeddings",
                            )?;
                            values.push(value.clone());
                        }
                        (modality, _) => {
                            return Err(Error::UnsupportedArchitecture(format!(
                                "token-only Qwen TP does not accept {} tensor input",
                                modality.as_str()
                            )))
                        }
                    }
                }
                concatenate_axis(&values, 1, stream)?
            }
        };
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(v)) => Some(v),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen hybrid decoder requires an array causal mask".into(),
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: QwenHybridForwardContext {
                mask,
                parts: Vec::new(),
                vision_jobs: Vec::new(),
                needs_assembly: false,
                draft_hidden: None,
            },
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        if self.vision.is_some() {
            eredu_runtime::ExecutionGraph::chain(["vision_encoder", "text_decoder"])
                .map_err(Into::into)
        } else {
            eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
        }
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_name(group)
            .is_ok_and(|name| name != "vision_encoder" || !context.vision_jobs.is_empty())
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_name(group)? {
            "vision_encoder" => Ok(self
                .vision
                .as_ref()
                .expect("vision group")
                .config
                .layer_count()),
            "text_decoder" => Ok(self.args.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        if self.execution_group_name(group)? == "vision_encoder" {
            Ok(QwenHybridLayer::Vision(Box::new(QwenVisionBlock::new(
                &self.vision.as_ref().expect("vision group").config,
                index,
                stream,
            )?)))
        } else {
            Ok(QwenHybridLayer::Text(Box::new(TransformerBlock::new(
                &self.args, index, stream,
            )?)))
        }
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if matches!(layer, QwenHybridLayer::Vision(_)) {
            return Ok(layer);
        }
        let QwenHybridLayer::Text(block) = &mut layer else {
            unreachable!();
        };
        let resident::FeedForward::Moe(moe) = &mut block.mlp else {
            return Err(Error::Parallel(
                "Qwen hybrid PP+EP requires routed MoE decoder layers".into(),
            ));
        };
        let mut local_args = self.args.clone();
        local_args.num_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local Qwen hybrid expert count exceeds i32".into()))?
        };
        moe.experts = Experts::new(&local_args, index, stream)?;
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
        if matches!(layer, QwenHybridLayer::Vision(_)) {
            return Ok(layer);
        }
        let QwenHybridLayer::Text(block) = &mut layer else {
            unreachable!();
        };
        let resident::FeedForward::Moe(moe) = &mut block.mlp else {
            return Err(Error::Parallel(
                "Qwen hybrid TP+PP+EP requires routed MoE decoder layers".into(),
            ));
        };
        let mut local_args = self.args.clone();
        local_args.moe_intermediate_size = moe.experts.intermediate_dim;
        local_args.num_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_expert_count())
                .map_err(|_| Error::Parallel("local Qwen hybrid expert count exceeds i32".into()))?
        };
        moe.experts = Experts::new(&local_args, index, stream)?;
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
        if !self.args.is_moe() {
            return Err(Error::Parallel(format!(
                "{} PP+EP requires a routed MoE text checkpoint",
                self.args.model_type
            )));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_experts as usize,
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
            register_qwen_hybrid_layer_parallel_plan(planner, &layer, index, &self.args)?;
        }
        if let Some(mtp) = &self.mtp {
            register_replicated_module(planner, &mtp.pre_fc_norm_hidden, "mtp.pre_fc_norm_hidden")?;
            register_replicated_module(
                planner,
                &mtp.pre_fc_norm_embedding,
                "mtp.pre_fc_norm_embedding",
            )?;
            register_replicated_module(planner, &mtp.fc, "mtp.fc")?;
            register_replicated_module(planner, &mtp.norm, "mtp.norm")?;
            for (index, layer) in mtp.layers.iter().enumerate() {
                register_qwen_hybrid_layer_parallel_plan_at(
                    planner,
                    layer,
                    &format!("mtp.layers.{index}"),
                    &self.args,
                )?;
            }
        }
        if let Some(vision) = &self.vision {
            for group in vision_parallel_parameter_groups(&vision.config, "visual", stream)? {
                planner.register(group)?;
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
        let local_semantic = |target: &str, global: i32| -> Result<i32, Error> {
            let tensor = layout.tensor(target).ok_or_else(|| {
                Error::Parallel(format!("missing Qwen hybrid TP layout for {target}"))
            })?;
            let units = tensor.logical_units().ok_or_else(|| {
                Error::Parallel(format!(
                    "Qwen hybrid TP layout for {target} has no logical domain"
                ))
            })?;
            let range = tensor.logical_range().ok_or_else(|| {
                Error::Parallel(format!(
                    "Qwen hybrid TP layout for {target} has no logical range"
                ))
            })?;
            let global = usize::try_from(global).map_err(|_| {
                Error::Parallel(format!("Qwen hybrid {target} geometry is invalid"))
            })?;
            if units == 0 || !global.is_multiple_of(units) {
                return Err(Error::Parallel(format!(
                    "Qwen hybrid {target} global geometry {global} is incompatible with {units} planner units"
                )));
            }
            let local = range
                .len()
                .checked_mul(global / units)
                .ok_or_else(|| Error::Parallel(format!("Qwen hybrid local {target} overflowed")))?;
            i32::try_from(local)
                .map_err(|_| Error::Parallel(format!("Qwen hybrid local {target} exceeds i32")))
        };
        let mut geometry = Vec::with_capacity(self.args.layer_schedule.len());
        for (index, policy) in self.args.layer_schedule.iter().copied().enumerate() {
            let prefix = format!("model.layers.{index}");
            let attention = match policy {
                LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
                    let target = format!("{prefix}.self_attn.q_proj.weight");
                    resident::ParallelAttentionGeometry::Full {
                        query_heads: local_semantic(&target, self.args.num_attention_heads)?,
                        kv_heads: local_semantic(&target, self.args.num_key_value_heads)?,
                    }
                }
                LayerPolicy::LinearAttention => {
                    let target = format!("{prefix}.linear_attn.in_proj_qkv.weight");
                    resident::ParallelAttentionGeometry::Linear {
                        key_heads: local_semantic(&target, self.args.linear_num_key_heads)?,
                        value_heads: local_semantic(&target, self.args.linear_num_value_heads)?,
                    }
                }
                LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                    return Err(Error::Parallel(
                        "Qwen hybrid TP does not support sliding self-attention".into(),
                    ));
                }
            };
            let feed_forward = if self.args.is_moe() {
                let routed = format!("{prefix}.mlp.experts.gate_up_proj");
                let shared = format!("{prefix}.mlp.shared_expert.gate_proj.weight");
                resident::ParallelFeedForwardGeometry::Moe {
                    routed_intermediate: local_semantic(&routed, self.args.moe_intermediate_size)?,
                    shared_intermediate: local_semantic(
                        &shared,
                        self.args.shared_expert_intermediate_size,
                    )?,
                }
            } else {
                let target = format!("{prefix}.mlp.gate_proj.weight");
                resident::ParallelFeedForwardGeometry::Dense {
                    intermediate: local_semantic(&target, self.args.intermediate_size)?,
                }
            };
            geometry.push(resident::ParallelLayerGeometry {
                attention,
                feed_forward,
            });
        }
        if let Some(mtp) = &mut self.mtp {
            for index in 0..mtp.layers.len() {
                let prefix = format!("mtp.layers.{index}");
                let query = format!("{prefix}.self_attn.q_proj.weight");
                let attention = resident::ParallelAttentionGeometry::Full {
                    query_heads: local_semantic(&query, self.args.num_attention_heads)?,
                    kv_heads: local_semantic(&query, self.args.num_key_value_heads)?,
                };
                let feed_forward = if self.args.is_moe() {
                    let routed = format!("{prefix}.mlp.experts.gate_up_proj");
                    let shared = format!("{prefix}.mlp.shared_expert.gate_proj.weight");
                    resident::ParallelFeedForwardGeometry::Moe {
                        routed_intermediate: local_semantic(
                            &routed,
                            self.args.moe_intermediate_size,
                        )?,
                        shared_intermediate: local_semantic(
                            &shared,
                            self.args.shared_expert_intermediate_size,
                        )?,
                    }
                } else {
                    let dense = format!("{prefix}.mlp.gate_proj.weight");
                    resident::ParallelFeedForwardGeometry::Dense {
                        intermediate: local_semantic(&dense, self.args.intermediate_size)?,
                    }
                };
                mtp.layers[index] = TransformerBlock::new_mtp_parallel_layerwise(
                    &self.args,
                    index,
                    resident::ParallelLayerGeometry {
                        attention,
                        feed_forward,
                    },
                    stream,
                )?;
            }
        }
        self.parallel_geometry = Some(geometry);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            self.args.vocab_size as usize,
            self.args.hidden_size,
            self.args.quantization,
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.quantization,
                context,
                stream,
            )?);
        }
        if let Some(vision) = &mut self.vision {
            configure_vision_parallel_static(vision, "visual", layout, stream)?;
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
        if self.execution_group_name(group)? == "vision_encoder" {
            let vision = &self.vision.as_ref().expect("vision group").config;
            return Ok(QwenHybridLayer::Vision(Box::new(
                new_parallel_vision_block(vision, "visual", index, layout, stream)?,
            )));
        }
        let _ = layout;
        let geometry = self
            .parallel_geometry
            .as_ref()
            .and_then(|geometry| geometry.get(index))
            .copied()
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "Qwen hybrid local geometry is unavailable for layer {index}"
                ))
            })?;
        Ok(QwenHybridLayer::Text(Box::new(
            TransformerBlock::new_parallel_layerwise(&self.args, index, geometry, stream)?,
        )))
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if self.execution_group_name(group).ok() == Some("vision_encoder") {
            format!("visual.blocks.{index}")
        } else {
            format!("model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if self.execution_group_name(group).ok() == Some("vision_encoder") {
            format!("qwen_hybrid.vision.{index:05}")
        } else {
            format!("qwen_hybrid.layer.{index:05}")
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        let bindings = self
            .binding_plan_for_module(
                layer,
                &prefix,
                store,
                (self.execution_group_name(group)? == "text_decoder").then_some(index),
            )?
            .build_bindings(store)?;
        Ok(
            if self.sparse_expert_cache && self.execution_group_name(group)? == "text_decoder" {
                bindings
                    .into_iter()
                    .filter(|binding| !binding.name().starts_with("mlp.experts."))
                    .collect()
            } else {
                bindings
            },
        )
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
        if self.execution_group_name(group)? == "vision_encoder" {
            return self.layer_bindings(group, index, &global, store);
        }
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
        match (self.execution_group_name(group)?, layer) {
            ("vision_encoder", QwenHybridLayer::Vision(block)) => {
                let vision = self.vision.as_mut().expect("vision group");
                for job in &mut context.vision_jobs {
                    job.hidden = vision.forward_block(
                        block,
                        index,
                        job.hidden.clone(),
                        &job.state,
                        stream,
                    )?;
                    vision.capture_deepstack(index, &job.hidden, &mut job.state, stream)?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("text_decoder", QwenHybridLayer::Text(block)) if self.sparse_expert_cache => {
                let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Qwen hybrid sparse expert cache was not initialized".into(),
                    )
                })?;
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                Ok(block.forward_sparse_experts(
                    BlockInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: Some(&mut cache.layers[index]),
                    },
                    stream,
                    |flat, indices, weights, stream| {
                        expert_cache
                            .execute_routes_bounded(
                                ExpertRouteBatch::new(index, flat, indices, weights, pass),
                                stream,
                                |flat, acquired, weights, stream| {
                                    let started = Instant::now();
                                    let mut compact_args = self.args.clone();
                                    compact_args.num_experts = acquired.identities().len() as i32;
                                    if let Some(quantization) = expert_cache.weight_quantization() {
                                        compact_args.quantization = Some(quantization);
                                        compact_args.quantization_config = None;
                                        compact_args.quantized_weight_configs = None;
                                    }
                                    let mut bank = Experts::new(&compact_args, index, stream)?;
                                    bank.gate_up_proj = Param::new(
                                        acquired.compact_binding("gate_up_proj", stream).map_err(
                                            |error| Exception::custom(error.to_string()),
                                        )?,
                                    );
                                    bank.gate_up_proj_scale_inv = Param::new(
                                        acquired
                                            .optional_compact_binding(
                                                "gate_up_proj_scale_inv",
                                                stream,
                                            )
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    bank.gate_up_proj_scales = Param::new(
                                        acquired
                                            .optional_compact_binding("gate_up_proj_scales", stream)
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    bank.gate_up_proj_biases = Param::new(
                                        acquired
                                            .optional_compact_binding("gate_up_proj_biases", stream)
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    bank.down_proj = Param::new(
                                        acquired.compact_binding("down_proj", stream).map_err(
                                            |error| Exception::custom(error.to_string()),
                                        )?,
                                    );
                                    bank.down_proj_scale_inv = Param::new(
                                        acquired
                                            .optional_compact_binding("down_proj_scale_inv", stream)
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    bank.down_proj_scales = Param::new(
                                        acquired
                                            .optional_compact_binding("down_proj_scales", stream)
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    bank.down_proj_biases = Param::new(
                                        acquired
                                            .optional_compact_binding("down_proj_biases", stream)
                                            .map_err(|error| {
                                                Exception::custom(error.to_string())
                                            })?,
                                    );
                                    expert_cache.record_compact_bank(
                                        pass,
                                        acquired.scratch_bytes(),
                                        started.elapsed(),
                                    )?;
                                    Ok(bank.forward_chunked(
                                        flat,
                                        acquired.compact_routes(),
                                        weights,
                                        stream,
                                    )?)
                                },
                            )
                            .map_err(|error| Exception::custom(error.to_string()))
                    },
                )?)
            }
            ("text_decoder", QwenHybridLayer::Text(block)) => Ok(block.forward(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                stream,
            )?),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen hybrid execution unit does not match group {group}"
            ))),
        }
    }

    fn forward_layer_with_observer<
        O: crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    >(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Self::Layer,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
        observer: &mut O,
    ) -> Result<Array, Error> {
        let prefix = self.layer_checkpoint_prefix(group, index);
        if !self.sparse_expert_cache
            && self.execution_group_name(group)? == "text_decoder"
            && matches!(layer, QwenHybridLayer::Text(_))
        {
            let QwenHybridLayer::Text(block) = layer else {
                unreachable!()
            };
            return Ok(block.forward_with_observer(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                stream,
                &prefix,
                observer,
            )?);
        }
        observer.observe(&format!("{prefix}.input"), hidden)?;
        let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
        observer.observe(&format!("{prefix}.output"), &output)?;
        Ok(observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output))
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
        match (self.execution_group_name(group)?, layer) {
            ("text_decoder", QwenHybridLayer::Text(block)) => Ok(block.forward_tensor_parallel(
                BlockInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: Some(&mut cache.layers[index]),
                },
                tp_group,
                execution.stream(),
            )?),
            _ => Err(Error::Parallel(
                "token-only Qwen TP received a non-text execution unit".into(),
            )),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if self.execution_group_name(group).ok() == Some("text_decoder") {
            cache.layers[index].retained_arrays()
        } else {
            Vec::new()
        }
    }

    fn retained_context_arrays<'a>(
        &self,
        context: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Vec<&'a Array> {
        context
            .vision_jobs
            .iter()
            .flat_map(|job| std::iter::once(&job.hidden).chain(job.state.retained_arrays()))
            .collect()
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let group_name = self.execution_group_name(group)?;
        let hidden = match dependency_outputs {
            [] => initial_hidden,
            [dependency] => dependency,
            _ => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen hybrid execution group {group_name} received {} dependency outputs",
                    dependency_outputs.len()
                )))
            }
        };
        let should_assemble = context.needs_assembly && group_name == "text_decoder";
        if !should_assemble {
            return Ok(hidden.clone());
        }
        if let Some(vision) = &mut self.vision {
            for job in &mut context.vision_jobs {
                let output = vision.finish(&job.hidden, &mut job.state, stream)?;
                job.hidden = output.embeddings;
            }
        }
        let assembled = context
            .parts
            .iter()
            .map(|part| match part {
                QwenHybridPreparedPart::Ready(value) => value,
                QwenHybridPreparedPart::Vision(job) => &context.vision_jobs[*job].hidden,
            })
            .collect::<Vec<_>>();
        let hidden = concatenate_axis(&assembled, 1, stream)?;
        context.mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "Qwen hybrid decoder requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        context.needs_assembly = false;
        Ok(hidden)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &Array,
        _cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        if self.execution_group_name(group)? == "text_decoder" {
            context.draft_hidden = Some(hidden.clone());
        }
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
            &mut self.embedding,
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

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        self.vision.is_none()
            && (key.starts_with("visual.")
                || key.starts_with("vision_tower.")
                || key.starts_with("model.visual.")
                || key.starts_with("model.vision_tower."))
    }
}

/// Shared Qwen hybrid token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, QwenHybridLayerwiseModel, Cache, S>;
