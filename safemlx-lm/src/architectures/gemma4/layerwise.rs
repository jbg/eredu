//! Text-decoder bounded layer execution for Gemma 4 checkpoints.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
    path::Path,
    sync::Arc,
    time::Instant,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        concatenate_axis, indexing::TryIndexOp, r#where, tanh, GgufCheckpoint, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    transforms::async_eval_with_event,
    Array, Stream,
};

use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheTopology,
};

use crate::{
    api::{
        common::generation::CausalLm,
        gemma4::{
            self as resident, AttentionInput, Cache, Gemma4Embedding, Gemma4TextModel, ModelArgs,
            TransformerBlock,
        },
        gemma4_audio::{
            AudioLayer, Gemma4AudioConfig, Gemma4AudioLayerwiseStatic, Gemma4AudioTower,
        },
        gemma4_multimodal::{Gemma4ClippedLinear, Gemma4ModalityEmbedder},
        gemma4_vision::{
            Gemma4VisionConfig, Gemma4VisionLayerwiseState, Gemma4VisionLayerwiseStatic,
            Gemma4VisionTower, VisionLayer,
        },
        input,
    },
    error::Error,
    nn::{
        parallel::{LinearParallelism, ParallelLinear, VocabParallelLmHead},
        tensor::create_causal_mask,
    },
    runtime::attention::AttentionPolicy,
    runtime::cache::{residency::PagedCacheOptions, KeyValueCache},
    runtime::checkpoint::binding::{
        build_module_bindings_with_recipes, canonical_checkpoint_name, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    runtime::checkpoint::{
        quantization::WeightQuantization,
        recipe::DerivedWeightRecipe,
        store::{GgufWeightStore, TensorSelection, WeightStore},
    },
    runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, partitioned_projection_members,
        register_partitioned_projection_group, register_projection_module,
        register_replicated_module, MemberSharding, ParallelPlanBuilder, ParameterGroupSpec,
        ParameterMemberSpec, ParameterRole, ProjectionSharding,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_safetensors_layerwise_model, load_tensor_parallel_layerwise_model,
        open_safetensors_weight_store, transformed_module_weight_store, ArchitectureAdapter,
        LayerWeightResidency, LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter,
        StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        AcquiredExperts, ExpertCache, ExpertCacheError, ExpertCacheReport, ExpertCatalogEntry,
        ExpertIdentity, ExpertPass, ExpertRouteBatch,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "gemma4.static.embedding";
const PER_LAYER_EMBEDDING_UNIT: &str = "gemma4.static.per_layer_embedding";
const PER_LAYER_PROJECTION_UNIT: &str = "gemma4.static.per_layer_projection";
const PER_LAYER_NORM_UNIT: &str = "gemma4.static.per_layer_norm";
const NORM_UNIT: &str = "gemma4.static.norm";
const HEAD_UNIT: &str = "gemma4.static.output";
const VISION_STATIC_UNIT: &str = "gemma4.static.vision";
const VISION_EMBED_UNIT: &str = "gemma4.static.vision_embed";
const AUDIO_STATIC_UNIT: &str = "gemma4.static.audio";
const AUDIO_EMBED_UNIT: &str = "gemma4.static.audio_embed";

fn gemma_clipped_projection_members(
    projections: &[(&Gemma4ClippedLinear, &str, ProjectionSharding)],
    preferred_units: usize,
) -> Result<(usize, Vec<ParameterMemberSpec>), Error> {
    let owned = projections
        .iter()
        .map(|(projection, prefix, sharding)| {
            (&projection.linear, format!("{prefix}.linear"), *sharding)
        })
        .collect::<Vec<_>>();
    let linear = owned
        .iter()
        .map(|(projection, prefix, sharding)| (*projection, prefix.as_str(), *sharding))
        .collect::<Vec<_>>();
    let (units, mut members) = partitioned_projection_members(&linear, preferred_units)?;
    for (projection, prefix, _) in projections {
        for (name, value) in [
            ("input_min", projection.input_min.as_ref()),
            ("input_max", projection.input_max.as_ref()),
            ("output_min", projection.output_min.as_ref()),
            ("output_max", projection.output_max.as_ref()),
        ] {
            members.push(array_parameter_member(
                format!("{prefix}.{name}"),
                value,
                MemberSharding::Replicated,
            )?);
        }
    }
    Ok((units, members))
}

fn register_gemma_clipped_projection_group(
    planner: &mut ParallelPlanBuilder,
    logical_name: &str,
    role: ParameterRole,
    projections: &[(&Gemma4ClippedLinear, &str, ProjectionSharding)],
    preferred_units: usize,
) -> Result<(), Error> {
    let (units, members) = gemma_clipped_projection_members(projections, preferred_units)?;
    planner.register(ParameterGroupSpec::partitioned(
        logical_name,
        role,
        units,
        members,
    )?)
}

fn register_gemma_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &TransformerBlock,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    let q = format!("{prefix}.self_attn.q_proj");
    let o = format!("{prefix}.self_attn.o_proj");
    let mut projection_names = vec![
        (&attention.q_proj, q, ProjectionSharding::Column),
        (&attention.o_proj, o, ProjectionSharding::Row),
    ];
    if let Some(projection) = &attention.k_proj {
        projection_names.push((
            projection,
            format!("{prefix}.self_attn.k_proj"),
            ProjectionSharding::Column,
        ));
    }
    if let Some(projection) = &attention.v_proj {
        projection_names.push((
            projection,
            format!("{prefix}.self_attn.v_proj"),
            ProjectionSharding::Column,
        ));
    }
    let projections = projection_names
        .iter()
        .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
        .collect::<Vec<_>>();
    register_partitioned_projection_group(
        planner,
        &format!("{prefix}.self_attn.heads"),
        ParameterRole::AttentionHeads,
        &projections,
        layer.layer_policy.num_key_value_heads.get() as usize,
    )?;
    register_replicated_module(
        planner,
        &attention.q_norm,
        &format!("{prefix}.self_attn.q_norm"),
    )?;
    if let Some(norm) = &attention.k_norm {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.k_norm"))?;
    }
    register_replicated_module(
        planner,
        &attention.rope,
        &format!("{prefix}.self_attn.rope"),
    )?;
    let gate = format!("{prefix}.mlp.gate_proj");
    let up = format!("{prefix}.mlp.up_proj");
    let down = format!("{prefix}.mlp.down_proj");
    register_partitioned_projection_group(
        planner,
        &format!("{prefix}.mlp.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (
                &layer.mlp.gate_proj,
                gate.as_str(),
                ProjectionSharding::Column,
            ),
            (&layer.mlp.up_proj, up.as_str(), ProjectionSharding::Column),
            (&layer.mlp.down_proj, down.as_str(), ProjectionSharding::Row),
        ],
        layer.mlp.hidden_dim as usize,
    )?;
    if let Some(router) = &layer.router {
        register_replicated_module(planner, router, &format!("{prefix}.router"))?;
    }
    if let Some(experts) = &layer.experts {
        let expert_prefix = format!("{prefix}.experts.switch_glu");
        let intermediate = experts.switch_glu.gate_proj.output_dim as usize;
        let down = &experts.switch_glu.down_proj;
        let alignment = down
            .quantization
            .or(down.iquant)
            .map_or(Ok(1usize), |quantization| {
                usize::try_from(quantization.group_size()).map_err(|_| {
                    Error::Parallel("Gemma expert quantization group exceeds usize".into())
                })
            })?;
        let units = aligned_partition_units(
            &format!("{expert_prefix}.intermediate"),
            intermediate,
            1,
            alignment,
        )?;
        let mut members = Vec::new();
        for (name, projection, axis) in [
            ("gate_proj", &experts.switch_glu.gate_proj, 1usize),
            ("up_proj", &experts.switch_glu.up_proj, 1usize),
            ("down_proj", &experts.switch_glu.down_proj, 2usize),
        ] {
            members.push(array_parameter_member(
                format!("{expert_prefix}.{name}.weight"),
                projection.weight.as_ref(),
                MemberSharding::Partitioned { axis },
            )?);
            for (companion, value) in [
                ("scales", projection.scales.as_ref().as_ref()),
                ("biases", projection.biases.as_ref().as_ref()),
            ] {
                if let Some(value) = value {
                    members.push(array_parameter_member(
                        format!("{expert_prefix}.{name}.{companion}"),
                        value,
                        MemberSharding::Partitioned { axis },
                    )?);
                }
            }
        }
        planner.register(ParameterGroupSpec::partitioned(
            format!("{expert_prefix}.intermediate"),
            ParameterRole::ExpertIntermediate,
            units,
            members,
        )?)?;
    }
    if let Some(projection) = &layer.per_layer_input_gate {
        register_projection_module(
            planner,
            projection,
            &format!("{prefix}.per_layer_input_gate"),
            ProjectionSharding::Replicated,
        )?;
    }
    if let Some(projection) = &layer.per_layer_projection {
        register_projection_module(
            planner,
            projection,
            &format!("{prefix}.per_layer_projection"),
            ProjectionSharding::Replicated,
        )?;
    }
    for (name, norm) in [
        ("input_layernorm", Some(&layer.input_layernorm)),
        (
            "post_attention_layernorm",
            Some(&layer.post_attention_layernorm),
        ),
        (
            "pre_feedforward_layernorm",
            Some(&layer.pre_feedforward_layernorm),
        ),
        (
            "post_feedforward_layernorm",
            Some(&layer.post_feedforward_layernorm),
        ),
        (
            "post_per_layer_input_norm",
            layer.post_per_layer_input_norm.as_ref(),
        ),
        (
            "post_feedforward_layernorm_1",
            layer.post_feedforward_layernorm_1.as_ref(),
        ),
        (
            "pre_feedforward_layernorm_2",
            layer.pre_feedforward_layernorm_2.as_ref(),
        ),
        (
            "post_feedforward_layernorm_2",
            layer.post_feedforward_layernorm_2.as_ref(),
        ),
    ] {
        if let Some(norm) = norm {
            register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
        }
    }
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.layer_scalar"),
        ParameterRole::Replicated,
        [ParameterMemberSpec::new(
            format!("{prefix}.layer_scalar"),
            [1],
            MemberSharding::Replicated,
        )],
    )?)?;
    Ok(())
}

fn register_gemma_vision_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &VisionLayer,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    let q = format!("{prefix}.self_attn.q_proj");
    let k = format!("{prefix}.self_attn.k_proj");
    let v = format!("{prefix}.self_attn.v_proj");
    let o = format!("{prefix}.self_attn.o_proj");
    register_gemma_clipped_projection_group(
        planner,
        &format!("{prefix}.self_attn.heads"),
        ParameterRole::AttentionHeads,
        &[
            (&attention.q_proj, q.as_str(), ProjectionSharding::Column),
            (&attention.k_proj, k.as_str(), ProjectionSharding::Column),
            (&attention.v_proj, v.as_str(), ProjectionSharding::Column),
            (&attention.o_proj, o.as_str(), ProjectionSharding::Row),
        ],
        attention.num_kv_heads as usize,
    )?;
    for (name, norm) in [("q_norm", &attention.q_norm), ("k_norm", &attention.k_norm)] {
        register_replicated_module(planner, norm, &format!("{prefix}.self_attn.{name}"))?;
    }
    let mlp = &layer.mlp;
    let gate = format!("{prefix}.mlp.gate_proj");
    let up = format!("{prefix}.mlp.up_proj");
    let down = format!("{prefix}.mlp.down_proj");
    register_gemma_clipped_projection_group(
        planner,
        &format!("{prefix}.mlp.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (&mlp.gate_proj, gate.as_str(), ProjectionSharding::Column),
            (&mlp.up_proj, up.as_str(), ProjectionSharding::Column),
            (&mlp.down_proj, down.as_str(), ProjectionSharding::Row),
        ],
        mlp.gate_proj.output_dim as usize,
    )?;
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
        (
            "pre_feedforward_layernorm",
            &layer.pre_feedforward_layernorm,
        ),
        (
            "post_feedforward_layernorm",
            &layer.post_feedforward_layernorm,
        ),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    Ok(())
}

fn register_gemma_audio_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &AudioLayer,
    prefix: &str,
) -> Result<(), Error> {
    let attention = &layer.self_attn;
    let names = [
        ("q_proj", &attention.q_proj, ProjectionSharding::Column),
        ("k_proj", &attention.k_proj, ProjectionSharding::Column),
        ("v_proj", &attention.v_proj, ProjectionSharding::Column),
        ("post", &attention.post, ProjectionSharding::Row),
    ];
    let owned = names
        .iter()
        .map(|(name, projection, sharding)| {
            (*projection, format!("{prefix}.self_attn.{name}"), *sharding)
        })
        .collect::<Vec<_>>();
    let projections = owned
        .iter()
        .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
        .collect::<Vec<_>>();
    let (units, mut members) =
        gemma_clipped_projection_members(&projections, attention.heads as usize)?;
    let relative = format!("{prefix}.self_attn.relative_k_proj");
    let (units, relative_members) = partitioned_projection_members(
        &[(
            &attention.relative_k_proj,
            relative.as_str(),
            ProjectionSharding::Column,
        )],
        units,
    )?;
    members.extend(relative_members);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.self_attn.heads"),
        ParameterRole::AttentionHeads,
        units,
        members,
    )?)?;
    planner.register(ParameterGroupSpec::new(
        format!("{prefix}.self_attn.per_dim_scale"),
        ParameterRole::Replicated,
        [array_parameter_member(
            format!("{prefix}.self_attn.per_dim_scale"),
            attention.per_dim_scale.as_ref(),
            MemberSharding::Replicated,
        )?],
    )?)?;

    let intermediate = layer.feed_forward1.ffw_layer_1.output_dim as usize;
    let mut ff_names = Vec::new();
    for (block_name, block) in [
        ("feed_forward1", &layer.feed_forward1),
        ("feed_forward2", &layer.feed_forward2),
    ] {
        ff_names.push((
            &block.ffw_layer_1,
            format!("{prefix}.{block_name}.ffw_layer_1"),
            ProjectionSharding::Column,
        ));
        ff_names.push((
            &block.ffw_layer_2,
            format!("{prefix}.{block_name}.ffw_layer_2"),
            ProjectionSharding::Row,
        ));
        register_replicated_module(
            planner,
            &block.pre_layer_norm,
            &format!("{prefix}.{block_name}.pre_layer_norm"),
        )?;
        register_replicated_module(
            planner,
            &block.post_layer_norm,
            &format!("{prefix}.{block_name}.post_layer_norm"),
        )?;
    }
    let ff_projections = ff_names
        .iter()
        .map(|(projection, name, sharding)| (*projection, name.as_str(), *sharding))
        .collect::<Vec<_>>();
    register_gemma_clipped_projection_group(
        planner,
        &format!("{prefix}.feed_forward.intermediate"),
        ParameterRole::FeedForwardIntermediate,
        &ff_projections,
        intermediate,
    )?;

    let convolution = &layer.lconv1d;
    let end = format!("{prefix}.lconv1d.linear_end");
    let (channel_units, mut channel_members) = gemma_clipped_projection_members(
        &[(
            &convolution.linear_end,
            end.as_str(),
            ProjectionSharding::Row,
        )],
        convolution.linear_end.input_dim as usize,
    )?;
    for (name, parameter) in convolution.linear_start.parameters().flatten() {
        let shape = parameter
            .shape()
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let sharding = if name.as_ref() == "linear.weight" {
            let channels = shape[0] / 2;
            MemberSharding::PartitionedSegments {
                axis: 0,
                segments: vec![0..channels, channels..2 * channels],
            }
        } else {
            MemberSharding::Replicated
        };
        channel_members.push(ParameterMemberSpec::new(
            format!("{prefix}.lconv1d.linear_start.{name}"),
            shape,
            sharding,
        ));
    }
    channel_members.push(array_parameter_member(
        format!("{prefix}.lconv1d.depthwise_conv1d.weight"),
        convolution.depthwise_conv1d.weight.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    channel_members.push(array_parameter_member(
        format!("{prefix}.lconv1d.conv_norm.weight"),
        convolution.conv_norm.weight.as_ref(),
        MemberSharding::Partitioned { axis: 0 },
    )?);
    planner.register(ParameterGroupSpec::partitioned(
        format!("{prefix}.lconv1d.channels"),
        ParameterRole::Channels,
        channel_units,
        channel_members,
    )?)?;
    register_replicated_module(
        planner,
        &convolution.pre_layer_norm,
        &format!("{prefix}.lconv1d.pre_layer_norm"),
    )?;
    for (name, norm) in [
        ("norm_pre_attn", &layer.norm_pre_attn),
        ("norm_post_attn", &layer.norm_post_attn),
        ("norm_out", &layer.norm_out),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    Ok(())
}

/// Gemma 4 multimodal model using bounded residency for media and text blocks.
pub struct Gemma4LayerwiseModel {
    execution: LayerwiseModel<Gemma4LayerwiseAdapter>,
}

impl Gemma4LayerwiseModel {
    /// Returns normalized Gemma 4 text arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    /// Returns canonical parameter and residency metadata.
    pub fn metadata(&self) -> &crate::LayerwiseModelMetadata {
        self.execution.metadata()
    }

    pub(crate) fn bind_parallel_topology(&mut self, topology: crate::MlxParallelContext) {
        self.execution.bind_parallel_topology(topology);
    }

    pub(crate) fn media_accounting(
        &self,
    ) -> (
        Option<&Gemma4VisionConfig>,
        Option<&Gemma4AudioConfig>,
        bool,
        bool,
        bool,
    ) {
        let adapter = self.execution.adapter();
        (
            adapter.vision.as_ref().map(|vision| &vision.config),
            adapter.audio_config.as_ref(),
            adapter.image_token_id.is_some(),
            adapter.audio_token_id.is_some(),
            adapter.video_token_id.is_some(),
        )
    }

    /// Creates an empty Gemma 4 generation cache.
    pub fn new_cache(&self) -> Cache {
        let mut cache = Cache::new(self.args());
        cache.rank = self.execution.prompt_cache_rank_identity();
        cache
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    pub(crate) fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
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
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Returns independent routed-expert cache telemetry when enabled.
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
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Runs the text decoder while preserving alternating and shared KV state.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Gemma4Input::Decode(inputs), cache, stream)
    }

    /// Runs text decode through the canonical observer contract.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution
            .forward_with_observer(Gemma4Input::Decode(inputs), cache, stream, observer)
    }

    /// Runs typed prefill through the canonical observer contract.
    pub fn prefill_input_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution
            .forward_with_observer(Gemma4Input::Prefill(input), cache, stream, observer)
    }

    /// Runs multimodal prefill through rank-local vision and audio groups.
    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(Gemma4Input::Prefill(input), cache, group, stream)
    }

    /// Runs decode on a TP-loaded Gemma multimodal model.
    pub(crate) fn decode_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(Gemma4Input::Decode(tokens), cache, group, stream)
    }

    /// Runs text decode while delegating routed experts to a topology-scoped executor.
    pub(crate) fn decode_with_expert_executor<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.forward_with_expert_executor_input(
            Gemma4Input::Decode(tokens),
            cache,
            &mut execute,
            stream,
        )
    }

    fn forward_with_expert_executor_input<F>(
        &mut self,
        input: Gemma4Input<'_>,
        cache: &mut Cache,
        execute: &mut F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_with_layer_executor(
            input,
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| {
                if adapter.execution_group_name(group)? != "text_decoder" {
                    return adapter
                        .forward_layer(group, index, layer, hidden, cache, context, stream);
                }
                let Gemma4Layer::Text(layer) = layer else {
                    return Err(Error::Parallel(format!(
                        "Gemma 4 external-expert unit does not match text layer {index}"
                    )));
                };
                let per_layer_input = context
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                    .transpose()?;
                let mask = context
                    .sliding_masks
                    .as_ref()
                    .and_then(|masks| masks.get(&layer.layer_policy.attention))
                    .or(context.mask.as_ref());
                Ok(layer.forward_with_expert_executor(
                    AttentionInput {
                        x: hidden,
                        mask,
                        cache: cache.kv[index].as_mut(),
                        position_offset: context.position_offset,
                        per_layer_input: per_layer_input.as_ref(),
                        shared_kv: Some(&mut context.shared_kv),
                        disable_generated_mask: false,
                        generated_sliding_window: None,
                    },
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Runs TP-sharded decode while delegating routed experts to EP.
    pub(crate) fn decode_tensor_expert_parallel<F>(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        tensor_group: &safemlx::distributed::Group,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Error>
    where
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        self.execution.forward_tensor_parallel_with_layer_executor(
            Gemma4Input::Decode(tokens),
            cache,
            tensor_group,
            stream,
            |adapter, group, index, layer, hidden, cache, context, execution| {
                if adapter.execution_group_name(group)? != "text_decoder" {
                    return adapter.forward_layer_with_execution(
                        group, index, layer, hidden, cache, context, execution,
                    );
                }
                let Gemma4Layer::Text(layer) = layer else {
                    return Err(Error::Parallel(format!(
                        "Gemma 4 TP+EP unit does not match text layer {index}"
                    )));
                };
                let stream = execution.stream();
                let per_layer_input = context
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                    .transpose()?;
                let mask = context
                    .sliding_masks
                    .as_ref()
                    .and_then(|masks| masks.get(&layer.layer_policy.attention))
                    .or(context.mask.as_ref());
                let group = execution.group().ok_or_else(|| {
                    Error::Parallel("Gemma 4 TP+EP execution has no TP subgroup".into())
                })?;
                Ok(layer.forward_tensor_with_expert_executor(
                    AttentionInput {
                        x: hidden,
                        mask,
                        cache: cache.kv[index].as_mut(),
                        position_offset: context.position_offset,
                        per_layer_input: per_layer_input.as_ref(),
                        shared_kv: Some(&mut context.shared_kv),
                        disable_generated_mask: false,
                        generated_sliding_window: None,
                    },
                    group,
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    pub(crate) fn prefill_mtp(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        self.forward_mtp(Gemma4Input::Prefill(input), cache, stream)
    }

    pub(crate) fn verify_mtp(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        self.forward_mtp(Gemma4Input::Decode(tokens), cache, stream)
    }

    fn forward_mtp(
        &mut self,
        input: Gemma4Input<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<resident::Gemma4StepOutput, Exception> {
        let (logits, context) = self
            .execution
            .forward_with_context_hook(input, cache, stream, |_, _, _| Ok(()))
            .map_err(|error| Exception::custom(error.to_string()))?;
        let hidden = context.draft_hidden.ok_or_else(|| {
            Exception::custom("Gemma 4 layerwise pass did not retain target draft state")
        })?;
        Ok(resident::Gemma4StepOutput {
            logits,
            hidden,
            shared_kv_states: context.shared_kv,
        })
    }

    pub(crate) fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        self.execution
            .adapter()
            .mtp_embedding_snapshot(stream, copy)
    }

    /// Clears temporary media and decoder blocks from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_all_device_groups()
    }
}

impl CausalLm<Cache> for Gemma4LayerwiseModel {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.execution
            .forward(Gemma4Input::Prefill(input), cache, stream)
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

/// Loads Gemma 4 text and configured media towers through generalized residency.
pub fn load_gemma4_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    crate::backend::mlx::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Gemma4,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let (args, vision, image_token_id, video_token_id, audio, audio_token_id) =
        resident::get_gemma4_model_config(model_dir)?;
    let adapter = Gemma4LayerwiseAdapter::new(
        args,
        vision,
        image_token_id,
        video_token_id,
        audio,
        audio_token_id,
        stream,
    )?;
    Ok(Gemma4LayerwiseModel {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn execute_transformed_gemma4_model(
    model_dir: &Path,
    model: resident::Model,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let (_, vision, _, _, audio, _) = resident::get_gemma4_model_config(model_dir)?;
    execute_transformed_gemma4_model_with_modalities(model, vision, audio, stream, weights_stream)
}

pub(crate) fn execute_transformed_gemma4_model_with_modalities(
    model: resident::Model,
    vision: Option<Gemma4VisionConfig>,
    audio: Option<Gemma4AudioConfig>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let adapter = Gemma4LayerwiseAdapter::new(
        model.args.clone(),
        vision,
        model.image_token_id,
        model.video_token_id,
        audio,
        model.audio_token_id,
        stream,
    )?;
    let store = transformed_module_weight_store(&model)?;
    Ok(Gemma4LayerwiseModel {
        execution: load_layerwise_model(
            store,
            adapter,
            LayerWeightResidency::FullyResident,
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Gemma 4 with rank-local vision and audio execution groups.
pub(crate) fn load_gemma4_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let mmproj = resident::open_sibling_mmproj(model_dir)?;
        return load_gemma4_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            mmproj.as_ref(),
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    crate::backend::mlx::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Gemma4,
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let (args, vision, image_token_id, video_token_id, audio, audio_token_id) =
        resident::get_gemma4_model_config(model_dir)?;
    let adapter = Gemma4LayerwiseAdapter::new(
        args,
        vision,
        image_token_id,
        video_token_id,
        audio,
        audio_token_id,
        stream,
    )?;
    Ok(Gemma4LayerwiseModel {
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

pub(crate) fn load_gemma4_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::Gemma4MmprojGguf>,
    options: LayerWeightResidency,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Gemma4LayerwiseModel, Vec<u32>), Error> {
    let residency = options.weight_residency();
    crate::backend::mlx::structural::validate_gguf(
        crate::api::GgufArchitecture::Gemma4,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gemma4_gguf_checkpoint(checkpoint, metadata, mmproj, None)?;
    let store = gemma4_gguf_store(checkpoint, mmproj, options.max_mapped_shards())?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        Gemma4LayerwiseAdapter::new(
            prepared.args,
            prepared.vision_config,
            prepared.image_token_id,
            prepared.video_token_id,
            prepared.audio_config,
            prepared.audio_token_id,
            stream,
        )?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((Gemma4LayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn load_gemma4_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: Option<&resident::Gemma4MmprojGguf>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Gemma4LayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::structural::validate_gguf(
        crate::api::GgufArchitecture::Gemma4,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )
    .into_loader_result()?;
    let prepared = resident::prepare_gemma4_gguf_checkpoint(checkpoint, metadata, mmproj, None)?;
    let has_routed_experts = prepared.args.num_experts.is_some();
    let adapter = Gemma4LayerwiseAdapter::new(
        prepared.args,
        prepared.vision_config,
        prepared.image_token_id,
        prepared.video_token_id,
        prepared.audio_config,
        prepared.audio_token_id,
        stream,
    )?;
    let store = gemma4_gguf_store(checkpoint, mmproj, residency.max_mapped_shards())?;
    if let Some(options) = residency.expert_cache() {
        if !has_routed_experts {
            return Err(Error::UnsupportedArchitecture(
                "independent expert caching requires a Gemma 4 MoE GGUF checkpoint".into(),
            ));
        }
        let mut adapter = adapter;
        adapter.external_experts = true;
        let mut execution = load_layerwise_model_with_quantization(
            store,
            adapter,
            residency.layers(),
            quantization,
            stream,
            weights_stream,
        )?;
        let store = execution.checkpoint_store_arc();
        let entries = gemma4_expert_catalog(execution.adapter().args(), store.as_ref())?;
        execution.adapter_mut().expert_cache = Some(match quantization {
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
        return Ok((Gemma4LayerwiseModel { execution }, prepared.eos_token_ids));
    }
    let execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((Gemma4LayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn gemma4_gguf_store(
    checkpoint: &GgufCheckpoint,
    mmproj: Option<&resident::Gemma4MmprojGguf>,
    max_mapped_shards: usize,
) -> Result<Arc<dyn WeightStore + Send + Sync>, Error> {
    let mut builder = GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(checkpoint.clone(), resident::translate_gguf_weight_name)?;
    if let Some(mmproj) = mmproj {
        builder = builder.add_checkpoint(
            mmproj.checkpoint.clone(),
            resident::translate_mmproj_weight_name,
        )?;
    }
    Ok(Arc::new(builder.build()?))
}

/// Builds the non-expert Gemma execution base used by EP and TP+EP.
pub(crate) fn load_gemma4_sparse_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let adapter = Gemma4LayerwiseAdapter::new_external_experts(args, stream)?;
    Ok(Gemma4LayerwiseModel {
        execution: load_layerwise_model(store, adapter, non_expert.into(), stream, weights_stream)?,
    })
}

/// Builds TP-sharded non-expert Gemma execution with external routed experts.
pub(crate) fn load_gemma4_sparse_tp_ep_base_with_store(
    store: Arc<dyn WeightStore + Send + Sync>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Gemma4LayerwiseModel, Error> {
    let adapter = Gemma4LayerwiseAdapter::new_external_experts(args, stream)?;
    Ok(Gemma4LayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            store,
            adapter,
            non_expert.into(),
            build,
            stream,
            weights_stream,
        )?,
    })
}

/// Returns one independently leasable unit for every Gemma routed expert.
pub(crate) fn gemma4_expert_catalog(
    args: &ModelArgs,
    store: &dyn WeightStore,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    gemma4_expert_catalog_for_layers(args, store, 0..args.num_hidden_layers as usize, None)
}

/// Builds stage-local Gemma expert recipes under an optional TP layout.
pub(crate) fn gemma4_expert_catalog_for_layers(
    args: &ModelArgs,
    store: &dyn WeightStore,
    layers: impl IntoIterator<Item = usize>,
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let global_experts = usize::try_from(args.num_experts.ok_or_else(|| {
        Error::UnsupportedArchitecture("Gemma 4 MoE config has no expert count".into())
    })?)
    .map_err(|_| Error::UnsupportedArchitecture("Gemma 4 expert count is negative".into()))?;
    let intermediate = usize::try_from(args.moe_intermediate_size.ok_or_else(|| {
        Error::UnsupportedArchitecture("Gemma 4 MoE config has no expert width".into())
    })?)
    .map_err(|_| Error::UnsupportedArchitecture("Gemma 4 expert width is negative".into()))?;
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut entries = Vec::new();
    for layer in layers {
        if args.layer_policy(layer).is_none_or(|policy| {
            policy.feed_forward != resident::FeedForwardPolicy::DenseWithSparseMoe
        }) {
            continue;
        }
        let logical_prefix = format!("model.language_model.layers.{layer}.experts.switch_glu");
        let alternate_prefix = format!("language_model.model.layers.{layer}.experts.switch_glu");
        let source_prefix = if keys.iter().any(|key| key.starts_with(&logical_prefix)) {
            logical_prefix.as_str()
        } else if keys.iter().any(|key| key.starts_with(&alternate_prefix)) {
            alternate_prefix.as_str()
        } else {
            logical_prefix.as_str()
        };
        let fused = format!("{source_prefix}.gate_up_proj");
        for expert in 0..global_experts {
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            };
            let mut bindings = Vec::new();
            for (projection, half) in [("gate_proj", Some(0usize)), ("up_proj", Some(1usize))] {
                for suffix in ["weight", "scales", "biases"] {
                    let separate = format!("{source_prefix}.{projection}.{suffix}");
                    let recipe = if keys.contains(&separate) {
                        Some(DerivedWeightRecipe::source(separate, selection.clone()))
                    } else {
                        let source = format!("{fused}.{suffix}");
                        keys.contains(&source).then(|| DerivedWeightRecipe::Select {
                            input: Box::new(DerivedWeightRecipe::source(source, selection.clone())),
                            selection: TensorSelection::Range {
                                axis: 1,
                                start: half.expect("gate/up half") * intermediate,
                                end: (half.expect("gate/up half") + 1) * intermediate,
                            },
                        })
                    };
                    if let Some(recipe) = recipe {
                        bindings.push(gemma_expert_recipe_binding(
                            &format!("{projection}.{suffix}"),
                            recipe,
                            store,
                        )?);
                    } else if suffix == "weight" {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Gemma 4 checkpoint is missing {projection} for expert {expert} in layer {layer}"
                        )));
                    }
                }
            }
            for suffix in ["weight", "scales", "biases"] {
                let source = format!("{source_prefix}.down_proj.{suffix}");
                if keys.contains(&source) {
                    bindings.push(gemma_expert_recipe_binding(
                        &format!("down_proj.{suffix}"),
                        DerivedWeightRecipe::source(source, selection.clone()),
                        store,
                    )?);
                } else if suffix == "weight" {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Gemma 4 checkpoint is missing down_proj for expert {expert} in layer {layer}"
                    )));
                }
            }
            let bindings = match layout {
                Some(layout) => crate::runtime::execution::layerwise::shard_layer_bindings(
                    bindings,
                    &logical_prefix,
                    store,
                    layout,
                )?,
                None => bindings,
            };
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Gemma 4 expert byte total overflowed".into())
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

fn gemma_expert_recipe_binding(
    name: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<WeightBinding, Error> {
    let bytes = recipe.infer(store)?.byte_len();
    Ok(WeightBinding::from_recipe(name, recipe, bytes)?)
}

pub(crate) fn execute_acquired_gemma_experts(
    args: &ModelArgs,
    layer: usize,
    hidden: &Array,
    acquired: &AcquiredExperts,
    _weights: &Array,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Exception> {
    if acquired.is_empty() {
        return Err(Exception::custom(
            ExpertCacheError::EmptyRoutedBank {
                architecture: "Gemma 4",
            }
            .to_string(),
        ));
    }
    let started = Instant::now();
    let intermediate = args
        .moe_intermediate_size
        .ok_or_else(|| Exception::custom("Gemma 4 MoE config has no expert width"))?;
    let mut packed_args = args.clone();
    if let Some(quantization) = cache.weight_quantization() {
        packed_args.quantized = true;
        packed_args.weight_quantization = Some(quantization);
        packed_args.quantization_group_size = quantization.group_size();
        packed_args.quantization_bits = quantization.bits();
        packed_args.quantized_weights = None;
        packed_args.quantized_weight_configs = None;
    }
    let mut bank = resident::GemmaExperts::new(
        &packed_args,
        layer,
        acquired.identities().len() as i32,
        intermediate,
        stream,
    )?;
    for (projection, target) in [
        ("gate_proj", &mut bank.switch_glu.gate_proj),
        ("up_proj", &mut bank.switch_glu.up_proj),
        ("down_proj", &mut bank.switch_glu.down_proj),
    ] {
        target.weight = Param::new(
            acquired
                .compact_binding(&format!("{projection}.weight"), stream)
                .map_err(|error| Exception::custom(error.to_string()))?,
        );
        target.scales = Param::new(
            acquired
                .optional_compact_binding(&format!("{projection}.scales"), stream)
                .map_err(|error| Exception::custom(error.to_string()))?,
        );
        target.biases = Param::new(
            acquired
                .optional_compact_binding(&format!("{projection}.biases"), stream)
                .map_err(|error| Exception::custom(error.to_string()))?,
        );
        target.cache_native_view()?;
    }
    cache
        .record_compact_bank(acquired.pass(), acquired.scratch_bytes(), started.elapsed())
        .map_err(|error| Exception::custom(error.to_string()))?;
    let routes = acquired.compact_routes().reshape(&[-1, 1], stream)?;
    let unit_weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    bank.forward(hidden, &routes, &unit_weights, stream)
}

/// Adapter for Gemma 4 per-layer inputs and shared-KV attention blocks.
pub struct Gemma4LayerwiseAdapter {
    args: ModelArgs,
    embedding: Gemma4Embedding,
    per_layer_embedding: Option<Gemma4Embedding>,
    per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    per_layer_norm: Option<nn::RmsNorm>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    vision: Option<Gemma4VisionLayerwiseStatic>,
    vision_config: Option<Gemma4VisionConfig>,
    embed_vision: Option<Gemma4ModalityEmbedder>,
    audio: Option<Gemma4AudioLayerwiseStatic>,
    embed_audio: Option<Gemma4ModalityEmbedder>,
    vision_depth: usize,
    audio_depth: usize,
    audio_config: Option<Gemma4AudioConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    audio_token_id: Option<i32>,
    parallel_vocabulary: Option<Range<usize>>,
    parallel_per_layer_vocabulary: Option<Range<usize>>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_per_layer_projection: Option<ParallelLinear>,
    parallel_text_geometry: Option<Vec<resident::ParallelLayerGeometry>>,
    external_experts: bool,
    expert_cache: Option<ExpertCache>,
}

impl Gemma4LayerwiseAdapter {
    /// Exports only one execution group's payload, preserving job order.
    pub(crate) fn pipeline_group_ingress_arrays(
        &self,
        group: &str,
        state: &Gemma4PipelineIngressState,
    ) -> Result<Vec<Array>, Error> {
        match group {
            "vision_encoder" => Ok(state
                .forward
                .context
                .vision_jobs
                .iter()
                .map(|job| job.hidden.clone())
                .collect()),
            "audio_encoder" => Ok(state
                .forward
                .context
                .audio_jobs
                .iter()
                .map(|job| job.hidden.clone())
                .collect()),
            _ => Err(Error::Parallel(format!(
                "Gemma has no routed media payload for execution group {group:?}"
            ))),
        }
    }

    /// Imports independent image/audio job activations from the previous PP owner.
    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut Gemma4PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let vision = state.forward.context.vision_jobs.len();
        let audio = state.forward.context.audio_jobs.len();
        if arrays.len() != vision + audio {
            return Err(Error::Parallel(format!(
                "Gemma distributed encoder payload has {} jobs, expected {} image plus {} audio jobs",
                arrays.len(), vision, audio
            )));
        }
        let (vision_arrays, audio_arrays) = arrays.split_at(vision);
        for (job, hidden) in state
            .forward
            .context
            .vision_jobs
            .iter_mut()
            .zip(vision_arrays)
        {
            job.hidden = hidden.clone();
            job.state.working_dtype = hidden.dtype();
        }
        for (job, hidden) in state
            .forward
            .context
            .audio_jobs
            .iter_mut()
            .zip(audio_arrays)
        {
            job.hidden = hidden.clone();
        }
        state.forward.hidden = vision_arrays
            .first()
            .or_else(|| audio_arrays.first())
            .cloned()
            .unwrap_or_else(|| state.forward.hidden.clone());
        Ok(())
    }

    /// Imports only one execution group's payload without disturbing an
    /// independently in-flight sibling root.
    pub(crate) fn replace_pipeline_group_ingress_arrays(
        &self,
        group: &str,
        state: &mut Gemma4PipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        match group {
            "vision_encoder" => {
                if arrays.len() != state.forward.context.vision_jobs.len() {
                    return Err(Error::Parallel(format!(
                        "Gemma vision payload has {} jobs, expected {}",
                        arrays.len(),
                        state.forward.context.vision_jobs.len()
                    )));
                }
                for (job, hidden) in state.forward.context.vision_jobs.iter_mut().zip(arrays) {
                    job.state.working_dtype = hidden.dtype();
                    job.hidden = hidden;
                }
            }
            "audio_encoder" => {
                if arrays.len() != state.forward.context.audio_jobs.len() {
                    return Err(Error::Parallel(format!(
                        "Gemma audio payload has {} jobs, expected {}",
                        arrays.len(),
                        state.forward.context.audio_jobs.len()
                    )));
                }
                for (job, hidden) in state.forward.context.audio_jobs.iter_mut().zip(arrays) {
                    job.hidden = hidden;
                }
            }
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma has no routed media payload for execution group {group:?}"
                )))
            }
        }
        state.forward.hidden = state
            .forward
            .context
            .vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .or_else(|| {
                state
                    .forward
                    .context
                    .audio_jobs
                    .first()
                    .map(|job| job.hidden.clone())
            })
            .unwrap_or_else(|| state.forward.hidden.clone());
        Ok(())
    }

    /// Creates the text-only adapter used by text pipeline stages.
    pub(crate) fn new_text(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        Self::new(args, None, None, None, None, None, stream)
    }

    /// Creates a text adapter whose routed expert payloads live outside layers.
    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new_text(args, stream)?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    /// Creates the semantic adapter used by a Gemma pipeline ingress stage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_pipeline(
        args: ModelArgs,
        vision_config: Option<Gemma4VisionConfig>,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<Gemma4AudioConfig>,
        audio_token_id: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        Self::new(
            args,
            vision_config,
            image_token_id,
            video_token_id,
            audio_config,
            audio_token_id,
            stream,
        )
    }

    /// Creates a multimodal pipeline adapter with independently managed experts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_pipeline_external_experts(
        args: ModelArgs,
        vision_config: Option<Gemma4VisionConfig>,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<Gemma4AudioConfig>,
        audio_token_id: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut adapter = Self::new_pipeline(
            args,
            vision_config,
            image_token_id,
            video_token_id,
            audio_config,
            audio_token_id,
            stream,
        )?;
        adapter.external_experts = true;
        Ok(adapter)
    }

    /// Returns the execution-group coordinates of configured media towers.
    pub(crate) fn pipeline_media_groups(&self) -> Vec<(usize, usize)> {
        let mut groups = Vec::new();
        let mut group = 0;
        if self.vision_depth > 0 {
            groups.push((group, self.vision_depth));
            group += 1;
        }
        if self.audio_depth > 0 {
            groups.push((group, self.audio_depth));
        }
        groups
    }

    /// Returns the unique sliding-window mask order used in pipeline payloads.
    pub(crate) fn pipeline_mask_windows(&self) -> Vec<std::num::NonZeroU32> {
        self.args
            .layer_schedule
            .iter()
            .filter_map(|policy| policy.attention.window())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Selects one configured static target for pipeline-owned materialization.
    pub(crate) fn pipeline_static_mut(&mut self, role: &str) -> Option<&mut dyn ModuleParameters> {
        match role {
            "embedding" => Some(&mut self.embedding),
            "per_layer_embedding" => self
                .per_layer_embedding
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "per_layer_projection" => {
                if let Some(module) = &mut self.parallel_per_layer_projection {
                    Some(module.inner_mut())
                } else {
                    self.per_layer_projection
                        .as_mut()
                        .map(|module| module as &mut dyn ModuleParameters)
                }
            }
            "per_layer_norm" => self
                .per_layer_norm
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "vision" => self
                .vision
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "vision_embed" => self
                .embed_vision
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "audio" => self
                .audio
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            "audio_embed" => self
                .embed_audio
                .as_mut()
                .map(|module| module as &mut dyn ModuleParameters),
            _ => None,
        }
    }

    /// Builds exact direct or derived bindings for one text decoder block.
    pub(crate) fn text_layer_bindings(
        &self,
        index: usize,
        layer: &TransformerBlock,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.bindings(
            layer,
            &format!("model.language_model.layers.{index}"),
            store,
        )
    }

    /// Builds one rank-local text block from the configured semantic TP plan.
    pub(crate) fn new_cartesian_text_layer(
        &self,
        index: usize,
        layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
        assignment: Option<&crate::runtime::distributed::expert::ExpertAssignment>,
        stream: &Stream,
    ) -> Result<TransformerBlock, Error> {
        match self.new_cartesian_layer(
            self.pipeline_text_group(),
            index,
            layout,
            assignment,
            stream,
        )? {
            Gemma4Layer::Text(layer) => Ok(*layer),
            _ => Err(Error::Parallel(format!(
                "Gemma 4 text planner returned a non-text layer at index {index}"
            ))),
        }
    }

    /// Resolves ordinary or TP-local bindings for one text block.
    pub(crate) fn cartesian_text_layer_bindings(
        &self,
        index: usize,
        _layer: &TransformerBlock,
        store: &dyn WeightStore,
        layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_cartesian_text_layer(index, None, None, stream)?;
        let bindings = self.text_layer_bindings(index, &global, store)?;
        let Some(layout) = layout else {
            return Ok(bindings);
        };
        crate::runtime::execution::layerwise::shard_layer_bindings(
            bindings,
            &format!("model.language_model.layers.{index}"),
            store,
            layout,
        )
    }

    fn new(
        args: ModelArgs,
        vision_config: Option<Gemma4VisionConfig>,
        image_token_id: Option<i32>,
        video_token_id: Option<i32>,
        audio_config: Option<Gemma4AudioConfig>,
        audio_token_id: Option<i32>,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let text = Gemma4TextModel::new(&args, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(
                crate::nn::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.quantization_for("lm_head.weight"),
                    stream,
                )?,
            )
        };
        let vision_tower = vision_config
            .clone()
            .map(|config| Gemma4VisionTower::new(config, stream))
            .transpose()?;
        let vision_depth = vision_config
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let vision = vision_tower.map(Gemma4VisionLayerwiseStatic::from_tower);
        let embed_vision = vision_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.hidden_size,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        let audio_tower = audio_config
            .as_ref()
            .map(|config| Gemma4AudioTower::new(config, stream))
            .transpose()?;
        let audio_depth = audio_config
            .as_ref()
            .map_or(0, |config| config.num_hidden_layers as usize);
        let audio = audio_tower.map(Gemma4AudioLayerwiseStatic::from_tower);
        let embed_audio = audio_config
            .as_ref()
            .map(|config| {
                Gemma4ModalityEmbedder::new(
                    config.output_proj_dims,
                    args.hidden_size,
                    config.rms_norm_eps,
                    false,
                    args.weight_quantization(),
                    stream,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            embedding: text.embed_tokens,
            per_layer_embedding: text.embed_tokens_per_layer,
            per_layer_projection: text.per_layer_model_projection,
            per_layer_norm: text.per_layer_projection_norm,
            norm: text.norm,
            lm_head,
            vision,
            vision_config,
            embed_vision,
            audio,
            embed_audio,
            vision_depth,
            audio_depth,
            audio_config,
            image_token_id,
            video_token_id,
            audio_token_id,
            parallel_vocabulary: None,
            parallel_per_layer_vocabulary: None,
            parallel_lm_head: None,
            parallel_per_layer_projection: None,
            parallel_text_geometry: None,
            external_experts: false,
            expert_cache: None,
        })
    }

    /// Returns normalized Gemma 4 text arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Returns the planner-derived text cache geometry after TP configuration.
    pub(crate) fn parallel_text_cache_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        let geometry = self.parallel_text_geometry.as_ref().ok_or_else(|| {
            Error::Parallel("Gemma 4 text TP geometry has not been configured".into())
        })?;
        resident::prompt_cache_layer_layout_with_geometry(&self.args, geometry)
    }

    fn mtp_embedding_snapshot(
        &self,
        stream: &Stream,
        copy: bool,
    ) -> Result<Gemma4Embedding, Exception> {
        let mut embedding = self.embedding.clone();
        if copy {
            async_eval_with_event(embedding.materialization_arrays())?.synchronize()?;
            embedding.copy_to_stream(stream)?;
            embedding.native = embedding
                .native
                .as_ref()
                .map(|native| native.copy_to_stream(stream))
                .transpose()?;
            async_eval_with_event(embedding.materialization_arrays())?.synchronize()?;
        }
        Ok(embedding)
    }

    fn recipes_for(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> BTreeMap<String, DerivedWeightRecipe> {
        let normalized = normalized_checkpoint_keys(store);
        let keys = store.keys();
        let parameters = module.parameters().flatten();
        let mut recipes = BTreeMap::new();
        if let Some(intermediate) = self.args.moe_intermediate_size {
            let fused = format!("{prefix}.experts.switch_glu.gate_up_proj");
            for suffix in ["weight", "scales", "biases"] {
                let source = format!("{fused}.{suffix}");
                if !keys.contains(&source) {
                    continue;
                }
                for (projection, start, end) in [
                    ("gate_proj", 0usize, intermediate as usize),
                    (
                        "up_proj",
                        intermediate as usize,
                        (2 * intermediate) as usize,
                    ),
                ] {
                    recipes.insert(
                        format!("experts.switch_glu.{projection}.{suffix}"),
                        DerivedWeightRecipe::Select {
                            input: Box::new(DerivedWeightRecipe::source(
                                source.clone(),
                                TensorSelection::Full,
                            )),
                            selection: TensorSelection::Range {
                                axis: 1,
                                start,
                                end,
                            },
                        },
                    );
                }
            }
        }
        for local_name in parameters.keys() {
            if recipes.contains_key(local_name.as_ref()) {
                continue;
            }
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if keys.contains(&destination) || keys.contains(&canonical) {
                continue;
            }
            if let Some(raw) = normalized.get(&canonical) {
                recipes.insert(
                    local_name.to_string(),
                    DerivedWeightRecipe::Cast {
                        input: Box::new(DerivedWeightRecipe::source(
                            raw.clone(),
                            TensorSelection::Full,
                        )),
                        dtype: parameters
                            .get(local_name)
                            .expect("parameter came from the same flattened tree")
                            .dtype(),
                    },
                );
            }
        }
        recipes
    }

    fn bindings(
        &self,
        module: &impl ModuleParameters,
        prefix: &str,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        let mut recipes = self.recipes_for(module, prefix, store);
        if self.external_experts && prefix.starts_with("model.language_model.layers.") {
            let parameters = module.parameters().flatten();
            recipes.retain(|name, _| parameters.contains_key(name.as_str()));
        }
        let mut bindings = build_module_bindings_with_recipes(module, prefix, store, recipes)?;
        if self.external_experts && prefix.starts_with("model.language_model.layers.") {
            bindings.retain(|binding| !binding.name().starts_with("experts."));
        }
        Ok(bindings)
    }

    fn prepare_per_layer_inputs_with_execution(
        &mut self,
        input_ids: &Array,
        hidden: &Array,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        match (
            self.per_layer_embedding.as_mut(),
            self.per_layer_projection.as_mut(),
            self.per_layer_norm.as_mut(),
        ) {
            (Some(token_embedding), Some(projection), Some(norm)) => {
                let ple = self.args.hidden_size_per_layer_input;
                let token_identity = match (
                    execution.and_then(|execution| execution.group()),
                    self.parallel_per_layer_vocabulary.as_ref(),
                ) {
                    (Some(group), Some(range)) => {
                        token_embedding.forward_tensor_parallel(input_ids, range, group, stream)?
                    }
                    _ => token_embedding.forward(input_ids, stream)?,
                }
                .multiply(Array::from_f32((ple as f32).sqrt()), stream)?
                .reshape(
                    &[
                        input_ids.dim(0),
                        input_ids.dim(1),
                        self.args.num_hidden_layers,
                        ple,
                    ],
                    stream,
                )?;
                let projected = match (execution, self.parallel_per_layer_projection.as_mut()) {
                    (Some(execution), Some(projection)) => {
                        let local = projection.forward(hidden, execution)?;
                        let group = execution.group().ok_or_else(|| {
                            Error::Parallel("Gemma per-layer projection requires TP group".into())
                        })?;
                        let widths = vec![local.dim(-1) as usize; execution.size()];
                        safemlx::distributed::all_gather_uneven_axis(
                            &local, -1, &widths, group, stream,
                        )?
                    }
                    _ => projection.forward(hidden, stream)?,
                }
                .multiply(
                    Array::from_f32((self.args.hidden_size as f32).sqrt().recip()),
                    stream,
                )?
                .reshape(
                    &[
                        hidden.dim(0),
                        hidden.dim(1),
                        self.args.num_hidden_layers,
                        ple,
                    ],
                    stream,
                )?;
                Ok(Some(
                    norm.forward(&projected, stream)?
                        .add(token_identity, stream)?
                        .multiply(Array::from_f32(2.0_f32.powf(-0.5)), stream)?,
                ))
            }
            (None, None, None) => Ok(None),
            _ => Err(Error::UnsupportedArchitecture(
                "Gemma 4 per-layer input modules are incomplete".into(),
            )),
        }
    }

    fn prepare_per_layer_inputs(
        &mut self,
        input_ids: &Array,
        hidden: &Array,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        self.prepare_per_layer_inputs_with_execution(input_ids, hidden, None, stream)
    }

    fn media_safe_per_layer_ids(&self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        let mut output = tokens.clone();
        for token_id in [
            self.image_token_id,
            self.video_token_id,
            self.audio_token_id,
        ]
        .into_iter()
        .flatten()
        {
            let mask = output.eq(Array::from_int(token_id), stream)?;
            output = r#where(
                &mask,
                Array::from_int(self.args.pad_token_id),
                &output,
                stream,
            )?;
        }
        Ok(output)
    }
}

fn normalized_checkpoint_keys(store: &dyn WeightStore) -> BTreeMap<String, String> {
    store
        .keys()
        .into_iter()
        .map(|raw| {
            let canonical = canonical_checkpoint_name(&raw);
            let runtime = canonical
                .strip_prefix("language_model.model.")
                .map(|rest| format!("model.language_model.{rest}"))
                .or_else(|| {
                    [
                        ("vision_tower.", "model.vision_tower."),
                        ("embed_vision.", "model.embed_vision."),
                        ("audio_tower.", "model.audio_tower."),
                        ("embed_audio.", "model.embed_audio."),
                    ]
                    .into_iter()
                    .find_map(|(source, destination)| {
                        canonical
                            .strip_prefix(source)
                            .map(|rest| format!("{destination}{rest}"))
                    })
                })
                .unwrap_or(canonical);
            (runtime, raw)
        })
        .collect()
}

/// Input mode for typed prefill and cached text decode.
pub enum Gemma4Input<'a> {
    /// Ordered multimodal prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text tokens for cached decode.
    Decode(&'a Array),
}

enum Gemma4PreparedPart {
    Ready { tokens: Array, embeddings: Array },
    Vision { token_id: u32, job: usize },
    Audio { token_id: u32, job: usize },
}

struct Gemma4VisionJob {
    hidden: Array,
    state: Gemma4VisionLayerwiseState,
}

struct Gemma4AudioJob {
    hidden: Array,
    valid: i32,
}

/// One leased Gemma 4 media or text unit.
pub enum Gemma4Layer {
    /// Vision transformer block.
    Vision(Box<VisionLayer>),
    /// Audio conformer-style block.
    Audio(Box<AudioLayer>),
    /// Text transformer block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for Gemma4Layer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Vision(x) => x.num_parameters(),
            Self::Audio(x) => x.num_parameters(),
            Self::Text(x) => x.num_parameters(),
        }
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(x) => x.parameters(),
            Self::Audio(x) => x.parameters(),
            Self::Text(x) => x.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Vision(x) => x.parameters_mut(),
            Self::Audio(x) => x.parameters_mut(),
            Self::Text(x) => x.parameters_mut(),
        }
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Vision(x) => x.trainable_parameters(),
            Self::Audio(x) => x.trainable_parameters(),
            Self::Text(x) => x.trainable_parameters(),
        }
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(x) => x.freeze_parameters(recursive),
            Self::Audio(x) => x.freeze_parameters(recursive),
            Self::Text(x) => x.freeze_parameters(recursive),
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Vision(x) => x.unfreeze_parameters(recursive),
            Self::Audio(x) => x.unfreeze_parameters(recursive),
            Self::Text(x) => x.unfreeze_parameters(recursive),
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(x) => x.all_frozen(),
            Self::Audio(x) => x.all_frozen(),
            Self::Text(x) => x.all_frozen(),
        }
    }
    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Vision(x) => x.any_frozen(),
            Self::Audio(x) => x.any_frozen(),
            Self::Text(x) => x.any_frozen(),
        }
    }
}

/// Transient Gemma 4 values shared across one multimodal decoder pass.
pub struct Gemma4ForwardContext {
    per_layer_inputs: Option<Array>,
    mask: Option<Array>,
    sliding_masks: Option<HashMap<AttentionPolicy, Array>>,
    position_offset: i32,
    shared_kv: HashMap<AttentionPolicy, (Array, Array)>,
    parts: Vec<Gemma4PreparedPart>,
    vision_jobs: Vec<Gemma4VisionJob>,
    audio_jobs: Vec<Gemma4AudioJob>,
    tokens: Option<Array>,
    needs_assembly: bool,
    draft_hidden: Option<Array>,
}

/// Opaque semantic state retained while a pipeline ingress stage executes its
/// configured media roots.
pub(crate) struct Gemma4PipelineIngressState {
    cache: Cache,
    forward: LayerwiseForwardState<Gemma4ForwardContext>,
}

/// Decoder-ready tensors produced by Gemma's shared multimodal semantics.
pub(crate) struct Gemma4PipelineIngressOutput {
    pub(crate) hidden: Array,
    pub(crate) per_layer_inputs: Option<Array>,
    pub(crate) full_mask: Option<Array>,
    pub(crate) sliding_masks: Vec<Array>,
}

impl Gemma4LayerwiseAdapter {
    pub(crate) fn execution_group_name(&self, group: usize) -> Result<&'static str, Error> {
        let mut index = 0;
        if self.vision_depth > 0 {
            if group == index {
                return Ok("vision_encoder");
            }
            index += 1;
        }
        if self.audio_depth > 0 {
            if group == index {
                return Ok("audio_encoder");
            }
            index += 1;
        }
        if group == index {
            Ok("text_decoder")
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 has no execution group {group}"
            )))
        }
    }

    pub(crate) fn pipeline_text_group(&self) -> usize {
        usize::from(self.vision_depth > 0) + usize::from(self.audio_depth > 0)
    }

    /// Starts typed pipeline ingress through the same adapter lifecycle used
    /// by resident and bounded layerwise execution.
    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        input: input::ModelInput<'_>,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressState, Error> {
        let mut cache = Cache::new(&self.args);
        let forward = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .begin_forward_with_execution(Gemma4Input::Prefill(input), &mut cache, execution)?,
            _ => self.begin_forward(Gemma4Input::Prefill(input), &mut cache, stream)?,
        };
        Ok(Gemma4PipelineIngressState { cache, forward })
    }

    /// Builds parameter-free image/audio job state for a downstream PP owner.
    /// Patch/subsample projections, text embedding, modality projectors, and
    /// final assembly remain on their placement-declared static owner.
    pub(crate) fn begin_pipeline_continuation(
        &self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressState, Error> {
        input::validate(input)?;
        let mut vision_jobs = Vec::new();
        let mut audio_jobs = Vec::new();
        for part in input.parts {
            match (part.modality, part.payload) {
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(pixels),
                ) => {
                    let positions = part.metadata.patch_position_ids.ok_or_else(|| {
                        Error::Parallel(
                            "Gemma vision continuation omitted patch_position_ids".into(),
                        )
                    })?;
                    let vision = self.vision.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Gemma vision continuation requires vision_config".into(),
                        )
                    })?;
                    vision_jobs.push(Gemma4VisionJob {
                        hidden: pixels.clone(),
                        state: vision.continuation_state(pixels, positions, stream)?,
                    });
                }
                (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                    let mask = part.metadata.audio_mask.ok_or_else(|| {
                        Error::Parallel("Gemma audio continuation omitted audio_mask".into())
                    })?;
                    if mask.shape().len() != 2
                        || mask.dim(0) != features.dim(0)
                        || mask.dim(1) != features.dim(1)
                    {
                        return Err(Error::Parallel(format!(
                            "Gemma audio continuation mask {:?} does not match {:?}",
                            mask.shape(),
                            features.shape()
                        )));
                    }
                    let valid_frames = mask.sum(None, stream)?.item::<i32>(stream);
                    audio_jobs.push(Gemma4AudioJob {
                        hidden: features.clone(),
                        valid: (valid_frames + 3) / 4,
                    });
                }
                _ => {}
            }
        }
        let hidden = vision_jobs
            .first()
            .map(|job| job.hidden.clone())
            .or_else(|| audio_jobs.first().map(|job| job.hidden.clone()))
            .or_else(|| {
                input.parts.first().map(|part| match part.payload {
                    input::InputPayload::TokenIds(value)
                    | input::InputPayload::Tensor(value)
                    | input::InputPayload::Embeddings(value) => value.clone(),
                })
            })
            .ok_or_else(|| Error::Parallel("Gemma continuation has no payload".into()))?;
        Ok(Gemma4PipelineIngressState {
            cache: Cache::new(&self.args),
            forward: LayerwiseForwardState {
                hidden,
                context: Gemma4ForwardContext {
                    per_layer_inputs: None,
                    mask: None,
                    sliding_masks: None,
                    position_offset: 0,
                    shared_kv: HashMap::new(),
                    parts: Vec::new(),
                    vision_jobs,
                    audio_jobs,
                    tokens: None,
                    needs_assembly: true,
                    draft_hidden: None,
                },
            },
        })
    }

    /// Returns whether a configured media group has work for this input.
    pub(crate) fn should_execute_pipeline_group(
        &self,
        group: usize,
        state: &Gemma4PipelineIngressState,
    ) -> bool {
        self.should_execute_group(group, &state.forward.context)
    }

    /// Executes one resident or leased media block using the canonical Gemma
    /// layerwise hooks.
    pub(crate) fn forward_pipeline_media_layer(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Gemma4Layer,
        state: &mut Gemma4PipelineIngressState,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Vec<Array>, Error> {
        state.forward.hidden = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .forward_layer_with_execution(
                    group,
                    index,
                    layer,
                    &state.forward.hidden,
                    &mut state.cache,
                    &mut state.forward.context,
                    execution,
                )?,
            _ => self.forward_layer(
                group,
                index,
                layer,
                &state.forward.hidden,
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        Ok(std::iter::once(state.forward.hidden.clone())
            .chain(
                self.retained_context_arrays(&state.forward.context, group, index)
                    .into_iter()
                    .cloned(),
            )
            .collect())
    }

    /// Completes media roots and assembles the exact decoder ingress tensors.
    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: Gemma4PipelineIngressState,
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Gemma4PipelineIngressOutput, Error> {
        let text_group = self.pipeline_text_group();
        state.forward.hidden = match execution {
            Some(execution) if execution.is_tensor_parallel() => self
                .begin_execution_group_with_execution(
                    text_group,
                    &state.forward.hidden,
                    &[],
                    &mut state.cache,
                    &mut state.forward.context,
                    execution,
                )?,
            _ => self.begin_execution_group(
                text_group,
                &state.forward.hidden,
                &[],
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        let sliding_masks = self
            .pipeline_mask_windows()
            .into_iter()
            .filter_map(|window| {
                state
                    .forward
                    .context
                    .sliding_masks
                    .as_ref()
                    .and_then(|masks| masks.get(&AttentionPolicy::Sliding { window }).cloned())
            })
            .collect();
        Ok(Gemma4PipelineIngressOutput {
            hidden: state.forward.hidden,
            per_layer_inputs: state.forward.context.per_layer_inputs,
            full_mask: state.forward.context.mask,
            sliding_masks,
        })
    }
}

impl LoadTimeQuantizableAdapter for Gemma4LayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantized = true;
        args.weight_quantization = Some(quantization);
        args.quantization_group_size = quantization.group_size();
        args.quantization_bits = quantization.bits();
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
        let mut vision_config = self.vision_config.clone();
        if let Some(config) = &mut vision_config {
            config.weight_quantization = Some(quantization);
        }
        let mut audio_config = self.audio_config.clone();
        if let Some(config) = &mut audio_config {
            config.weight_quantization = Some(quantization);
        }
        let mut adapter = Self::new(
            args,
            vision_config,
            self.image_token_id,
            self.video_token_id,
            audio_config,
            self.audio_token_id,
            stream,
        )?;
        adapter.external_experts = self.external_experts;
        Ok(adapter)
    }
}

fn ignores_gemma4_checkpoint_key(key: &str) -> bool {
    key.starts_with("rope_freqs.")
        || [
            "multi_modal_projector.",
            "model.multi_modal_projector.",
            "model.vision_embedder.",
        ]
        .iter()
        .any(|prefix| key.starts_with(prefix))
}

impl ArchitectureAdapter for Gemma4LayerwiseAdapter {
    type Input<'a> = Gemma4Input<'a>;
    type Cache = Cache;
    type Layer = Gemma4Layer;
    type ForwardContext = Gemma4ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        self.args.weight_quantization()
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        if target.contains("vision_tower.") {
            return target.ends_with("patch_embedder.input_proj.weight")
                && self
                    .vision_config
                    .as_ref()
                    .and_then(|config| {
                        config.weight_quantization.filter(|quantization| {
                            let input = 3 * config.patch_size * config.patch_size;
                            input % quantization.group_size() == 0 && input % 32 == 0
                        })
                    })
                    .is_some();
        }
        if target.contains("audio_tower.") {
            let input = if target.ends_with("subsample_conv_projection.input_proj_linear.weight") {
                self.audio_config
                    .as_ref()
                    .map(|config| 32 * config.subsampling_conv_channels[1])
            } else if target.ends_with("output_proj.weight") {
                self.audio_config.as_ref().map(|config| config.hidden_size)
            } else {
                None
            };
            return input
                .zip(
                    self.audio_config
                        .as_ref()
                        .and_then(|config| config.weight_quantization),
                )
                .is_some_and(|(input, quantization)| {
                    input % quantization.group_size() == 0 && input % 32 == 0
                });
        }
        true
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.num_hidden_layers as usize;
        let layer_layout = match topology {
            Some(topology) if topology.tensor_parallel_size > 1 => {
                let geometry = self.parallel_text_geometry.as_ref().ok_or_else(|| {
                    Error::Parallel(
                        "Gemma 4 parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?;
                resident::prompt_cache_layer_layout_with_geometry(&self.args, geometry)
            }
            _ => resident::prompt_cache_layer_layout(&self.args),
        }?;
        Ok(PromptCacheModelIdentity {
            model_family: "gemma4".into(),
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

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn WeightStore,
        select: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<StaticUnitBindings>, Error> {
        let mut units = Vec::new();
        if select(EMBEDDING_UNIT) {
            units.push(StaticUnitBindings::new(
                EMBEDDING_UNIT,
                self.bindings(&self.embedding, "model.language_model.embed_tokens", store)?,
            )?);
        }
        if select(PER_LAYER_EMBEDDING_UNIT) {
            if let Some(module) = &self.per_layer_embedding {
                units.push(StaticUnitBindings::new(
                    PER_LAYER_EMBEDDING_UNIT,
                    self.bindings(module, "model.language_model.embed_tokens_per_layer", store)?,
                )?);
            }
        }
        if select(PER_LAYER_PROJECTION_UNIT) {
            if let Some(module) = &self.per_layer_projection {
                units.push(StaticUnitBindings::new(
                    PER_LAYER_PROJECTION_UNIT,
                    self.bindings(
                        module,
                        "model.language_model.per_layer_model_projection",
                        store,
                    )?,
                )?);
            }
        }
        if select(PER_LAYER_NORM_UNIT) {
            if let Some(module) = &self.per_layer_norm {
                units.push(StaticUnitBindings::new(
                    PER_LAYER_NORM_UNIT,
                    self.bindings(
                        module,
                        "model.language_model.per_layer_projection_norm",
                        store,
                    )?,
                )?);
            }
        }
        if select(NORM_UNIT) {
            units.push(StaticUnitBindings::new(
                NORM_UNIT,
                self.bindings(&self.norm, "model.language_model.norm", store)?,
            )?);
        }
        if select(HEAD_UNIT) {
            if let Some(module) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    self.bindings(module, "lm_head", store)?,
                )?);
            }
        }
        if select(VISION_STATIC_UNIT) {
            if let Some(module) = &self.vision {
                units.push(StaticUnitBindings::new(
                    VISION_STATIC_UNIT,
                    self.bindings(module, "model.vision_tower", store)?,
                )?);
            }
        }
        if select(VISION_EMBED_UNIT) {
            if let Some(module) = &self.embed_vision {
                units.push(StaticUnitBindings::new(
                    VISION_EMBED_UNIT,
                    self.bindings(module, "model.embed_vision", store)?,
                )?);
            }
        }
        if select(AUDIO_STATIC_UNIT) {
            if let Some(module) = &self.audio {
                units.push(StaticUnitBindings::new(
                    AUDIO_STATIC_UNIT,
                    self.bindings(module, "model.audio_tower", store)?,
                )?);
            }
        }
        if select(AUDIO_EMBED_UNIT) {
            if let Some(module) = &self.embed_audio {
                units.push(StaticUnitBindings::new(
                    AUDIO_EMBED_UNIT,
                    self.bindings(module, "model.embed_audio", store)?,
                )?);
            }
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let mut index = 0;
        populate_module_from_lease(&mut self.embedding, &leases[index])?;
        index += 1;
        if let Some(module) = &mut self.per_layer_embedding {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if self.per_layer_projection.is_some() {
            if let Some(module) = &mut self.parallel_per_layer_projection {
                populate_module_from_lease(module.inner_mut(), &leases[index])?;
            } else if let Some(module) = &mut self.per_layer_projection {
                populate_module_from_lease(module, &leases[index])?;
            }
            index += 1;
        }
        if let Some(module) = &mut self.per_layer_norm {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        populate_module_from_lease(&mut self.norm, &leases[index])?;
        index += 1;
        if self.lm_head.is_some() {
            if let Some(module) = &mut self.parallel_lm_head {
                populate_module_from_lease(module.inner_mut(), &leases[index])?;
            } else if let Some(module) = &mut self.lm_head {
                populate_module_from_lease(module, &leases[index])?;
            }
            index += 1;
        }
        if let Some(module) = &mut self.vision {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.embed_vision {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.audio {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if let Some(module) = &mut self.embed_audio {
            populate_module_from_lease(module, &leases[index])?;
            index += 1;
        }
        if index != leases.len() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 adapter received {} static leases, consumed {index}",
                leases.len()
            )));
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.kv.is_empty() {
            cache.reset_kv(&self.args);
        }
        if cache.kv.len() != self.args.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 cache has {} layers, expected {}",
                cache.kv.len(),
                self.args.num_hidden_layers
            )));
        }
        Ok(())
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        if let Gemma4Input::Prefill(typed) = input {
            input::validate(typed)?;
            cache.token_ids.clear();
            cache.reset_kv(&self.args);
            let mut parts = Vec::with_capacity(typed.parts.len());
            let mut vision_jobs = Vec::new();
            let mut audio_jobs = Vec::new();
            let scale = Array::from_f32((self.args.hidden_size as f32).sqrt());
            for part in typed.parts {
                match (part.modality, part.payload) {
                    (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: tokens.clone(),
                            embeddings: self
                                .embedding
                                .forward(tokens, stream)?
                                .multiply(&scale, stream)?,
                        });
                    }
                    (
                        modality @ (input::Modality::Image | input::Modality::Video),
                        input::InputPayload::Tensor(pixels),
                    ) => {
                        let positions = part.metadata.patch_position_ids.ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 {} tensor input requires patch_position_ids metadata",
                                modality.as_str()
                            ))
                        })?;
                        let token_id = if modality == input::Modality::Image {
                            self.image_token_id
                        } else {
                            self.video_token_id
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        let (hidden, state) = self
                            .vision
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(format!(
                                    "gemma4 {} tensor input requires vision_config and vision weights",
                                    modality.as_str()
                                ))
                            })?
                            .begin(pixels, positions, stream)?;
                        let job = vision_jobs.len();
                        vision_jobs.push(Gemma4VisionJob { hidden, state });
                        parts.push(Gemma4PreparedPart::Vision { token_id, job });
                    }
                    (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                        let mask = part.metadata.audio_mask.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 audio tensor input requires audio_mask metadata".into(),
                            )
                        })?;
                        let token_id = self.audio_token_id.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 config does not define an audio token ID".into(),
                            )
                        })? as u32;
                        let (hidden, valid) = self
                            .audio
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                    "gemma4 audio tensor input requires audio_config and audio weights".into(),
                                )
                            })?
                            .begin(features, mask, stream)?;
                        let job = audio_jobs.len();
                        audio_jobs.push(Gemma4AudioJob { hidden, valid });
                        parts.push(Gemma4PreparedPart::Audio { token_id, job });
                    }
                    (
                        modality @ (input::Modality::Image
                        | input::Modality::Video
                        | input::Modality::Audio),
                        input::InputPayload::Embeddings(embeddings),
                    ) => {
                        input::ensure_hidden_size(
                            embeddings,
                            self.args.hidden_size,
                            "Gemma 4 media embeddings",
                        )?;
                        let token_id = match modality {
                            input::Modality::Image => self.image_token_id,
                            input::Modality::Video => self.video_token_id,
                            input::Modality::Audio => self.audio_token_id,
                            input::Modality::Text => unreachable!(),
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: input::token_ids_array(
                                &vec![token_id; embeddings.dim(1) as usize],
                                stream,
                            )?,
                            embeddings: embeddings.clone(),
                        });
                    }
                    (modality, _) => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "gemma4 layerwise input does not support {} payloads of this kind",
                            modality.as_str()
                        )));
                    }
                }
            }
            if self.vision_depth == 0 && self.audio_depth == 0 {
                let token_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { tokens, .. } => Some(tokens),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let embedding_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let tokens = concatenate_axis(&token_parts, 1, stream)?;
                let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
                cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
                let per_layer_inputs = self.prepare_per_layer_inputs(&tokens, &hidden, stream)?;
                let mask = (hidden.dim(1) > 1)
                    .then(|| create_causal_mask(hidden.dim(1), Some(0), None, None, stream))
                    .transpose()?;
                return Ok(LayerwiseForwardState {
                    hidden,
                    context: Gemma4ForwardContext {
                        per_layer_inputs,
                        mask,
                        sliding_masks: None,
                        position_offset: 0,
                        shared_kv: HashMap::new(),
                        parts,
                        vision_jobs,
                        audio_jobs,
                        tokens: Some(tokens),
                        needs_assembly: false,
                        draft_hidden: None,
                    },
                });
            }
            let hidden = vision_jobs
                .first()
                .map(|job| job.hidden.clone())
                .or_else(|| audio_jobs.first().map(|job| job.hidden.clone()))
                .or_else(|| {
                    parts.iter().find_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings.clone()),
                        _ => None,
                    })
                })
                .expect("validated non-empty Gemma 4 input");
            return Ok(LayerwiseForwardState {
                hidden,
                context: Gemma4ForwardContext {
                    per_layer_inputs: None,
                    mask: None,
                    sliding_masks: None,
                    position_offset: 0,
                    shared_kv: HashMap::new(),
                    parts,
                    vision_jobs,
                    audio_jobs,
                    tokens: None,
                    needs_assembly: true,
                    draft_hidden: None,
                },
            });
        }
        let Gemma4Input::Decode(tokens) = input else {
            unreachable!()
        };
        cache
            .token_ids
            .extend(resident::token_ids_from_array(tokens, stream)?);
        let hidden = self.embedding.forward(tokens, stream)?.multiply(
            Array::from_f32((self.args.hidden_size as f32).sqrt()),
            stream,
        )?;
        let position_offset = cache
            .kv
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mask = (hidden.dim(1) > 1)
            .then(|| create_causal_mask(hidden.dim(1), Some(position_offset), None, None, stream))
            .transpose()?;
        let per_layer_inputs = self.prepare_per_layer_inputs(tokens, &hidden, stream)?;
        Ok(LayerwiseForwardState {
            hidden,
            context: Gemma4ForwardContext {
                per_layer_inputs,
                mask,
                sliding_masks: None,
                position_offset,
                shared_kv: HashMap::new(),
                parts: Vec::new(),
                vision_jobs: Vec::new(),
                audio_jobs: Vec::new(),
                tokens: Some(tokens.clone()),
                needs_assembly: false,
                draft_hidden: None,
            },
        })
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let Some(group) = execution.group() else {
            return self.begin_forward(input, cache, execution.stream());
        };
        let stream = execution.stream();
        let vocabulary = self
            .parallel_vocabulary
            .as_ref()
            .ok_or_else(|| Error::Parallel("Gemma TP embedding was not configured".into()))?;
        if let Gemma4Input::Prefill(typed) = input {
            input::validate(typed)?;
            cache.token_ids.clear();
            cache.reset_kv(&self.args);
            let mut parts = Vec::with_capacity(typed.parts.len());
            let mut vision_jobs = Vec::new();
            let mut audio_jobs = Vec::new();
            let scale = Array::from_f32((self.args.hidden_size as f32).sqrt());
            for part in typed.parts {
                match (part.modality, part.payload) {
                    (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: tokens.clone(),
                            embeddings: self
                                .embedding
                                .forward_tensor_parallel(tokens, vocabulary, group, stream)?
                                .multiply(&scale, stream)?,
                        });
                    }
                    (
                        modality @ (input::Modality::Image | input::Modality::Video),
                        input::InputPayload::Tensor(pixels),
                    ) => {
                        let positions = part.metadata.patch_position_ids.ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 {} tensor input requires patch_position_ids metadata",
                                modality.as_str()
                            ))
                        })?;
                        let token_id = if modality == input::Modality::Image {
                            self.image_token_id
                        } else {
                            self.video_token_id
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        let (hidden, state) = self
                            .vision
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(format!(
                                    "gemma4 {} tensor input requires vision_config and vision weights",
                                    modality.as_str()
                                ))
                            })?
                            .begin_tensor_parallel(pixels, positions, group, stream)?;
                        let job = vision_jobs.len();
                        vision_jobs.push(Gemma4VisionJob { hidden, state });
                        parts.push(Gemma4PreparedPart::Vision { token_id, job });
                    }
                    (input::Modality::Audio, input::InputPayload::Tensor(features)) => {
                        let mask = part.metadata.audio_mask.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 audio tensor input requires audio_mask metadata".into(),
                            )
                        })?;
                        let token_id = self.audio_token_id.ok_or_else(|| {
                            Error::UnsupportedArchitecture(
                                "gemma4 config does not define an audio token ID".into(),
                            )
                        })? as u32;
                        let (hidden, valid) = self
                            .audio
                            .as_mut()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                    "gemma4 audio tensor input requires audio_config and audio weights".into(),
                                )
                            })?
                            .begin_tensor_parallel(features, mask, group, stream)?;
                        let job = audio_jobs.len();
                        audio_jobs.push(Gemma4AudioJob { hidden, valid });
                        parts.push(Gemma4PreparedPart::Audio { token_id, job });
                    }
                    (
                        modality @ (input::Modality::Image
                        | input::Modality::Video
                        | input::Modality::Audio),
                        input::InputPayload::Embeddings(embeddings),
                    ) => {
                        input::ensure_hidden_size(
                            embeddings,
                            self.args.hidden_size,
                            "Gemma 4 media embeddings",
                        )?;
                        let token_id = match modality {
                            input::Modality::Image => self.image_token_id,
                            input::Modality::Video => self.video_token_id,
                            input::Modality::Audio => self.audio_token_id,
                            input::Modality::Text => unreachable!(),
                        }
                        .ok_or_else(|| {
                            Error::UnsupportedArchitecture(format!(
                                "gemma4 config does not define a {} token ID",
                                modality.as_str()
                            ))
                        })? as u32;
                        parts.push(Gemma4PreparedPart::Ready {
                            tokens: input::token_ids_array(
                                &vec![token_id; embeddings.dim(1) as usize],
                                stream,
                            )?,
                            embeddings: embeddings.clone(),
                        });
                    }
                    (modality, _) => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "gemma4 layerwise input does not support {} payloads of this kind",
                            modality.as_str()
                        )));
                    }
                }
            }
            if self.vision_depth == 0 && self.audio_depth == 0 {
                let token_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { tokens, .. } => Some(tokens),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let embedding_parts = parts
                    .iter()
                    .filter_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let tokens = concatenate_axis(&token_parts, 1, stream)?;
                let hidden = concatenate_axis(&embedding_parts, 1, stream)?;
                cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
                let per_layer_inputs = self.prepare_per_layer_inputs_with_execution(
                    &tokens,
                    &hidden,
                    Some(execution),
                    stream,
                )?;
                let mask = (hidden.dim(1) > 1)
                    .then(|| create_causal_mask(hidden.dim(1), Some(0), None, None, stream))
                    .transpose()?;
                return Ok(LayerwiseForwardState {
                    hidden,
                    context: Gemma4ForwardContext {
                        per_layer_inputs,
                        mask,
                        sliding_masks: None,
                        position_offset: 0,
                        shared_kv: HashMap::new(),
                        parts,
                        vision_jobs,
                        audio_jobs,
                        tokens: Some(tokens),
                        needs_assembly: false,
                        draft_hidden: None,
                    },
                });
            }
            let hidden = vision_jobs
                .first()
                .map(|job| job.hidden.clone())
                .or_else(|| audio_jobs.first().map(|job| job.hidden.clone()))
                .or_else(|| {
                    parts.iter().find_map(|part| match part {
                        Gemma4PreparedPart::Ready { embeddings, .. } => Some(embeddings.clone()),
                        _ => None,
                    })
                })
                .expect("validated non-empty Gemma 4 input");
            return Ok(LayerwiseForwardState {
                hidden,
                context: Gemma4ForwardContext {
                    per_layer_inputs: None,
                    mask: None,
                    sliding_masks: None,
                    position_offset: 0,
                    shared_kv: HashMap::new(),
                    parts,
                    vision_jobs,
                    audio_jobs,
                    tokens: None,
                    needs_assembly: true,
                    draft_hidden: None,
                },
            });
        }
        let Gemma4Input::Decode(tokens) = input else {
            unreachable!()
        };
        cache
            .token_ids
            .extend(resident::token_ids_from_array(tokens, stream)?);
        let hidden = self
            .embedding
            .forward_tensor_parallel(tokens, vocabulary, group, stream)?
            .multiply(
                Array::from_f32((self.args.hidden_size as f32).sqrt()),
                stream,
            )?;
        let position_offset = cache
            .kv
            .iter()
            .flatten()
            .map(KeyValueCache::offset)
            .max()
            .unwrap_or(0);
        let mask = (hidden.dim(1) > 1)
            .then(|| create_causal_mask(hidden.dim(1), Some(position_offset), None, None, stream))
            .transpose()?;
        let per_layer_inputs =
            self.prepare_per_layer_inputs_with_execution(tokens, &hidden, Some(execution), stream)?;
        Ok(LayerwiseForwardState {
            hidden,
            context: Gemma4ForwardContext {
                per_layer_inputs,
                mask,
                sliding_masks: None,
                position_offset,
                shared_kv: HashMap::new(),
                parts: Vec::new(),
                vision_jobs: Vec::new(),
                audio_jobs: Vec::new(),
                tokens: Some(tokens.clone()),
                needs_assembly: false,
                draft_hidden: None,
            },
        })
    }

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        use crate::runtime::execution::layerwise::{ExecutionGroupDag, ExecutionGroupSpec};
        let mut groups = Vec::new();
        let mut ingress = Vec::new();
        if self.vision_depth > 0 {
            groups.push(ExecutionGroupSpec::root("vision_encoder"));
            ingress.push("vision_encoder");
        }
        if self.audio_depth > 0 {
            groups.push(ExecutionGroupSpec::root("audio_encoder"));
            ingress.push("audio_encoder");
        }
        if ingress.is_empty() {
            groups.push(ExecutionGroupSpec::root("text_decoder"));
        } else {
            groups.push(ExecutionGroupSpec::with_dependencies(
                "text_decoder",
                ingress,
            ));
        }
        ExecutionGroupDag::new(groups, "text_decoder")
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        self.execution_group_name(group).is_ok_and(|id| match id {
            "vision_encoder" => !context.vision_jobs.is_empty(),
            "audio_encoder" => !context.audio_jobs.is_empty(),
            _ => true,
        })
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match self.execution_group_name(group)? {
            "vision_encoder" => Ok(self.vision_depth),
            "audio_encoder" => Ok(self.audio_depth),
            "text_decoder" => Ok(self.args.num_hidden_layers as usize),
            _ => unreachable!(),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        match self.execution_group_name(group)? {
            "vision_encoder" => Ok(Gemma4Layer::Vision(Box::new(VisionLayer::new(
                &self.vision.as_ref().expect("vision group").config,
                stream,
            )?))),
            "audio_encoder" => Ok(Gemma4Layer::Audio(Box::new(AudioLayer::new(
                self.audio_config.as_ref().expect("audio group"),
                stream,
            )?))),
            "text_decoder" => {
                let mut layer = TransformerBlock::new(
                    &self.args,
                    *self.args.layer_policy(index).ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Gemma 4 has no attention policy for layer {index}"
                        ))
                    })?,
                    index,
                    stream,
                )?;
                if self.external_experts {
                    layer.experts = None;
                }
                Ok(Gemma4Layer::Text(Box::new(layer)))
            }
            _ => unreachable!(),
        }
    }

    fn register_parallel_parameters(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.embedding
            .register_tensor_parallel_parameters(planner, "model.language_model.embed_tokens")?;
        if let Some(embedding) = &self.per_layer_embedding {
            embedding.register_tensor_parallel_parameters(
                planner,
                "model.language_model.embed_tokens_per_layer",
            )?;
        }
        if let Some(projection) = &self.per_layer_projection {
            crate::nn::parallel::register_linear_parameter_group(
                planner,
                projection,
                "model.language_model.per_layer_model_projection",
                LinearParallelism::Column,
            )?;
        }
        if let Some(norm) = &self.per_layer_norm {
            crate::nn::parallel::register_replicated_parameter_group(
                planner,
                norm,
                "model.language_model.per_layer_projection_norm",
            )?;
        }
        crate::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            "model.language_model.norm",
        )?;
        if let Some(head) = &self.lm_head {
            planner.register(crate::nn::parallel::vocab_lm_head_parameter_group(
                head,
                "lm_head",
                self.args.hidden_size,
                self.args.vocab_size as usize,
                false,
            )?)?;
        }
        if let Some(vision) = &self.vision {
            let patch_prefix = "model.vision_tower.patch_embedder";
            let (units, mut members) = partitioned_projection_members(
                &[(
                    &vision.patch_embedder.input_proj,
                    &format!("{patch_prefix}.input_proj"),
                    ProjectionSharding::Column,
                )],
                vision.config.hidden_size as usize,
            )?;
            members.push(ParameterMemberSpec::new(
                format!("{patch_prefix}.position_embedding_table"),
                vision
                    .patch_embedder
                    .position_embedding_table
                    .shape()
                    .iter()
                    .map(|&dimension| dimension as usize)
                    .collect::<Vec<_>>(),
                MemberSharding::Partitioned { axis: 2 },
            ));
            planner.register(ParameterGroupSpec::partitioned(
                format!("{patch_prefix}.hidden"),
                ParameterRole::ColumnProjection,
                units,
                members,
            )?)?;
            for (name, value) in [
                ("std_bias", vision.std_bias.as_ref()),
                ("std_scale", vision.std_scale.as_ref()),
            ] {
                if let Some(value) = value {
                    planner.register(ParameterGroupSpec::new(
                        format!("model.vision_tower.{name}"),
                        ParameterRole::Replicated,
                        [ParameterMemberSpec::new(
                            format!("model.vision_tower.{name}"),
                            value
                                .shape()
                                .iter()
                                .map(|&dimension| dimension as usize)
                                .collect::<Vec<_>>(),
                            MemberSharding::Replicated,
                        )],
                    )?)?;
                }
            }
        }
        if let Some(embedder) = &self.embed_vision {
            register_partitioned_projection_group(
                planner,
                "model.embed_vision.input",
                ParameterRole::RowProjection,
                &[(
                    &embedder.embedding_projection,
                    "model.embed_vision.embedding_projection",
                    ProjectionSharding::Row,
                )],
                embedder.input_size as usize,
            )?;
        }
        if let Some(audio) = &self.audio {
            crate::nn::parallel::register_replicated_parameter_group(
                planner,
                &audio.subsample_conv_projection.layer0,
                "model.audio_tower.subsample_conv_projection.layer0",
            )?;
            crate::nn::parallel::register_replicated_parameter_group(
                planner,
                &audio.subsample_conv_projection.layer1,
                "model.audio_tower.subsample_conv_projection.layer1",
            )?;
            register_partitioned_projection_group(
                planner,
                "model.audio_tower.subsample_conv_projection.hidden",
                ParameterRole::ColumnProjection,
                &[(
                    &audio.subsample_conv_projection.input_proj_linear,
                    "model.audio_tower.subsample_conv_projection.input_proj_linear",
                    ProjectionSharding::Column,
                )],
                self.audio_config
                    .as_ref()
                    .expect("audio config")
                    .hidden_size as usize,
            )?;
            register_partitioned_projection_group(
                planner,
                "model.audio_tower.output_projection",
                ParameterRole::ColumnProjection,
                &[(
                    &audio.output_proj,
                    "model.audio_tower.output_proj",
                    ProjectionSharding::Column,
                )],
                self.audio_config
                    .as_ref()
                    .expect("audio config")
                    .output_proj_dims as usize,
            )?;
        }
        if let Some(embedder) = &self.embed_audio {
            register_partitioned_projection_group(
                planner,
                "model.embed_audio.input",
                ParameterRole::RowProjection,
                &[(
                    &embedder.embedding_projection,
                    "model.embed_audio.embedding_projection",
                    ProjectionSharding::Row,
                )],
                embedder.input_size as usize,
            )?;
        }
        if let Some(vision) = &self.vision {
            for index in 0..self.vision_depth {
                let layer = VisionLayer::new(&vision.config, stream)?;
                register_gemma_vision_layer_parallel_plan(
                    planner,
                    &layer,
                    &format!("model.vision_tower.encoder.layers.{index}"),
                )?;
            }
        }
        if let Some(config) = &self.audio_config {
            for index in 0..self.audio_depth {
                let layer = AudioLayer::new(config, stream)?;
                register_gemma_audio_layer_parallel_plan(
                    planner,
                    &layer,
                    &format!("model.audio_tower.layers.{index}"),
                )?;
            }
        }
        for index in 0..self.args.num_hidden_layers as usize {
            let layer = TransformerBlock::new(
                &self.args,
                *self.args.layer_policy(index).ok_or_else(|| {
                    Error::Parallel(format!("missing Gemma policy for layer {index}"))
                })?,
                index,
                stream,
            )?;
            register_gemma_layer_parallel_plan(
                planner,
                &layer,
                &format!("model.language_model.layers.{index}"),
            )?;
        }
        Ok(())
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let local_semantic = |targets: &[String], global: i32| -> Result<i32, Error> {
            let (target, tensor) = targets
                .iter()
                .find_map(|target| layout.tensor(target).map(|tensor| (target, tensor)))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "missing Gemma 4 TP layout for any of {}",
                        targets.join(", ")
                    ))
                })?;
            let units = tensor.logical_units().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical domain"
                ))
            })?;
            let range = tensor.logical_range().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical range"
                ))
            })?;
            let global = usize::try_from(global)
                .map_err(|_| Error::Parallel(format!("Gemma 4 {target} geometry is invalid")))?;
            if units == 0 || !global.is_multiple_of(units) {
                return Err(Error::Parallel(format!(
                    "Gemma 4 {target} global geometry {global} is incompatible with {units} planner units"
                )));
            }
            let local = range
                .len()
                .checked_mul(global / units)
                .ok_or_else(|| Error::Parallel(format!("Gemma 4 local {target} overflowed")))?;
            i32::try_from(local)
                .map_err(|_| Error::Parallel(format!("Gemma 4 local {target} exceeds i32")))
        };
        let semantic_range = |targets: &[&str], global: i32| -> Result<Range<usize>, Error> {
            let (target, tensor) = targets
                .iter()
                .find_map(|target| layout.tensor(target).map(|tensor| (*target, tensor)))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "missing Gemma 4 TP layout for any of {}",
                        targets.join(", ")
                    ))
                })?;
            let units = tensor.logical_units().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical domain"
                ))
            })?;
            let range = tensor.logical_range().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical range"
                ))
            })?;
            let global = usize::try_from(global)
                .map_err(|_| Error::Parallel(format!("Gemma 4 {target} geometry is invalid")))?;
            if units == 0 || !global.is_multiple_of(units) {
                return Err(Error::Parallel(format!(
                    "Gemma 4 {target} global geometry {global} is incompatible with {units} planner units"
                )));
            }
            let width = global / units;
            Ok(range.start * width..range.end * width)
        };
        let semantic_widths = |targets: &[&str], global: i32| -> Result<Vec<usize>, Error> {
            let (target, tensor) = targets
                .iter()
                .find_map(|target| layout.tensor(target).map(|tensor| (*target, tensor)))
                .ok_or_else(|| {
                    Error::Parallel(format!(
                        "missing Gemma 4 TP layout for any of {}",
                        targets.join(", ")
                    ))
                })?;
            let units = tensor.logical_units().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical domain"
                ))
            })?;
            let global = usize::try_from(global)
                .map_err(|_| Error::Parallel(format!("Gemma 4 {target} geometry is invalid")))?;
            if units == 0 || !global.is_multiple_of(units) {
                return Err(Error::Parallel(format!(
                    "Gemma 4 {target} global geometry {global} is incompatible with {units} planner units"
                )));
            }
            let width = global / units;
            (0..context.topology().tensor_parallel_size)
                .map(|rank| {
                    crate::core::balanced_contiguous_range(
                        units,
                        context.topology().tensor_parallel_size,
                        rank,
                        false,
                    )
                    .map(|range| range.len() * width)
                    .map_err(Error::from)
                })
                .collect()
        };
        let mut text_geometry = Vec::with_capacity(self.args.num_hidden_layers as usize);
        for index in 0..self.args.num_hidden_layers as usize {
            let policy = *self.args.layer_policy(index).ok_or_else(|| {
                Error::Parallel(format!("missing Gemma 4 policy for layer {index}"))
            })?;
            let prefix = format!("model.language_model.layers.{index}");
            let projection_targets = |suffix: &str| {
                vec![
                    format!("{prefix}.{suffix}.weight"),
                    format!("{prefix}.{suffix}.inner.weight"),
                ]
            };
            let attention = projection_targets("self_attn.q_proj");
            let dense = projection_targets("mlp.gate_proj");
            let expert_intermediate =
                if policy.feed_forward == resident::FeedForwardPolicy::DenseWithSparseMoe {
                    let global = self.args.moe_intermediate_size.ok_or_else(|| {
                        Error::Parallel(format!(
                            "Gemma 4 MoE layer {index} has no expert intermediate width"
                        ))
                    })?;
                    Some(local_semantic(
                        &[format!("{prefix}.experts.switch_glu.gate_proj.weight")],
                        global,
                    )?)
                } else {
                    None
                };
            text_geometry.push(resident::ParallelLayerGeometry {
                query_heads: local_semantic(&attention, self.args.num_attention_heads)?,
                kv_heads: local_semantic(&attention, policy.num_key_value_heads.get() as i32)?,
                dense_intermediate: local_semantic(&dense, policy.intermediate_size.get() as i32)?,
                expert_intermediate,
            });
        }
        self.parallel_text_geometry = Some(text_geometry);
        let topology = context.topology();
        let vocabulary = crate::core::balanced_contiguous_range(
            self.args.vocab_size as usize,
            topology.tensor_parallel_size,
            topology.tensor_parallel_rank,
            false,
        )?;
        self.embedding = Gemma4Embedding::unloaded(
            vocabulary.len() as i32,
            self.args.hidden_size,
            self.args
                .quantization_for("model.language_model.embed_tokens.weight"),
            stream,
        )?;
        self.parallel_vocabulary = Some(vocabulary);
        if self.per_layer_embedding.is_some() {
            let global = self
                .args
                .vocab_size_per_layer_input
                .unwrap_or(self.args.vocab_size) as usize;
            let range = crate::core::balanced_contiguous_range(
                global,
                topology.tensor_parallel_size,
                topology.tensor_parallel_rank,
                false,
            )?;
            self.per_layer_embedding = Some(Gemma4Embedding::unloaded(
                range.len() as i32,
                self.args.num_hidden_layers * self.args.hidden_size_per_layer_input,
                self.args
                    .quantization_for("model.language_model.embed_tokens_per_layer.weight"),
                stream,
            )?);
            self.parallel_per_layer_vocabulary = Some(range);
        }
        if self.per_layer_projection.is_some() {
            self.parallel_per_layer_projection = Some(ParallelLinear::unloaded(
                self.args.hidden_size,
                self.args.num_hidden_layers * self.args.hidden_size_per_layer_input,
                false,
                self.args
                    .quantization_for("model.language_model.per_layer_model_projection.weight"),
                LinearParallelism::Column,
                context,
                stream,
            )?);
        }
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                self.args.hidden_size,
                self.args.vocab_size as usize,
                self.args.quantization_for("lm_head.weight"),
                context,
                stream,
            )?);
        }
        if let Some(embedder) = &self.embed_vision {
            let targets = [
                "model.embed_vision.embedding_projection.weight",
                "model.embed_vision.embedding_projection.inner.weight",
            ];
            self.embed_vision = Some(Gemma4ModalityEmbedder::new_tensor_parallel(
                embedder.input_size,
                self.args.hidden_size,
                embedder.eps,
                false,
                self.args.weight_quantization(),
                semantic_range(&targets, embedder.input_size)?,
                stream,
            )?);
        }
        if let Some(vision) = &mut self.vision {
            let target = "model.vision_tower.patch_embedder.input_proj.weight";
            let range = semantic_range(&[target], vision.config.hidden_size)?;
            vision.patch_embedder =
                crate::api::gemma4_vision::VisionPatchEmbedder::new_tensor_parallel(
                    &vision.config,
                    range.len() as i32,
                    semantic_widths(&[target], vision.config.hidden_size)?,
                    stream,
                )?;
        }
        if let (Some(audio), Some(config)) = (&self.audio, &self.audio_config) {
            let input_target =
                "model.audio_tower.subsample_conv_projection.input_proj_linear.weight";
            let output_target = "model.audio_tower.output_proj.weight";
            self.audio = Some(Gemma4AudioLayerwiseStatic::new_tensor_parallel(
                audio,
                config,
                semantic_range(&[input_target], config.hidden_size)?.len() as i32,
                semantic_widths(&[input_target], config.hidden_size)?,
                semantic_range(&[output_target], config.output_proj_dims)?.len() as i32,
                stream,
            )?);
        }
        if let Some(embedder) = &self.embed_audio {
            let targets = [
                "model.embed_audio.embedding_projection.weight",
                "model.embed_audio.embedding_projection.inner.weight",
            ];
            self.embed_audio = Some(Gemma4ModalityEmbedder::new_tensor_parallel(
                embedder.input_size,
                self.args.hidden_size,
                embedder.eps,
                false,
                self.args.weight_quantization(),
                semantic_range(&targets, embedder.input_size)?,
                stream,
            )?);
        }
        Ok(())
    }

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let local_semantic = |target: &str, global: i32| -> Result<i32, Error> {
            let tensor = layout.tensor(target).ok_or_else(|| {
                Error::Parallel(format!("missing Gemma 4 TP layout for {target}"))
            })?;
            let units = tensor.logical_units().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical domain"
                ))
            })?;
            let range = tensor.logical_range().ok_or_else(|| {
                Error::Parallel(format!(
                    "Gemma 4 TP layout for {target} has no logical range"
                ))
            })?;
            let global = usize::try_from(global)
                .map_err(|_| Error::Parallel(format!("Gemma 4 {target} geometry is invalid")))?;
            if units == 0 || !global.is_multiple_of(units) {
                return Err(Error::Parallel(format!(
                    "Gemma 4 {target} global geometry {global} is incompatible with {units} planner units"
                )));
            }
            i32::try_from(range.len() * (global / units))
                .map_err(|_| Error::Parallel(format!("Gemma 4 local {target} exceeds i32")))
        };
        let id = self.execution_group_name(group)?;
        match id {
            "vision_encoder" => {
                let cfg = &self.vision.as_ref().expect("vision group").config;
                let prefix = format!("model.vision_tower.encoder.layers.{index}");
                let q = format!("{prefix}.self_attn.q_proj.linear.weight");
                let mlp = format!("{prefix}.mlp.gate_proj.linear.weight");
                Ok(Gemma4Layer::Vision(Box::new(
                    VisionLayer::new_tensor_parallel(
                        cfg,
                        local_semantic(&q, cfg.num_attention_heads)?,
                        local_semantic(&q, cfg.num_key_value_heads)?,
                        local_semantic(&mlp, cfg.intermediate_size)?,
                        stream,
                    )?,
                )))
            }
            "audio_encoder" => {
                let cfg = self.audio_config.as_ref().expect("audio group");
                let prefix = format!("model.audio_tower.layers.{index}");
                let q = format!("{prefix}.self_attn.q_proj.linear.weight");
                let ff = format!("{prefix}.feed_forward1.ffw_layer_1.linear.weight");
                let channels = format!("{prefix}.lconv1d.linear_end.linear.weight");
                Ok(Gemma4Layer::Audio(Box::new(
                    AudioLayer::new_tensor_parallel(
                        cfg,
                        local_semantic(&q, cfg.num_attention_heads)?,
                        local_semantic(&ff, 4 * cfg.hidden_size)?,
                        local_semantic(&channels, cfg.hidden_size)?,
                        stream,
                    )?,
                )))
            }
            "text_decoder" => {
                let _ = layout;
                let geometry = self
                    .parallel_text_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.get(index))
                    .copied()
                    .ok_or_else(|| {
                        Error::Parallel(format!(
                            "Gemma 4 local geometry is unavailable for layer {index}"
                        ))
                    })?;
                Ok(Gemma4Layer::Text(Box::new(
                    TransformerBlock::new_parallel_layerwise(&self.args, index, geometry, stream)?,
                )))
            }
            _ => unreachable!(),
        }
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if self.execution_group_name(group)? != "text_decoder" {
            return Ok(layer);
        }
        let Gemma4Layer::Text(text) = &mut layer else {
            unreachable!("validated Gemma text execution group")
        };
        if text.experts.is_some() {
            let local_experts = if self.external_experts {
                0
            } else {
                i32::try_from(assignment.local_expert_count())
                    .map_err(|_| Error::Parallel("local Gemma expert count exceeds i32".into()))?
            };
            let intermediate = self.args.moe_intermediate_size.ok_or_else(|| {
                Error::Parallel(format!("Gemma 4 MoE layer {index} has no expert width"))
            })?;
            text.experts = Some(resident::GemmaExperts::new(
                &self.args,
                index,
                local_experts,
                intermediate,
                stream,
            )?);
        }
        Ok(layer)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        if self.execution_group_name(group)? != "text_decoder" {
            return Ok(layer);
        }
        let Gemma4Layer::Text(text) = &mut layer else {
            unreachable!("validated Gemma text execution group")
        };
        if let Some(experts) = &text.experts {
            let local_experts = if self.external_experts {
                0
            } else {
                i32::try_from(assignment.local_expert_count())
                    .map_err(|_| Error::Parallel("local Gemma expert count exceeds i32".into()))?
            };
            let intermediate = experts.switch_glu.gate_proj.output_dim;
            text.experts = Some(resident::GemmaExperts::new(
                &self.args,
                index,
                local_experts,
                intermediate,
                stream,
            )?);
        }
        Ok(layer)
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) -> Result<Option<crate::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.external_experts {
            return Ok(None);
        }
        if !self
            .args
            .layer_schedule
            .iter()
            .any(|policy| policy.feed_forward == resident::FeedForwardPolicy::DenseWithSparseMoe)
        {
            return Err(Error::Parallel(
                "Gemma 4 expert parallelism requires routed MoE text layers".into(),
            ));
        }
        let experts = self.args.num_experts.ok_or_else(|| {
            Error::Parallel("Gemma 4 MoE config has no global expert count".into())
        })?;
        Ok(Some(
            crate::runtime::distributed::expert::ExpertAssignment::balanced(
                usize::try_from(experts)
                    .map_err(|_| Error::Parallel("Gemma 4 expert count is negative".into()))?,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let source = self.new_layer(group, index, stream)?;
        self.layer_bindings(group, index, &source, store)
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        match self.execution_group_name(group).ok() {
            Some("vision_encoder") => format!("model.vision_tower.encoder.layers.{index}"),
            Some("audio_encoder") => format!("model.audio_tower.layers.{index}"),
            _ => format!("model.language_model.layers.{index}"),
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        match self.execution_group_name(group).ok() {
            Some("vision_encoder") => format!("gemma4.vision.{index:05}"),
            Some("audio_encoder") => format!("gemma4.audio.{index:05}"),
            _ => format!("gemma4.layer.{index:05}"),
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
    ) -> Result<Vec<WeightBinding>, Error> {
        self.bindings(layer, &self.layer_checkpoint_prefix(group, index), store)
    }

    fn populate_layer(
        &self,
        group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if self.external_experts && self.execution_group_name(group)? == "text_decoder" {
            populate_module_from_lease_excluding(layer, lease, |name| {
                name.starts_with("experts.")
            })?;
        } else {
            populate_module_from_lease(layer, lease)?;
        }
        if !self.external_experts {
            if let Gemma4Layer::Text(block) = layer {
                if let Some(experts) = &mut block.experts {
                    for projection in [
                        &mut experts.switch_glu.gate_proj,
                        &mut experts.switch_glu.up_proj,
                        &mut experts.switch_glu.down_proj,
                    ] {
                        projection.cache_native_view()?;
                    }
                }
            }
        }
        Ok(())
    }

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.external_experts {
            store
                .keys()
                .into_iter()
                .filter(|key| {
                    key.starts_with("model.language_model.layers.")
                        && key.contains(".experts.switch_glu.")
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Vec<WeightBinding>, Error> {
        let global = self.new_layer(group, index, stream)?;
        crate::runtime::execution::layerwise::shard_layer_bindings(
            self.layer_bindings(group, index, &global, store)?,
            &self.layer_checkpoint_prefix(group, index),
            store,
            layout,
        )
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
            ("vision_encoder", Gemma4Layer::Vision(layer)) => {
                for job in &mut context.vision_jobs {
                    job.hidden = layer.forward(
                        &job.hidden,
                        &job.state.padding,
                        &job.state.cos,
                        &job.state.sin,
                        stream,
                    )?;
                }
                Ok(context.vision_jobs[0].hidden.clone())
            }
            ("audio_encoder", Gemma4Layer::Audio(layer)) => {
                for job in &mut context.audio_jobs {
                    job.hidden = layer.forward(&job.hidden, job.valid, stream)?;
                }
                Ok(context.audio_jobs[0].hidden.clone())
            }
            ("text_decoder", Gemma4Layer::Text(layer)) => {
                let per_layer_input = context
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                    .transpose()?;
                let mask = context
                    .sliding_masks
                    .as_ref()
                    .and_then(|masks| masks.get(&layer.layer_policy.attention))
                    .or(context.mask.as_ref());
                let input = AttentionInput {
                    x: hidden,
                    mask,
                    cache: cache.kv[index].as_mut(),
                    position_offset: context.position_offset,
                    per_layer_input: per_layer_input.as_ref(),
                    shared_kv: Some(&mut context.shared_kv),
                    disable_generated_mask: false,
                    generated_sliding_window: None,
                };
                if self.external_experts {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::Parallel(
                            "Gemma 4 external experts require topology execution or an initialized independent expert cache"
                                .into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    return Ok(layer.forward_with_expert_executor(
                        input,
                        stream,
                        |flat, indices, weights, stream| {
                            expert_cache
                                .execute_routes_bounded(
                                    ExpertRouteBatch::new(index, flat, indices, weights, pass),
                                    stream,
                                    |flat, acquired, weights, stream| {
                                        Ok(execute_acquired_gemma_experts(
                                            &self.args,
                                            index,
                                            flat,
                                            acquired,
                                            weights,
                                            expert_cache,
                                            stream,
                                        )?)
                                    },
                                )
                                .map_err(|error| Exception::custom(error.to_string()))
                        },
                    )?);
                }
                Ok(layer.forward(input, stream)?)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 execution unit does not match group {group}"
            ))),
        }
    }

    fn forward_layer_with_observer<O: crate::runtime::execution::inspection::ActivationObserver>(
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
        if self.external_experts && self.expert_cache.is_some() {
            observer.observe(&format!("{prefix}.input"), hidden)?;
            let output = self.forward_layer(group, index, layer, hidden, cache, context, stream)?;
            observer.observe(&format!("{prefix}.output"), &output)?;
            return Ok(observer
                .intervene(&format!("{prefix}.output"), &output)?
                .unwrap_or(output));
        }
        if self.execution_group_name(group)? == "text_decoder" {
            let Gemma4Layer::Text(layer) = layer else {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Gemma 4 execution unit does not match group {group}"
                )));
            };
            let per_layer_input = context
                .per_layer_inputs
                .as_ref()
                .map(|inputs| inputs.try_index_device((.., .., index as i32, ..), stream))
                .transpose()?;
            let mask = context
                .sliding_masks
                .as_ref()
                .and_then(|masks| masks.get(&layer.layer_policy.attention))
                .or(context.mask.as_ref());
            return Ok(layer.forward_with_observer(
                AttentionInput {
                    x: hidden,
                    mask,
                    cache: cache.kv[index].as_mut(),
                    position_offset: context.position_offset,
                    per_layer_input: per_layer_input.as_ref(),
                    shared_kv: Some(&mut context.shared_kv),
                    disable_generated_mask: false,
                    generated_sliding_window: None,
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
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
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
        let id = self.execution_group_name(group)?;
        if id == "vision_encoder" {
            if let Gemma4Layer::Vision(layer) = layer {
                for job in &mut context.vision_jobs {
                    job.hidden = layer.forward_tensor_parallel(
                        &job.hidden,
                        &job.state.padding,
                        &job.state.cos,
                        &job.state.sin,
                        tp_group,
                        execution.stream(),
                    )?;
                }
                return Ok(context.vision_jobs[0].hidden.clone());
            }
        } else if id == "audio_encoder" {
            if let Gemma4Layer::Audio(layer) = layer {
                for job in &mut context.audio_jobs {
                    job.hidden = layer.forward_tensor_parallel(
                        &job.hidden,
                        job.valid,
                        tp_group,
                        execution.stream(),
                    )?;
                }
                return Ok(context.audio_jobs[0].hidden.clone());
            }
        } else if id == "text_decoder" {
            if let Gemma4Layer::Text(layer) = layer {
                let per_layer_input = context
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| {
                        inputs.try_index_device((.., .., index as i32, ..), execution.stream())
                    })
                    .transpose()?;
                let mask = context
                    .sliding_masks
                    .as_ref()
                    .and_then(|masks| masks.get(&layer.layer_policy.attention))
                    .or(context.mask.as_ref());
                return Ok(layer.forward_tensor_parallel(
                    hidden,
                    mask,
                    cache.kv[index].as_mut(),
                    context.position_offset,
                    per_layer_input.as_ref(),
                    &mut context.shared_kv,
                    tp_group,
                    execution.stream(),
                )?);
            }
        }
        self.forward_layer(
            group,
            index,
            layer,
            hidden,
            cache,
            context,
            execution.stream(),
        )
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if self.execution_group_name(group).ok() == Some("text_decoder") {
            cache.kv[index]
                .as_ref()
                .map(KeyValueCache::retained_arrays)
                .unwrap_or_default()
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
        let mut arrays = context
            .shared_kv
            .values()
            .flat_map(|(keys, values)| [keys, values])
            .collect::<Vec<_>>();
        for job in &context.vision_jobs {
            arrays.push(&job.hidden);
            arrays.extend(job.state.retained_arrays());
        }
        arrays.extend(context.audio_jobs.iter().map(|job| &job.hidden));
        arrays
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
        if self.execution_group_name(group)? != "text_decoder" {
            return Ok(dependency_outputs.first().unwrap_or(initial_hidden).clone());
        }
        if !context.needs_assembly {
            return Ok(initial_hidden.clone());
        }
        if let (Some(vision), Some(embedder)) = (&self.vision, &mut self.embed_vision) {
            for job in &mut context.vision_jobs {
                job.hidden =
                    embedder.forward(&vision.finish(&job.hidden, &job.state, stream)?, stream)?;
            }
        }
        if let (Some(audio), Some(embedder)) = (&mut self.audio, &mut self.embed_audio) {
            for job in &mut context.audio_jobs {
                job.hidden =
                    embedder.forward(&audio.finish(&job.hidden, job.valid, stream)?, stream)?;
            }
        }
        let mut token_parts = Vec::with_capacity(context.parts.len());
        let mut embedding_parts = Vec::with_capacity(context.parts.len());
        for part in &context.parts {
            match part {
                Gemma4PreparedPart::Ready { tokens, embeddings } => {
                    token_parts.push(tokens.clone());
                    embedding_parts.push(embeddings.clone());
                }
                Gemma4PreparedPart::Vision { token_id, job } => {
                    let embeddings = context.vision_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(0) as usize * embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(if embeddings.dim(0) == 1 {
                        embeddings
                    } else {
                        embeddings.reshape(&[1, -1, embeddings.dim(2)], stream)?
                    });
                }
                Gemma4PreparedPart::Audio { token_id, job } => {
                    let embeddings = context.audio_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(embeddings);
                }
            }
        }
        let token_refs = token_parts.iter().collect::<Vec<_>>();
        let embedding_refs = embedding_parts.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&token_refs, 1, stream)?;
        let hidden = concatenate_axis(&embedding_refs, 1, stream)?;
        cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
        let per_layer_ids = self.media_safe_per_layer_ids(&tokens, stream)?;
        context.per_layer_inputs =
            self.prepare_per_layer_inputs(&per_layer_ids, &hidden, stream)?;
        let masks = resident::multimodal_attention_masks(
            &cache.token_ids,
            self.image_token_id.map(|id| id as u32),
            self.video_token_id.map(|id| id as u32),
            &self.args.layer_schedule,
        );
        context.mask = Some(masks.full);
        context.sliding_masks = Some(masks.sliding);
        context.tokens = Some(tokens);
        context.needs_assembly = false;
        Ok(hidden)
    }

    fn begin_execution_group_with_execution(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(tp_group) = execution.group() else {
            return self.begin_execution_group(
                group,
                initial_hidden,
                dependency_outputs,
                cache,
                context,
                execution.stream(),
            );
        };
        if self.execution_group_name(group)? != "text_decoder" {
            return Ok(dependency_outputs.first().unwrap_or(initial_hidden).clone());
        }
        if !context.needs_assembly {
            return Ok(initial_hidden.clone());
        }
        let stream = execution.stream();
        if let (Some(vision), Some(embedder)) = (&self.vision, &mut self.embed_vision) {
            for job in &mut context.vision_jobs {
                job.hidden = embedder.forward_tensor_parallel(
                    &vision.finish(&job.hidden, &job.state, stream)?,
                    tp_group,
                    stream,
                )?;
            }
        }
        if let (Some(audio), Some(embedder)) = (&mut self.audio, &mut self.embed_audio) {
            for job in &mut context.audio_jobs {
                job.hidden = embedder.forward_tensor_parallel(
                    &audio.finish_tensor_parallel(&job.hidden, job.valid, stream)?,
                    tp_group,
                    stream,
                )?;
            }
        }
        let mut token_parts = Vec::with_capacity(context.parts.len());
        let mut embedding_parts = Vec::with_capacity(context.parts.len());
        for part in &context.parts {
            match part {
                Gemma4PreparedPart::Ready { tokens, embeddings } => {
                    token_parts.push(tokens.clone());
                    embedding_parts.push(embeddings.clone());
                }
                Gemma4PreparedPart::Vision { token_id, job } => {
                    let embeddings = context.vision_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(0) as usize * embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(if embeddings.dim(0) == 1 {
                        embeddings
                    } else {
                        embeddings.reshape(&[1, -1, embeddings.dim(2)], stream)?
                    });
                }
                Gemma4PreparedPart::Audio { token_id, job } => {
                    let embeddings = context.audio_jobs[*job].hidden.clone();
                    token_parts.push(input::token_ids_array(
                        &vec![*token_id; embeddings.dim(1) as usize],
                        stream,
                    )?);
                    embedding_parts.push(embeddings);
                }
            }
        }
        let tokens = concatenate_axis(&token_parts.iter().collect::<Vec<_>>(), 1, stream)?;
        let hidden = concatenate_axis(&embedding_parts.iter().collect::<Vec<_>>(), 1, stream)?;
        cache.token_ids = resident::token_ids_from_array(&tokens, stream)?;
        let per_layer_ids = self.media_safe_per_layer_ids(&tokens, stream)?;
        context.per_layer_inputs = self.prepare_per_layer_inputs_with_execution(
            &per_layer_ids,
            &hidden,
            Some(execution),
            stream,
        )?;
        let masks = resident::multimodal_attention_masks(
            &cache.token_ids,
            self.image_token_id.map(|id| id as u32),
            self.video_token_id.map(|id| id as u32),
            &self.args.layer_schedule,
        );
        context.mask = Some(masks.full);
        context.sliding_masks = Some(masks.sliding);
        context.tokens = Some(tokens);
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
        let mut logits = match self.lm_head.as_mut() {
            Some(head) => head.forward(&hidden, stream)?,
            None => self.embedding.as_linear(&hidden, stream)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(&logits.divide(Array::from_f32(softcap), stream)?, stream)?
                .multiply(Array::from_f32(softcap), stream)?;
        }
        Ok(logits)
    }

    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<Array, Error> {
        let Some(group) = execution.group() else {
            return self.finish(hidden, cache, context, execution.stream());
        };
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let local = if let Some(head) = &mut self.parallel_lm_head {
            return head
                .forward(&hidden, execution)?
                .all_gather(execution)
                .and_then(|logits| {
                    if let Some(softcap) = self.args.final_logit_softcapping {
                        Ok(tanh(
                            &logits.divide(Array::from_f32(softcap), execution.stream())?,
                            execution.stream(),
                        )?
                        .multiply(Array::from_f32(softcap), execution.stream())?)
                    } else {
                        Ok(logits)
                    }
                });
        } else {
            self.embedding.as_linear(&hidden, execution.stream())?
        };
        let widths = (0..execution.size())
            .map(|rank| {
                crate::core::balanced_contiguous_range(
                    self.args.vocab_size as usize,
                    execution.size(),
                    rank,
                    false,
                )
                .map(|range| range.len())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut logits = safemlx::distributed::all_gather_uneven_axis(
            &local,
            -1,
            &widths,
            group,
            execution.stream(),
        )?;
        if let Some(softcap) = self.args.final_logit_softcapping {
            logits = tanh(
                &logits.divide(Array::from_f32(softcap), execution.stream())?,
                execution.stream(),
            )?
            .multiply(Array::from_f32(softcap), execution.stream())?;
        }
        Ok(logits)
    }

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        ignores_gemma4_checkpoint_key(key)
    }
}

/// Gemma 4 token generation using bounded text-layer execution.
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    crate::nn::generation::Generate<'a, Gemma4LayerwiseModel, Cache, S>;

#[cfg(test)]
mod tests {
    use crate::backend::mlx::{DeviceAssignment, MlxParallelContext};
    use std::{collections::HashMap, fs, path::Path};

    use safemlx::{
        distributed::{Backend, Group},
        module::ModuleParameters,
        ops::ones_dtype,
        Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
    };

    use super::*;
    use crate::{
        api::{
            common::generation::CausalLm,
            gemma4::{self as resident, Model, ModelInput},
            gemma4_audio::Gemma4AudioConfig,
            gemma4_vision::Gemma4VisionConfig,
            input as runtime_input,
        },
        core::residency::{MemoryTier, OffloadConfig},
        runtime::{
            cache::ConcatKeyValueCache,
            checkpoint::quantization::{AffineQuantization, WeightQuantization},
            distributed::parallel::{ParallelBuildContext, ShardingPolicy},
            execution::inspection::ActivationRecorder,
            execution::layerwise::{
                ArchitectureAdapter, LayerWeightResidency, LayerwiseLoadOptions,
                LoadTimeQuantizableAdapter,
            },
            residency::dense_stream::DenseDiskStreamLoadOptions,
        },
    };

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "gemma4",
                "hidden_size": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "pad_token_id": 0,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "head_dim": 4,
                "attention_bias": false,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 32,
                "num_kv_shared_layers": 1,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                "sliding_window": 8,
                "final_logit_softcapping": 4.0
            }
        })
    }

    #[test]
    fn strict_layerwise_loading_ignores_auxiliary_gguf_rope_frequencies() {
        assert!(super::ignores_gemma4_checkpoint_key("rope_freqs.weight"));
        assert!(!super::ignores_gemma4_checkpoint_key(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ));
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn load_time_adapter_packs_aligned_gemma4_media_projections() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["text_config"]["hidden_size"] = 32.into();
        value["text_config"]["intermediate_size"] = 64.into();
        value["text_config"]["num_attention_heads"] = 4.into();
        value["text_config"]["num_key_value_heads"] = 2.into();
        value["text_config"]["head_dim"] = 8.into();
        value["text_config"]["hidden_size_per_layer_input"] = 0.into();
        value["text_config"]["num_kv_shared_layers"] = 0.into();
        let args = resident::model_args_from_config_value(&value["text_config"]).unwrap();
        let vision = Gemma4VisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            patch_size: 2,
            pooling_kernel_size: 2,
            position_embedding_size: 4096,
            rms_norm_eps: 1e-6,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
        };
        let audio = Gemma4AudioConfig {
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            output_proj_dims: 32,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 4,
            attention_context_right: 0,
            attention_invalid_logits_value: -1.0e9,
            attention_logit_cap: 10.0,
            residual_weight: 0.5,
            rms_norm_eps: 1e-6,
            subsampling_conv_channels: vec![4, 4],
            weight_quantization: None,
        };
        let source = Gemma4LayerwiseAdapter::new(
            args.clone(),
            Some(vision.clone()),
            Some(20),
            Some(21),
            Some(audio.clone()),
            Some(22),
            gpu.stream(),
        )
        .unwrap();
        let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
        let target = source
            .load_time_quantized(quantization, gpu.stream())
            .unwrap();

        assert_eq!(target.quantization(), Some(quantization));
        assert!(matches!(
            target.vision.as_ref().unwrap().patch_embedder.input_proj,
            MaybeQuantized::Original(_)
        ));
        assert!(target
            .audio
            .as_ref()
            .unwrap()
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));
        let Gemma4Layer::Vision(vision_layer) = target.new_layer(0, 0, gpu.stream()).unwrap()
        else {
            panic!("Gemma 4 vision group must build a vision layer")
        };
        assert!(vision_layer
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));
        let Gemma4Layer::Audio(audio_layer) = target.new_layer(1, 0, gpu.stream()).unwrap() else {
            panic!("Gemma 4 audio group must build an audio layer")
        };
        assert!(audio_layer
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));

        let mut fixture = Model::new_with_modalities(
            args.clone(),
            Some(20),
            Some(vision.clone()),
            Some(21),
            Some(22),
            Some(audio.clone()),
            gpu.stream(),
        )
        .unwrap();
        initialize(&mut fixture, gpu.stream());
        let store = transformed_module_weight_store(&fixture).unwrap();
        let execution = load_layerwise_model_with_quantization(
            store,
            Gemma4LayerwiseAdapter::new(
                args,
                Some(vision),
                Some(20),
                Some(21),
                Some(audio),
                Some(22),
                gpu.stream(),
            )
            .unwrap(),
            LayerWeightResidency::FullyResident,
            Some(quantization),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let keys = execution.checkpoint_store_arc().keys();
        assert!(
            keys.iter()
                .any(|key| key
                    == "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.scales")
        );
        assert!(keys
            .iter()
            .any(|key| key == "model.audio_tower.layers.0.self_attn.q_proj.linear.scales"));
        assert!(keys.iter().any(
            |key| key == "model.audio_tower.subsample_conv_projection.input_proj_linear.scales"
        ));
        assert!(!keys
            .iter()
            .any(|key| key == "model.vision_tower.patch_embedder.input_proj.scales"));
        let report = execution.residency_report().unwrap();
        let materialization = report.materialization().unwrap();
        assert!(materialization.transformed_weights > 20);
        assert!(materialization.output_bytes < materialization.source_bytes_read);
        assert!(materialization.peak_planned_working_set_bytes <= materialization.output_bytes);

        let mut quantized = Gemma4LayerwiseModel { execution };
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let pixels = Array::zeros::<f32>(&[1, 4, 12], gpu.stream()).unwrap();
        let positions = Array::from_slice(&[0i32, 0, 0, 1, 1, 0, 1, 1], &[1, 4, 2]);
        let audio_features = Array::zeros::<f32>(&[1, 8, 128], gpu.stream()).unwrap();
        let audio_mask = Array::from_slice(&[true; 8], &[1, 8]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::patch_position_ids(&positions),
            ),
            runtime_input::InputPart::audio_tensor(
                &audio_features,
                runtime_input::InputMetadata::audio_mask(&audio_mask),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut dense_cache = Cache::default();
        let mut quantized_cache = quantized.new_cache();
        let expected = fixture
            .prefill_input_logits(typed, &mut dense_cache, gpu.stream())
            .unwrap();
        let actual = quantized
            .prefill_input_logits(typed, &mut quantized_cache, gpu.stream())
            .unwrap();
        assert!(actual
            .all_close(&expected, Some(2e-2), Some(2e-2), None, gpu.stream())
            .unwrap()
            .item::<bool>(gpu.stream()));

        value["image_token_id"] = serde_json::json!(20);
        value["video_token_id"] = serde_json::json!(21);
        value["audio_token_id"] = serde_json::json!(22);
        value["tie_word_embeddings"] = serde_json::json!(true);
        value["text_config"]["tie_word_embeddings"] = serde_json::json!(true);
        value["vision_config"] = serde_json::json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4096,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "standardize": false
        });
        value["audio_config"] = serde_json::json!({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "output_proj_dims": 32,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 4,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1.0e9,
            "attention_logit_cap": 10.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 1e-6,
            "subsampling_conv_channels": [4, 4]
        });
        let dir = tempfile::tempdir().unwrap();
        let arrays = fixture
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(name)
                    .replacen("model.language_model.", "language_model.model.", 1);
                (name, *value)
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let eager_quantized = resident::load_gemma4_model_quantized(
            dir.path(),
            quantization,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        assert!(eager_quantized
            .model
            .vision_tower
            .as_ref()
            .unwrap()
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));
        assert!(eager_quantized
            .model
            .audio_tower
            .as_ref()
            .unwrap()
            .parameters()
            .flatten()
            .values()
            .any(|parameter| parameter.dtype() == Dtype::Uint32));
    }

    fn initialize(model: &mut Model, stream: &Stream) {
        let mut names = model
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut params = model.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = params.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                ones_dtype(&shape, dtype, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.0005 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap()
            };
        }
    }

    fn write_fixture(dir: &Path, model: &Model) {
        let arrays = model
            .parameters()
            .flatten()
            .iter()
            .map(|(name, value)| {
                let name = crate::runtime::checkpoint::binding::canonical_checkpoint_name(name)
                    .replacen("model.language_model.", "language_model.model.", 1);
                (name, *value)
            })
            .collect::<Vec<_>>();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
    }

    fn assert_close(left: &Array, right: &Array) {
        let left = left.evaluated().unwrap();
        let right = right.evaluated().unwrap();
        assert_eq!(left.as_array().shape(), right.as_array().shape());
        for (left, right) in left.as_slice::<f32>().iter().zip(right.as_slice::<f32>()) {
            assert!((left - right).abs() <= 5e-5, "{left} != {right}");
        }
    }

    fn parity(depth: usize) {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args = resident::model_args_from_config_value(&config()["text_config"]).unwrap();
        args.tie_word_embeddings = false;
        let mut fixture = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);

        let mut eager =
            resident::load_gemma4_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_gemma4_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, depth).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut eager_cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let mut layerwise_cache = layerwise.new_cache();
        for tokens in [
            Array::from_slice(&[1u32, 2, 3], &[1, 3]),
            Array::from_slice(&[4u32], &[1, 1]),
            Array::from_slice(&[5u32], &[1, 1]),
        ] {
            let expected = eager
                .forward_logits(
                    ModelInput {
                        inputs: &tokens,
                        inputs_embeds: None,
                        per_layer_input_ids: None,
                        mask: None,
                        sliding_masks: None,
                        cache: &mut eager_cache,
                    },
                    false,
                    gpu.stream(),
                )
                .unwrap();
            let actual = layerwise
                .forward(&tokens, &mut layerwise_cache, gpu.stream())
                .unwrap();
            assert_close(&actual, &expected);
        }
        assert_eq!(
            layerwise_cache
                .kv
                .iter()
                .map(|cache| cache.as_ref().map_or(0, KeyValueCache::offset))
                .collect::<Vec<_>>(),
            vec![5, 5, 5, 0]
        );
        let report = layerwise.residency_report().unwrap();
        let resident_layers = report
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().starts_with("gemma4.layer."))
            .filter(|unit| unit.device_resident())
            .count();
        assert!(resident_layers <= depth);
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn tensor_parallel_dense_stream_loads_text_static_and_decoder_group() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args = resident::model_args_from_config_value(&config()["text_config"]).unwrap();
        args.tie_word_embeddings = false;
        let mut fixture = Model::new(args, gpu.stream()).unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);

        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let topology =
            MlxParallelContext::for_rank(0, 1, 1, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
                .unwrap();
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let model = load_gemma4_tensor_parallel_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(options),
            build,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(report
            .residency()
            .units()
            .iter()
            .filter(|unit| unit.id().as_str().contains("gemma4.layer."))
            .all(|unit| unit.planned_tier() == MemoryTier::Disk));
    }

    #[test]
    fn tensor_parallel_plan_derives_uneven_text_vision_and_audio_geometry() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["text_config"]["hidden_size"] = 12.into();
        value["text_config"]["intermediate_size"] = 7.into();
        value["text_config"]["num_attention_heads"] = 6.into();
        value["text_config"]["num_key_value_heads"] = 3.into();
        value["text_config"]["head_dim"] = 2.into();
        value["text_config"]["hidden_size_per_layer_input"] = 0.into();
        value["text_config"]["enable_moe_block"] = true.into();
        value["text_config"]["num_experts"] = 2.into();
        value["text_config"]["top_k_experts"] = 1.into();
        value["text_config"]["moe_intermediate_size"] = 5.into();
        let args = resident::model_args_from_config_value(&value["text_config"]).unwrap();
        let vision = Gemma4VisionConfig {
            hidden_size: 11,
            intermediate_size: 7,
            num_hidden_layers: 1,
            num_attention_heads: 6,
            num_key_value_heads: 3,
            head_dim: 2,
            patch_size: 2,
            pooling_kernel_size: 1,
            position_embedding_size: 4,
            rms_norm_eps: 1e-6,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
        };
        let audio = Gemma4AudioConfig {
            hidden_size: 12,
            num_hidden_layers: 1,
            num_attention_heads: 3,
            output_proj_dims: 7,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 4,
            attention_context_right: 0,
            attention_invalid_logits_value: -1.0e9,
            attention_logit_cap: 10.0,
            residual_weight: 0.5,
            rms_norm_eps: 1e-6,
            subsampling_conv_channels: vec![2, 2],
            weight_quantization: None,
        };

        for (rank, query_heads, kv_heads, dense, expert, vision_hidden, vision_mlp, audio_heads) in
            [(0, 4, 2, 4, 3, 6, 4, 2), (1, 2, 1, 3, 2, 5, 3, 1)]
        {
            let mut adapter = Gemma4LayerwiseAdapter::new(
                args.clone(),
                Some(vision.clone()),
                Some(20),
                Some(21),
                Some(audio.clone()),
                Some(22),
                execution.stream(),
            )
            .unwrap();
            let topology = MlxParallelContext::for_rank(
                rank,
                2,
                1,
                1,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap();
            let context = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let mut planner = context.planner();
            adapter
                .register_parallel_parameters(context, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            adapter
                .configure_parallel_static(context, &layout, execution.stream())
                .unwrap();

            let geometry = adapter.parallel_text_geometry.as_ref().unwrap();
            assert!(geometry.iter().all(|geometry| {
                geometry.query_heads == query_heads
                    && geometry.kv_heads == kv_heads
                    && geometry.dense_intermediate == dense
                    && geometry.expert_intermediate == Some(expert)
            }));
            assert_eq!(
                adapter
                    .vision
                    .as_ref()
                    .unwrap()
                    .patch_embedder
                    .position_embedding_table
                    .dim(-1),
                vision_hidden
            );
            assert_eq!(
                adapter
                    .embed_vision
                    .as_ref()
                    .unwrap()
                    .parallel_input_range
                    .as_ref()
                    .unwrap()
                    .len(),
                vision_hidden as usize
            );
            assert_eq!(
                adapter
                    .embed_audio
                    .as_ref()
                    .unwrap()
                    .parallel_input_range
                    .as_ref()
                    .unwrap()
                    .len(),
                if rank == 0 { 4 } else { 3 }
            );

            let Gemma4Layer::Vision(layer) = adapter
                .new_parallel_layer(0, 0, &layout, execution.stream())
                .unwrap()
            else {
                panic!("expected Gemma 4 vision layer")
            };
            assert_eq!(layer.self_attn.num_heads, query_heads);
            assert_eq!(layer.self_attn.num_kv_heads, kv_heads);
            assert_eq!(layer.mlp.gate_proj.output_dim, vision_mlp);

            let Gemma4Layer::Audio(layer) = adapter
                .new_parallel_layer(1, 0, &layout, execution.stream())
                .unwrap()
            else {
                panic!("expected Gemma 4 audio layer")
            };
            assert_eq!(layer.self_attn.heads, audio_heads);
            assert_eq!(layer.lconv1d.global_hidden_size, audio.hidden_size);

            let Gemma4Layer::Text(layer) = adapter
                .new_parallel_layer(2, 0, &layout, execution.stream())
                .unwrap()
            else {
                panic!("expected Gemma 4 text layer")
            };
            assert_eq!(layer.num_attention_heads, query_heads);
            assert_eq!(
                layer.layer_policy.num_key_value_heads.get(),
                kv_heads as u32
            );
            assert_eq!(layer.mlp.hidden_dim, dense);
            assert_eq!(
                layer
                    .experts
                    .as_ref()
                    .unwrap()
                    .switch_glu
                    .gate_proj
                    .output_dim,
                expert
            );

            let identity = adapter.prompt_cache_model_identity(Some(topology)).unwrap();
            match identity.layer_layout.get(0).unwrap() {
                crate::LayerCachePolicy::KeyValueWithFixedState {
                    num_key_value_heads,
                    ..
                } => assert_eq!(num_key_value_heads.get(), kv_heads as u32),
                policy => panic!("unexpected Gemma 4 cache policy {policy:?}"),
            }
        }
    }

    #[test]
    fn tensor_parallel_plan_keeps_gemma_expert_companions_block_aligned() {
        use crate::runtime::checkpoint::quantization::WeightQuantization;

        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut value = config();
        value["text_config"]["hidden_size"] = 12.into();
        value["text_config"]["num_attention_heads"] = 6.into();
        value["text_config"]["num_key_value_heads"] = 3.into();
        value["text_config"]["head_dim"] = 2.into();
        value["text_config"]["hidden_size_per_layer_input"] = 0.into();
        value["text_config"]["enable_moe_block"] = true.into();
        value["text_config"]["num_experts"] = 2.into();
        value["text_config"]["top_k_experts"] = 1.into();
        value["text_config"]["moe_intermediate_size"] = 96.into();
        let mut args = resident::model_args_from_config_value(&value["text_config"]).unwrap();
        args.num_hidden_layers = 1;
        args.layer_schedule =
            crate::LayerSchedule::new(1, vec![*args.layer_policy(0).unwrap()]).unwrap();
        args.quantized_weight_configs = Some(HashMap::from([(
            "model.language_model.layers.0.experts.switch_glu.down_proj.weight".into(),
            WeightQuantization::MxFp4,
        )]));

        for (rank, expert, packed, companions) in [(0, 64, 8, 2), (1, 32, 4, 1)] {
            let mut adapter = Gemma4LayerwiseAdapter::new(
                args.clone(),
                None,
                None,
                None,
                None,
                None,
                execution.stream(),
            )
            .unwrap();
            let topology = MlxParallelContext::for_rank(
                rank,
                2,
                1,
                1,
                DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap();
            let context = ParallelBuildContext::new(topology, ShardingPolicy::Require);
            let mut planner = context.planner();
            adapter
                .register_parallel_parameters(context, &mut planner, execution.stream())
                .unwrap();
            let (_, layout) = planner.finish().unwrap();
            adapter
                .configure_parallel_static(context, &layout, execution.stream())
                .unwrap();
            assert_eq!(
                adapter.parallel_text_geometry.as_ref().unwrap()[0].expert_intermediate,
                Some(expert)
            );
            let prefix = "model.language_model.layers.0.experts.switch_glu.down_proj";
            assert_eq!(
                layout
                    .tensor(&format!("{prefix}.weight"))
                    .unwrap()
                    .local_shape()[2],
                packed
            );
            assert_eq!(
                layout
                    .tensor(&format!("{prefix}.scales"))
                    .unwrap()
                    .local_shape()[2],
                companions
            );
        }
    }

    #[test]
    fn gemma4_per_layer_inputs_and_shared_kv_parity() {
        parity(1);
        parity(2);
    }

    #[test]
    fn gemma4_multimodal_vision_audio_and_text_group_parity() {
        let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let mut args = resident::model_args_from_config_value(&config()["text_config"]).unwrap();
        args.tie_word_embeddings = false;
        let vision = Gemma4VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 2,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-6,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
        };
        let audio = Gemma4AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            output_proj_dims: 8,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 4,
            attention_context_right: 0,
            attention_invalid_logits_value: -1.0e9,
            attention_logit_cap: 10.0,
            residual_weight: 0.5,
            rms_norm_eps: 1e-6,
            subsampling_conv_channels: vec![2, 2],
            weight_quantization: None,
        };
        let mut fixture = Model::new_with_modalities(
            args,
            Some(20),
            Some(vision),
            Some(21),
            Some(22),
            Some(audio),
            gpu.stream(),
        )
        .unwrap();
        initialize(&mut fixture, gpu.stream());
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), &fixture);
        let mut value = config();
        value["image_token_id"] = serde_json::json!(20);
        value["video_token_id"] = serde_json::json!(21);
        value["audio_token_id"] = serde_json::json!(22);
        value["vision_config"] = serde_json::json!({
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "standardize": false,
        });
        value["audio_config"] = serde_json::json!({
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 4,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1.0e9,
            "attention_logit_cap": 10.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 1e-6,
            "subsampling_conv_channels": [2, 2],
        });
        fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();

        let mut resident =
            resident::load_gemma4_model(dir.path(), gpu.stream(), cpu.stream()).unwrap();
        let mut layerwise = load_gemma4_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut serial = load_gemma4_layerwise_model(
            dir.path(),
            LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        serial.execution.force_serial_reference(true);
        let graph = layerwise.execution.execution_graph();
        assert_eq!(
            graph
                .groups()
                .iter()
                .map(|group| group.id())
                .collect::<Vec<_>>(),
            ["vision_encoder", "audio_encoder", "text_decoder"]
        );
        assert_eq!(graph.dependencies(2), Some([0, 1].as_slice()));
        assert_eq!(graph.output(), 2);
        let text = runtime_input::token_ids_array(&[1, 2], gpu.stream()).unwrap();
        let pixels = Array::zeros::<f32>(&[1, 4096, 12], gpu.stream()).unwrap();
        let positions = (0..64)
            .flat_map(|row| (0..64).flat_map(move |column| [row, column]))
            .collect::<Vec<i32>>();
        let positions = Array::from_slice(&positions, &[1, 4096, 2]);
        let audio_features = Array::zeros::<f32>(&[1, 8, 128], gpu.stream()).unwrap();
        let audio_mask = Array::from_slice(&[true; 8], &[1, 8]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::patch_position_ids(&positions),
            ),
            runtime_input::InputPart::audio_tensor(
                &audio_features,
                runtime_input::InputMetadata::audio_mask(&audio_mask),
            ),
        ];
        let typed = runtime_input::ModelInput::new(&parts);
        let mut resident_cache = Cache::default();
        let mut layerwise_cache = layerwise.new_cache();
        let mut serial_cache = serial.new_cache();
        let expected = resident
            .prefill_input_logits(typed, &mut resident_cache, gpu.stream())
            .unwrap();
        let actual = layerwise
            .prefill_input_logits(typed, &mut layerwise_cache, gpu.stream())
            .unwrap();
        let trace = layerwise.execution.ready_set_trace();
        let vision_stream = trace
            .submissions()
            .iter()
            .find_map(|&(group, _, stream)| (group == 0).then_some(stream))
            .unwrap();
        let audio_stream = trace
            .submissions()
            .iter()
            .find_map(|&(group, _, stream)| (group == 1).then_some(stream))
            .unwrap();
        assert_ne!(vision_stream, audio_stream);
        assert!(trace
            .independent_group_events()
            .iter()
            .any(|&(producer, consumer)| [producer, consumer] == [0, 1]));
        let serial_actual = serial
            .prefill_input_logits(typed, &mut serial_cache, gpu.stream())
            .unwrap();
        assert_close(&actual, &expected);
        assert_close(&actual, &serial_actual);
        let mut concurrent_observer = ActivationRecorder::default();
        let mut serial_observer = ActivationRecorder::default();
        let mut concurrent_observer_cache = layerwise.new_cache();
        let mut serial_observer_cache = serial.new_cache();
        layerwise
            .prefill_input_with_observer(
                typed,
                &mut concurrent_observer_cache,
                gpu.stream(),
                &mut concurrent_observer,
            )
            .unwrap();
        serial
            .prefill_input_with_observer(
                typed,
                &mut serial_observer_cache,
                gpu.stream(),
                &mut serial_observer,
            )
            .unwrap();
        assert_eq!(
            concurrent_observer
                .activations()
                .iter()
                .map(|activation| activation.name.as_str())
                .collect::<Vec<_>>(),
            serial_observer
                .activations()
                .iter()
                .map(|activation| activation.name.as_str())
                .collect::<Vec<_>>()
        );
        let report = layerwise.residency_report().unwrap();
        assert!(report
            .units()
            .iter()
            .any(|unit| unit.id().as_str().starts_with("gemma4.vision.")));
        assert!(report
            .units()
            .iter()
            .any(|unit| unit.id().as_str().starts_with("gemma4.audio.")));

        let dense_options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let mut dense = load_gemma4_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(dense_options),
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let mut dense_cache = dense.new_cache();
        let dense_actual = dense
            .prefill_input_logits(typed, &mut dense_cache, gpu.stream())
            .unwrap();
        assert_close(&dense_actual, &expected);
        assert!(dense
            .execution
            .ready_set_trace()
            .independent_group_events()
            .iter()
            .any(|&(producer, consumer)| [producer, consumer] == [0, 1]));
        let dense_report = dense.dense_stream_report().unwrap().unwrap();
        assert_eq!(dense_report.prefill_forwards(), 1);
        assert!(dense_report
            .execution_groups()
            .iter()
            .all(|group| group.completed_executions() == 1));

        let group = Group::init(false, Backend::Any).unwrap();
        assert_eq!(group.size(), 1);
        let topology =
            MlxParallelContext::for_rank(0, 1, 1, 1, DeviceAssignment::new(DeviceType::Gpu, 0))
                .unwrap();
        let build = ParallelBuildContext::new(topology, ShardingPolicy::Require);
        let options = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap();
        let tp = load_gemma4_tensor_parallel_layerwise_model(
            dir.path(),
            LayerWeightResidency::DenseDiskStream(options),
            build,
            gpu.stream(),
            cpu.stream(),
        )
        .unwrap();
        let report = tp.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.execution_groups().len(), 3);
    }
}
