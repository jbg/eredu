//! Bounded-residency execution for Kimi Linear safetensors and GGUF checkpoints.

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::{OffloadUnit, WeightBinding};

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Instant};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
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
        linear::{
            project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            unloaded_maybe_quantized_linear,
        },
        parallel::{
            planned_optional_partition_widths, register_swiglu_projection_group,
            SwiGluProjectionNames, VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::create_causal_mask,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::{
        cache::residency::PagedCacheOptions,
        checkpoint::{
            binding::{
                build_module_binding_plan_with_recipes,
                build_module_binding_plan_with_recipes_excluding, canonical_checkpoint_name,
                populate_module_from_lease, populate_module_from_lease_excluding,
            },
            binding_plan::{BindingPlan, PlannedBinding},
            quantization::should_quantize_on_load,
            recipe::DerivedWeightRecipe,
            store::{open_gguf_checkpoint_source, TensorSelection, WeightStoreBackend},
        },
        distributed::parallel::{
            aligned_partition_units, array_parameter_member, partitioned_projection_members,
            register_projection_module, register_replicated_module, MemberSharding,
            ParallelPlanBuilder, ParameterGroupSpec, ParameterRole, ProjectionSharding,
        },
        execution::layerwise::{
            load_layerwise_model, load_layerwise_model_with_quantization,
            load_tensor_parallel_layerwise_model, open_safetensors_weight_store,
            ArchitectureAdapter, LayerWeightResidency, LayerwiseForwardState, LayerwiseModel,
            LoadTimeQuantizableAdapter, StaticUnitBindings, WeightResidency,
        },
        residency::{
            expert_cache::{
                ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertCatalogEntry,
                ExpertIdentity, ExpertPass, ExpertRouteBatch,
            },
            manager::ResidentUnitLease,
        },
    },
};

use eredu_runtime::ResidencyReport;

use super::model::{
    self as resident, Cache, DecoderLayer, FeedForward, FeedForwardPolicy, ModelArgs,
};

const EMBEDDING_UNIT: &str = "kimi_linear.static.embedding";
const NORM_UNIT: &str = "kimi_linear.static.norm";
const HEAD_UNIT: &str = "kimi_linear.static.output";

fn register_kimi_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &DecoderLayer,
    args: &ModelArgs,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    match &layer.self_attn {
        resident::Attention::Kda(attention) => {
            let attention_prefix = format!("{prefix}.self_attn");
            let q_prefix = format!("{attention_prefix}.q_proj");
            let k_prefix = format!("{attention_prefix}.k_proj");
            let v_prefix = format!("{attention_prefix}.v_proj");
            let f_prefix = format!("{attention_prefix}.f_b_proj");
            let b_prefix = format!("{attention_prefix}.b_proj");
            let g_prefix = format!("{attention_prefix}.g_b_proj");
            let o_prefix = format!("{attention_prefix}.o_proj");
            let preferred_heads = usize::try_from(attention.num_heads)
                .map_err(|_| Error::Parallel("Kimi KDA head count exceeds usize".into()))?;
            let (head_units, mut members) = partitioned_projection_members(
                &[
                    (
                        &attention.q_proj,
                        q_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.k_proj,
                        k_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.v_proj,
                        v_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.f_b_proj,
                        f_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.b_proj,
                        b_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.g_b_proj,
                        g_prefix.as_str(),
                        ProjectionSharding::Column,
                    ),
                    (
                        &attention.o_proj,
                        o_prefix.as_str(),
                        ProjectionSharding::Row,
                    ),
                ],
                preferred_heads,
            )?;
            for (name, convolution) in [
                ("q_conv1d", &attention.q_conv1d),
                ("k_conv1d", &attention.k_conv1d),
                ("v_conv1d", &attention.v_conv1d),
            ] {
                members.push(array_parameter_member(
                    format!("{attention_prefix}.{name}.weight"),
                    convolution.weight.as_ref(),
                    MemberSharding::Partitioned { axis: 0 },
                )?);
            }
            for (name, projection) in [
                ("f_a_proj", &attention.f_a_proj),
                ("g_a_proj", &attention.g_a_proj),
            ] {
                register_projection_module(
                    planner,
                    projection,
                    &format!("{prefix}.self_attn.{name}"),
                    ProjectionSharding::Replicated,
                )?;
            }
            for (name, value, axis) in [
                ("A_log", attention.A_log.as_ref(), 2usize),
                ("dt_bias", attention.dt_bias.as_ref(), 0usize),
            ] {
                members.push(array_parameter_member(
                    format!("{attention_prefix}.{name}"),
                    value,
                    MemberSharding::Partitioned { axis },
                )?);
            }
            register_replicated_module(
                planner,
                &attention.o_norm,
                &format!("{prefix}.self_attn.o_norm"),
            )?;
            planner.register(ParameterGroupSpec::partitioned(
                format!("{attention_prefix}.heads"),
                ParameterRole::AttentionHeads,
                head_units,
                members,
            )?)?;
        }
        resident::Attention::Mla(attention) => {
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
                .map_err(|_| Error::Parallel("Kimi MLA head count exceeds usize".into()))?;
            let (mut head_units, mut members) =
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
                let (packed_units, packed_members) =
                    partitioned_projection_members(&packed, head_units)?;
                head_units = packed_units;
                members.extend(packed_members);
            }
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
            planner.register(ParameterGroupSpec::partitioned(
                format!("{attention_prefix}.heads"),
                ParameterRole::AttentionHeads,
                head_units,
                members,
            )?)?;
            for (name, norm) in [
                ("q_a_layernorm", attention.q_a_layernorm.as_ref()),
                ("kv_a_layernorm", Some(&attention.kv_a_layernorm)),
            ] {
                if let Some(norm) = norm {
                    register_replicated_module(
                        planner,
                        norm,
                        &format!("{prefix}.self_attn.{name}"),
                    )?;
                }
            }
            register_replicated_module(
                planner,
                &attention.rope,
                &format!("{prefix}.self_attn.rope"),
            )?;
        }
    }
    for (name, norm) in [
        ("input_layernorm", &layer.input_layernorm),
        ("post_attention_layernorm", &layer.post_attention_layernorm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }
    let register_swiglu = |planner: &mut ParallelPlanBuilder,
                           mlp: &crate::backend::mlx::nn::layers::SwiGluMlp,
                           prefix: &str,
                           intermediate: i32|
     -> Result<(), Error> {
        register_swiglu_projection_group(
            planner,
            prefix,
            SwiGluProjectionNames {
                gate: "gate_proj",
                up: "up_proj",
                down: "down_proj",
            },
            &mlp.gate_proj,
            &mlp.up_proj,
            &mlp.down_proj,
            intermediate,
        )
    };
    match &layer.mlp {
        resident::FeedForward::Dense(mlp) => register_swiglu(
            planner,
            mlp,
            &format!("{prefix}.mlp"),
            args.intermediate_size,
        )?,
        resident::FeedForward::Moe(moe) => {
            let source_prefix = format!("{prefix}.mlp");
            register_replicated_module(planner, &moe.gate, &format!("{source_prefix}.gate"))?;
            register_swiglu(
                planner,
                &moe.shared_experts,
                &format!("{source_prefix}.shared_experts"),
                args.moe_intermediate_size * args.num_shared_experts,
            )?;
            let experts = &moe.experts;
            let intermediate = usize::try_from(experts.intermediate_dim)
                .map_err(|_| Error::Parallel("Kimi expert width exceeds usize".into()))?;
            let segments = vec![0..intermediate, intermediate..2 * intermediate];
            let down_alignment =
                experts
                    .down_affine
                    .or(experts.down_iquant)
                    .map_or(Ok(1usize), |quantization| {
                        usize::try_from(quantization.group_size()).map_err(|_| {
                            Error::Parallel("Kimi expert quantization group exceeds usize".into())
                        })
                    })?;
            let expert_units = aligned_partition_units(
                &format!("{source_prefix}.experts.intermediate"),
                intermediate,
                1,
                down_alignment,
            )?;
            let mut members = vec![array_parameter_member(
                format!("{source_prefix}.experts.gate_up_proj"),
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
                        format!("{source_prefix}.experts.{name}"),
                        value,
                        MemberSharding::PartitionedSegments {
                            axis: 1,
                            segments: segments.clone(),
                        },
                    )?);
                }
            }
            members.push(array_parameter_member(
                format!("{source_prefix}.experts.down_proj"),
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
                        format!("{source_prefix}.experts.{name}"),
                        value,
                        MemberSharding::Partitioned { axis: 2 },
                    )?);
                }
            }
            planner.register(ParameterGroupSpec::partitioned(
                format!("{source_prefix}.experts.intermediate"),
                ParameterRole::ExpertIntermediate,
                expert_units,
                members,
            )?)?;
        }
    }
    Ok(())
}

/// Kimi Linear causal LM with a bounded decoder-layer window.
pub struct KimiLinearLayerwiseModel {
    execution: LayerwiseModel<KimiLinearLayerwiseAdapter>,
}

impl KimiLinearLayerwiseModel {
    /// Returns the validated architecture configuration.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Creates an empty heterogeneous KDA/MLA cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.args())
    }

    /// Creates device-resident or blockwise-paged MLA state. KDA's bounded
    /// convolution and recurrent tensors remain resident under either policy.
    pub fn new_cache_with_options(
        &self,
        policy: crate::backend::mlx::runtime::cache::residency::CacheResidencyPolicy,
    ) -> Result<Cache, Error> {
        Cache::new_with_options_and_rank(
            self.args(),
            policy,
            self.execution.prompt_cache_rank_identity(),
        )
        .map_err(Into::into)
    }

    /// Returns aggregate live MLA paging observations, if paging is enabled.
    pub fn cache_residency_report(
        &self,
        cache: &Cache,
    ) -> Result<Option<crate::backend::mlx::runtime::cache::residency::CacheResidencyReport>, Error>
    {
        cache.residency_report().map_err(Into::into)
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(
        &self,
    ) -> Option<&crate::backend::mlx::runtime::execution::layerwise::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns the exact rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    /// Returns the cache-relevant architecture fingerprint for this rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> String {
        resident::prompt_cache_architecture_fingerprint(self.args())
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

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn eredu_checkpoint::store::CheckpointSource) {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Returns current weight-residency telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Returns disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(
        &self,
    ) -> Result<
        Option<crate::backend::mlx::runtime::execution::layerwise::DenseDiskStreamReport>,
        Error,
    > {
        self.execution.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(&self) -> Result<Option<ExpertCacheReport>, Error> {
        self.execution
            .adapter()
            .expert_cache
            .as_ref()
            .map(ExpertCache::report)
            .transpose()
            .map_err(Error::from)
    }

    /// Runs the embedding, hybrid decoder stack, norm, and output projection.
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution.forward(inputs, cache, stream)
    }

    /// Runs the canonical execution path with stable per-layer observation points.
    pub fn forward_with_observer(
        &mut self,
        inputs: &Array,
        cache: &mut Cache,
        stream: &Stream,
        observer: &mut dyn crate::backend::mlx::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        self.execution
            .forward_with_observer(inputs, cache, stream, observer)
    }

    /// Runs a rank-local tensor-parallel KDA/MLA forward pass.
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
        mask: Option<&Array>,
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
                    hidden,
                    mask.or(context.mask.as_ref()),
                    Some(&mut cache.layers[index]),
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Runs TP-sharded KDA/MLA and dense/shared projections while delegating
    /// sparse routed experts to the matching EP subgroup.
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
        self.execution.forward_tensor_parallel_with_layer_executor(
            inputs,
            cache,
            tensor_group,
            stream,
            |_adapter, _group, index, layer, hidden, cache, context, execution| {
                let tp_group = execution.group().ok_or_else(|| {
                    Error::Parallel(
                        "Kimi Linear TP+EP execution requires an active TP group".into(),
                    )
                })?;
                Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    mask.or(context.mask.as_ref()),
                    Some(&mut cache.layers[index]),
                    tp_group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Evicts temporary decoder layers from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalModel<Cache> for KimiLinearLayerwiseModel {
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

/// Loads Kimi Linear through the shared generalized execution engine.
pub fn load_kimi_linear_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("Kimi Linear", args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = KimiLinearLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(KimiLinearLayerwiseModel {
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

/// Loads Kimi Linear through the generalized tensor-parallel engine.
pub(crate) fn load_kimi_linear_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_kimi_linear_gguf_tensor_parallel_model(
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
    Ok(KimiLinearLayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            KimiLinearLayerwiseAdapter::new(args, stream)?,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_kimi_linear_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(KimiLinearLayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
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
    let execution = load_tensor_parallel_layerwise_model(
        store,
        KimiLinearLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((
        KimiLinearLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

pub(crate) fn load_kimi_linear_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(KimiLinearLayerwiseModel, Vec<u32>), Error> {
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
            load_kimi_linear_sparse_with_store(
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
        KimiLinearLayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((
        KimiLinearLayerwiseModel { execution },
        prepared.eos_token_ids,
    ))
}

/// Loads only the replicated Kimi Linear GGUF parameters needed by sparse
/// expert-parallel execution and returns the shared lazy checkpoint store.
pub fn load_kimi_linear_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    args.validate()?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Kimi Linear independent expert cache",
                args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let source_adapter = KimiLinearLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let store = open_safetensors_weight_store(model_dir, non_expert.layers().max_mapped_shards())?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        source_adapter,
        non_expert,
        quantize_on_load,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = kimi_expert_catalog(&args, store.as_ref())?;
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
    Ok(KimiLinearLayerwiseModel { execution })
}

fn load_kimi_linear_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args.clone(), stream)?;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries = kimi_expert_catalog(&args, store.as_ref())?;
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
    Ok(KimiLinearLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Kimi execution base used by distributed EP.
pub(crate) fn load_kimi_linear_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    args.validate()?;
    let adapter = KimiLinearLayerwiseAdapter::new_sparse(args, stream)?;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(KimiLinearLayerwiseModel { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_kimi_linear_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<KimiLinearLayerwiseModel, Error> {
    args.validate()?;
    let mut adapter = KimiLinearLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(KimiLinearLayerwiseModel { execution })
}

/// Adapter for Kimi's heterogeneous KDA/MLA decoder layers.
pub struct KimiLinearLayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_cache_geometry: Option<Vec<resident::KimiLayerCacheGeometry>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl KimiLinearLayerwiseAdapter {
    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                args.weight_quantization_for("lm_head.weight"),
                stream,
            )?)
        };
        Ok(Self {
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
            lm_head,
            parallel_embedding: None,
            parallel_lm_head: None,
            parallel_cache_geometry: None,
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

    /// Returns the architecture configuration.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
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
        let gguf = store.source_diagnostics()?.backend == WeightStoreBackend::Gguf;
        let mut recipes = BTreeMap::new();

        for local_name in layer.parameters().flatten().keys() {
            let destination = format!("{prefix}.{local_name}");
            let canonical = canonical_checkpoint_name(&destination);
            if let Some(raw) = normalized.get(&canonical) {
                let mut recipe = DerivedWeightRecipe::source(raw.clone(), TensorSelection::Full);
                if local_name.ends_with("q_conv1d.weight")
                    || local_name.ends_with("k_conv1d.weight")
                    || local_name.ends_with("v_conv1d.weight")
                {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: vec![
                            (self.args.kda_config.num_heads * self.args.kda_config.head_dim)
                                as usize,
                            1,
                            self.args.kda_config.short_conv_kernel_size as usize,
                        ],
                    };
                } else if local_name.ends_with("A_log") {
                    recipe = DerivedWeightRecipe::Reshape {
                        input: Box::new(recipe),
                        shape: vec![1, 1, self.args.kda_config.num_heads as usize, 1],
                    };
                    if gguf {
                        recipe = DerivedWeightRecipe::NegLog {
                            input: Box::new(recipe),
                        };
                    }
                } else if keys.contains(&destination) || keys.contains(&canonical) {
                    continue;
                }
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            if let Some(recipe) =
                self.split_expert_recipe(local_name.as_ref(), &prefix, &normalized)?
            {
                recipes.insert(local_name.to_string(), recipe);
                continue;
            }
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear checkpoint is missing runtime parameter {canonical}"
            )));
        }
        Ok(recipes)
    }

    fn split_expert_recipe(
        &self,
        local_name: &str,
        prefix: &str,
        normalized: &BTreeMap<String, String>,
    ) -> Result<Option<DerivedWeightRecipe>, Error> {
        let component = if local_name == "mlp.experts.gate_up_proj" {
            Some(("gate_up", "weight", ""))
        } else if local_name == "mlp.experts.gate_up_proj_scales" {
            Some(("gate_up", "scales", "_scales"))
        } else if local_name == "mlp.experts.gate_up_proj_biases" {
            Some(("gate_up", "biases", "_biases"))
        } else if local_name == "mlp.experts.down_proj" {
            Some(("down", "weight", ""))
        } else {
            None
        };
        let Some((projection, checkpoint_component, packed_suffix)) = component else {
            return Ok(None);
        };
        if projection == "gate_up" {
            let gate = normalized.get(&format!("{prefix}.mlp.experts.gate_proj{packed_suffix}"));
            let up = normalized.get(&format!("{prefix}.mlp.experts.up_proj{packed_suffix}"));
            match (gate, up) {
                (Some(gate), Some(up)) => {
                    return Ok(Some(DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate.clone(), TensorSelection::Full),
                            DerivedWeightRecipe::source(up.clone(), TensorSelection::Full),
                        ],
                    }));
                }
                (None, None) => {}
                _ => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Kimi Linear checkpoint layer {prefix} has mismatched packed gate/up expert tensors"
                    )));
                }
            }
        }
        if checkpoint_component != "weight" {
            return Ok(None);
        }
        let mut experts = Vec::with_capacity(self.args.num_experts as usize);
        for expert in 0..self.args.num_experts {
            let source = |name: &str| -> Result<DerivedWeightRecipe, Error> {
                let runtime =
                    format!("{prefix}.mlp.experts.{expert}.{name}.{checkpoint_component}");
                let raw = normalized.get(&runtime).ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Kimi Linear checkpoint is missing expert tensor {runtime}"
                    ))
                })?;
                Ok(DerivedWeightRecipe::source(
                    raw.clone(),
                    TensorSelection::Full,
                ))
            };
            experts.push(if projection == "gate_up" {
                DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![source("w1")?, source("w3")?],
                }
            } else {
                source("w2")?
            });
        }
        Ok(Some(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: experts,
        }))
    }
}

impl LoadTimeQuantizableAdapter for KimiLinearLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
        if self.sparse_expert_cache {
            Self::new_sparse(args, stream)
        } else {
            Self::new(args, stream)
        }
    }
}

fn normalized_checkpoint_keys(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| {
            let runtime = canonical_checkpoint_name(&raw).replace(".block_sparse_moe.", ".mlp.");
            (runtime, raw)
        })
        .collect()
}

/// Per-forward causal state shared by all MLA layers.
pub struct KimiLinearForwardContext {
    mask: Option<Array>,
}

impl ArchitectureAdapter for KimiLinearLayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = KimiLinearForwardContext;

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

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let geometry = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_cache_geometry.clone().ok_or_else(|| {
                    Error::Parallel(
                        "Kimi parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?
            }
            _ => self
                .args
                .layer_schedule
                .iter()
                .map(|policy| resident::KimiLayerCacheGeometry {
                    kda_heads: (policy.attention == resident::AttentionKind::Kda)
                        .then_some(self.args.kda_config.num_heads),
                })
                .collect(),
        };
        let layer_count = self.args.num_hidden_layers as usize;
        Ok(PromptCacheModelIdentity {
            model_family: "kimi_linear".into(),
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
            layer_layout: resident::prompt_cache_layer_layout_with_geometry(&self.args, &geometry)
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
        let rank = descriptor.topology.cache_rank_identity();
        resident::Model::save_prompt_cache_with_rank(
            cache,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            rank,
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
            if let Some(lm_head) = &self.lm_head {
                units.push(StaticUnitBindings::new(
                    HEAD_UNIT,
                    build_module_binding_plan_with_recipes(
                        lm_head,
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

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear adapter received {} static leases, expected {expected}",
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
        } else if let Some(lm_head) = &mut self.lm_head {
            populate_module_from_lease(lm_head, &leases[2])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.layers.is_empty() {
            *cache = Cache::new(&self.args);
        }
        cache.validate(&self.args.layer_schedule)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        let hidden = self.embedding.forward(input, stream)?;
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
        Ok(LayerwiseForwardState {
            hidden,
            context: KimiLinearForwardContext { mask },
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
        let offset = cache.offset();
        let mask = if hidden.dim(1) > 1 && offset > 0 {
            Some(create_causal_mask(
                hidden.dim(1),
                Some(offset),
                None,
                None,
                execution.stream(),
            )?)
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: KimiLinearForwardContext { mask },
        })
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Into::into)
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            Ok(self.args.layer_schedule.len())
        } else {
            Err(Error::UnsupportedArchitecture(format!(
                "Kimi Linear decoder has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        Ok(DecoderLayer::new(&self.args, index, stream)?)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if let FeedForward::Moe(moe) = &mut layer.mlp {
            let prefix = format!("model.layers.{index}.mlp.experts");
            moe.experts = crate::backend::mlx::nn::moe::PackedSwiGluExperts::new(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local Kimi expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                self.args.moe_intermediate_size,
                self.args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                self.args
                    .weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?;
        }
        Ok(layer)
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        if let FeedForward::Moe(moe) = &mut layer.mlp {
            let prefix = format!("model.layers.{index}.mlp.experts");
            let intermediate = moe.experts.intermediate_dim;
            moe.experts = crate::backend::mlx::nn::moe::PackedSwiGluExperts::new(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local Kimi expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                intermediate,
                self.args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                self.args
                    .weight_quantization_for(&format!("{prefix}.down_proj")),
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
        if self.args.num_experts <= 0
            || !self
                .args
                .layer_schedule
                .iter()
                .any(|policy| policy.feed_forward == FeedForwardPolicy::SparseMoe)
        {
            return Err(Error::Parallel(
                "Kimi Linear PP+EP requires a checkpoint with sparse MoE layers".into(),
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
        for index in 0..self.args.layer_schedule.len() {
            let layer = DecoderLayer::new(&self.args, index, stream)?;
            register_kimi_layer_parallel_plan(planner, &layer, &self.args, index)?;
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let kda_heads = planned_optional_partition_widths(
            layout,
            self.args
                .layer_schedule
                .iter()
                .map(|policy| policy.attention == resident::AttentionKind::Kda),
            self.args.kda_config.head_dim,
            "model.layers",
            "self_attn.q_proj",
        )?;
        self.parallel_cache_geometry = Some(
            kda_heads
                .into_iter()
                .map(|kda_heads| resident::KimiLayerCacheGeometry { kda_heads })
                .collect(),
        );
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
    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let prefix = format!("model.layers.{index}");
        let find = |n: &str| {
            layout
                .tensor(&format!("{prefix}.{n}.weight"))
                .or_else(|| layout.tensor(&format!("{prefix}.{n}.inner.weight")))
        };
        let mut args = self.args.clone();
        let attention_kind = self
            .args
            .layer_policy(index)
            .ok_or_else(|| Error::Parallel(format!("missing Kimi layer policy {index}")))?
            .attention;
        match attention_kind {
            resident::AttentionKind::Kda => {
                let q = find("self_attn.q_proj").ok_or_else(|| {
                    Error::Parallel(format!("missing KDA query layout for {prefix}"))
                })?;
                let width = i32::try_from(q.local_shape()[0]).map_err(|_| {
                    Error::Parallel("Kimi local KDA query width exceeds i32".into())
                })?;
                if width % self.args.kda_config.head_dim != 0 {
                    return Err(Error::Parallel(format!(
                        "Kimi local KDA query width {width} splits head dimension {}",
                        self.args.kda_config.head_dim
                    )));
                }
                args.kda_config.num_heads = width / self.args.kda_config.head_dim;
            }
            resident::AttentionKind::Mla => {
                let q = find(if self.args.q_lora_rank.is_some() {
                    "self_attn.q_b_proj"
                } else {
                    "self_attn.q_proj"
                })
                .ok_or_else(|| Error::Parallel(format!("missing MLA query layout for {prefix}")))?;
                let head_width = self.args.qk_nope_head_dim + self.args.qk_rope_head_dim;
                let width = i32::try_from(q.local_shape()[0]).map_err(|_| {
                    Error::Parallel("Kimi local MLA query width exceeds i32".into())
                })?;
                if width % head_width != 0 {
                    return Err(Error::Parallel(format!(
                        "Kimi local MLA query width {width} splits head width {head_width}"
                    )));
                }
                args.num_attention_heads = width / head_width;
            }
        }
        let dense_intermediate = find("mlp.gate_proj")
            .map(|value| {
                i32::try_from(value.local_shape()[0])
                    .map_err(|_| Error::Parallel("Kimi local dense width exceeds i32".into()))
            })
            .transpose()?
            .unwrap_or(args.intermediate_size);
        let routed_intermediate = layout
            .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
            .map(|value| {
                let packed = i32::try_from(value.local_shape()[1])
                    .map_err(|_| Error::Parallel("Kimi local routed expert width exceeds i32".into()))?;
                if packed % 2 != 0 {
                    return Err(Error::Parallel(format!(
                        "Kimi local packed expert width {packed} does not contain equal gate/up segments"
                    )));
                }
                Ok(packed / 2)
            })
            .transpose()?
            .unwrap_or(args.moe_intermediate_size);
        let shared_intermediate = find("mlp.shared_experts.gate_proj")
            .map(|value| {
                i32::try_from(value.local_shape()[0]).map_err(|_| {
                    Error::Parallel("Kimi local shared expert width exceeds i32".into())
                })
            })
            .transpose()?
            .unwrap_or(args.moe_intermediate_size * args.num_shared_experts);
        Ok(DecoderLayer::new_with_widths(
            &args,
            index,
            dense_intermediate,
            routed_intermediate,
            shared_intermediate,
            stream,
        )?)
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("kimi_linear.layer.{index:05}")
    }

    fn layer_bindings(
        &self,
        _group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, Error> {
        let plan = build_module_binding_plan_with_recipes_excluding(
            layer,
            &format!("model.layers.{index}"),
            store,
            self.recipes_for_layer(layer, index, store)?,
            |name| self.sparse_expert_cache && name.starts_with("mlp.experts."),
        )?;
        Ok(plan.build_bindings(store)?)
    }
    fn parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        layout: &crate::backend::mlx::runtime::distributed::parallel::LocalModelLayout,
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
                .filter(|key| {
                    key.contains(".mlp.experts.") || key.contains(".block_sparse_moe.experts.")
                })
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
        let policy = self.args.layer_policy(index).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Kimi Linear layer schedule has no policy for layer {index}"
            ))
        })?;
        if self.sparse_expert_cache && policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Kimi Linear sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_sparse_experts(
                hidden,
                context.mask.as_ref(),
                Some(&mut cache.layers[index]),
                stream,
                |flat, indices, weights, stream| {
                    expert_cache
                        .execute_routes_bounded(
                            ExpertRouteBatch::new(index, flat, indices, weights, pass),
                            stream,
                            |flat, acquired, weights, stream| {
                                let started = Instant::now();
                                let load_time = expert_cache.weight_quantization();
                                let gate_up_quantization = load_time.or_else(|| {
                                    self.args.weight_quantization_for(&format!(
                                        "model.layers.{index}.mlp.experts.gate_up_proj"
                                    ))
                                });
                                let down_quantization = load_time.or_else(|| {
                                    self.args.weight_quantization_for(&format!(
                                        "model.layers.{index}.mlp.experts.down_proj"
                                    ))
                                });
                                let mut bank =
                                    crate::backend::mlx::nn::moe::PackedSwiGluExperts::new(
                                        acquired.identities().len() as i32,
                                        self.args.hidden_size,
                                        self.args.moe_intermediate_size,
                                        gate_up_quantization,
                                        down_quantization,
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
        let mut observer = None;
        Ok(layer.forward_impl(
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
            stream,
            &format!("model.layers.{index}"),
            &mut observer,
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
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
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
        key.starts_with("model.mtp.")
    }
}

pub(crate) fn kimi_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    kimi_expert_catalog_for_layers(args, store, 0..args.layer_schedule.len())
}

pub(crate) fn kimi_expert_catalog_for_layers(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layers: std::ops::Range<usize>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let normalized = normalized_checkpoint_keys(store);
    let mut entries = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if !layers.contains(&layer) || policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for expert in 0..args.num_experts as usize {
            let mut planned = Vec::new();
            for (binding_name, recipe) in [
                (
                    "gate_up_proj",
                    expert_projection_recipe(&normalized, &prefix, expert, "gate_up_proj")?,
                ),
                (
                    "down_proj",
                    expert_projection_recipe(&normalized, &prefix, expert, "down_proj")?,
                ),
            ] {
                let metadata = recipe.infer(store)?;
                planned.push(PlannedBinding {
                    target_name: binding_name.into(),
                    expected_shape: metadata.shape().to_vec(),
                    expected_dtype: metadata.dtype().clone(),
                    recipe,
                });
            }
            for (name, projection, suffix) in [
                ("gate_up_proj_scales", "gate_up_proj", "_scales"),
                ("gate_up_proj_biases", "gate_up_proj", "_biases"),
                ("down_proj_scales", "down_proj", "_scales"),
                ("down_proj_biases", "down_proj", "_biases"),
            ] {
                if let Some(recipe) = optional_expert_component_recipe(
                    &normalized,
                    &prefix,
                    expert,
                    projection,
                    suffix,
                )? {
                    let metadata = recipe.infer(store)?;
                    planned.push(PlannedBinding {
                        target_name: name.into(),
                        expected_shape: metadata.shape().to_vec(),
                        expected_dtype: metadata.dtype().clone(),
                        recipe,
                    });
                }
            }
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture(
                        "Kimi Linear expert byte total overflowed".into(),
                    )
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn expert_projection_recipe(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projection: &str,
) -> Result<DerivedWeightRecipe, Error> {
    if let Some(raw) = normalized.get(&format!("{prefix}.{projection}")) {
        return Ok(DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            },
        ));
    }
    if projection == "gate_up_proj" {
        let gate = normalized.get(&format!("{prefix}.gate_proj"));
        let up = normalized.get(&format!("{prefix}.up_proj"));
        match (gate, up) {
            (Some(gate), Some(up)) => {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                return Ok(DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate.clone(), selection.clone()),
                        DerivedWeightRecipe::source(up.clone(), selection),
                    ],
                });
            }
            (None, None) => {}
            _ => {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Kimi Linear checkpoint {prefix} has mismatched packed gate/up expert tensors"
                )));
            }
        }
        let source = |name: &str| -> Result<DerivedWeightRecipe, Error> {
            let runtime = format!("{prefix}.{expert}.{name}.weight");
            let raw = normalized.get(&runtime).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Kimi Linear checkpoint is missing expert tensor {runtime}"
                ))
            })?;
            Ok(DerivedWeightRecipe::source(
                raw.clone(),
                TensorSelection::Full,
            ))
        };
        return Ok(DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source("w1")?, source("w3")?],
            }],
        });
    }
    let runtime = format!("{prefix}.{expert}.w2.weight");
    let raw = normalized.get(&runtime).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "Kimi Linear checkpoint is missing expert tensor {runtime}"
        ))
    })?;
    Ok(DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![DerivedWeightRecipe::source(
            raw.clone(),
            TensorSelection::Full,
        )],
    })
}

fn optional_expert_component_recipe(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projection: &str,
    suffix: &str,
) -> Result<Option<DerivedWeightRecipe>, Error> {
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    if let Some(raw) = normalized.get(&format!("{prefix}.{projection}{suffix}")) {
        return Ok(Some(DerivedWeightRecipe::source(raw.clone(), selection)));
    }
    if projection != "gate_up_proj" {
        return Ok(None);
    }
    let gate = normalized.get(&format!("{prefix}.gate_proj{suffix}"));
    let up = normalized.get(&format!("{prefix}.up_proj{suffix}"));
    match (gate, up) {
        (Some(gate), Some(up)) => Ok(Some(DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::source(gate.clone(), selection.clone()),
                DerivedWeightRecipe::source(up.clone(), selection),
            ],
        })),
        (None, None) => Ok(None),
        _ => Err(Error::UnsupportedArchitecture(format!(
            "Kimi Linear checkpoint {prefix} has mismatched packed gate/up expert components {suffix:?}"
        ))),
    }
}

/// Token generation over a bounded Kimi model.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    crate::backend::mlx::nn::generation::Generate<'a, KimiLinearLayerwiseModel, Cache, S>;
