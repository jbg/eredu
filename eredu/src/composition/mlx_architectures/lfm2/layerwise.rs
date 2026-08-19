//! Unified fully resident and bounded layer execution for LFM2/LFM2.5.

use eredu_runtime::{
    ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, LayerWeightResidency,
    NonExpertWeightResidency, WeightResidency,
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
        self as common,
        linear::project_logits_maybe_quantized,
        moe::PackedSwiGluExperts,
        parallel::{
            planned_optional_kv_head_layout, planned_optional_partition_widths,
            register_gated_depthwise_conv_group, register_gqa_projection_group,
            register_swiglu_projection_group, GqaProjectionNames, SwiGluProjectionNames,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::KeyValueCache,
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, build_module_binding_plan_with_recipes_excluding,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    backend::mlx::runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    backend::mlx::runtime::checkpoint::store::{open_gguf_checkpoint_source, TensorSelection},
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load,
        recipe::{recipe_dtype_from_mlx, DerivedWeightRecipe},
    },
    backend::mlx::runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_replicated_module,
        ParallelPlanBuilder,
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
    composition::mlx_architectures::lfm2::model::{
        self as resident, Cache, DecoderLayer, FeedForwardPolicy, LayerCache, ModelArgs,
        OperatorPolicy,
    },
};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};

use eredu_runtime::ResidencyReport;

const EMBEDDING_UNIT: &str = "lfm2.static.embedding";
const NORM_UNIT: &str = "lfm2.static.norm";
const HEAD_UNIT: &str = "lfm2.static.output";

fn lfm2_conv_weight_recipe(
    checkpoint_key: &str,
    checkpoint_shape: &[usize],
    parameter_shape: &[i32],
    parameter_dtype: Dtype,
) -> Option<DerivedWeightRecipe> {
    let source = || {
        Box::new(DerivedWeightRecipe::source(
            checkpoint_key,
            TensorSelection::Full,
        ))
    };
    let target_shape = parameter_shape
        .iter()
        .map(|dimension| *dimension as usize)
        .collect::<Vec<_>>();
    let layout_recipe = if checkpoint_shape.len() == 2 {
        DerivedWeightRecipe::Reshape {
            input: source(),
            shape: target_shape,
        }
    } else if target_shape.len() == 3
        && checkpoint_shape == [target_shape[0], target_shape[2], target_shape[1]]
        && checkpoint_shape != target_shape
    {
        DerivedWeightRecipe::Transpose {
            input: source(),
            axes: vec![0, 2, 1],
        }
    } else {
        return None;
    };
    // Direct checkpoint bindings adopt the checkpoint dtype when they replace
    // an unloaded placeholder. Derived bindings validate their output against
    // the placeholder first, so make that conversion explicit. In particular,
    // released MLX LFM2 checkpoints use BF16 while DepthwiseConv1d's unloaded
    // kernel is F32.
    Some(DerivedWeightRecipe::Cast {
        input: Box::new(layout_recipe),
        dtype: recipe_dtype_from_mlx(parameter_dtype),
    })
}

fn register_lfm2_layer_parallel_plan(
    planner: &mut ParallelPlanBuilder,
    layer: &DecoderLayer,
    args: &ModelArgs,
    index: usize,
) -> Result<(), Error> {
    let prefix = format!("model.layers.{index}");
    if let Some(attention) = &layer.self_attn {
        register_gqa_projection_group(
            planner,
            &format!("{prefix}.self_attn"),
            GqaProjectionNames {
                query: "q_proj",
                key: "k_proj",
                value: "v_proj",
                output: "out_proj",
            },
            &attention.q_proj,
            &attention.k_proj,
            &attention.v_proj,
            &attention.out_proj,
            attention.n_heads,
            attention.n_kv_heads,
            attention.head_dim,
        )?;
        for (name, module) in [
            ("q_layernorm", &attention.q_layernorm),
            ("k_layernorm", &attention.k_layernorm),
        ] {
            register_replicated_module(planner, module, &format!("{prefix}.self_attn.{name}"))?;
        }
        register_replicated_module(
            planner,
            &attention.rope,
            &format!("{prefix}.self_attn.rope"),
        )?;
    }
    if let Some(conv) = &layer.conv {
        register_gated_depthwise_conv_group(
            planner,
            &format!("{prefix}.conv"),
            &conv.in_proj,
            &conv.conv,
            &conv.out_proj,
            args.hidden_size,
        )?;
    }
    for (name, norm) in [
        ("operator_norm", &layer.operator_norm),
        ("ffn_norm", &layer.ffn_norm),
    ] {
        register_replicated_module(planner, norm, &format!("{prefix}.{name}"))?;
    }

    let feed_forward = &layer.feed_forward;
    if feed_forward.is_moe {
        let gate = feed_forward.gate.as_ref().ok_or_else(|| {
            Error::Parallel(format!("LFM2 layer {index} is missing its MoE gate"))
        })?;
        register_replicated_module(planner, gate, &format!("{prefix}.feed_forward.gate"))?;
        if let Some(bias) = feed_forward.expert_bias.as_ref().as_ref() {
            planner.register(ParameterGroupSpec::new(
                format!("{prefix}.feed_forward.expert_bias"),
                ParameterRole::Replicated,
                [array_parameter_member(
                    format!("{prefix}.feed_forward.expert_bias"),
                    bias,
                    MemberSharding::Replicated,
                )?],
            )?)?;
        }
        let experts = feed_forward.experts.as_ref().ok_or_else(|| {
            Error::Parallel(format!("LFM2 layer {index} is missing its expert bank"))
        })?;
        let expert_prefix = format!("{prefix}.feed_forward.experts");
        let intermediate = usize::try_from(experts.intermediate_dim)
            .map_err(|_| Error::Parallel("LFM2 expert width exceeds usize".into()))?;
        let down_alignment =
            experts
                .down_affine
                .or(experts.down_iquant)
                .map_or(Ok(1usize), |quantization| {
                    usize::try_from(quantization.group_size()).map_err(|_| {
                        Error::Parallel("LFM2 expert quantization group exceeds usize".into())
                    })
                })?;
        let expert_units = aligned_partition_units(
            &format!("{expert_prefix}.intermediate"),
            intermediate,
            1,
            down_alignment,
        )?;
        let segments = vec![0..intermediate, intermediate..2 * intermediate];
        let mut members = vec![array_parameter_member(
            format!("{expert_prefix}.gate_up_proj"),
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
                    format!("{expert_prefix}.{name}"),
                    value,
                    MemberSharding::PartitionedSegments {
                        axis: 1,
                        segments: segments.clone(),
                    },
                )?);
            }
        }
        members.push(array_parameter_member(
            format!("{expert_prefix}.down_proj"),
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
                    format!("{expert_prefix}.{name}"),
                    value,
                    MemberSharding::Partitioned { axis: 2 },
                )?);
            }
        }
        planner.register(ParameterGroupSpec::partitioned(
            format!("{expert_prefix}.intermediate"),
            ParameterRole::ExpertIntermediate,
            expert_units,
            members,
        )?)?;
    } else {
        let w1 = feed_forward
            .w1
            .as_ref()
            .ok_or_else(|| Error::Parallel(format!("LFM2 dense layer {index} is missing w1")))?;
        let w3 = feed_forward
            .w3
            .as_ref()
            .ok_or_else(|| Error::Parallel(format!("LFM2 dense layer {index} is missing w3")))?;
        let w2 = feed_forward
            .w2
            .as_ref()
            .ok_or_else(|| Error::Parallel(format!("LFM2 dense layer {index} is missing w2")))?;
        register_swiglu_projection_group(
            planner,
            &format!("{prefix}.feed_forward"),
            SwiGluProjectionNames {
                gate: "w1",
                up: "w3",
                down: "w2",
            },
            w1,
            w3,
            w2,
            args.dense_intermediate_size,
        )?;
    }
    Ok(())
}

/// LFM2/LFM2.5 causal LM with host-backed decoder blocks.
pub struct Lfm2LayerwiseModel {
    execution: LayerwiseModel<Lfm2LayerwiseAdapter>,
}

impl Lfm2LayerwiseModel {
    /// Returns the validated model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
    }

    /// Creates heterogeneous attention and convolution state.
    pub fn new_cache(&self) -> Cache {
        self.execution.adapter().new_cache()
    }

    /// Creates resident heterogeneous state or pages only growing attention
    /// blocks while convolution history remains rank-local on device.
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

    /// Returns aggregate live attention paging telemetry, if enabled.
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

    /// Returns this rank's exact prompt-cache state layout.
    pub fn prompt_cache_layer_layout(
        &self,
    ) -> Result<crate::LayerSchedule<crate::LayerCachePolicy>, Error> {
        self.execution.prompt_cache_layer_layout()
    }

    /// Returns the cache-relevant architecture fingerprint for this rank.
    pub fn prompt_cache_architecture_fingerprint(&self) -> Result<String, Error> {
        self.execution.prompt_cache_architecture_fingerprint()
    }

    /// Returns the complete rank-local prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
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

    /// Runs the hybrid decoder while preserving recurrent and KV state.
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
                Ok(layer.forward_with_expert_executor(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    stream,
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Runs TP-sharded hybrid operators and dense blocks while delegating
    /// sparse routed experts to the matching EP subgroup.
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
                    Error::Parallel("LFM2 TP+EP execution requires an active TP group".into())
                })?;
                Ok(layer.forward_tensor_with_expert_executor(
                    hidden,
                    context.mask.as_ref(),
                    Some(&mut cache.layers[index]),
                    tp_group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?)
            },
        )
    }

    /// Clears temporary decoder copies from the execution device.
    pub fn clear_device_layer_window(&self) -> Result<(), Error> {
        self.execution.clear_device_group("text_decoder")
    }
}

impl CausalModel<Cache> for Lfm2LayerwiseModel {
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

/// Loads dense or MoE LFM2 through generalized parameter residency.
pub fn load_lfm2_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load("LFM2", args.weight_quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = Lfm2LayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(Lfm2LayerwiseModel {
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

/// Loads dense or MoE LFM2 through the generalized tensor-parallel engine.
pub(crate) fn load_lfm2_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        return load_lfm2_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    Ok(Lfm2LayerwiseModel {
        execution: load_tensor_parallel_layerwise_model(
            open_safetensors_weight_store(model_dir, options.max_mapped_shards())?,
            Lfm2LayerwiseAdapter::new(resident::get_model_args(model_dir)?, stream)?,
            options,
            build,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_lfm2_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Lfm2LayerwiseModel, Vec<u32>), Error> {
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let is_moe = prepared.args.model_type == "lfm2_moe";
    let gguf_plan =
        super::checkpoint::gguf_plan(&prepared.args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            move |name| resident::translate_gguf_weight_name(name, is_moe),
            options.max_mapped_shards(),
        )?);
    let execution = load_tensor_parallel_layerwise_model(
        store,
        Lfm2LayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((Lfm2LayerwiseModel { execution }, prepared.eos_token_ids))
}

pub(crate) fn load_lfm2_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Lfm2LayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_gguf_checkpoint(checkpoint, metadata, weights_stream)?;
    let args = prepared.args;
    let is_moe = args.model_type == "lfm2_moe";
    let gguf_plan = super::checkpoint::gguf_plan(&args).map_err(Error::UnsupportedArchitecture)?;
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(open_gguf_checkpoint_source(
            checkpoint.clone(),
            &gguf_plan,
            |name| resident::translate_gguf_weight_name(name, is_moe),
            residency.max_mapped_shards(),
        )?);
    if let Some(expert_options) = residency.expert_cache() {
        return Ok((
            load_lfm2_gguf_sparse_with_store(
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
        Lfm2LayerwiseAdapter::new(args, stream)?,
        residency.layers(),
        quantization,
        stream,
        weights_stream,
    )?;
    Ok((Lfm2LayerwiseModel { execution }, prepared.eos_token_ids))
}

fn load_lfm2_gguf_sparse_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    if !args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an LFM2 MoE GGUF checkpoint".into(),
        ));
    }
    let mut adapter = Lfm2LayerwiseAdapter::new(args.clone(), stream)?;
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
    let entries = lfm2_expert_catalog(&args, checkpoint_store.as_ref())?;
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
    Ok(Lfm2LayerwiseModel { execution })
}

/// Builds the streamed nonexpert LFM2 execution base used by distributed EP.
pub(crate) fn load_lfm2_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let mut adapter = Lfm2LayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(Lfm2LayerwiseModel { execution })
}

/// Builds the shared TP-sharded nonexpert base used by combined TP+EP.
pub(crate) fn load_lfm2_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let mut adapter = Lfm2LayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(Lfm2LayerwiseModel { execution })
}

/// Loads MoE LFM2 with independently cached experts and bounded non-expert units.
pub fn load_lfm2_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Lfm2LayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_model_args(model_dir)?;
    if !args.has_sparse_moe_layers() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires an LFM2 MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "LFM2 independent expert cache",
                args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut source_adapter = Lfm2LayerwiseAdapter::new(args.clone(), stream)?;
    source_adapter.sparse_expert_cache = true;
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
    let entries = lfm2_expert_catalog(&args, store.as_ref())?;
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
    Ok(Lfm2LayerwiseModel { execution })
}

/// Adapter shared by dense, MoE, attention, and short-convolution LFM2 blocks.
pub struct Lfm2LayerwiseAdapter {
    args: ModelArgs,
    embedding: MaybeQuantized<nn::Embedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_cache_geometry: Option<Vec<resident::Lfm2LayerCacheGeometry>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl Lfm2LayerwiseAdapter {
    /// Creates metadata-only pinned modules.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let embedding = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let norm = nn::RmsNorm::unloaded(args.hidden_size, args.norm_eps, Dtype::Float32, stream)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
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
            parallel_cache_geometry: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    /// Creates an adapter whose routed expert banks are supplied by an
    /// independent residency component rather than layer payloads.
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
        Cache::new(&self.args).expect("validated LFM2 layer schedule")
    }

    fn split_expert_recipes(
        &self,
        index: usize,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
        if self
            .args
            .layer_policy(index)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer schedule has no policy for layer {index}"
                ))
            })?
            .feed_forward
            != FeedForwardPolicy::SparseMoe
        {
            return Ok(BTreeMap::new());
        }
        let runtime_prefix = format!("model.layers.{index}.feed_forward.experts");
        let keys = store.source_keys();
        if keys.contains(&format!("{runtime_prefix}.gate_up_proj")) {
            return Ok(BTreeMap::new());
        }
        if keys.contains(&format!("{runtime_prefix}.gate_proj"))
            && keys.contains(&format!("{runtime_prefix}.up_proj"))
        {
            let mut recipes = BTreeMap::from([(
                "feed_forward.experts.gate_up_proj".into(),
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(
                            format!("{runtime_prefix}.gate_proj"),
                            TensorSelection::Full,
                        ),
                        DerivedWeightRecipe::source(
                            format!("{runtime_prefix}.up_proj"),
                            TensorSelection::Full,
                        ),
                    ],
                },
            )]);
            for suffix in ["_scales", "_biases"] {
                let gate = format!("{runtime_prefix}.gate_proj{suffix}");
                let up = format!("{runtime_prefix}.up_proj{suffix}");
                if keys.contains(&gate) && keys.contains(&up) {
                    recipes.insert(
                        format!("feed_forward.experts.gate_up_proj{suffix}"),
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
            return Ok(recipes);
        }
        let mut gate_up = Vec::with_capacity(self.args.num_experts as usize);
        let mut down = Vec::with_capacity(self.args.num_experts as usize);
        for expert in 0..self.args.num_experts {
            let gate = expert_source(store, &runtime_prefix, expert, &["w1", "gate_proj"])?;
            let up = expert_source(store, &runtime_prefix, expert, &["w3", "up_proj"])?;
            let down_source = expert_source(store, &runtime_prefix, expert, &["w2", "down_proj"])?;
            gate_up.push(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![gate, up],
            });
            down.push(down_source);
        }
        Ok(BTreeMap::from([
            (
                "feed_forward.experts.gate_up_proj".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: gate_up,
                },
            ),
            (
                "feed_forward.experts.down_proj".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: down,
                },
            ),
        ]))
    }
}

fn expert_source(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, Error> {
    let keys = store.source_keys();
    let key = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| keys.contains(candidate))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "LFM2 checkpoint is missing expert {expert} projection under {prefix}"
            ))
        })?;
    Ok(DerivedWeightRecipe::source(key, TensorSelection::Full))
}

/// Per-forward attention mask shared by the temporary hybrid blocks.
pub struct Lfm2ForwardContext {
    mask: Option<Array>,
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

impl LoadTimeQuantizableAdapter for Lfm2LayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.weight_quantization = Some(quantization);
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for Lfm2LayerwiseAdapter {
    type Input<'a> = &'a Array;
    type Cache = Cache;
    type Layer = DecoderLayer;
    type ForwardContext = Lfm2ForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn safetensors_checkpoint_plan(
        &self,
    ) -> Result<crate::backend::mlx::runtime::execution::layerwise::ArchitectureCheckpointPlan, Error>
    {
        super::checkpoint::safetensors_plan(&self.args, true)
            .map_err(Error::UnsupportedArchitecture)
            .map(Into::into)
    }

    fn quantization(&self) -> Option<eredu_checkpoint::WeightQuantization> {
        self.args.weight_quantization
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = usize::try_from(self.args.num_hidden_layers)
            .map_err(|_| Exception::custom("invalid LFM2 cache layer count"))?;
        let geometry = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_cache_geometry.clone().ok_or_else(|| {
                    Error::Parallel(
                        "LFM2 parallel cache identity requested before local layout configuration"
                            .into(),
                    )
                })?
            }
            _ => self
                .args
                .layer_schedule
                .iter()
                .map(|policy| match policy.operator {
                    OperatorPolicy::CausalConvolution => resident::Lfm2LayerCacheGeometry {
                        kv_heads: None,
                        convolution_channels: Some(self.args.hidden_size),
                    },
                    OperatorPolicy::SelfAttention(_) => resident::Lfm2LayerCacheGeometry {
                        kv_heads: Some(self.args.num_key_value_heads),
                        convolution_channels: None,
                    },
                })
                .collect(),
        };
        Ok(PromptCacheModelIdentity {
            model_family: "lfm2".into(),
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
        _options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Self::Cache, PromptCacheManifest), Error> {
        resident::Model::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity.clone(),
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
                    "model.embedding_norm",
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

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "LFM2 adapter received {} static leases, expected {expected}",
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
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[2])?;
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
                "LFM2 cache has {} layers, expected {}",
                cache.layers.len(),
                self.args.num_hidden_layers
            )));
        }
        for (index, layer_cache) in cache.layers.iter().enumerate() {
            let policy = self.args.layer_schedule.get(index).ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "LFM2 layer schedule has no policy for layer {index}"
                ))
            })?;
            let matches = matches!(
                (policy.operator, layer_cache),
                (OperatorPolicy::CausalConvolution, LayerCache::Conv(_))
                    | (OperatorPolicy::SelfAttention(_), LayerCache::Attention(_))
            );
            if !matches {
                return Err(Error::UnsupportedArchitecture(format!(
                    "LFM2 cache kind does not match the layer schedule at layer {index}"
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
        let hidden = self.embedding.forward(input, stream)?;
        let mask = if hidden.dim(1) > 1 {
            let offset_cache = vec![Some(OffsetOnlyCache(cache.offset()))];
            match create_attention_mask(&hidden, &offset_cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Error::UnsupportedArchitecture(
                        "LFM2 requires an array causal mask".into(),
                    ));
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Lfm2ForwardContext { mask },
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
                        "LFM2 requires an array causal mask".into(),
                    ))
                }
                None => None,
            }
        } else {
            None
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Lfm2ForwardContext { mask },
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
                "LFM2 has no execution group {group}"
            )))
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        self.layer_count(group)?;
        let index = i32::try_from(index)
            .map_err(|_| Error::UnsupportedArchitecture("LFM2 layer index exceeds i32".into()))?;
        DecoderLayer::new(&self.args, index, stream)
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        let mut layer = self.new_layer(group, index, stream)?;
        if layer.feed_forward.is_moe {
            let prefix = format!("model.layers.{index}.feed_forward.experts");
            layer.feed_forward.experts = Some(PackedSwiGluExperts::new(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local LFM2 expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                self.args.moe_intermediate_size,
                self.args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                self.args
                    .weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?);
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
        if layer.feed_forward.is_moe {
            let prefix = format!("model.layers.{index}.feed_forward.experts");
            let intermediate = layer
                .feed_forward
                .experts
                .as_ref()
                .ok_or_else(|| Error::Parallel(format!("LFM2 layer {index} has no expert bank")))?
                .intermediate_dim;
            layer.feed_forward.experts = Some(PackedSwiGluExperts::new(
                if self.sparse_expert_cache {
                    0
                } else {
                    i32::try_from(assignment.local_expert_count()).map_err(|_| {
                        Error::Parallel("local LFM2 expert count exceeds i32".into())
                    })?
                },
                self.args.hidden_size,
                intermediate,
                self.args
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                self.args
                    .weight_quantization_for(&format!("{prefix}.down_proj")),
                stream,
            )?);
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
        if !self.args.has_sparse_moe_layers() {
            return Err(Error::Parallel(
                "LFM2 PP+EP requires a checkpoint with sparse MoE layers".into(),
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
            "model.embedding_norm",
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
            let layer = DecoderLayer::new(&self.args, index as i32, stream)?;
            register_lfm2_layer_parallel_plan(planner, &layer, &self.args, index)?;
        }
        Ok(())
    }
    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let head_dim = self.args.hidden_size / self.args.num_attention_heads;
        let kv_heads = planned_optional_kv_head_layout(
            layout,
            self.args
                .layer_schedule
                .iter()
                .map(|policy| matches!(policy.operator, OperatorPolicy::SelfAttention(_))),
            head_dim,
            "model.layers",
        )?;
        let convolution_channels = planned_optional_partition_widths(
            layout,
            self.args
                .layer_schedule
                .iter()
                .map(|policy| policy.operator == OperatorPolicy::CausalConvolution),
            1,
            "model.layers",
            "conv.conv",
        )?;
        self.parallel_cache_geometry = Some(
            kv_heads
                .into_iter()
                .zip(convolution_channels)
                .map(
                    |(kv_heads, convolution_channels)| resident::Lfm2LayerCacheGeometry {
                        kv_heads,
                        convolution_channels,
                    },
                )
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
        let mut args = self.args.clone();
        let head_dim = self.args.hidden_size / self.args.num_attention_heads;
        if let Some(q) = find("self_attn.q_proj") {
            let width = i32::try_from(q.local_shape()[0])
                .map_err(|_| Error::Parallel("LFM2 local query width exceeds i32".into()))?;
            if width % head_dim != 0 {
                return Err(Error::Parallel(format!(
                    "LFM2 local query width {width} splits head dimension {head_dim}"
                )));
            }
            args.num_attention_heads = width / head_dim;
        }
        if let Some(k) = find("self_attn.k_proj") {
            let width = i32::try_from(k.local_shape()[0])
                .map_err(|_| Error::Parallel("LFM2 local key width exceeds i32".into()))?;
            if width % head_dim != 0 {
                return Err(Error::Parallel(format!(
                    "LFM2 local key width {width} splits head dimension {head_dim}"
                )));
            }
            args.num_key_value_heads = width / head_dim;
        }
        let dense = find("feed_forward.w1")
            .map(|value| {
                i32::try_from(value.local_shape()[0])
                    .map_err(|_| Error::Parallel("LFM2 local dense width exceeds i32".into()))
            })
            .transpose()?
            .unwrap_or(args.dense_intermediate_size);
        let moe = layout
            .tensor(&format!("{prefix}.feed_forward.experts.gate_up_proj"))
            .map(|value| {
                let packed = i32::try_from(value.local_shape()[1])
                    .map_err(|_| Error::Parallel("LFM2 local expert width exceeds i32".into()))?;
                if packed % 2 != 0 {
                    return Err(Error::Parallel(format!(
                        "LFM2 packed expert width {packed} does not contain equal gate/up segments"
                    )));
                }
                Ok(packed / 2)
            })
            .transpose()?
            .unwrap_or(args.moe_intermediate_size);
        let convolution_channels = layout
            .tensor(&format!("{prefix}.conv.conv.weight"))
            .map(|value| {
                i32::try_from(value.local_shape()[0])
                    .map_err(|_| Error::Parallel("LFM2 local convolution width exceeds i32".into()))
            })
            .transpose()?;
        DecoderLayer::new_with_widths(
            &args,
            index as i32,
            dense,
            moe,
            Some(head_dim),
            convolution_channels,
            stream,
        )
    }

    fn layer_checkpoint_prefix(&self, _group: usize, index: usize) -> String {
        format!("model.layers.{index}")
    }

    fn layer_unit_name(&self, _group: usize, index: usize) -> String {
        format!("lfm2.layer.{index:05}")
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
                |name| name.starts_with("feed_forward.experts."),
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
        let mut recipes = self.split_expert_recipes(index, store)?;
        let conv_key = format!("{prefix}.conv.conv.weight");
        if let Ok(metadata) = store.source_metadata(&conv_key) {
            if let Some(parameter) = layer.parameters().flatten().get("conv.conv.weight") {
                if let Some(recipe) = lfm2_conv_weight_recipe(
                    &conv_key,
                    &metadata.logical_shape,
                    parameter.shape(),
                    parameter.dtype(),
                ) {
                    recipes.insert("conv.conv.weight".into(), recipe);
                }
            }
        }
        Ok(build_module_binding_plan_with_recipes_excluding(
            layer,
            &prefix,
            store,
            recipes,
            |name| self.sparse_expert_cache && name.starts_with("feed_forward.experts."),
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
                if target.contains("feed_forward.experts.") {
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
                .filter(|key| key.contains(".feed_forward.experts."))
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
                "LFM2 layer schedule has no policy for layer {index}"
            ))
        })?;
        if self.sparse_expert_cache && policy.feed_forward == FeedForwardPolicy::SparseMoe {
            let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "LFM2 sparse expert cache was not initialized".into(),
                )
            })?;
            let pass = if hidden.dim(1) > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            return Ok(layer.forward_with_expert_executor(
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
                                let prefix = format!("model.layers.{index}.feed_forward.experts");
                                let load_time = expert_cache.weight_quantization();
                                let mut bank = PackedSwiGluExperts::new(
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
                                bank.down_proj = Param::new(
                                    acquired
                                        .compact_binding("down_proj", stream)
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
            hidden,
            context.mask.as_ref(),
            Some(&mut cache.layers[index]),
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
}

pub(crate) fn lfm2_expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let keys = store
        .source_keys()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut entries = Vec::new();
    for (layer, _) in args
        .layer_schedule
        .iter()
        .enumerate()
        .filter(|(_, policy)| policy.feed_forward == FeedForwardPolicy::SparseMoe)
    {
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..args.num_experts as usize {
            let identity = ExpertIdentity::new(layer, expert);
            let mut planned = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", &packed_gate_up),
                    ("down_proj", &packed_down),
                ] {
                    let recipe = DerivedWeightRecipe::source(
                        key.clone(),
                        TensorSelection::Range {
                            axis: 0,
                            start: expert,
                            end: expert + 1,
                        },
                    );
                    planned.push(planned_binding(name, recipe, store)?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        let recipe = DerivedWeightRecipe::source(
                            key,
                            TensorSelection::Range {
                                axis: 0,
                                start: expert,
                                end: expert + 1,
                            },
                        );
                        planned.push(planned_binding(name, recipe, store)?);
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
                for (name, recipe) in [
                    (
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
                    ),
                    (
                        "down_proj",
                        DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    ),
                ] {
                    planned.push(planned_binding(name, recipe, store)?);
                }
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        let recipe = DerivedWeightRecipe::Concatenate {
                            axis: 1,
                            inputs: vec![
                                DerivedWeightRecipe::source(gate, selection.clone()),
                                DerivedWeightRecipe::source(up, selection.clone()),
                            ],
                        };
                        planned.push(planned_binding(
                            format!("gate_up_proj{suffix}"),
                            recipe,
                            store,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        let recipe = DerivedWeightRecipe::source(down, selection.clone());
                        planned.push(planned_binding(
                            format!("down_proj{suffix}"),
                            recipe,
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
                        "split LFM2 experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                            .into(),
                    ));
                }
                let gate = expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?;
                let up = expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?;
                let down = expert_source(store, &prefix, expert as i32, &["w2", "down_proj"])?;
                for (name, recipe) in [
                    (
                        "gate_up_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![gate, up],
                            }],
                        },
                    ),
                    (
                        "down_proj",
                        DerivedWeightRecipe::Stack {
                            axis: 0,
                            inputs: vec![down],
                        },
                    ),
                ] {
                    planned.push(planned_binding(name, recipe, store)?);
                }
            }
            let bindings = BindingPlan::new(planned)
                .and_then(|plan| plan.build_bindings(store))
                .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("LFM2 expert byte total overflowed".into())
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

fn planned_binding(
    name: impl Into<String>,
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

/// LFM2 token generation iterator using bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Lfm2LayerwiseModel, Cache, S>;
