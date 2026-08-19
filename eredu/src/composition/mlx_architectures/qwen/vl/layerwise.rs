//! Shared bounded layer execution for dense and MoE Qwen3-VL models.

use eredu_runtime::LayerWeightResidency;

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::CausalModel;
use eredu_runtime::WeightBinding;

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
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        zeros_dtype, GgufCheckpoint, GgufMetadataValue,
    },
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
        attention::AttentionInput,
        parallel::{
            planned_kv_head_layout, vocab_embedding_parameter_group, vocab_lm_head_parameter_group,
            VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::KeyValueCache,
    backend::mlx::runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, populate_module_from_lease,
        populate_module_from_lease_excluding,
    },
    backend::mlx::runtime::checkpoint::store::TensorSelection,
    backend::mlx::runtime::checkpoint::{
        quantization::should_quantize_on_load, recipe::DerivedWeightRecipe,
    },
    backend::mlx::runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_tensor_parallel_layerwise_model, open_safetensors_weight_store, ArchitectureAdapter,
        LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter, StaticUnitBindings,
        WeightResidency,
    },
    backend::mlx::runtime::media::input,
    backend::mlx::runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheLoadOptions, ExpertCacheReport, ExpertPass, ExpertRouteBatch,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
    composition::mlx_architectures::qwen::dense::{
        layerwise::{qwen_text_layer_bindings, register_qwen_layer_parallel_plan},
        Decoder, Experts as QwenExperts, FeedForward, TransformerBlock,
    },
    composition::mlx_architectures::qwen::{
        vl::model::{self as resident, Cache, ModelArgs},
        vl::vision::{
            configure_vision_parallel_static, grid_thw_from_array, new_parallel_vision_block,
            vision_parallel_parameter_groups, QwenVisionBlock, QwenVisionLayerwiseState,
            QwenVisionLayerwiseStatic, QwenVisionTransformer,
        },
    },
};
use eredu_runtime::PagedCacheOptions;

use eredu_runtime::ResidencyReport;

const VISION_STATIC_UNIT: &str = "qwen3_vl.static.vision";
const EMBEDDING_UNIT: &str = "qwen3_vl.static.embedding";
const NORM_UNIT: &str = "qwen3_vl.static.norm";
const HEAD_UNIT: &str = "qwen3_vl.static.output";

/// Dense or MoE Qwen3-VL with independent vision and text residency windows.
pub struct Qwen3VlLayerwiseModel {
    execution: LayerwiseModel<Qwen3VlLayerwiseAdapter>,
}

impl Qwen3VlLayerwiseModel {
    /// Returns the parsed multimodal model arguments.
    pub fn args(&self) -> &ModelArgs {
        self.execution.adapter().args()
    }

    pub(crate) fn bind_parallel_topology(
        &mut self,
        topology: crate::backend::mlx::MlxParallelContext,
    ) {
        self.execution.bind_parallel_topology(topology);
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

    /// Returns the public architecture type.
    pub fn model_type(&self) -> &'static str {
        if self.args().text_config.model_type == "qwen3_vl_moe_text" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    /// Creates empty KV and multimodal position state.
    pub fn new_cache(&self) -> Cache {
        Cache::default()
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

    /// Returns the complete rank-local multimodal prompt-cache identity.
    pub fn prompt_cache_model_identity(&self) -> Result<PromptCacheModelIdentity, Error> {
        self.execution.prompt_cache_model_identity()
    }

    /// Persists a compatible multimodal prefix cache.
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

    pub(crate) fn save_prompt_cache_with_validated_identity(
        &self,
        cache: &mut Cache,
        directory: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Error> {
        self.execution.save_prompt_cache_with_validated_identity(
            cache,
            directory,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        )
    }

    /// Restores a compatible multimodal prefix cache.
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

    pub(crate) fn load_prompt_cache_with_validated_identity(
        &self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        options: PagedCacheOptions,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Error> {
        self.execution.load_prompt_cache_with_validated_identity(
            directory,
            expected,
            identity,
            prefix_token_ids,
            options,
            stream,
        )
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

    /// Returns the persistent checkpoint store.
    pub fn checkpoint_store(&self) -> &(dyn eredu_checkpoint::store::CheckpointSource) {
        self.execution.checkpoint_store()
    }

    pub(crate) fn checkpoint_store_arc(
        &self,
    ) -> Arc<dyn eredu_checkpoint::store::CheckpointSource> {
        self.execution.checkpoint_store_arc()
    }

    /// Runs typed multimodal prefill through vision and text execution groups.
    pub fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Prefill(input), cache, stream)
    }

    /// Runs a text decode step using cached multimodal RoPE state.
    pub fn decode(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward(Qwen3VlInput::Decode(tokens), cache, stream)
    }

    /// Runs a multimodal prefill through rank-local vision execution units.
    pub(crate) fn prefill_tensor_parallel(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(Qwen3VlInput::Prefill(input), cache, group, stream)
    }

    /// Runs text decode on a TP-loaded multimodal model.
    pub(crate) fn decode_tensor_parallel(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.execution
            .forward_tensor_parallel(Qwen3VlInput::Decode(tokens), cache, group, stream)
    }

    /// Runs streamed text layers while delegating routed experts to a caller.
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
        self.execution.forward_with_layer_executor(
            Qwen3VlInput::Decode(tokens),
            cache,
            stream,
            |adapter, group, index, layer, hidden, cache, context, stream| match (group, layer) {
                (0, Qwen3VlLayer::Vision(block)) => {
                    let Some(state) = context.vision.as_mut() else {
                        return Ok(hidden.clone());
                    };
                    let output = adapter.vision.forward_block(
                        block,
                        index,
                        hidden.clone(),
                        state,
                        stream,
                    )?;
                    adapter
                        .vision
                        .capture_deepstack(index, &output, state, stream)?;
                    Ok(output)
                }
                (1, Qwen3VlLayer::Text(block)) => {
                    let mut output = block.forward_sparse_experts_with_rotary(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: cache.kv[index].as_mut(),
                        },
                        &context.cos,
                        &context.sin,
                        stream,
                        |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                    )?;
                    if let Some(features) = context.deepstack_features.get(index) {
                        let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                        let features = features.try_index_device((0, .., ..), stream)?;
                        let aligned = masked_scatter(
                            &base,
                            context.visual_mask.as_ref().expect("DeepStack visual mask"),
                            features,
                            stream,
                        )?;
                        output = output.add(aligned, stream)?;
                    }
                    Ok(output)
                }
                _ => Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-VL execution unit does not match group {group}"
                ))),
            },
        )
    }

    /// Runs the TP-sharded MRoPE decoder while delegating routed experts to
    /// the matching-coordinate EP exchange group.
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
            Qwen3VlInput::Decode(tokens),
            cache,
            tensor_group,
            stream,
            |_adapter, group, index, layer, hidden, cache, context, execution| {
                if group != 1 {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL TP+EP decoder received non-text execution group {group}"
                    )));
                }
                let Qwen3VlLayer::Text(block) = layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL TP+EP decoder received a vision unit at text layer {index}"
                    )));
                };
                let tp_group = execution.group().ok_or_else(|| {
                    Error::Parallel("Qwen3-VL TP+EP execution requires an active TP group".into())
                })?;
                let mut output = block.forward_sparse_experts_with_rotary_tensor_parallel(
                    AttentionInput {
                        x: hidden,
                        mask: context.mask.as_ref(),
                        cache: cache.kv[index].as_mut(),
                    },
                    &context.cos,
                    &context.sin,
                    tp_group,
                    execution.stream(),
                    |hidden, ids, weights, stream| execute(index, hidden, ids, weights, stream),
                )?;
                if let Some(features) = context.deepstack_features.get(index) {
                    let base = zeros_dtype(output.shape(), output.dtype(), execution.stream())?;
                    let features = features.try_index_device((0, .., ..), execution.stream())?;
                    let aligned = masked_scatter(
                        &base,
                        context.visual_mask.as_ref().expect("DeepStack visual mask"),
                        features,
                        execution.stream(),
                    )?;
                    output = output.add(aligned, execution.stream())?;
                }
                Ok(output)
            },
        )
    }

    /// Clears temporary copies for one execution group.
    pub fn clear_device_group(&self, group: &str) -> Result<(), Error> {
        self.execution.clear_device_group(group)
    }
}

impl CausalModel<Cache> for Qwen3VlLayerwiseModel {
    type Tensor = Array;
    type Input<'a> = input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.prefill(input, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        self.decode(input_tokens, cache, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Loads either Qwen3-VL architecture through shared generalized residency.
pub fn load_qwen3_vl_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3-VL",
                args.text_config.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let adapter = Qwen3VlLayerwiseAdapter::new(args, stream)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    Ok(Qwen3VlLayerwiseModel {
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

/// Loads Qwen3-VL with rank-local vision execution groups.
pub(crate) fn load_qwen3_vl_tensor_parallel_layerwise_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let vision_path = resident::find_qwen3_vl_mmproj(model_dir)?;
        let vision_checkpoint = GgufCheckpoint::open(vision_path)?;
        let vision_metadata =
            crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
        return load_qwen3_vl_gguf_tensor_parallel_model(
            &checkpoint,
            &metadata,
            (&vision_checkpoint, &vision_metadata),
            options,
            build,
            stream,
            weights_stream,
        )
        .map(|(model, _)| model);
    }
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    let adapter = Qwen3VlLayerwiseAdapter::new(args, stream)?;
    Ok(Qwen3VlLayerwiseModel {
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

pub(crate) fn load_qwen3_vl_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision: (&GgufCheckpoint, &HashMap<String, GgufMetadataValue>),
    options: LayerWeightResidency,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Qwen3VlLayerwiseModel, Vec<u32>), Error> {
    let (vision_checkpoint, vision_metadata) = vision;
    crate::backend::mlx::runtime::execution::layerwise::validate_gguf_layerwise_source(
        checkpoint, metadata, options,
    )?;
    let prepared = resident::prepare_qwen3_vl_gguf_checkpoint(
        checkpoint,
        metadata,
        vision_checkpoint,
        vision_metadata,
    )?;
    let store = qwen3_vl_gguf_store(
        checkpoint,
        vision_checkpoint,
        &prepared.args,
        options.max_mapped_shards(),
    )?;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        Qwen3VlLayerwiseAdapter::new(prepared.args, stream)?,
        options,
        build,
        stream,
        weights_stream,
    )?;
    Ok((Qwen3VlLayerwiseModel { execution }, prepared.eos_token_ids))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_qwen3_vl_gguf_layerwise_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision_checkpoint: &GgufCheckpoint,
    vision_metadata: &HashMap<String, GgufMetadataValue>,
    residency: WeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Qwen3VlLayerwiseModel, Vec<u32>), Error> {
    let prepared = resident::prepare_qwen3_vl_gguf_checkpoint(
        checkpoint,
        metadata,
        vision_checkpoint,
        vision_metadata,
    )?;
    let store = qwen3_vl_gguf_store(
        checkpoint,
        vision_checkpoint,
        &prepared.args,
        residency.max_mapped_shards(),
    )?;
    let args = prepared.args;
    let execution = if let Some(expert_options) = residency.expert_cache() {
        load_qwen3_vl_gguf_sparse_execution(
            store,
            args,
            expert_options,
            residency.layers(),
            quantization,
            stream,
            weights_stream,
        )?
    } else {
        load_layerwise_model_with_quantization(
            store,
            Qwen3VlLayerwiseAdapter::new(args, stream)?,
            residency.layers(),
            quantization,
            stream,
            weights_stream,
        )?
    };
    Ok((Qwen3VlLayerwiseModel { execution }, prepared.eos_token_ids))
}

fn load_qwen3_vl_gguf_sparse_execution(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: resident::ModelArgs,
    options: ExpertCacheLoadOptions,
    non_expert: LayerWeightResidency,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseModel<Qwen3VlLayerwiseAdapter>, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires qwen3vlmoe GGUF".into(),
        ));
    }
    let mut adapter = Qwen3VlLayerwiseAdapter::new(args.clone(), stream)?;
    adapter.sparse_expert_cache = true;
    let mut execution = load_layerwise_model_with_quantization(
        store,
        adapter,
        non_expert,
        quantization,
        stream,
        weights_stream,
    )?;
    let store = execution.checkpoint_store_arc();
    let entries =
        crate::composition::mlx_architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
            &args.text_config,
            store.as_ref(),
            "model.language_model.layers",
        )?;
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
    Ok(execution)
}

pub(crate) fn qwen3_vl_gguf_store(
    checkpoint: &GgufCheckpoint,
    vision_checkpoint: &GgufCheckpoint,
    args: &resident::ModelArgs,
    max_mapped_shards: usize,
) -> Result<Arc<dyn eredu_checkpoint::store::CheckpointSource>, Error> {
    let deepstack = args.vision_config.deepstack_layers();
    let is_moe = args.text_config.is_moe();
    let text_variant = if is_moe {
        crate::composition::mlx_architectures::qwen::dense::checkpoint::GgufVariant::Qwen3Moe
    } else {
        crate::composition::mlx_architectures::qwen::dense::checkpoint::GgufVariant::Qwen3
    };
    let text_plan = crate::composition::mlx_architectures::qwen::dense::checkpoint::gguf_plan(
        &args.text_config,
        text_variant,
    )
    .map_err(Error::UnsupportedArchitecture)?;
    let vision_plan =
        super::checkpoint::projector_gguf_plan(&args.vision_config, args.text_config.hidden_size)
            .map_err(Error::UnsupportedArchitecture)?;
    Ok(Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(max_mapped_shards)?
            .add_checkpoint(checkpoint.catalog().clone(), &text_plan, move |name| {
                let name =
                    crate::composition::mlx_architectures::qwen::dense::translate_gguf_weight_name(
                        name, is_moe,
                    );
                name.strip_prefix("model.")
                    .map(|name| format!("model.language_model.{name}"))
                    .unwrap_or(name)
            })?
            .add_checkpoint(
                vision_checkpoint.catalog().clone(),
                &vision_plan,
                move |name| resident::translate_qwen3_vl_mmproj_name(name, &deepstack),
            )?
            .build()?,
    ))
}

/// Loads Qwen3-VL-MoE with independently cached experts and bounded non-expert units.
pub fn load_qwen3_vl_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::backend::mlx::runtime::execution::layerwise::NonExpertWeightResidency,
    options: ExpertCacheLoadOptions,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    let model_dir = model_dir.as_ref();
    let args = resident::get_qwen3_vl_model_args(model_dir)?;
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "sparse expert caching requires a Qwen3-VL-MoE checkpoint".into(),
        ));
    }
    let quantize_on_load = quantization
        .map(|requested| {
            should_quantize_on_load(
                "Qwen3-VL independent expert cache",
                args.text_config.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let mut source_adapter = Qwen3VlLayerwiseAdapter::new(args.clone(), stream)?;
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
    let entries =
        crate::composition::mlx_architectures::qwen::dense::layerwise::qwen3_expert_catalog_at(
            &args.text_config,
            store.as_ref(),
            "model.language_model.layers",
        )?;
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
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Builds the streamed nonexpert Qwen3-VL execution base used by distributed EP.
pub(crate) fn load_qwen3_vl_sparse_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "streamed sparse expert parallelism requires Qwen3-VL-MoE".into(),
        ));
    }
    let mut adapter = Qwen3VlLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_layerwise_model(store, adapter, non_expert, stream, weights_stream)?;
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Builds the TP-sharded nonexpert Qwen3-VL-MoE base used by combined TP+EP.
pub(crate) fn load_qwen3_vl_sparse_tp_ep_base_with_store(
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    args: ModelArgs,
    non_expert: impl Into<LayerWeightResidency>,
    build: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Qwen3VlLayerwiseModel, Error> {
    if !args.text_config.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "combined tensor/expert parallelism requires Qwen3-VL-MoE".into(),
        ));
    }
    let mut adapter = Qwen3VlLayerwiseAdapter::new(args, stream)?;
    adapter.sparse_expert_cache = true;
    let execution = load_tensor_parallel_layerwise_model(
        store,
        adapter,
        non_expert,
        build,
        stream,
        weights_stream,
    )?;
    Ok(Qwen3VlLayerwiseModel { execution })
}

/// Family-specific input distinguishing typed prefill from cached decode.
pub enum Qwen3VlInput<'a> {
    /// Ordered text and visual prompt parts.
    Prefill(input::ModelInput<'a>),
    /// Text token IDs for a cached decode step.
    Decode(&'a Array),
}

enum PreparedPart {
    Text(Array),
    Visual(i32),
}

/// Per-forward vision preparation and multimodal text state.
pub struct Qwen3VlForwardContext {
    tokens: Array,
    parts: Vec<PreparedPart>,
    vision: Option<QwenVisionLayerwiseState>,
    mask: Option<Array>,
    cos: Array,
    sin: Array,
    visual_mask: Option<Array>,
    deepstack_features: Vec<Array>,
}

/// Architecture-authored immutable state relayed with pipeline activations.
///
/// DeepStack features are already aligned to the assembled decoder sequence,
/// so the transport layer only needs stable `[batch, sequence, hidden]`
/// tensors and never needs to understand visual placeholder semantics.
pub(crate) struct Qwen3VlPipelinePrepared {
    pub(crate) hidden: Array,
    pub(crate) cos: Array,
    pub(crate) sin: Array,
    pub(crate) rope_delta: i32,
    pub(crate) deepstack_features: Vec<Array>,
}

/// Opaque prepared state routed between placed Qwen3-VL vision owners.
pub(crate) struct Qwen3VlPipelineIngressState {
    cache: Cache,
    forward: LayerwiseForwardState<Qwen3VlForwardContext>,
}

/// One temporary unit from either the vision or text group.
pub enum Qwen3VlLayer {
    /// Vision transformer block.
    Vision(Box<QwenVisionBlock>),
    /// Dense or sparse-MoE Qwen3 decoder block.
    Text(Box<TransformerBlock>),
}

impl ModuleParameters for Qwen3VlLayer {
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

/// Shared dense/MoE multimodal adapter.
pub struct Qwen3VlLayerwiseAdapter {
    args: ModelArgs,
    vision: QwenVisionLayerwiseStatic,
    embedding: MaybeQuantized<nn::Embedding>,
    parallel_embedding: Option<VocabParallelEmbedding>,
    norm: nn::RmsNorm,
    lm_head: Option<MaybeQuantized<nn::Linear>>,
    parallel_lm_head: Option<VocabParallelLmHead>,
    parallel_kv_heads: Option<Vec<i32>>,
    sparse_expert_cache: bool,
    expert_cache: Option<ExpertCache>,
}

impl Qwen3VlLayerwiseAdapter {
    /// Starts placed Qwen3-VL ingress before any vision block executes.
    pub(crate) fn begin_pipeline_ingress(
        &mut self,
        typed: input::ModelInput<'_>,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelineIngressState, Error> {
        let mut cache = Cache::default();
        let forward = self.prepare_prefill(typed, &mut cache, execution, stream)?;
        Ok(Qwen3VlPipelineIngressState { cache, forward })
    }

    /// Rebuilds only parameter-free vision scheduling state on a downstream
    /// PP owner. Patch/text embeddings and projectors stay on their declared
    /// static owners; the previous encoder owner supplies the hidden tensor.
    pub(crate) fn begin_pipeline_continuation(
        &self,
        typed: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelineIngressState, Error> {
        input::validate(typed)?;
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        for part in typed.parts {
            if matches!(
                part.modality,
                input::Modality::Image | input::Modality::Video
            ) {
                let input::InputPayload::Tensor(tensor) = part.payload else {
                    return Err(Error::Parallel(
                        "Qwen3-VL continuation requires tensor media payloads".into(),
                    ));
                };
                let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                    Error::Parallel("Qwen3-VL continuation omitted qwen_grid_thw".into())
                })?;
                pixels.push(tensor.clone());
                grids.push(grid.clone());
            }
        }
        if pixels.is_empty() {
            return Err(Error::Parallel(
                "Qwen3-VL continuation requires at least one visual payload".into(),
            ));
        }
        let pixel_refs = pixels.iter().collect::<Vec<_>>();
        let grid_refs = grids.iter().collect::<Vec<_>>();
        let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
        let grids = concatenate_axis(&grid_refs, 0, stream)?;
        let vision = self.vision.continuation_state(&pixels, &grids, stream)?;
        let hidden = zeros_dtype(
            &[pixels.dim(0), self.args.vision_config.hidden_size],
            pixels.dtype(),
            stream,
        )?;
        let empty = zeros_dtype(
            &[1, 0, self.args.text_config.head_dim],
            pixels.dtype(),
            stream,
        )?;
        Ok(Qwen3VlPipelineIngressState {
            cache: Cache::default(),
            forward: LayerwiseForwardState {
                hidden,
                context: Qwen3VlForwardContext {
                    tokens: Array::from_slice(&[] as &[u32], &[1, 0]),
                    parts: Vec::new(),
                    vision: Some(vision),
                    mask: None,
                    cos: empty.clone(),
                    sin: empty,
                    visual_mask: None,
                    deepstack_features: Vec::new(),
                },
            },
        })
    }

    /// Returns whether this request contains model-native visual work.
    pub(crate) fn pipeline_ingress_active(&self, state: &Qwen3VlPipelineIngressState) -> bool {
        state.forward.context.vision.is_some()
    }

    /// Exports variable DeepStack state and the evolving vision activation.
    pub(crate) fn pipeline_ingress_arrays(
        &self,
        state: &Qwen3VlPipelineIngressState,
    ) -> Vec<Array> {
        std::iter::once(state.forward.hidden.clone())
            .chain(
                state
                    .forward
                    .context
                    .vision
                    .iter()
                    .flat_map(QwenVisionLayerwiseState::retained_arrays)
                    .cloned(),
            )
            .collect()
    }

    /// Imports activation and DeepStack features from the previous PP owner.
    pub(crate) fn replace_pipeline_ingress_arrays(
        &self,
        state: &mut Qwen3VlPipelineIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let (hidden, features) = arrays.split_first().ok_or_else(|| {
            Error::Parallel("Qwen3-VL placed ingress omitted vision hidden state".into())
        })?;
        state.forward.hidden = hidden.clone();
        if let Some(vision) = &mut state.forward.context.vision {
            vision.replace_deepstack_features(features.to_vec());
        } else if !features.is_empty() {
            return Err(Error::Parallel(
                "Qwen3-VL text-only ingress received DeepStack state".into(),
            ));
        }
        Ok(())
    }

    /// Executes one placed vision block.
    pub(crate) fn forward_pipeline_vision_layer(
        &mut self,
        index: usize,
        layer: &mut Qwen3VlLayer,
        state: &mut Qwen3VlPipelineIngressState,
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

    /// Finalizes merger/projector state into decoder-facing payload tensors.
    pub(crate) fn finish_pipeline_ingress(
        &mut self,
        mut state: Qwen3VlPipelineIngressState,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let hidden = match execution {
            Some(execution) => self.begin_execution_group_with_execution(
                1,
                &state.forward.hidden,
                &[state.forward.hidden.clone()],
                &mut state.cache,
                &mut state.forward.context,
                execution,
            )?,
            None => self.begin_execution_group(
                1,
                &state.forward.hidden,
                &[state.forward.hidden.clone()],
                &mut state.cache,
                &mut state.forward.context,
                stream,
            )?,
        };
        let deepstack_features = state
            .forward
            .context
            .deepstack_features
            .iter()
            .map(|features| {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                masked_scatter(
                    &base,
                    state
                        .forward
                        .context
                        .visual_mask
                        .as_ref()
                        .expect("DeepStack visual mask"),
                    features.try_index_device((0, .., ..), stream)?,
                    stream,
                )
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos: state.forward.context.cos,
            sin: state.forward.context.sin,
            rope_delta: state.cache.rope_delta,
            deepstack_features,
        })
    }

    pub(crate) fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let text = Decoder::new(&args.text_config, stream)?;
        let lm_head = if args.text_config.tie_word_embeddings {
            None
        } else {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.text_config.hidden_size,
                    args.text_config.vocab_size,
                    args.text_config
                        .quantization
                        .or(args.text_config.quantization_config),
                    stream,
                )?,
            )
        };
        Ok(Self {
            args,
            vision: QwenVisionLayerwiseStatic::from_transformer(visual),
            embedding: text.embed_tokens,
            parallel_embedding: None,
            norm: text.norm,
            lm_head,
            parallel_lm_head: None,
            parallel_kv_heads: None,
            sparse_expert_cache: false,
            expert_cache: None,
        })
    }

    pub(crate) fn new_external_experts(args: ModelArgs, stream: &Stream) -> Result<Self, Error> {
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = true;
        Ok(adapter)
    }

    /// Returns parsed multimodal arguments.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub(crate) fn vision_mut(&mut self) -> &mut QwenVisionLayerwiseStatic {
        &mut self.vision
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

    /// Executes the local multimodal reference path and returns decoder-facing
    /// state. Distributed execution uses the same architecture-owned methods
    /// through the generic placement and transport plan.
    pub(crate) fn prepare_pipeline_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        vision_layers: &mut [Qwen3VlLayer],
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let mut cache = Cache::default();
        let mut state = self.prepare_prefill(typed, &mut cache, execution, stream)?;
        if state.context.vision.is_some() {
            if vision_layers.len() != self.args.vision_config.layer_count() {
                return Err(Error::Parallel(format!(
                    "Qwen3-VL local reference owns {} vision blocks, expected {}",
                    vision_layers.len(),
                    self.args.vision_config.layer_count()
                )));
            }
            for (index, layer) in vision_layers.iter_mut().enumerate() {
                state.hidden = if let Some(execution) = execution {
                    self.forward_layer_with_execution(
                        0,
                        index,
                        layer,
                        &state.hidden,
                        &mut cache,
                        &mut state.context,
                        execution,
                    )?
                } else {
                    self.forward_layer(
                        0,
                        index,
                        layer,
                        &state.hidden,
                        &mut cache,
                        &mut state.context,
                        stream,
                    )?
                };
            }
        }
        let hidden = if let Some(execution) = execution {
            self.begin_execution_group_with_execution(
                1,
                &state.hidden,
                &[state.hidden.clone()],
                &mut cache,
                &mut state.context,
                execution,
            )?
        } else {
            self.begin_execution_group(
                1,
                &state.hidden,
                &[state.hidden.clone()],
                &mut cache,
                &mut state.context,
                stream,
            )?
        };
        let deepstack_features = state
            .context
            .deepstack_features
            .iter()
            .map(|features| {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                masked_scatter(
                    &base,
                    state
                        .context
                        .visual_mask
                        .as_ref()
                        .expect("DeepStack visual mask"),
                    features.try_index_device((0, .., ..), stream)?,
                    stream,
                )
                .map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos: state.context.cos,
            sin: state.context.sin,
            rope_delta: cache.rope_delta,
            deepstack_features,
        })
    }

    /// Embeds a text-only pipeline step and reconstructs MRoPE from the
    /// persisted multimodal position delta.
    pub(crate) fn prepare_pipeline_tokens(
        &mut self,
        tokens: &Array,
        offset: i32,
        rope_delta: i32,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<Qwen3VlPipelinePrepared, Error> {
        let hidden = match (&mut self.parallel_embedding, execution) {
            (Some(embedding), Some(execution)) => embedding.forward(tokens, execution)?,
            _ => self.embedding.forward(tokens, stream)?,
        };
        let start = offset + rope_delta;
        let positions = [
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
        ];
        let (cos, sin) = resident::mrope_embeddings(
            &positions,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let deepstack_features = (0..self.args.vision_config.deepstack_layer_count())
            .map(|_| zeros_dtype(hidden.shape(), hidden.dtype(), stream).map_err(Error::from))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Qwen3VlPipelinePrepared {
            hidden,
            cos,
            sin,
            rope_delta,
            deepstack_features,
        })
    }

    fn prepare_prefill(
        &mut self,
        typed: input::ModelInput<'_>,
        cache: &mut Cache,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Qwen3VlForwardContext>, Error> {
        input::validate(typed)?;
        let mut token_parts = Vec::new();
        let mut prepared_parts = Vec::new();
        let mut pixels = Vec::new();
        let mut grids = Vec::new();
        let merge = self.args.vision_config.spatial_merge_size;
        for part in typed.parts {
            match (part.modality, part.payload) {
                (input::Modality::Text, input::InputPayload::TokenIds(tokens)) => {
                    token_parts.push(tokens.clone());
                    let embedding = match (&mut self.parallel_embedding, execution) {
                        (Some(embedding), Some(execution)) => {
                            embedding.forward(tokens, execution)?
                        }
                        _ => self.embedding.forward(tokens, stream)?,
                    };
                    prepared_parts.push(PreparedPart::Text(embedding));
                }
                (
                    input::Modality::Image | input::Modality::Video,
                    input::InputPayload::Tensor(tensor),
                ) => {
                    let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                        Error::UnsupportedArchitecture(format!(
                            "Qwen3-VL {} input requires qwen_grid_thw metadata",
                            part.modality.as_str()
                        ))
                    })?;
                    let merged = grid_thw_from_array(grid, stream)?
                        .into_iter()
                        .map(|(t, h, w)| t * (h / merge) * (w / merge))
                        .sum::<i32>();
                    let token_id = if part.modality == input::Modality::Image {
                        self.args.image_token_id
                    } else {
                        self.args.video_token_id
                    };
                    token_parts.push(input::token_ids_array(
                        &vec![token_id; merged as usize],
                        stream,
                    )?);
                    prepared_parts.push(PreparedPart::Visual(merged));
                    pixels.push(tensor.clone());
                    grids.push(grid.clone());
                }
                (modality, _) => {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen3-VL layerwise input does not support {} payloads of this kind",
                        modality.as_str()
                    )));
                }
            }
        }
        let token_refs = token_parts.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&token_refs, 1, stream)?;
        let (position_ids, rope_delta) =
            resident::multimodal_position_ids(typed, merge, tokens.dim(1), stream)?;
        cache.rope_delta = rope_delta;
        let (cos, sin) = resident::mrope_embeddings(
            &position_ids,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let (hidden, vision) = if pixels.is_empty() {
            let hidden = prepared_parts
                .iter()
                .filter_map(|part| match part {
                    PreparedPart::Text(value) => Some(value),
                    PreparedPart::Visual(_) => None,
                })
                .collect::<Vec<_>>();
            (concatenate_axis(&hidden, 1, stream)?, None)
        } else {
            let pixel_refs = pixels.iter().collect::<Vec<_>>();
            let grid_refs = grids.iter().collect::<Vec<_>>();
            let pixels = concatenate_axis(&pixel_refs, 0, stream)?;
            let grids = concatenate_axis(&grid_refs, 0, stream)?;
            let (hidden, state) = self.vision.begin(&pixels, &grids, stream)?;
            (hidden, Some(state))
        };
        Ok(LayerwiseForwardState {
            hidden,
            context: Qwen3VlForwardContext {
                tokens,
                parts: prepared_parts,
                vision,
                mask: None,
                cos,
                sin,
                visual_mask: None,
                deepstack_features: Vec::new(),
            },
        })
    }

    fn prepare_decode(
        &mut self,
        tokens: &Array,
        cache: &mut Cache,
        execution: Option<
            &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<'_>,
        >,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Qwen3VlForwardContext>, Error> {
        let hidden = match (&mut self.parallel_embedding, execution) {
            (Some(embedding), Some(execution)) => embedding.forward(tokens, execution)?,
            _ => self.embedding.forward(tokens, stream)?,
        };
        let start = cache
            .kv
            .first()
            .and_then(Option::as_ref)
            .map(KeyValueCache::offset)
            .unwrap_or(0)
            + cache.rope_delta;
        let positions = [
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
            (start..start + tokens.dim(1)).collect(),
        ];
        let (cos, sin) = resident::mrope_embeddings(
            &positions,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        Ok(LayerwiseForwardState {
            hidden,
            context: Qwen3VlForwardContext {
                tokens: tokens.clone(),
                parts: Vec::new(),
                vision: None,
                mask: None,
                cos,
                sin,
                visual_mask: None,
                deepstack_features: Vec::new(),
            },
        })
    }
}

impl LoadTimeQuantizableAdapter for Qwen3VlLayerwiseAdapter {
    fn load_time_quantized(
        &self,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<Self, Error> {
        let mut args = self.args.clone();
        args.text_config.quantization = Some(quantization);
        args.text_config.quantization_config = None;
        args.text_config.quantized_weight_configs = None;
        args.vision_config
            .apply_load_time_quantization(quantization);
        let mut adapter = Self::new(args, stream)?;
        adapter.sparse_expert_cache = self.sparse_expert_cache;
        Ok(adapter)
    }
}

impl ArchitectureAdapter for Qwen3VlLayerwiseAdapter {
    type Input<'a> = Qwen3VlInput<'a>;
    type Cache = Cache;
    type Layer = Qwen3VlLayer;
    type ForwardContext = Qwen3VlForwardContext;

    fn model_type(&self) -> &str {
        if self.args.text_config.model_type == "qwen3_vl_moe_text" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
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
        self.args
            .text_config
            .quantization
            .or(self.args.text_config.quantization_config)
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        let vision = target
            .strip_prefix("model.visual.")
            .or_else(|| target.split_once(".visual.").map(|(_, suffix)| suffix));
        match vision {
            Some(target) => self
                .args
                .vision_config
                .quantized_weight_configs
                .contains_key(target),
            None => true,
        }
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::backend::mlx::MlxParallelContext>,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let layer_count = self.args.text_config.num_hidden_layers as usize;
        let local_kv_heads = match topology {
            Some(topology) if topology.is_axis_active(crate::ParallelAxis::Tensor) => {
                self.parallel_kv_heads.clone().ok_or_else(|| {
                    Error::Parallel(
                    "Qwen3-VL parallel cache identity requested before local layout configuration"
                        .into(),
                )
                })?
            }
            _ => vec![self.args.text_config.num_key_value_heads; layer_count],
        };
        Ok(PromptCacheModelIdentity {
            model_family: "qwen3_vl".into(),
            effective_model_type: self.model_type().into(),
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
            layer_layout: resident::prompt_cache_layer_layout_with_kv_heads(
                &self.args,
                &local_kv_heads,
            )?,
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
        let patch = "model.visual.patch_embed.proj.weight";
        let vision_recipes = if store.source_keys().contains(&format!("{patch}.1")) {
            BTreeMap::from([(
                "patch_embed.proj.weight".to_string(),
                DerivedWeightRecipe::Stack {
                    axis: 2,
                    inputs: vec![
                        DerivedWeightRecipe::source(patch, TensorSelection::Full),
                        DerivedWeightRecipe::source(format!("{patch}.1"), TensorSelection::Full),
                    ],
                },
            )])
        } else {
            BTreeMap::new()
        };
        let mut units = vec![
            StaticUnitBindings::new(
                VISION_STATIC_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.vision,
                    "model.visual",
                    store,
                    vision_recipes,
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                EMBEDDING_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.embedding,
                    "model.language_model.embed_tokens",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
            StaticUnitBindings::new(
                NORM_UNIT,
                build_module_binding_plan_with_recipes(
                    &self.norm,
                    "model.language_model.norm",
                    store,
                    BTreeMap::new(),
                )?
                .build_bindings(store)?,
            )?,
        ];
        if let Some(head) = &self.lm_head {
            units.push(StaticUnitBindings::new(
                HEAD_UNIT,
                build_module_binding_plan_with_recipes(head, "lm_head", store, BTreeMap::new())?
                    .build_bindings(store)?,
            )?);
        }
        Ok(units)
    }

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected = if self.lm_head.is_some() { 4 } else { 3 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        populate_module_from_lease(&mut self.vision, &leases[0])?;
        if let Some(embedding) = &mut self.parallel_embedding {
            populate_module_from_lease(embedding.inner_mut(), &leases[1])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[1])?;
        }
        populate_module_from_lease(&mut self.norm, &leases[2])?;
        if let Some(head) = &mut self.parallel_lm_head {
            populate_module_from_lease(head.inner_mut(), &leases[3])?;
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[3])?;
        }
        Ok(())
    }

    fn validate_cache(&self, cache: &mut Cache) -> Result<(), Error> {
        if cache.kv.is_empty() {
            cache.kv = (0..self.args.text_config.num_hidden_layers)
                .map(|_| Some(Default::default()))
                .collect();
        }
        if cache.kv.len() != self.args.text_config.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL cache has {} layers, expected {}",
                cache.kv.len(),
                self.args.text_config.num_hidden_layers
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
        match input {
            Qwen3VlInput::Prefill(input) => self.prepare_prefill(input, cache, None, stream),
            Qwen3VlInput::Decode(tokens) => self.prepare_decode(tokens, cache, None, stream),
        }
    }

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        cache: &mut Self::Cache,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        match input {
            Qwen3VlInput::Prefill(input) => {
                self.prepare_prefill(input, cache, Some(execution), execution.stream())
            }
            Qwen3VlInput::Decode(tokens) => {
                self.prepare_decode(tokens, cache, Some(execution), execution.stream())
            }
        }
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(["vision_encoder", "text_decoder"]).map_err(Into::into)
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        group != 0 || context.vision.is_some()
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self.args.vision_config.layer_count()),
            1 => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
        match group {
            0 => Ok(Qwen3VlLayer::Vision(Box::new(QwenVisionBlock::new(
                &self.args.vision_config,
                index,
                stream,
            )?))),
            1 => Ok(Qwen3VlLayer::Text(Box::new(
                TransformerBlock::new_for_layer(&self.args.text_config, index as i32, stream)?,
            ))),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            ))),
        }
    }

    fn parallel_parameter_groups(
        &self,
        _context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
    ) -> Result<Vec<eredu_runtime::ParameterGroupSpec>, Error> {
        let mut groups = vec![vocab_embedding_parameter_group(
            &self.embedding,
            "model.language_model.embed_tokens",
            self.args.text_config.vocab_size as usize,
            self.args.text_config.hidden_size,
            false,
        )?];
        if let Some(head) = &self.lm_head {
            groups.push(vocab_lm_head_parameter_group(
                head,
                "lm_head",
                self.args.text_config.hidden_size,
                self.args.text_config.vocab_size as usize,
                false,
            )?);
        }
        Ok(groups)
    }

    fn configure_parallel_static(
        &mut self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        layout: &eredu_runtime::LocalModelLayout,
        stream: &Stream,
    ) -> Result<(), Error> {
        let config = &self.args.text_config;
        self.parallel_kv_heads = Some(planned_kv_head_layout(
            layout,
            config.num_hidden_layers as usize,
            config.head_dim,
            "model.language_model.layers",
        )?);
        self.parallel_embedding = Some(VocabParallelEmbedding::unloaded(
            config.vocab_size as usize,
            config.hidden_size,
            config.quantization.or(config.quantization_config),
            context,
            stream,
        )?);
        if self.lm_head.is_some() {
            self.parallel_lm_head = Some(VocabParallelLmHead::unloaded(
                config.hidden_size,
                config.vocab_size as usize,
                config.quantization.or(config.quantization_config),
                context,
                stream,
            )?);
        }
        configure_vision_parallel_static(&mut self.vision, "model.visual", layout, stream)?;
        Ok(())
    }

    fn register_parallel_parameters(
        &self,
        context: crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::backend::mlx::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        for group in self.parallel_parameter_groups(context)? {
            planner.register(group)?;
        }
        for group in
            vision_parallel_parameter_groups(&self.args.vision_config, "model.visual", stream)?
        {
            planner.register(group)?;
        }
        for index in 0..self.args.text_config.num_hidden_layers as usize {
            let layer =
                TransformerBlock::new_for_layer(&self.args.text_config, index as i32, stream)?;
            register_qwen_layer_parallel_plan(
                planner,
                &layer,
                &self.args.text_config,
                &format!("model.language_model.layers.{index}"),
            )?;
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
        if group != 0 {
            let prefix = format!("model.language_model.layers.{index}");
            let planned = |name: &str| {
                layout
                    .tensor(&format!("{prefix}.{name}.weight"))
                    .or_else(|| layout.tensor(&format!("{prefix}.{name}.inner.weight")))
            };
            let mut config = self.args.text_config.clone();
            let query = planned("self_attn.q_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP query for {prefix}")))?;
            let key = planned("self_attn.k_proj")
                .ok_or_else(|| Error::Parallel(format!("missing TP key for {prefix}")))?;
            config.num_attention_heads = i32::try_from(query.local_shape()[0])
                .map_err(|_| Error::Parallel("Qwen local query width exceeds i32".into()))?
                / config.head_dim;
            config.num_key_value_heads = i32::try_from(key.local_shape()[0])
                .map_err(|_| Error::Parallel("Qwen local key width exceeds i32".into()))?
                / config.head_dim;
            if config.is_moe() {
                let experts = layout
                    .tensor(&format!("{prefix}.mlp.experts.gate_up_proj"))
                    .ok_or_else(|| Error::Parallel(format!("missing TP experts for {prefix}")))?;
                config.moe_intermediate_size = i32::try_from(experts.local_shape()[1] / 2)
                    .map_err(|_| Error::Parallel("Qwen local expert width exceeds i32".into()))?;
            } else {
                let gate = planned("mlp.gate_proj")
                    .ok_or_else(|| Error::Parallel(format!("missing TP MLP for {prefix}")))?;
                config.intermediate_size = i32::try_from(gate.local_shape()[0])
                    .map_err(|_| Error::Parallel("Qwen local MLP width exceeds i32".into()))?;
            }
            return Ok(Qwen3VlLayer::Text(Box::new(
                TransformerBlock::new_for_layer(&config, index as i32, stream)?,
            )));
        }
        let config = &self.args.vision_config;
        Ok(Qwen3VlLayer::Vision(Box::new(new_parallel_vision_block(
            config,
            "model.visual",
            index,
            layout,
            stream,
        )?)))
    }

    fn new_expert_parallel_layer(
        &self,
        group: usize,
        index: usize,
        assignment: &crate::backend::mlx::runtime::distributed::expert::ExpertAssignment,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        if group != 1 || !self.args.text_config.is_moe() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL expert-local construction requires an MoE text layer, got group {group}"
            )));
        }
        let mut layer = self.new_layer(group, index, stream)?;
        let Qwen3VlLayer::Text(block) = &mut layer else {
            unreachable!("validated text group")
        };
        let FeedForward::Moe(moe) = &mut block.mlp else {
            return Err(Error::Parallel(format!(
                "Qwen3-VL text layer {index} is not an MoE layer"
            )));
        };
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen3-VL expert count exceeds i32".into()))?
        };
        let prefix = format!("model.language_model.layers.{index}.mlp.experts");
        moe.experts = QwenExperts::new(
            local_experts,
            self.args.text_config.hidden_size,
            self.args.text_config.moe_intermediate_size,
            self.args
                .text_config
                .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
            self.args
                .text_config
                .weight_quantization_for(&format!("{prefix}.down_proj")),
            stream,
        )?;
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
        if group != 1 || !self.args.text_config.is_moe() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL combined tensor/expert construction requires an MoE text layer, got group {group}"
            )));
        }
        let mut layer = self.new_parallel_layer(group, index, layout, stream)?;
        let Qwen3VlLayer::Text(block) = &mut layer else {
            unreachable!("validated text group")
        };
        let FeedForward::Moe(moe) = &mut block.mlp else {
            return Err(Error::Parallel(format!(
                "Qwen3-VL text layer {index} is not an MoE layer"
            )));
        };
        let local_experts = if self.sparse_expert_cache {
            0
        } else {
            i32::try_from(assignment.local_global_expert_ids().len())
                .map_err(|_| Error::Parallel("local Qwen3-VL expert count exceeds i32".into()))?
        };
        let local_intermediate = moe.experts.intermediate_dim;
        let prefix = format!("model.language_model.layers.{index}.mlp.experts");
        moe.experts = QwenExperts::new(
            local_experts,
            self.args.text_config.hidden_size,
            local_intermediate,
            self.args
                .text_config
                .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
            self.args
                .text_config
                .weight_quantization_for(&format!("{prefix}.down_proj")),
            stream,
        )?;
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
        if !self.args.text_config.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen3-VL has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::backend::mlx::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.text_config.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("model.visual.blocks.{index}")
        } else {
            format!("model.language_model.layers.{index}")
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("qwen3_vl.vision.{index:05}")
        } else {
            format!("qwen3_vl.text.{index:05}")
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
        if group == 1 {
            let Qwen3VlLayer::Text(layer) = layer else {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-VL text execution group contains a vision layer at {index}"
                )));
            };
            qwen_text_layer_bindings(
                layer,
                &self.args.text_config,
                &prefix,
                store,
                self.sparse_expert_cache,
            )
        } else {
            Ok(
                build_module_binding_plan_with_recipes(layer, &prefix, store, BTreeMap::new())?
                    .build_bindings(store)?,
            )
        }
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
        if group != 1 {
            return Err(Error::Parallel(format!(
                "Qwen3-VL expert-local bindings require text group 1, got {group}"
            )));
        }
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

    fn ignores_checkpoint_key(&self, key: &str) -> bool {
        crate::composition::mlx_architectures::qwen::dense::checkpoint::is_redundant_tied_output_head_key(
            &self.args.text_config,
            key,
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
        match (group, layer) {
            (0, Qwen3VlLayer::Vision(block)) => {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let output =
                    self.vision
                        .forward_block(block, index, hidden.clone(), state, stream)?;
                self.vision
                    .capture_deepstack(index, &output, state, stream)?;
                Ok(output)
            }
            (1, Qwen3VlLayer::Text(block)) => {
                let mut output = if self.sparse_expert_cache {
                    let expert_cache = self.expert_cache.as_ref().ok_or_else(|| {
                        Error::UnsupportedArchitecture(
                            "Qwen3-VL sparse expert cache was not initialized".into(),
                        )
                    })?;
                    let pass = if hidden.dim(1) > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    block.forward_sparse_experts_with_rotary(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: cache.kv[index].as_mut(),
                        },
                        &context.cos,
                        &context.sin,
                        stream,
                        |flat, indices, weights, stream| {
                            expert_cache
                                .execute_routes_bounded(
                                    ExpertRouteBatch::new(index, flat, indices, weights, pass),
                                    stream,
                                    |flat, acquired, weights, stream| {
                                        let started = Instant::now();
                                        let prefix = format!(
                                            "model.language_model.layers.{index}.mlp.experts"
                                        );
                                        let args = &self.args.text_config;
                                        let load_time = expert_cache.weight_quantization();
                                        let mut bank = QwenExperts::new(
                                            acquired.identities().len() as i32,
                                            args.hidden_size,
                                            args.moe_intermediate_size,
                                            load_time.or_else(|| {
                                                args.weight_quantization_for(&format!(
                                                    "{prefix}.gate_up_proj"
                                                ))
                                            }),
                                            load_time.or_else(|| {
                                                args.weight_quantization_for(&format!(
                                                    "{prefix}.down_proj"
                                                ))
                                            }),
                                            stream,
                                        )?;
                                        bank.gate_up_proj = Param::new(
                                            acquired
                                                .compact_binding("gate_up_proj", stream)
                                                .map_err(|error| {
                                                    Exception::custom(error.to_string())
                                                })?,
                                        );
                                        bank.gate_up_proj_scales = Param::new(
                                            acquired
                                                .optional_compact_binding(
                                                    "gate_up_proj_scales",
                                                    stream,
                                                )
                                                .map_err(|error| {
                                                    Exception::custom(error.to_string())
                                                })?,
                                        );
                                        bank.gate_up_proj_biases = Param::new(
                                            acquired
                                                .optional_compact_binding(
                                                    "gate_up_proj_biases",
                                                    stream,
                                                )
                                                .map_err(|error| {
                                                    Exception::custom(error.to_string())
                                                })?,
                                        );
                                        bank.down_proj = Param::new(
                                            acquired.compact_binding("down_proj", stream).map_err(
                                                |error| Exception::custom(error.to_string()),
                                            )?,
                                        );
                                        bank.down_proj_scales = Param::new(
                                            acquired
                                                .optional_compact_binding(
                                                    "down_proj_scales",
                                                    stream,
                                                )
                                                .map_err(|error| {
                                                    Exception::custom(error.to_string())
                                                })?,
                                        );
                                        bank.down_proj_biases = Param::new(
                                            acquired
                                                .optional_compact_binding(
                                                    "down_proj_biases",
                                                    stream,
                                                )
                                                .map_err(|error| {
                                                    Exception::custom(error.to_string())
                                                })?,
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
                    )?
                } else {
                    block.forward_with_rotary_embeddings(
                        AttentionInput {
                            x: hidden,
                            mask: context.mask.as_ref(),
                            cache: cache.kv[index].as_mut(),
                        },
                        &context.cos,
                        &context.sin,
                        stream,
                    )?
                };
                if let Some(features) = context.deepstack_features.get(index) {
                    let base = zeros_dtype(output.shape(), output.dtype(), stream)?;
                    let features = features.try_index_device((0, .., ..), stream)?;
                    let aligned = masked_scatter(
                        &base,
                        context.visual_mask.as_ref().expect("DeepStack visual mask"),
                        features,
                        stream,
                    )?;
                    output = output.add(aligned, stream)?;
                }
                Ok(output)
            }
            _ => Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL execution unit does not match group {group}"
            ))),
        }
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
        if group == 0 {
            if let Qwen3VlLayer::Vision(block) = layer {
                let Some(state) = context.vision.as_mut() else {
                    return Ok(hidden.clone());
                };
                let output = self.vision.forward_block_tensor_parallel(
                    block,
                    index,
                    hidden.clone(),
                    state,
                    tp_group,
                    execution.stream(),
                )?;
                self.vision.capture_deepstack_tensor_parallel(
                    index,
                    &output,
                    state,
                    tp_group,
                    execution.stream(),
                )?;
                return Ok(output);
            }
        } else if let Qwen3VlLayer::Text(block) = layer {
            return Ok(block.forward_with_rotary_embeddings_tensor_parallel(
                AttentionInput {
                    x: hidden,
                    mask: context.mask.as_ref(),
                    cache: cache.kv[index].as_mut(),
                },
                &context.cos,
                &context.sin,
                tp_group,
                execution.stream(),
            )?);
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
        if group == 1 {
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
        context
            .vision
            .as_ref()
            .map(QwenVisionLayerwiseState::retained_arrays)
            .unwrap_or_default()
            .into_iter()
            .chain(context.deepstack_features.iter())
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
        if group == 0 {
            return Ok(initial_hidden.clone());
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            )));
        }
        let hidden = dependency_outputs.first().unwrap_or(initial_hidden);
        let hidden = if let Some(mut state) = context.vision.take() {
            let output = self.vision.finish(hidden, &mut state, stream)?;
            context.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(context.parts.len());
            for part in &context.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            let refs = assembled.iter().collect::<Vec<_>>();
            concatenate_axis(&refs, 1, stream)?
        } else {
            hidden.clone()
        };
        context.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3-VL layerwise execution requires an explicit causal mask".into(),
                ));
            }
            None => None,
        };
        context.visual_mask = if context.deepstack_features.is_empty() {
            None
        } else {
            Some(
                context
                    .tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &context
                            .tokens
                            .eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        Ok(hidden)
    }

    fn begin_execution_group_with_execution(
        &mut self,
        group: usize,
        initial_hidden: &Array,
        dependency_outputs: &[Array],
        cache: &mut Self::Cache,
        context: &mut Self::ForwardContext,
        execution: &crate::backend::mlx::runtime::distributed::parallel::ParallelExecutionContext<
            '_,
        >,
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
        if group == 0 {
            return Ok(initial_hidden.clone());
        }
        if group != 1 {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL has no execution group {group}"
            )));
        }
        let hidden = dependency_outputs.first().unwrap_or(initial_hidden);
        let stream = execution.stream();
        let hidden = if let Some(mut state) = context.vision.take() {
            let output = self
                .vision
                .finish_tensor_parallel(hidden, &mut state, tp_group, stream)?;
            context.deepstack_features = output.deepstack_features;
            let mut visual_offset = 0;
            let mut assembled = Vec::with_capacity(context.parts.len());
            for part in &context.parts {
                match part {
                    PreparedPart::Text(embedding) => assembled.push(embedding.clone()),
                    PreparedPart::Visual(len) => {
                        assembled.push(output.embeddings.try_index_device(
                            (.., visual_offset..visual_offset + *len, ..),
                            stream,
                        )?);
                        visual_offset += *len;
                    }
                }
            }
            concatenate_axis(&assembled.iter().collect::<Vec<_>>(), 1, stream)?
        } else {
            hidden.clone()
        };
        context.mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Error::UnsupportedArchitecture(
                    "Qwen3-VL layerwise execution requires an explicit causal mask".into(),
                ));
            }
            None => None,
        };
        context.visual_mask = if context.deepstack_features.is_empty() {
            None
        } else {
            Some(
                context
                    .tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &context
                            .tokens
                            .eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        Ok(hidden)
    }

    fn finish(
        &mut self,
        hidden: &Array,
        _cache: &mut Self::Cache,
        _context: &Self::ForwardContext,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let hidden = self.norm.forward(hidden, stream)?;
        Ok(common::linear::project_logits_maybe_quantized(
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
        if self.parallel_embedding.is_none() {
            return self.finish(hidden, cache, context, execution.stream());
        }
        let hidden = self.norm.forward(hidden, execution.stream())?;
        let logits = if let Some(head) = &mut self.parallel_lm_head {
            head.forward(&hidden, execution)?
        } else {
            self.parallel_embedding
                .as_mut()
                .expect("parallel embedding")
                .project_logits(&hidden, execution)?
        };
        logits.all_gather(execution)
    }
}

/// Qwen3-VL generation using shared vision/text bounded layer execution.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Qwen3VlLayerwiseModel, Cache, S>;
