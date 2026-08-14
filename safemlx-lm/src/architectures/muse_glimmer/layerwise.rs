//! Bounded layer execution for the shared Muse-Glimmer decoder.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::indexing::{NewAxis, TryIndexOp},
    ops::{concatenate_axis, zeros_dtype, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::{
    self as resident,
    vision::{VisionBlock, VisionConfig, VisionState, VisionStatic},
    DecoderConfig, Experts, FeedForward, TransformerBlock,
};
use crate::{
    api::{
        common::{
            attention::AttentionInput,
            generation::CausalLm,
            linear::{
                build_unloaded_maybe_quantized_lm_head_with_quantization,
                project_logits_maybe_quantized, unloaded_maybe_quantized_embedding,
            },
        },
        input,
    },
    error::Error,
    nn::{
        parallel::{
            planned_kv_head_layout, register_gqa_projection_group, register_linear_parameter_group,
            register_swiglu_projection_group, GqaProjectionNames, LinearParallelism,
            SwiGluProjectionNames, VocabParallelEmbedding, VocabParallelLmHead,
        },
        tensor::{create_attention_mask, AttentionMask},
    },
    runtime::cache::residency::{
        CacheResidencyPolicy, CacheResidencyReport, LayerCachePolicy, PagedCacheOptions,
        PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
        PromptCacheTopology,
    },
    runtime::cache::{
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache, SlidingKeyValueCache,
    },
    runtime::checkpoint::binding::{
        build_module_binding_plan_with_recipes, build_module_binding_plan_with_recipes_excluding,
        populate_module_from_lease, populate_module_from_lease_excluding,
    },
    runtime::checkpoint::binding_plan::{BindingPlan, PlannedBinding},
    runtime::checkpoint::store::{GgufWeightStore, TensorSelection, WeightStore},
    runtime::checkpoint::{
        quantization::{should_quantize_on_load, WeightQuantization},
        recipe::DerivedWeightRecipe,
    },
    runtime::distributed::parallel::{
        aligned_partition_units, array_parameter_member, register_replicated_module,
        MemberSharding, ParallelPlanBuilder, ParameterGroupSpec, ParameterRole,
    },
    runtime::execution::layerwise::{
        load_layerwise_model, load_layerwise_model_with_quantization,
        load_safetensors_layerwise_model, load_tensor_parallel_layerwise_model,
        open_safetensors_weight_store, ArchitectureAdapter, LayerWeightResidency,
        LayerwiseForwardState, LayerwiseModel, LoadTimeQuantizableAdapter, SharedWeightStore,
        StaticUnitBindings, WeightResidency,
    },
    runtime::residency::expert_cache::{
        ExpertCache, ExpertCacheError, ExpertCacheLoadOptions, ExpertCacheReport,
        ExpertCatalogEntry, ExpertIdentity, ExpertPass, ExpertRouteBatch,
    },
    runtime::residency::manager::{OffloadUnit, ResidencyReport, ResidentUnitLease, WeightBinding},
};

const EMBEDDING_UNIT: &str = "muse_glimmer.static.embedding";
const NORM_UNIT: &str = "muse_glimmer.static.norm";
const HEAD_UNIT: &str = "muse_glimmer.static.output";
const VISION_STATIC_UNIT: &str = "muse_glimmer.static.vision";
const VISION_PATCH_WEIGHT: &str = "vision_tower.patch_embedder.patch_embedding.weight";

fn vision_static_bindings(
    vision: &VisionStatic,
    store: &dyn WeightStore,
) -> Result<Vec<WeightBinding>, Error> {
    let checkpoint_key = format!("model.{VISION_PATCH_WEIGHT}");
    let mut recipes = BTreeMap::new();
    if store
        .metadata(&checkpoint_key)
        .is_ok_and(|metadata| metadata.shape.len() == 4)
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

/// Architecture-owned KV cache accepted by the canonical Muse-Glimmer adapter.
pub enum MuseGlimmerLayerwiseCache {
    /// Append-only device KV caches.
    Concat(Vec<Option<ConcatKeyValueCache>>),
    /// Sliding device KV caches used by expert-parallel execution.
    Sliding(Vec<Option<SlidingKeyValueCache>>),
    /// Paged KV caches used by expert-parallel execution.
    Paged(Vec<Option<PagedKeyValueCache>>),
}

/// Host-backed Muse-Glimmer causal LM.
pub struct LayerwiseDecoder {
    execution: LayerwiseModel<MuseGlimmerLayerwiseAdapter>,
}

impl LayerwiseDecoder {
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
                crate::runtime::execution::inspection::ActivationObserverProxy(&mut observer);
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
            CacheResidencyPolicy::Device => Ok(MuseGlimmerLayerwiseCache::Concat(self.new_cache())),
            CacheResidencyPolicy::Paged(options) => {
                let manager = crate::CacheResidencyManager::new(options)
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let caches = resident::new_paged_cache_with_manager(
                    self.args(),
                    manager,
                    self.execution.prompt_cache_rank_identity(),
                )?;
                Ok(MuseGlimmerLayerwiseCache::Paged(caches))
            }
        }
    }

    /// Returns aggregate live KV paging observations, if paging is enabled.
    pub fn cache_residency_report(
        &self,
        cache: &MuseGlimmerLayerwiseCache,
    ) -> Result<Option<CacheResidencyReport>, Error> {
        match cache {
            MuseGlimmerLayerwiseCache::Paged(caches) => caches
                .iter()
                .flatten()
                .next()
                .map(PagedKeyValueCache::report)
                .transpose()
                .map_err(Into::into),
            MuseGlimmerLayerwiseCache::Concat(_) | MuseGlimmerLayerwiseCache::Sliding(_) => {
                Ok(None)
            }
        }
    }

    /// Returns rank-local generalized parallel information when applicable.
    pub fn parallel_info(&self) -> Option<&crate::ParallelModelInfo> {
        self.execution.parallel_info()
    }

    /// Returns generalized parameter-residency and memory metadata.
    pub fn residency_metadata(&self) -> &crate::LayerwiseModelMetadata {
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
        let mut owned = MuseGlimmerLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.execution.save_prompt_cache(
            &mut owned,
            destination,
            descriptor,
            prefix_token_ids,
            options,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Concat(owned) = owned else {
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
        let MuseGlimmerLayerwiseCache::Concat(cache) = cache else {
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
    ) -> Result<Option<crate::runtime::execution::layerwise::DenseDiskStreamReport>, Error> {
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
    pub fn checkpoint_store(&self) -> &(dyn WeightStore + Send + Sync) {
        self.execution.checkpoint_store()
    }

    /// Runs a rank-local tensor-parallel forward pass through the generalized
    /// execution-group engine.
    pub fn forward_tensor_parallel(
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

    /// Runs Qwen2/Qwen2.5 or Qwen3 with a standard KV cache.
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let mut owned = MuseGlimmerLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.execution.forward(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Concat(owned) = owned else {
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
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned = MuseGlimmerLayerwiseCache::Concat(std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let MuseGlimmerLayerwiseCache::Concat(owned) = owned else {
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
        observer: &mut dyn crate::runtime::execution::inspection::ActivationObserver,
    ) -> Result<Array, Error> {
        let mut owned = MuseGlimmerLayerwiseCache::Paged(std::mem::take(cache));
        let result = self.execution.forward_with_observer(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
            observer,
        );
        let MuseGlimmerLayerwiseCache::Paged(owned) = owned else {
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
        let mut owned = MuseGlimmerLayerwiseCache::Paged(std::mem::take(cache));
        let result = self.execution.forward(
            MuseGlimmerAdapterInput::Decode { inputs, mask },
            &mut owned,
            stream,
        );
        let MuseGlimmerLayerwiseCache::Paged(owned) = owned else {
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

impl CausalLm<Vec<Option<ConcatKeyValueCache>>> for LayerwiseDecoder {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<ConcatKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut owned = MuseGlimmerLayerwiseCache::Concat(std::mem::take(cache));
        let result = self
            .execution
            .forward(MuseGlimmerAdapterInput::Prefill(input), &mut owned, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream);
        let MuseGlimmerLayerwiseCache::Concat(owned) = owned else {
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

impl CausalLm<Vec<Option<PagedKeyValueCache>>> for LayerwiseDecoder {
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<PagedKeyValueCache>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mut owned = MuseGlimmerLayerwiseCache::Paged(std::mem::take(cache));
        let result = self
            .execution
            .forward(MuseGlimmerAdapterInput::Prefill(input), &mut owned, stream)
            .map_err(|error| Exception::custom(error.to_string()))?
            .try_index_device((.., -1, ..), stream);
        let MuseGlimmerLayerwiseCache::Paged(owned) = owned else {
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

/// Loads Qwen2/Qwen2.5 or Qwen3 through the generalized residency engine.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    let args = resident::load_config(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let adapter = MuseGlimmerLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_safetensors_layerwise_model(
            model_dir,
            adapter,
            options,
            stream,
            weights_stream,
        )?,
    })
}

pub(crate) fn load_safetensors_quantized_residency(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let args = resident::load_config(model_dir)?;
    let store = open_safetensors_weight_store(model_dir, options.max_mapped_shards())?;
    if !should_quantize_on_load(
        "dense Qwen residency",
        args.weight_quantization(),
        quantization,
    )? {
        return Ok(LayerwiseDecoder {
            execution: load_layerwise_model(
                store,
                MuseGlimmerLayerwiseAdapter::new(args, stream)?,
                options,
                stream,
                weights_stream,
            )?,
        });
    }
    let source_adapter = MuseGlimmerLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
        execution: load_layerwise_model_with_quantization(
            store,
            source_adapter,
            options,
            Some(quantization),
            stream,
            weights_stream,
        )?,
    })
}

/// Loads Muse-Glimmer checkpoints through the generalized
/// tensor-parallel execution-group engine.
pub fn load_tensor_parallel_model(
    model_dir: impl AsRef<Path>,
    options: impl Into<LayerWeightResidency>,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LayerwiseDecoder, Error> {
    let model_dir = model_dir.as_ref();
    let options = options.into();
    let residency = options.weight_residency();
    if model_dir
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(model_dir)?;
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
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
    crate::api::structural::validate_safetensors_load_path(
        args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::default().with_weight_residency(residency),
    )?;
    let adapter = MuseGlimmerLayerwiseAdapter::new(args, stream)?;
    Ok(LayerwiseDecoder {
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

pub(crate) fn load_gguf_tensor_parallel_model(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    options: LayerWeightResidency,
    build: crate::runtime::distributed::parallel::ParallelBuildContext,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(LayerwiseDecoder, Vec<u32>), Error> {
    crate::runtime::execution::layerwise::validate_gguf_layerwise_source(
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
    let store = muse_gguf_store(checkpoint, mmproj.as_ref(), options.max_mapped_shards())?;
    let execution = load_tensor_parallel_layerwise_model(
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
    let store = muse_gguf_store(checkpoint, mmproj.as_ref(), residency.max_mapped_shards())?;

    if let Some(expert_options) = residency.expert_cache() {
        let _ = expert_options;
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer is dense and does not support sparse expert-cache residency".into(),
        ));
    }
    let execution = load_layerwise_model_with_quantization(
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
    crate::api::structural::validate_muse_glimmer_projector_gguf(
        checkpoint,
        metadata,
        &mmproj.checkpoint,
        &mmproj.metadata,
    )
    .into_loader_result()?;
    let mut vision = VisionConfig::from_gguf_metadata(&mmproj.metadata, args.hidden_size)?;
    let configs = crate::runtime::checkpoint::load::gguf_quantization_configs(
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
    max_mapped_shards: usize,
) -> Result<Arc<dyn WeightStore + Send + Sync>, Error> {
    let mut builder = GgufWeightStore::builder()
        .max_cached_readers(max_mapped_shards)?
        .add_checkpoint(checkpoint.clone(), |name| {
            resident::translate_gguf_weight_name(name, false)
        })?;
    if let Some(mmproj) = mmproj {
        builder = builder.add_checkpoint(
            mmproj.checkpoint.clone(),
            resident::translate_mmproj_store_weight_name,
        )?;
    }
    Ok(Arc::new(builder.build()?))
}

pub(crate) fn prepare_gguf_pipeline_source(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    max_mapped_shards: usize,
) -> Result<(DecoderConfig, SharedWeightStore), Error> {
    let mmproj = open_mmproj_for_checkpoint(checkpoint)?;
    let (mut args, _) =
        resident::prepare_gguf_checkpoint(checkpoint, metadata, "muse-glimmer", false)?;
    apply_mmproj_config(checkpoint, metadata, &mut args, mmproj.as_ref())?;
    let store = muse_gguf_store(checkpoint, mmproj.as_ref(), max_mapped_shards)?;
    Ok((args, store))
}

/// Loads sparse Qwen3 with independently cached experts and bounded non-expert units.
pub fn load_qwen3_expert_cache_model(
    model_dir: impl AsRef<Path>,
    non_expert: crate::NonExpertWeightResidency,
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
    let mut execution = load_layerwise_model_with_quantization(
        store,
        source_adapter,
        non_expert,
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
        MuseGlimmerLayerwiseCache::Concat(
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
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
        let entries = crate::architectures::qwen::vl::vision::grid_thw_from_array(&grids, stream)?;
        let patches = entries.iter().map(|(t, h, w)| t * h * w).sum::<i32>();
        let state = vision.continuation_state(&grids, stream)?;
        let hidden = zeros_dtype(
            &[patches, vision.config.hidden_size],
            Dtype::Float32,
            stream,
        )?;
        Ok(MuseGlimmerPipelineIngressState {
            cache: self.pipeline_cache(),
            forward: LayerwiseForwardState {
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
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
    forward: LayerwiseForwardState<MuseGlimmerForwardContext>,
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

impl LoadTimeQuantizableAdapter for MuseGlimmerLayerwiseAdapter {
    fn load_time_quantized(
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

impl ArchitectureAdapter for MuseGlimmerLayerwiseAdapter {
    type Input<'a> = MuseGlimmerAdapterInput<'a>;
    type Cache = MuseGlimmerLayerwiseCache;
    type Layer = MuseGlimmerLayer;
    type ForwardContext = MuseGlimmerForwardContext;

    fn model_type(&self) -> &str {
        &self.args.model_type
    }

    fn quantization(&self) -> Option<crate::runtime::checkpoint::quantization::WeightQuantization> {
        self.args.quantization.or(self.args.quantization_config)
    }

    fn quantizes_static_binding(&self, binding: &WeightBinding) -> bool {
        let target = binding.logical_target().unwrap_or(binding.checkpoint_key());
        quantizes_static_target(self.args.vision_config.as_ref(), target)
    }

    fn prompt_cache_model_identity(
        &self,
        topology: Option<crate::ParallelTopology>,
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
                PromptCacheTopology::for_parallel_topology,
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
        match cache {
            MuseGlimmerLayerwiseCache::Concat(cache) => resident::save_prompt_cache(
                &self.args,
                cache,
                destination,
                descriptor,
                prefix_token_ids,
                options,
                stream,
            )
            .map_err(Into::into),
            MuseGlimmerLayerwiseCache::Paged(caches) => {
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
            MuseGlimmerLayerwiseCache::Sliding(_) => Err(Error::Parallel(
                "Muse-Glimmer sliding-cache persistence is unsupported; use concat or paged cache state".into(),
            )),
        }
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
        let (cache, manifest) = resident::load_prompt_cache_with_identity(
            &self.args,
            directory,
            expected,
            prefix_token_ids,
            identity,
            stream,
        )?;
        Ok((MuseGlimmerLayerwiseCache::Concat(cache), manifest))
    }

    fn validate_cache(&self, cache: &mut Self::Cache) -> Result<(), Error> {
        let expected = usize::try_from(self.args.num_hidden_layers).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer layer count {} is invalid",
                self.args.num_hidden_layers
            ))
        })?;
        match cache {
            MuseGlimmerLayerwiseCache::Concat(caches) => {
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
            MuseGlimmerLayerwiseCache::Sliding(caches) => {
                validate_muse_glimmer_cache(caches, expected)
            }
            MuseGlimmerLayerwiseCache::Paged(caches) => {
                validate_muse_glimmer_cache(caches, expected)
            }
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
        match input {
            MuseGlimmerAdapterInput::Decode { inputs, mask } => {
                let hidden = self.embedding.forward(inputs, stream)?;
                let hidden =
                    resident::rms_norm_without_scale(&hidden, self.args.rms_norm_eps, stream)?;
                Ok(LayerwiseForwardState {
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

    fn begin_forward_with_execution<'a>(
        &mut self,
        input: Self::Input<'a>,
        _cache: &mut Self::Cache,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
    ) -> Result<LayerwiseForwardState<Self::ForwardContext>, Error> {
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
                Ok(LayerwiseForwardState {
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

    fn execution_graph(
        &self,
    ) -> Result<crate::runtime::execution::layerwise::ExecutionGroupDag, Error> {
        crate::runtime::execution::layerwise::ExecutionGroupDag::chain([
            "vision_encoder",
            "text_decoder",
        ])
    }

    fn should_execute_group(&self, group: usize, context: &Self::ForwardContext) -> bool {
        group != 0 || context.vision.is_some()
    }

    fn layer_count(&self, group: usize) -> Result<usize, Error> {
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

    fn static_units(&self, store: &dyn WeightStore) -> Result<Vec<StaticUnitBindings>, Error> {
        self.selected_static_units(store, &|_| true)
    }

    fn selected_static_units(
        &self,
        store: &dyn WeightStore,
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

    fn populate_static(&mut self, leases: &[ResidentUnitLease]) -> Result<(), Error> {
        let expected =
            usize::from(self.vision.is_some()) + if self.lm_head.is_some() { 3 } else { 2 };
        if leases.len() != expected {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer adapter received {} static leases, expected {expected}",
                leases.len()
            )));
        }
        let mut cursor = 0;
        if let Some(vision) = &mut self.vision {
            populate_module_from_lease(vision, &leases[cursor])?;
            cursor += 1;
        }
        if let Some(embedding) = &mut self.parallel_embedding {
            populate_module_from_lease(embedding.inner_mut(), &leases[cursor])?;
        } else {
            populate_module_from_lease(&mut self.embedding, &leases[cursor])?;
        }
        cursor += 1;
        populate_module_from_lease(&mut self.norm, &leases[cursor])?;
        cursor += 1;
        if let Some(head) = &mut self.parallel_lm_head {
            populate_module_from_lease(head.inner_mut(), &leases[cursor])?;
        } else if let Some(head) = &mut self.lm_head {
            populate_module_from_lease(head, &leases[cursor])?;
        }
        Ok(())
    }

    fn new_layer(&self, group: usize, index: usize, stream: &Stream) -> Result<Self::Layer, Error> {
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

    fn register_parallel_parameters(
        &self,
        _context: crate::runtime::distributed::parallel::ParallelBuildContext,
        planner: &mut crate::runtime::distributed::parallel::ParallelPlanBuilder,
        stream: &Stream,
    ) -> Result<(), Error> {
        let root = self.language_model_root();
        planner.register(crate::nn::parallel::vocab_embedding_parameter_group(
            &self.embedding,
            &format!("{root}.embed_tokens"),
            self.args.vocab_size as usize,
            self.args.hidden_size,
            false,
        )?)?;
        crate::nn::parallel::register_replicated_parameter_group(
            planner,
            &self.norm,
            &format!("{root}.norm"),
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

    fn configure_parallel_static(
        &mut self,
        context: crate::runtime::distributed::parallel::ParallelBuildContext,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
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

    fn new_parallel_layer(
        &self,
        group: usize,
        index: usize,
        layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        stream: &Stream,
    ) -> Result<Self::Layer, Error> {
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

    fn new_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(
            "Muse-Glimmer is dense and does not support expert parallelism".into(),
        ))
    }

    fn new_tensor_expert_parallel_layer(
        &self,
        _group: usize,
        _index: usize,
        _layout: &crate::runtime::distributed::parallel::LocalModelLayout,
        _assignment: &crate::runtime::distributed::expert::ExpertAssignment,
        _stream: &Stream,
    ) -> Result<Self::Layer, Error> {
        Err(Error::Parallel(
            "Muse-Glimmer is dense and does not support TP+EP".into(),
        ))
    }

    fn expert_parallel_assignment(
        &self,
        topology: crate::runtime::distributed::topology::ParallelTopology,
    ) -> Result<Option<crate::runtime::distributed::expert::ExpertAssignment>, Error> {
        if topology.expert_parallel_size == 1 && !self.sparse_expert_cache {
            return Ok(None);
        }
        if !self.args.is_moe() {
            return Err(Error::Parallel(
                "dense Qwen has no routed experts for expert-parallel ownership".into(),
            ));
        }
        Ok(Some(
            crate::runtime::distributed::expert::ExpertAssignment::balanced(
                self.args.num_experts as usize,
                topology.expert_parallel_size,
                topology.expert_parallel_rank,
            )?,
        ))
    }

    fn layer_checkpoint_prefix(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("model.vision_tower.layers.{index}")
        } else {
            format!("{}.layers.{index}", self.language_model_root())
        }
    }

    fn layer_unit_name(&self, group: usize, index: usize) -> String {
        if group == 0 {
            format!("muse_glimmer.vision.{index:05}")
        } else {
            format!("muse_glimmer.text.{index:05}")
        }
    }

    fn populate_layer(
        &self,
        group: usize,
        _index: usize,
        layer: &mut Self::Layer,
        lease: &ResidentUnitLease,
    ) -> Result<(), Error> {
        if group == 1 && self.sparse_expert_cache {
            Ok(populate_module_from_lease_excluding(
                layer,
                lease,
                |name| name.starts_with("mlp.experts."),
            )?)
        } else {
            Ok(populate_module_from_lease(layer, lease)?)
        }
    }

    fn layer_bindings(
        &self,
        group: usize,
        index: usize,
        layer: &Self::Layer,
        store: &dyn WeightStore,
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

    fn expert_parallel_layer_bindings(
        &self,
        group: usize,
        index: usize,
        _layer: &Self::Layer,
        store: &dyn WeightStore,
        assignment: &crate::runtime::distributed::expert::ExpertAssignment,
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

    fn additional_consumed_checkpoint_keys(&self, store: &dyn WeightStore) -> Vec<String> {
        if self.sparse_expert_cache {
            store
                .keys()
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
            MuseGlimmerLayerwiseCache::Concat(caches) => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index]
                    .as_mut()
                    .expect("validated Muse-Glimmer cache"),
                context,
                stream,
            ),
            MuseGlimmerLayerwiseCache::Sliding(caches) => self.forward_cached_layer(
                index,
                layer,
                hidden,
                caches[index]
                    .as_mut()
                    .expect("validated Muse-Glimmer cache"),
                context,
                stream,
            ),
            MuseGlimmerLayerwiseCache::Paged(caches) => self.forward_cached_layer(
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
            MuseGlimmerLayerwiseCache::Concat(caches) => layer.forward_with_observer(
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
            MuseGlimmerLayerwiseCache::Sliding(caches) => layer.forward_with_observer(
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
            MuseGlimmerLayerwiseCache::Paged(caches) => layer.forward_with_observer(
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
            MuseGlimmerLayerwiseCache::Concat(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
            MuseGlimmerLayerwiseCache::Sliding(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
            MuseGlimmerLayerwiseCache::Paged(caches) => Ok(layer.forward_tensor_parallel(
                hidden,
                context.mask.as_ref(),
                caches[index].as_mut(),
                tp_group,
                execution.stream(),
            )?),
        }
    }

    fn retained_arrays<'a>(
        &self,
        cache: &'a Self::Cache,
        group: usize,
        index: usize,
    ) -> Vec<&'a Array> {
        if group == 0 {
            return Vec::new();
        }
        match cache {
            MuseGlimmerLayerwiseCache::Concat(caches) => retained_cache_arrays(caches, index),
            MuseGlimmerLayerwiseCache::Sliding(caches) => retained_cache_arrays(caches, index),
            MuseGlimmerLayerwiseCache::Paged(caches) => retained_cache_arrays(caches, index),
        }
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
            MuseGlimmerLayerwiseCache::Concat(caches) => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
            MuseGlimmerLayerwiseCache::Sliding(caches) => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
            MuseGlimmerLayerwiseCache::Paged(caches) => muse_glimmer_attention_mask(
                &hidden,
                context.requested_mask.as_ref(),
                caches,
                stream,
            )?,
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

    fn finish_with_execution(
        &mut self,
        hidden: &Array,
        cache: &mut Self::Cache,
        context: &Self::ForwardContext,
        execution: &crate::runtime::distributed::parallel::ParallelExecutionContext<'_>,
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
        execution: Option<&crate::runtime::distributed::parallel::ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<LayerwiseForwardState<MuseGlimmerForwardContext>, Error> {
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
                        crate::architectures::qwen::vl::vision::grid_thw_from_array(grid, stream)?;
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
            return Ok(LayerwiseForwardState {
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
        Ok(LayerwiseForwardState {
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
    store: &dyn WeightStore,
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
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
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
    store: &dyn WeightStore,
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
    store: &dyn WeightStore,
    layer_root: &str,
    layout: Option<&crate::runtime::distributed::parallel::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
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
                Some(layout) => crate::runtime::execution::layerwise::shard_layer_bindings(
                    bindings, &prefix, store, layout,
                )?,
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
    store: &dyn WeightStore,
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
mod tests {
    use serde_json::json;

    use super::*;
    use crate::runtime::checkpoint::{quantization::AffineQuantization, store::MemoryWeightStore};

    fn tiny_vision_config() -> VisionConfig {
        VisionConfig::from_hf_value(
            &json!({
                "model_type": "muse_glimmer_vision",
                "hidden_act": "gelu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "patch_size": 2,
                "patch_temporal": 1,
                "merge_size": 2,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "max_position_embeddings": 4,
                "layer_norm_eps": 1e-5,
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }),
            24,
        )
        .unwrap()
    }

    fn vision_store(vision: &VisionStatic, patch_weight: Array) -> MemoryWeightStore {
        let mut arrays = vision
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| (format!("model.{name}"), value.clone()))
            .collect::<BTreeMap<_, _>>();
        assert!(arrays
            .insert(format!("model.{VISION_PATCH_WEIGHT}"), patch_weight)
            .is_some());
        MemoryWeightStore::new(arrays).unwrap()
    }

    #[test]
    fn dflash_fuses_raw_target_residuals_in_configured_order() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let first = Array::from_slice(&[3.0_f32, 4.0], &[1, 1, 2]);
        let second = Array::from_slice(&[12.0_f32, 5.0], &[1, 1, 2]);
        let mut captured = BTreeMap::from([(1, first), (13, second)]);

        let actual = concatenate_dflash_target_states(&mut captured, &[13, 1], stream).unwrap();
        let expected = Array::from_slice(&[12.0_f32, 5.0, 3.0, 4.0], &[1, 1, 4]);

        assert!(actual
            .all_close(&expected, Some(0.0), Some(0.0), None, stream)
            .unwrap()
            .item::<bool>(stream));
        assert!(captured.is_empty());
    }

    #[test]
    fn vision_static_binding_flattens_collapsed_gguf_patch_kernel() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let vision = VisionStatic::new(tiny_vision_config(), 8, stream).unwrap();

        let flat = zeros_dtype(&[4, 12], Dtype::Float32, stream).unwrap();
        let flat_store = vision_store(&vision, flat);
        let flat_bindings = vision_static_bindings(&vision, &flat_store).unwrap();
        assert!(flat_bindings
            .iter()
            .find(|binding| binding.name() == VISION_PATCH_WEIGHT)
            .unwrap()
            .recipe()
            .is_none());

        let collapsed = zeros_dtype(&[4, 3, 2, 2], Dtype::Float32, stream).unwrap();
        let collapsed_store = vision_store(&vision, collapsed);
        let collapsed_bindings = vision_static_bindings(&vision, &collapsed_store).unwrap();
        let binding = collapsed_bindings
            .iter()
            .find(|binding| binding.name() == VISION_PATCH_WEIGHT)
            .unwrap();
        assert!(matches!(
            binding.recipe(),
            Some(DerivedWeightRecipe::Reshape { shape, .. }) if shape == &[4, 12]
        ));
        assert_eq!(
            binding
                .source_recipe()
                .infer(&collapsed_store)
                .unwrap()
                .shape(),
            &[4, 12]
        );
    }

    #[test]
    fn load_time_quantization_skips_unaligned_static_vision_weights() {
        let mut vision = VisionConfig::from_hf_value(
            &json!({
                "model_type": "muse_glimmer_vision",
                "hidden_act": "gelu",
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 32,
                "pos_emb_width": 32,
                "max_position_embeddings": 1024,
                "layer_norm_eps": 1e-5,
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }),
            6656,
        )
        .unwrap();
        vision.apply_load_time_quantization(WeightQuantization::Affine(
            AffineQuantization::new(64, 4).unwrap(),
        ));

        assert!(!quantizes_static_target(
            Some(&vision),
            "model.vision_tower.patch_embedder.patch_embedding.weight",
        ));
        assert!(quantizes_static_target(
            Some(&vision),
            "model.vision_adapter.fc1.weight",
        ));
        assert!(quantizes_static_target(
            Some(&vision),
            "model.vision_projection.weight",
        ));
        assert!(!quantizes_static_target(
            Some(&vision),
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
        ));
        assert!(quantizes_static_target(
            Some(&vision),
            "model.language_model.embed_tokens.weight",
        ));
    }
}
