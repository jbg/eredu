//! Qwen3-VL conditional-generation model support.

use eredu_runtime::{CausalModel, RuntimeState, StateError, StateLayout};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{
        concatenate_axis,
        indexing::{masked_scatter, TryIndexOp},
        zeros_dtype, GgufCheckpoint, GgufMetadataArray, GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Stream,
};
use serde_json::Value;

use super::vision::VisionConfigSource;
pub use super::vision::{
    QwenVisionTransformer, VisionAttentionPolicy, VisionConfig, VisionLayerPolicy,
};

use crate::core::cache::{
    derive_prompt_cache_architecture_fingerprint, PromptCacheDescriptor, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::{
        self as common,
        attention::AttentionInput,
        tensor::{create_attention_mask, AttentionMask},
    },
    backend::mlx::runtime::cache::{
        residency::{
            open_prompt_cache_snapshot, save_prompt_cache_snapshot, CacheBlockArrays,
            PromptCacheSnapshotBlock, PromptCacheStateArray,
        },
        ConcatKeyValueCache, KeyValueCache,
    },
    backend::mlx::runtime::checkpoint::load::gguf_quantization_configs,
    backend::mlx::runtime::media::input as runtime_input,
    composition::mlx_architectures::qwen::{dense as dense_qwen, vl::vision::grid_thw_from_array},
    core::attention::LayerSchedule,
    core::cache::{
        LayerCachePolicy, StateTensorDimension, StateTensorDtype, StateTensorOwner,
        StateTensorPolicy, StateTensorRole,
    },
};

#[derive(Debug, Clone)]
/// Parsed Qwen3-VL configuration.
pub struct ModelArgs {
    /// Text decoder configuration shared with Qwen3.
    pub text_config: dense_qwen::DecoderConfig,
    /// Vision encoder configuration shared across Qwen VL models.
    pub vision_config: VisionConfig,
    /// Placeholder token used for image embeddings.
    pub image_token_id: u32,
    /// Placeholder token used for video embeddings.
    pub video_token_id: u32,
    /// Interleaved temporal/height/width RoPE sections.
    pub mrope_section: Vec<i32>,
}

fn parse_model_args_value(mut value: Value) -> Result<ModelArgs, Error> {
    let object = value.as_object_mut().ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3_vl config must be a JSON object".into())
    })?;
    let model_type = object
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>")
        .to_string();
    if !matches!(model_type.as_str(), "qwen3_vl" | "qwen3_vl_moe") {
        return Err(Error::UnsupportedModelType(model_type));
    }
    let image_token_id = object
        .get("image_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing image_token_id".into())
        })?;
    let video_token_id = object
        .get("video_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing video_token_id".into())
        })?;
    let vision_config = serde_json::from_value::<VisionConfigSource>(
        object.get("vision_config").cloned().ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3_vl config is missing vision_config".into())
        })?,
    )?
    .normalize_qwen3_vl()?;
    let top_level_quantization = object.get("quantization").cloned();
    let top_level_quantization_config = object.get("quantization_config").cloned();
    let mut text_value = object.get("text_config").cloned().ok_or_else(|| {
        Error::UnsupportedArchitecture("qwen3_vl config is missing text_config".into())
    })?;
    let text_object = text_value.as_object_mut().ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("{model_type} text_config must be a JSON object"))
    })?;
    if !text_object.contains_key("tie_word_embeddings") {
        if let Some(tie_word_embeddings) = object.get("tie_word_embeddings").cloned() {
            text_object.insert("tie_word_embeddings".into(), tie_word_embeddings);
        }
    }
    if let Some(quantization) = top_level_quantization {
        text_object.insert("quantization".into(), quantization);
    }
    if let Some(quantization) = top_level_quantization_config {
        text_object.insert("quantization_config".into(), quantization);
    }
    let rope = text_object
        .get_mut("rope_scaling")
        .and_then(Value::as_object_mut);
    let mrope_section = rope
        .as_ref()
        .and_then(|rope| rope.get("mrope_section"))
        .and_then(Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_i64().and_then(|value| i32::try_from(value).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_else(|| vec![24, 20, 20]);
    if let Some(rope) = rope {
        rope.remove("mrope_section");
        rope.remove("mrope_interleaved");
    }
    let normalized_text_model_type = if model_type == "qwen3_vl_moe" {
        "qwen3_vl_moe_text"
    } else {
        "qwen3_vl_text"
    };
    let text_config =
        dense_qwen::qwen3_text_config_from_hf_value(&text_value, normalized_text_model_type)?;
    if model_type == "qwen3_vl_moe" && text_config.num_experts <= 0 {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_vl_moe text_config must define routed experts".into(),
        ));
    }
    if mrope_section.len() != 3 || mrope_section.iter().any(|&section| section < 0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl mrope_section must contain three non-negative values, got {mrope_section:?}"
        )));
    }
    if mrope_section.iter().sum::<i32>() != text_config.head_dim / 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl mrope_section {mrope_section:?} does not cover half of head_dim {}",
            text_config.head_dim
        )));
    }
    let args = ModelArgs {
        text_config,
        vision_config,
        image_token_id,
        video_token_id,
        mrope_section,
    };
    validate_qwen3_vl_model_args(&args)?;
    Ok(args)
}

pub(crate) fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    parse_model_args_value(config.clone())
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "qwen3_vl",
        [
            (
                "text",
                dense_qwen::prompt_cache_architecture_fingerprint(&args.text_config),
            ),
            ("image_token", args.image_token_id.to_string()),
            ("video_token", args.video_token_id.to_string()),
            ("mrope_section", format!("{:?}", args.mrope_section)),
        ],
    )
}

#[cfg(test)]
pub(crate) fn prompt_cache_layer_layout(
    args: &ModelArgs,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let kv_heads =
        vec![args.text_config.num_key_value_heads; args.text_config.num_hidden_layers as usize];
    prompt_cache_layer_layout_with_kv_heads(args, &kv_heads)
}

pub(crate) fn prompt_cache_layer_layout_with_kv_heads(
    args: &ModelArgs,
    kv_heads: &[i32],
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let cache_error = |error: crate::core::cache::CachePolicyError| {
        Error::UnsupportedArchitecture(error.to_string())
    };
    let layers = args.text_config.num_hidden_layers as usize;
    if kv_heads.len() != layers {
        return Err(Error::Parallel(format!(
            "Qwen3-VL cache geometry has {} layers, expected {layers}",
            kv_heads.len()
        )));
    }
    let policies = (0..layers)
        .map(|layer| {
            let attention = *args
                .text_config
                .attention_schedule
                .get(layer)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "Qwen3-VL text attention schedule has no layer {layer}"
                    ))
                })?;
            if layer == 0 {
                LayerCachePolicy::key_value_with_fixed_state(
                    attention,
                    kv_heads[layer],
                    args.text_config.head_dim,
                    vec![StateTensorPolicy::new(
                        StateTensorRole::PositionDelta,
                        vec![StateTensorDimension::Scalar],
                        StateTensorDtype::Int32,
                        crate::MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(cache_error)?],
                )
                .map_err(cache_error)
            } else {
                LayerCachePolicy::key_value(attention, kv_heads[layer], args.text_config.head_dim)
                    .map_err(cache_error)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

#[cfg(test)]
pub(crate) fn state_layout(args: &ModelArgs) -> Result<StateLayout, Error> {
    StateLayout::new(prompt_cache_layer_layout_with_kv_heads(
        args,
        &vec![args.text_config.num_key_value_heads; args.text_config.num_hidden_layers as usize],
    )?)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

pub(crate) fn state_layout_with_kv_heads(
    args: &ModelArgs,
    kv_heads: &[i32],
) -> Result<StateLayout, Error> {
    StateLayout::new(prompt_cache_layer_layout_with_kv_heads(args, kv_heads)?)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn validate_qwen3_vl_model_args(args: &ModelArgs) -> Result<(), Error> {
    let vision = &args.vision_config;
    if vision.hidden_size <= 0
        || vision.intermediate_size <= 0
        || vision.num_heads <= 0
        || vision.num_position_embeddings <= 0
        || vision.in_channels <= 0
        || vision.patch_size <= 0
        || vision.spatial_merge_size <= 0
        || vision.temporal_patch_size <= 0
        || vision.window_size <= 0
        || vision.out_hidden_size <= 0
    {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_vl vision geometry must be positive".into(),
        ));
    }
    if vision.hidden_size % vision.num_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl vision hidden size {} is not divisible by {} attention heads",
            vision.hidden_size, vision.num_heads
        )));
    }
    if vision.out_hidden_size != args.text_config.hidden_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3_vl vision output size {} does not match text hidden size {}",
            vision.out_hidden_size, args.text_config.hidden_size
        )));
    }
    Ok(())
}

/// Reads Qwen3-VL arguments from a Hugging Face model directory.
pub fn get_qwen3_vl_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    parse_model_args_value(serde_json::from_reader(std::fs::File::open(
        model_dir.as_ref().join("config.json"),
    )?)?)
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3-VL vision encoder and Qwen3 language decoder.
pub struct Qwen3VLModel {
    #[param]
    /// Vision tower.
    pub visual: QwenVisionTransformer,
    #[param]
    /// Qwen3-compatible language model body.
    pub language_model: dense_qwen::Decoder,
}

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3-VL conditional-generation model.
pub struct Model {
    /// Parsed model configuration.
    pub args: ModelArgs,
    #[param]
    /// Model body matching the public checkpoint parameter tree.
    pub model: Qwen3VLModel,
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

/// Generation state for Qwen3-VL, including multimodal RoPE offset state.
#[derive(Debug, Clone)]
pub struct Cache {
    layout: StateLayout,
    /// Per-layer key/value caches.
    pub kv: Vec<Option<ConcatKeyValueCache>>,
    pub(crate) rope_delta: i32,
}

impl Cache {
    pub(crate) fn new(args: &ModelArgs) -> Self {
        Self::new_with_kv_heads(
            args,
            &vec![
                args.text_config.num_key_value_heads;
                args.text_config.num_hidden_layers as usize
            ],
        )
        .expect("validated Qwen3-VL state geometry")
    }

    pub(crate) fn new_with_kv_heads(args: &ModelArgs, kv_heads: &[i32]) -> Result<Self, Error> {
        Ok(Self {
            layout: state_layout_with_kv_heads(args, kv_heads)?,
            kv: (0..args.text_config.num_hidden_layers)
                .map(|_| Some(ConcatKeyValueCache::default()))
                .collect(),
            rope_delta: 0,
        })
    }

    pub(crate) fn validate(&self, args: &ModelArgs) -> Result<(), Error> {
        if self.kv.len() != args.text_config.num_hidden_layers as usize {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen3-VL cache has {} layers, expected {}",
                self.kv.len(),
                args.text_config.num_hidden_layers
            )));
        }
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.kv
            .iter_mut()
            .flatten()
            .for_each(ConcatKeyValueCache::clear);
        self.rope_delta = 0;
    }
}

impl RuntimeState<crate::backend::mlx::nn::shared::MlxBackend> for Cache {
    type RetainedValues<'a> = std::vec::IntoIter<&'a Array>;

    fn layout(&self) -> &StateLayout {
        &self.layout
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
        self.kv
            .get(address.index())
            .ok_or(StateError::UnknownLayer {
                layer: address.index(),
                count: self.kv.len(),
            })
            .map(|cache| {
                cache
                    .as_ref()
                    .map(KeyValueCache::retained_arrays)
                    .unwrap_or_default()
                    .into_iter()
            })
    }
}

struct PreparedPrefill {
    tokens: Array,
    embeddings: Array,
    position_ids: [Vec<i32>; 3],
    rope_delta: i32,
    deepstack_features: Vec<Array>,
}

impl Model {
    /// Creates an unloaded Qwen3-VL model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let visual = QwenVisionTransformer::new_deepstack(args.vision_config.clone(), stream)?;
        let language_model = dense_qwen::Decoder::new(&args.text_config, stream)?;
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
            model: Qwen3VLModel {
                visual,
                language_model,
            },
            lm_head,
        })
    }

    /// Returns the effective model type.
    pub fn model_type(&self) -> &str {
        if self.args.text_config.model_type == "qwen3_vl_moe_text" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        }
    }

    /// Creates an empty cache.
    pub fn new_cache(&self) -> Cache {
        Cache::new(&self.args)
    }

    pub(crate) fn save_prompt_cache(
        cache: &Cache,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        let end = i64::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Qwen-VL prompt length exceeds i64"))?;
        let rank = descriptor.topology.cache_rank_identity();
        let mut blocks = Vec::with_capacity(cache.kv.len());
        for (layer, cache) in cache.kv.iter().enumerate() {
            let cache = cache.as_ref().ok_or_else(|| {
                Exception::custom("Qwen-VL prompt cache is missing a decoder layer")
            })?;
            if cache.offset() as i64 != end {
                return Err(Exception::custom(
                    "Qwen-VL cache offset does not match the persisted prefix",
                ));
            }
            let (keys, values) = cache
                .snapshot_arrays(stream)?
                .ok_or_else(|| Exception::custom("Qwen-VL key/value state is missing"))?;
            blocks.push(PromptCacheSnapshotBlock {
                global_layer: layer,
                start: end - i64::from(keys.dim(-2)),
                end,
                rank,
                arrays: CacheBlockArrays::KeyValue { keys, values },
            });
        }
        let position_delta = Array::from_slice(&[cache.rope_delta], &[1]);
        let state = [PromptCacheStateArray {
            owner: StateTensorOwner::Layer(0),
            role: StateTensorRole::PositionDelta,
            array: &position_delta,
        }];
        save_prompt_cache_snapshot(
            destination,
            descriptor,
            prefix_token_ids,
            blocks,
            &state,
            options,
        )
        .map_err(|error| Exception::custom(error.to_string()))
    }

    #[cfg(test)]
    pub(crate) fn load_prompt_cache(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let layer_count = args.text_config.num_hidden_layers as usize;
        let identity = PromptCacheModelIdentity {
            model_family: "qwen3_vl".into(),
            effective_model_type: if args.text_config.model_type == "qwen3_vl_moe_text" {
                "qwen3_vl_moe".into()
            } else {
                "qwen3_vl".into()
            },
            architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
            layer_count,
            global_layer_start: 0,
            global_layer_end: layer_count,
            sink_tokens: 0,
            layer_prefix_offsets: vec![0; layer_count],
            topology: Default::default(),
            layer_layout: prompt_cache_layer_layout(args)
                .map_err(|error| Exception::custom(error.to_string()))?,
        };
        Self::load_prompt_cache_with_identity(
            args,
            directory,
            expected,
            prefix_token_ids,
            &identity,
            stream,
        )
    }

    pub(crate) fn load_prompt_cache_with_identity(
        args: &ModelArgs,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        identity: &PromptCacheModelIdentity,
        stream: &Stream,
    ) -> Result<(Cache, PromptCacheManifest), Exception> {
        let (blocks, state, manifest) =
            open_prompt_cache_snapshot(directory, expected, identity, prefix_token_ids, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        let mut blocks = blocks
            .into_iter()
            .map(|block| (block.global_layer, block))
            .collect::<BTreeMap<_, _>>();
        let mut state = state
            .into_iter()
            .map(|state| ((state.owner, state.role), state.array))
            .collect::<BTreeMap<_, _>>();
        let rope_delta = state
            .remove(&(StateTensorOwner::Layer(0), StateTensorRole::PositionDelta))
            .ok_or_else(|| Exception::custom("Qwen-VL position delta is missing"))?
            .try_item::<i32>(stream)?;
        let mut cache = Cache {
            layout: StateLayout::new(identity.layer_layout.clone())
                .map_err(|error| Exception::custom(error.to_string()))?,
            kv: (0..args.text_config.num_hidden_layers)
                .map(|layer| {
                    let window = args
                        .text_config
                        .attention_schedule
                        .get(layer as usize)
                        .and_then(|policy| policy.window())
                        .map(|window| {
                            i32::try_from(window.get())
                                .expect("validated Qwen-VL attention window fits i32")
                        });
                    Some(match window {
                        Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
                        None => ConcatKeyValueCache::new(),
                    })
                })
                .collect(),
            rope_delta,
        };
        let end = i32::try_from(prefix_token_ids.len())
            .map_err(|_| Exception::custom("Qwen-VL prompt length exceeds i32"))?;
        for (layer, cache) in cache.kv.iter_mut().enumerate() {
            let cache = cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Qwen-VL decoder cache layer is missing"))?;
            let block = blocks
                .remove(&layer)
                .ok_or_else(|| Exception::custom("Qwen-VL prompt-cache block is missing"))?;
            match block.arrays {
                CacheBlockArrays::KeyValue { keys, values } => {
                    cache.restore_resident(keys, values, end)?;
                }
                _ => return Err(Exception::custom("Qwen-VL prompt-cache kind mismatch")),
            }
        }
        if !blocks.is_empty() || !state.is_empty() {
            return Err(Exception::custom(
                "Qwen-VL prompt cache has unexpected state",
            ));
        }
        Ok((cache, manifest))
    }

    fn prepare_typed_prefill(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<PreparedPrefill, Exception> {
        let modality_tokens = [
            runtime_input::ModalityToken {
                modality: runtime_input::Modality::Image,
                token_id: self.args.image_token_id,
            },
            runtime_input::ModalityToken {
                modality: runtime_input::Modality::Video,
                token_id: self.args.video_token_id,
            },
        ];
        let deepstack_count = self.args.vision_config.deepstack_layer_count();
        let mut collected = (0..deepstack_count)
            .map(|_| Vec::new())
            .collect::<Vec<Vec<Array>>>();
        let embed_tokens = &mut self.model.language_model.embed_tokens;
        let visual = &mut self.model.visual;
        let prepared = runtime_input::prepare_decoder_prefill(
            input,
            &modality_tokens,
            self.args.text_config.hidden_size,
            "qwen3_vl",
            stream,
            |tokens, stream| embed_tokens.forward(tokens, stream),
            |part, stream| {
                let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                    Exception::custom(format!(
                        "qwen3_vl {} input requires qwen_grid_thw metadata",
                        part.modality.as_str()
                    ))
                })?;
                let tensor = match part.payload {
                    runtime_input::InputPayload::Tensor(tensor) => tensor,
                    runtime_input::InputPayload::Embeddings(_) => {
                        return Err(Exception::custom(
                            "qwen3_vl requires model-native visual tensors because DeepStack features cannot be reconstructed from final embeddings",
                        ));
                    }
                    runtime_input::InputPayload::TokenIds(_) => {
                        return Err(Exception::custom(
                            "qwen3_vl visual input does not accept token-id payloads",
                        ));
                    }
                };
                let output = visual.forward_features(tensor, grid, stream)?;
                if output.deepstack_features.len() != collected.len() {
                    return Err(Exception::custom(format!(
                        "qwen3_vl vision tower returned {} DeepStack features, expected {}",
                        output.deepstack_features.len(),
                        collected.len()
                    )));
                }
                for (layer, feature) in output.deepstack_features.into_iter().enumerate() {
                    collected[layer].push(feature);
                }
                Ok(vec![output.embeddings])
            },
        )?;
        let tokens = prepared.tokens().clone();
        let embeddings = match prepared.embeddings() {
            Some(embeddings) => embeddings.clone(),
            None => self
                .model
                .language_model
                .embed_tokens
                .forward(&tokens, stream)?,
        };
        let (position_ids, rope_delta) = multimodal_position_ids(
            input,
            self.args.vision_config.spatial_merge_size,
            tokens.dim(1),
            stream,
        )?;
        let deepstack_features = if collected
            .first()
            .is_some_and(|features| features.is_empty())
        {
            Vec::new()
        } else {
            collected
                .into_iter()
                .map(|features| concatenate_axis(&features, 1, stream))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(PreparedPrefill {
            tokens,
            embeddings,
            position_ids,
            rope_delta,
            deepstack_features,
        })
    }

    fn forward_prepared(
        &mut self,
        prepared: PreparedPrefill,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        cache.rope_delta = prepared.rope_delta;
        self.forward_embeddings(
            &prepared.tokens,
            prepared.embeddings,
            &prepared.position_ids,
            &prepared.deepstack_features,
            cache,
            stream,
        )
    }

    fn forward_embeddings(
        &mut self,
        tokens: &Array,
        mut hidden: Array,
        position_ids: &[Vec<i32>; 3],
        deepstack_features: &[Array],
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let mask = match create_attention_mask(&hidden, &cache.kv, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Some(mask),
            Some(AttentionMask::Causal) => {
                return Err(Exception::custom(
                    "qwen3_vl requires an explicit causal mask",
                ));
            }
            None => None,
        };
        if cache.kv.is_empty() {
            cache.kv = (0..self.model.language_model.layers.len())
                .map(|_| Some(ConcatKeyValueCache::default()))
                .collect();
        }
        let (cos, sin) = mrope_embeddings(
            position_ids,
            self.args.text_config.head_dim,
            self.args.text_config.rope_theta,
            &self.args.mrope_section,
        );
        let visual_mask = if deepstack_features.is_empty() {
            None
        } else {
            Some(
                tokens
                    .eq(Array::from_int(self.args.image_token_id as i32), stream)?
                    .logical_or(
                        &tokens.eq(Array::from_int(self.args.video_token_id as i32), stream)?,
                        stream,
                    )?,
            )
        };
        for (layer_index, (layer, layer_cache)) in self
            .model
            .language_model
            .layers
            .iter_mut()
            .zip(cache.kv.iter_mut())
            .enumerate()
        {
            hidden = layer.forward_with_rotary_embeddings(
                AttentionInput {
                    x: &hidden,
                    mask: mask.as_ref(),
                    cache: layer_cache.as_mut(),
                },
                &cos,
                &sin,
                stream,
            )?;
            if let Some(features) = deepstack_features.get(layer_index) {
                let base = zeros_dtype(hidden.shape(), hidden.dtype(), stream)?;
                let features = features.try_index_device((0, .., ..), stream)?;
                let aligned = masked_scatter(
                    &base,
                    visual_mask.as_ref().expect("DeepStack visual mask"),
                    features,
                    stream,
                )?;
                hidden = hidden.add(aligned, stream)?;
            }
        }
        let hidden = self.model.language_model.norm.forward(&hidden, stream)?;
        common::linear::project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.language_model.embed_tokens,
            &hidden,
            stream,
        )
    }
}

pub(crate) fn multimodal_position_ids(
    input: runtime_input::ModelInput<'_>,
    spatial_merge_size: i32,
    expected_len: i32,
    stream: &Stream,
) -> Result<([Vec<i32>; 3], i32), Exception> {
    let mut positions = [Vec::new(), Vec::new(), Vec::new()];
    let mut current = 0;
    for part in input.parts {
        match (part.modality, part.payload) {
            (runtime_input::Modality::Text, runtime_input::InputPayload::TokenIds(tokens)) => {
                for position in current..current + tokens.dim(1) {
                    for axis in &mut positions {
                        axis.push(position);
                    }
                }
                current += tokens.dim(1);
            }
            (runtime_input::Modality::Image | runtime_input::Modality::Video, _) => {
                let grid = part.metadata.qwen_grid_thw.ok_or_else(|| {
                    Exception::custom("qwen3_vl visual input requires qwen_grid_thw metadata")
                })?;
                for (t, h, w) in grid_thw_from_array(grid, stream)? {
                    let h = h / spatial_merge_size;
                    let w = w / spatial_merge_size;
                    for temporal in 0..t {
                        for height in 0..h {
                            for width in 0..w {
                                positions[0].push(current + temporal);
                                positions[1].push(current + height);
                                positions[2].push(current + width);
                            }
                        }
                    }
                    current += h.max(w);
                }
            }
            _ => {
                return Err(Exception::custom(format!(
                    "qwen3_vl does not support {} input",
                    part.modality.as_str()
                )));
            }
        }
    }
    if positions[0].len() as i32 != expected_len {
        return Err(Exception::custom(format!(
            "qwen3_vl position metadata describes {} tokens, prepared input has {expected_len}",
            positions[0].len()
        )));
    }
    let max_position = positions
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .unwrap_or(0);
    Ok((positions, max_position + 1 - expected_len))
}

pub(crate) fn mrope_embeddings(
    position_ids: &[Vec<i32>; 3],
    head_dim: i32,
    theta: f32,
    sections: &[i32],
) -> (Array, Array) {
    let (cos, sin) = mrope_values(position_ids, head_dim, theta, sections);
    let len = position_ids[0].len() as i32;
    (
        Array::from_slice(&cos, &[1, len, head_dim]),
        Array::from_slice(&sin, &[1, len, head_dim]),
    )
}

fn mrope_values(
    position_ids: &[Vec<i32>; 3],
    head_dim: i32,
    theta: f32,
    sections: &[i32],
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv_freq = (0..half)
        .map(|index| 1.0 / theta.powf(2.0 * index as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    let len = position_ids[0].len();
    let mut cos = Vec::with_capacity(len * head_dim as usize);
    let mut sin = Vec::with_capacity(len * head_dim as usize);
    for ((&temporal, &height), &width) in position_ids[0]
        .iter()
        .zip(&position_ids[1])
        .zip(&position_ids[2])
    {
        let token_positions = [temporal, height, width];
        let mut angles = Vec::with_capacity(half as usize);
        for (index, inv) in inv_freq.iter().enumerate() {
            let axis = if index % 3 == 1 && index < (sections[1] * 3) as usize {
                1
            } else if index % 3 == 2 && index < (sections[2] * 3) as usize {
                2
            } else {
                0
            };
            angles.push(token_positions[axis] as f32 * inv);
        }
        for angle in angles.iter().chain(angles.iter()) {
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

pub(crate) struct PreparedQwen3VlGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_qwen3_vl_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision_checkpoint: &GgufCheckpoint,
    vision_metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<PreparedQwen3VlGguf, Error> {
    let architecture = dense_qwen::gguf_string(metadata, "general.architecture")?;
    let is_moe = match architecture.as_str() {
        "qwen3vl" => false,
        "qwen3vlmoe" => true,
        _ => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF architecture {architecture:?}; this loader supports qwen3vl and qwen3vlmoe"
            )))
        }
    };
    crate::composition::mlx::structural::validate_qwen3_vl_projector_gguf(
        checkpoint,
        metadata,
        vision_checkpoint,
        vision_metadata,
    )
    .into_loader_result()?;
    let (mut text_config, eos_token_ids) =
        dense_qwen::prepare_gguf_checkpoint(checkpoint, metadata, &architecture, is_moe)?;
    text_config.model_type = if is_moe {
        "qwen3_vl_moe_text"
    } else {
        "qwen3_vl_text"
    }
    .into();
    let args =
        qwen3_vl_args_from_gguf_catalog(text_config, metadata, vision_checkpoint, vision_metadata)?;
    Ok(PreparedQwen3VlGguf {
        args,
        eos_token_ids,
    })
}

/// Builds the complete Qwen3-VL geometry from GGUF catalogs without reading payload bytes.
pub(crate) fn qwen3_vl_args_from_gguf_catalog(
    text_config: dense_qwen::DecoderConfig,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision_checkpoint: &GgufCheckpoint,
    vision_metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    validate_qwen3_vl_mmproj(vision_metadata)?;
    let (mrope_section, image_token_id, video_token_id) =
        validate_qwen3_vl_text_gguf_catalog(&text_config, metadata)?;
    let vision_config =
        qwen_vision_config_from_gguf_catalog(vision_checkpoint, vision_metadata, "Qwen3-VL")?;
    validate_qwen3_vl_vision_geometry(&text_config, metadata, &vision_config)?;
    Ok(ModelArgs {
        text_config,
        vision_config,
        image_token_id,
        video_token_id,
        mrope_section,
    })
}

/// Builds the shared Qwen vision geometry and checkpoint-native projection
/// formats from a llama.cpp-style `clip` projector catalog.
pub(crate) fn qwen_vision_config_from_gguf_catalog(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    family: &str,
) -> Result<VisionConfig, Error> {
    validate_qwen3_vl_mmproj(metadata)?;
    let deepstack_visual_indexes = gguf_deepstack_layers(metadata)?;
    let hidden_size = dense_qwen::gguf_i32_catalog(metadata, "clip.vision.embedding_length")?;
    let position_layout = checkpoint
        .catalog()
        .tensors()
        .find(|tensor| tensor.descriptor().name == "v.position_embd.weight")
        .and_then(|tensor| tensor.outputs().first())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "{family} mmproj is missing v.position_embd.weight"
            ))
        })?;
    if position_layout.shape.len() != 2 || position_layout.shape[1] != hidden_size as u64 {
        return Err(Error::UnsupportedArchitecture(format!(
            "unexpected {family} position embedding shape {:?}",
            position_layout.shape
        )));
    }
    let depth = dense_qwen::gguf_i32_catalog(metadata, "clip.vision.block_count")?;
    let depth = usize::try_from(depth)
        .ok()
        .filter(|depth| *depth > 0)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("{family} GGUF vision depth must be positive"))
        })?;
    let mut layer_policies = vec![
        VisionLayerPolicy {
            attention: VisionAttentionPolicy::Full,
            deepstack_merger: None,
        };
        depth
    ];
    let mut seen = std::collections::BTreeSet::new();
    for (merger, layer) in deepstack_visual_indexes.into_iter().enumerate() {
        let index = usize::try_from(layer)
            .ok()
            .filter(|index| *index < depth)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "{family} GGUF DeepStack layer {layer} is outside vision depth {depth}"
                ))
            })?;
        if !seen.insert(index) {
            return Err(Error::UnsupportedArchitecture(format!(
                "{family} GGUF DeepStack layer {layer} is duplicated"
            )));
        }
        layer_policies[index].deepstack_merger = Some(u32::try_from(merger).map_err(|_| {
            Error::UnsupportedArchitecture(format!("{family} GGUF has too many DeepStack layers"))
        })?);
    }
    let deepstack = layer_policies
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| policy.deepstack_merger.map(|order| (order, layer as i32)))
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect::<Vec<_>>();
    let quantized_weight_configs = gguf_quantization_configs(checkpoint, |name| {
        let translated = translate_qwen3_vl_mmproj_name(name, &deepstack);
        translated
            .strip_prefix("model.visual.")
            .unwrap_or(&translated)
            .to_string()
    })?;
    for format in quantized_weight_configs.values() {
        format.validate()?;
    }
    Ok(VisionConfig {
        layer_schedule: LayerSchedule::new(depth, layer_policies).map_err(|error| {
            Error::UnsupportedArchitecture(format!("{family} GGUF vision {error}"))
        })?,
        hidden_size,
        hidden_act: "gelu_pytorch_tanh".into(),
        intermediate_size: dense_qwen::gguf_i32_catalog(
            metadata,
            "clip.vision.feed_forward_length",
        )?,
        num_heads: dense_qwen::gguf_i32_catalog(metadata, "clip.vision.attention.head_count")?,
        num_position_embeddings: i32::try_from(position_layout.shape[0]).map_err(|_| {
            Error::UnsupportedArchitecture(format!("{family} position count exceeds i32"))
        })?,
        in_channels: 3,
        patch_size: dense_qwen::gguf_i32_catalog(metadata, "clip.vision.patch_size")?,
        spatial_merge_size: dense_qwen::gguf_i32_catalog(
            metadata,
            "clip.vision.spatial_merge_size",
        )?,
        temporal_patch_size: 2,
        window_size: 112,
        out_hidden_size: dense_qwen::gguf_i32_catalog(metadata, "clip.vision.projection_dim")?,
        quantized_weight_configs,
    })
}

pub(crate) fn validate_qwen3_vl_text_gguf_catalog(
    text_config: &dense_qwen::DecoderConfig,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(Vec<i32>, u32, u32), Error> {
    if !text_config.tie_word_embeddings {
        return Err(Error::UnsupportedArchitecture(
            "qwen3vl GGUF with an untied output head is not supported".into(),
        ));
    }
    let architecture = dense_qwen::gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "qwen3vl" | "qwen3vlmoe") {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3-VL text validation requires qwen3vl or qwen3vlmoe, got {architecture:?}"
        )));
    }
    let mrope_key = format!("{architecture}.rope.dimension_sections");
    let mrope_section = gguf_integer_array(metadata, &mrope_key, Some(3))?;
    if mrope_section.len() != 3 || mrope_section.iter().any(|&section| section < 0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF RoPE sections must contain three non-negative values, got {mrope_section:?}"
        )));
    }
    if mrope_section.iter().sum::<i32>() != text_config.head_dim / 2 {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF RoPE sections {mrope_section:?} do not cover half of head_dim {}",
            text_config.head_dim
        )));
    }
    Ok((
        mrope_section,
        gguf_token_id(metadata, "<|image_pad|>")?,
        gguf_token_id(metadata, "<|video_pad|>")?,
    ))
}

pub(crate) fn validate_qwen3_vl_vision_geometry(
    text_config: &dense_qwen::DecoderConfig,
    metadata: &HashMap<String, GgufMetadataValue>,
    vision_config: &VisionConfig,
) -> Result<(), Error> {
    if vision_config.hidden_size <= 0
        || vision_config.intermediate_size <= 0
        || vision_config.num_heads <= 0
        || vision_config.patch_size <= 0
        || vision_config.spatial_merge_size <= 0
    {
        return Err(Error::UnsupportedArchitecture(
            "qwen3vl GGUF vision geometry must be positive".into(),
        ));
    }
    let architecture = dense_qwen::gguf_string(metadata, "general.architecture")?;
    let deepstack_key = format!("{architecture}.n_deepstack_layers");
    if let Some(value) = metadata.get(&deepstack_key) {
        let expected = value.as_i64().ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {deepstack_key:?} has the wrong type"
            ))
        })?;
        if expected != vision_config.deepstack_layer_count() as i64 {
            return Err(Error::UnsupportedArchitecture(format!(
                "qwen3vl GGUF expects {expected} DeepStack layers, but its mmproj contains {}",
                vision_config.deepstack_layer_count()
            )));
        }
    }
    if vision_config.out_hidden_size != text_config.hidden_size {
        return Err(Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF projector output {} does not match language hidden size {}",
            vision_config.out_hidden_size, text_config.hidden_size
        )));
    }
    Ok(())
}

pub(crate) fn validate_qwen3_vl_mmproj(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let architecture = dense_qwen::gguf_string(metadata, "general.architecture")?;
    let projector = dense_qwen::gguf_string(metadata, "clip.projector_type")?;
    if architecture != "clip" || projector != "qwen3vl_merger" {
        return Err(Error::UnsupportedArchitecture(format!(
            "expected a qwen3vl GGUF vision projector, got architecture {architecture:?} and projector {projector:?}"
        )));
    }
    Ok(())
}

fn gguf_integer_array(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
    take: Option<usize>,
) -> Result<Vec<i32>, Error> {
    let values = metadata
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .and_then(GgufMetadataArray::to_i64_vec)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "GGUF metadata is missing integer array {key:?}"
            ))
        })?;
    let values = take.map_or(values.as_slice(), |count| {
        &values[..values.len().min(count)]
    });
    values
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF metadata value in {key:?} exceeds i32"
                ))
            })
        })
        .collect()
}

fn gguf_deepstack_layers(metadata: &HashMap<String, GgufMetadataValue>) -> Result<Vec<i32>, Error> {
    let layers = match metadata.get("clip.vision.is_deepstack_layers") {
        Some(GgufMetadataValue::Array(GgufMetadataArray::Bool(layers))) => layers,
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF metadata key \"clip.vision.is_deepstack_layers\" has the wrong type".into(),
            ));
        }
        None => {
            return Err(Error::UnsupportedArchitecture(
                "qwen3vl mmproj is missing DeepStack layer metadata".into(),
            ));
        }
    };
    layers
        .iter()
        .enumerate()
        .filter_map(|(index, &enabled)| enabled.then_some(i32::try_from(index)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::UnsupportedArchitecture("DeepStack layer index exceeds i32".into()))
}

pub(crate) fn gguf_token_id(
    metadata: &HashMap<String, GgufMetadataValue>,
    token: &str,
) -> Result<u32, Error> {
    let tokens = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("qwen3vl GGUF is missing tokenizer.ggml.tokens".into())
        })?;
    let index = tokens
        .iter()
        .position(|candidate| candidate == token)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("qwen3vl GGUF tokenizer is missing {token:?}"))
        })?;
    u32::try_from(index)
        .map_err(|_| Error::UnsupportedArchitecture("qwen3vl token id exceeds u32".into()))
}

pub(crate) fn translate_qwen3_vl_mmproj_name(name: &str, deepstack_layers: &[i32]) -> String {
    const ROOTS: [(&str, &str); 6] = [
        ("v.position_embd", "model.visual.pos_embed"),
        ("v.patch_embd", "model.visual.patch_embed.proj"),
        ("v.post_ln", "model.visual.merger.norm"),
        ("mm.0", "model.visual.merger.linear_fc1"),
        ("mm.2", "model.visual.merger.linear_fc2"),
        ("v.blk", "model.visual.blocks"),
    ];
    if let Some(rest) = name.strip_prefix("v.deepstack.") {
        if let Some((layer, suffix)) = rest.split_once('.') {
            if let Ok(layer) = layer.parse::<i32>() {
                if let Some(index) = deepstack_layers.iter().position(|&value| value == layer) {
                    let suffix =
                        suffix
                            .replacen("fc1", "linear_fc1", 1)
                            .replacen("fc2", "linear_fc2", 1);
                    return format!("model.visual.deepstack_merger_list.{index}.{suffix}");
                }
            }
        }
    }
    for (source, target) in ROOTS {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, target, 1);
            if source == "v.blk" {
                translated = translated
                    .replace(".attn_out.", ".attn.proj.")
                    .replace(".attn_qkv.", ".attn.qkv.")
                    .replace(".ffn_up.", ".mlp.linear_fc1.")
                    .replace(".ffn_down.", ".mlp.linear_fc2.")
                    .replace(".ln1.", ".norm1.")
                    .replace(".ln2.", ".norm2.");
            }
            return translated;
        }
    }
    name.to_string()
}

/// Loads one shared Qwen vision projector into an architecture-selected root.
/// Finds the dense sibling mmproj used by the single-path dense or MoE loader.
pub(crate) fn find_qwen3_vl_mmproj(gguf_file: &Path) -> Result<PathBuf, Error> {
    crate::backend::mlx::runtime::checkpoint::gguf::find_sibling_mmproj(gguf_file, "qwen3vl")?
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "qwen3vl GGUF requires a nearby mmproj file relative to {}",
                gguf_file.display()
            ))
        })
}

impl CausalModel<Cache> for Model {
    type Tensor = Array;
    type Input<'a> = runtime_input::ModelInput<'a>;
    type Error = Exception;

    fn prefill_input_logits(
        &mut self,
        input: runtime_input::ModelInput<'_>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prepared = self.prepare_typed_prefill(input, stream)?;
        self.forward_prepared(prepared, cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let embeddings = self
            .model
            .language_model
            .embed_tokens
            .forward(input_tokens, stream)?;
        let start = cache
            .kv
            .first()
            .and_then(Option::as_ref)
            .map(crate::backend::mlx::runtime::cache::KeyValueCache::offset)
            .unwrap_or(0)
            + cache.rope_delta;
        let positions = [
            (start..start + input_tokens.dim(1)).collect(),
            (start..start + input_tokens.dim(1)).collect(),
            (start..start + input_tokens.dim(1)).collect(),
        ];
        self.forward_embeddings(input_tokens, embeddings, &positions, &[], cache, stream)?
            .try_index_device((.., -1, ..), stream)
    }
}

/// Qwen3-VL generation iterator.
pub type Generate<'a, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;
