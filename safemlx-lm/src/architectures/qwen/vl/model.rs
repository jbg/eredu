//! Qwen3-VL conditional-generation model support.

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

use crate::{
    api::{
        common::{self, attention::AttentionInput, generation::CausalLm},
        input as runtime_input,
        qwen_vl::grid_thw_from_array,
    },
    architectures::qwen::dense as dense_qwen,
    error::Error,
    nn::tensor::{create_attention_mask, AttentionMask},
    runtime::attention::LayerSchedule,
    runtime::cache::{
        residency::{
            derive_prompt_cache_architecture_fingerprint, open_prompt_cache_snapshot,
            save_prompt_cache_snapshot, CacheBlockArrays, LayerCachePolicy, PromptCacheDescriptor,
            PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
            PromptCacheSnapshotBlock, PromptCacheStateArray, StateTensorDimension,
            StateTensorDtype, StateTensorOwner, StateTensorPolicy, StateTensorRole,
        },
        ConcatKeyValueCache, KeyValueCache,
    },
    runtime::checkpoint::load::gguf_quantization_configs,
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
    let cache_error = |error: crate::runtime::cache::residency::CacheResidencyError| {
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
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// Per-layer key/value caches.
    pub kv: Vec<Option<ConcatKeyValueCache>>,
    pub(crate) rope_delta: i32,
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
        Cache::default()
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
    crate::api::structural::validate_qwen3_vl_projector_gguf(
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
    crate::runtime::checkpoint::gguf::find_sibling_mmproj(gguf_file, "qwen3vl")?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!(
            "qwen3vl GGUF requires a nearby mmproj file relative to {}",
            gguf_file.display()
        ))
    })
}

impl CausalLm<Cache> for Model {
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
            .map(crate::runtime::cache::KeyValueCache::offset)
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
pub type Generate<'a, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Cache, S>;

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use safemlx::{
        module::ModuleParameters,
        ops::{indexing::TryIndexOp, GgufMetadataArray, GgufMetadataValue},
        Array, Device, DeviceType, ExecutionContext,
    };
    use serde_json::json;

    use crate::api::{common::generation::CausalLm, input as runtime_input};

    fn tiny_args() -> super::ModelArgs {
        let text_config = crate::architectures::qwen::dense::DecoderConfig {
            model_type: "qwen3_vl_text".into(),
            hidden_size: 12,
            num_hidden_layers: 1,
            intermediate_size: 24,
            num_attention_heads: 1,
            rms_norm_eps: 1e-6,
            vocab_size: 32,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            head_dim: 12,
            tie_word_embeddings: true,
            rope_scaling: Some(HashMap::new()),
            hidden_act: "silu".into(),
            attention_dropout: 0.0,
            attention_bias: Some(false),
            mlp_bias: Some(false),
            attention_schedule: crate::runtime::attention::LayerSchedule::all_full(1).unwrap(),
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            moe_intermediate_size: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            quantized_weight_configs: None,
        };
        let vision_config = super::VisionConfig {
            layer_schedule: crate::runtime::attention::LayerSchedule::new(
                1,
                vec![super::VisionLayerPolicy {
                    attention: super::VisionAttentionPolicy::Full,
                    deepstack_merger: Some(0),
                }],
            )
            .unwrap(),
            hidden_size: 8,
            hidden_act: "gelu_pytorch_tanh".into(),
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 16,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 8,
            out_hidden_size: 12,
            quantized_weight_configs: Default::default(),
        };
        super::ModelArgs {
            text_config,
            vision_config,
            image_token_id: 30,
            video_token_id: 31,
            mrope_section: vec![2, 2, 2],
        }
    }

    fn tiny_model(stream: &safemlx::Stream) -> super::Model {
        super::Model::new(tiny_args(), stream).unwrap()
    }

    #[test]
    fn prompt_cache_layout_records_multimodal_position_delta() {
        use crate::runtime::cache::residency::{LayerCachePolicy, StateTensorRole};

        let layout = super::prompt_cache_layer_layout(&tiny_args()).unwrap();
        let LayerCachePolicy::KeyValueWithFixedState { tensors, .. } = layout.get(0).unwrap()
        else {
            panic!("Qwen-VL layer zero must carry its position delta");
        };
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].role, StateTensorRole::PositionDelta);
    }

    #[test]
    fn prompt_cache_layout_accepts_exact_rank_local_kv_geometry() {
        use crate::runtime::{attention::LayerSchedule, cache::residency::LayerCachePolicy};

        let mut args = tiny_args();
        args.text_config.num_hidden_layers = 2;
        args.text_config.attention_schedule = LayerSchedule::all_full(2).unwrap();
        let layout = super::prompt_cache_layer_layout_with_kv_heads(&args, &[2, 1]).unwrap();
        let LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            ..
        } = layout.get(0).unwrap()
        else {
            panic!("Qwen-VL layer zero must carry its position delta");
        };
        assert_eq!(num_key_value_heads.get(), 2);
        let LayerCachePolicy::KeyValue {
            num_key_value_heads,
            ..
        } = layout.get(1).unwrap()
        else {
            panic!("Qwen-VL layer one must carry ordinary KV state");
        };
        assert_eq!(num_key_value_heads.get(), 1);
        assert!(super::prompt_cache_layer_layout_with_kv_heads(&args, &[2]).is_err());
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn schema_v4_multimodal_position_state_save_reload_parity() {
        use crate::runtime::cache::{
            residency::{PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology},
            KeyValueCache,
        };

        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let args = tiny_args();
        let mut kv = crate::runtime::cache::ConcatKeyValueCache::new();
        let values = (0..48).map(|value| value as f32).collect::<Vec<_>>();
        let keys = Array::from_slice(&values, &[1, 1, 4, 12]);
        kv.update_and_fetch(keys.clone(), keys, stream).unwrap();
        let cache = super::Cache {
            kv: vec![Some(kv)],
            rope_delta: 9,
        };
        let prefix_ids = [1_u32, 30, 30, 2];
        let layout = super::prompt_cache_layer_layout(&args).unwrap();
        let descriptor = PromptCacheDescriptor {
            model_family: "qwen3_vl".into(),
            effective_model_type: "qwen3_vl".into(),
            checkpoint_fingerprint: "zero-fixture".into(),
            prefix_content_fingerprint: "image:fixture-a;tokens:1,30,30,2".into(),
            architecture_fingerprint: super::prompt_cache_architecture_fingerprint(&args),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            batch_size: 1,
            layer_prefix_offsets: vec![0; layout.len()],
            layer_layout: layout,
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("prompt-cache");
        super::Model::save_prompt_cache(
            &cache,
            &destination,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            stream,
        )
        .unwrap();
        let (restored, _) =
            super::Model::load_prompt_cache(&args, &destination, &descriptor, &prefix_ids, stream)
                .unwrap();
        assert_eq!(restored.rope_delta, 9);
        assert_eq!(restored.kv[0].as_ref().unwrap().offset(), 4);

        let mut wrong_media = descriptor;
        wrong_media.prefix_content_fingerprint = "image:fixture-b;tokens:1,30,30,2".into();
        assert!(super::Model::load_prompt_cache(
            &args,
            &destination,
            &wrong_media,
            &prefix_ids,
            stream,
        )
        .is_err());
    }

    #[test]
    fn parses_qwen3_vl_2b_config_shape() {
        let mut config = json!({
            "model_type":"qwen3_vl","image_token_id":151655,"video_token_id":151656,
            "text_config":{
                "model_type":"qwen3_vl_text","hidden_size":2048,"num_hidden_layers":28,
                "intermediate_size":6144,"num_attention_heads":16,"rms_norm_eps":0.000001,
                "vocab_size":151936,"num_key_value_heads":8,"max_position_embeddings":262144,
                "rope_theta":5000000.0,"head_dim":128,"tie_word_embeddings":true,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}
            },
            "vision_config":{
                "depth":24,"hidden_size":1024,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":4096,"num_heads":16,"num_position_embeddings":2304,
                "in_channels":3,"patch_size":16,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":2048,
                "deepstack_visual_indexes":[5,11,17]
            }
        });
        let args = super::parse_model_args_value(config.clone()).unwrap();
        assert_eq!(args.text_config.hidden_size, 2048);
        assert_eq!(args.vision_config.deepstack_layers(), vec![5, 11, 17]);
        assert_eq!(args.mrope_section, vec![24, 20, 20]);
        assert!(args.text_config.quantization.is_none());

        config["quantization"] = json!({"group_size": 64, "bits": 4, "mode": "affine"});
        let args = super::parse_model_args_value(config).unwrap();
        assert_eq!(
            args.text_config.quantization,
            Some(crate::runtime::checkpoint::quantization::AffineQuantization::default().into())
        );
    }

    #[test]
    fn parses_qwen3_vl_moe_config_shape() {
        let config = json!({
            "model_type":"qwen3_vl_moe","image_token_id":151655,"video_token_id":151656,
            "tie_word_embeddings":false,
            "text_config":{
                "model_type":"qwen3_vl_moe_text","hidden_size":2048,"num_hidden_layers":48,
                "intermediate_size":6144,"num_attention_heads":32,"rms_norm_eps":0.000001,
                "vocab_size":151936,"num_key_value_heads":4,"max_position_embeddings":262144,
                "rope_theta":5000000.0,"head_dim":128,
                "moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,
                "norm_topk_prob":true,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}
            },
            "vision_config":{
                "depth":27,"hidden_size":1152,"hidden_act":"gelu_pytorch_tanh",
                "intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,
                "in_channels":3,"patch_size":16,"spatial_merge_size":2,
                "temporal_patch_size":2,"out_hidden_size":2048,
                "deepstack_visual_indexes":[8,16,24]
            }
        });
        let args = super::parse_model_args_value(config).unwrap();
        assert_eq!(args.text_config.model_type, "qwen3_vl_moe_text");
        assert_eq!(args.text_config.num_experts, 128);
        assert_eq!(args.text_config.num_experts_per_tok, 8);
        assert!(!args.text_config.tie_word_embeddings);
        assert_eq!(args.vision_config.deepstack_layers(), vec![8, 16, 24]);
    }

    #[test]
    fn interleaved_mrope_uses_height_and_width_slots() {
        let positions = [vec![1], vec![2], vec![3]];
        let (values, _) = super::mrope_values(&positions, 12, 10_000.0, &[2, 2, 2]);
        assert!((values[0] - 1.0f32.cos()).abs() < 1e-6);
        let height_angle = 2.0 / 10_000.0f32.powf(2.0 / 12.0);
        assert!((values[1] - height_angle.cos()).abs() < 1e-6);
        let width_angle = 3.0 / 10_000.0f32.powf(4.0 / 12.0);
        assert!((values[2] - width_angle.cos()).abs() < 1e-6);

        let (values, _) = super::mrope_values(&positions, 14, 10_000.0, &[3, 2, 2]);
        let temporal_tail_angle = 1.0 / 10_000.0f32.powf(12.0 / 14.0);
        assert!((values[6] - temporal_tail_angle.cos()).abs() < 1e-6);
    }

    #[test]
    fn translates_llama_cpp_qwen3_vl_mmproj_names() {
        let deepstack = [5, 11, 17];
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("v.blk.7.attn_qkv.weight", &deepstack),
            "model.visual.blocks.7.attn.qkv.weight"
        );
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("mm.2.bias", &deepstack),
            "model.visual.merger.linear_fc2.bias"
        );
        assert_eq!(
            super::translate_qwen3_vl_mmproj_name("v.deepstack.11.fc1.weight", &deepstack),
            "model.visual.deepstack_merger_list.1.linear_fc1.weight"
        );
    }

    #[test]
    fn parses_qwen3_vl_gguf_deepstack_and_mrope_metadata() {
        let metadata = HashMap::from([
            (
                "qwen3vl.rope.dimension_sections".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2, 2, 0])),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![false, true, false, true])),
            ),
        ]);
        assert_eq!(
            super::gguf_integer_array(&metadata, "qwen3vl.rope.dimension_sections", Some(3))
                .unwrap(),
            vec![2, 2, 2]
        );
        assert_eq!(super::gguf_deepstack_layers(&metadata).unwrap(), vec![1, 3]);
    }

    #[test]
    fn discovers_dense_qwen3_vl_mmproj_sibling() {
        let dir = std::env::temp_dir().join(format!(
            "safemlx-qwen3vl-mmproj-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let model = dir.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let dense = dir.join("mmproj-Qwen3VL-2B-Instruct-F16.gguf");
        let quantized = dir.join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf");
        std::fs::File::create(&model).unwrap();
        std::fs::File::create(&dense).unwrap();
        std::fs::File::create(&quantized).unwrap();
        assert_eq!(super::find_qwen3_vl_mmproj(&model).unwrap(), dense);
        let quantization_dir = dir.join("Q4_K_M");
        std::fs::create_dir(&quantization_dir).unwrap();
        let sharded_model = quantization_dir.join("model-00001-of-00002.gguf");
        std::fs::File::create(&sharded_model).unwrap();
        assert_eq!(super::find_qwen3_vl_mmproj(&sharded_model).unwrap(), dense);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn strict_loads_q8_qwen_vision_from_synthetic_gguf_checkpoints() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut source_args = tiny_args();
        source_args.text_config.num_hidden_layers = 2;
        source_args.text_config.attention_schedule =
            crate::runtime::attention::LayerSchedule::all_full(2).unwrap();
        source_args.vision_config.hidden_size = 32;
        source_args.vision_config.intermediate_size = 32;
        source_args.vision_config.num_heads = 4;
        let mut source = super::Model::new(source_args, stream).unwrap();
        for (name, parameter) in source.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.001), stream).unwrap()
            };
        }
        let mut arrays = HashMap::new();
        let mut vision_arrays = HashMap::new();
        for (name, value) in source.parameters().flatten() {
            if let Some(name) = name.strip_prefix("model.language_model.") {
                let name = name
                    .replace("layers.", "blk.")
                    .replace("self_attn.q_norm", "attn_q_norm")
                    .replace("self_attn.k_norm", "attn_k_norm")
                    .replace("self_attn.q_proj", "attn_q")
                    .replace("self_attn.k_proj", "attn_k")
                    .replace("self_attn.v_proj", "attn_v")
                    .replace("self_attn.o_proj", "attn_output")
                    .replace("input_layernorm", "attn_norm")
                    .replace("post_attention_layernorm", "ffn_norm")
                    .replace("mlp.gate_proj", "ffn_gate")
                    .replace("mlp.down_proj", "ffn_down")
                    .replace("mlp.up_proj", "ffn_up");
                let name = match name.as_str() {
                    "embed_tokens.weight" => "token_embd.weight".into(),
                    "norm.weight" => "output_norm.weight".into(),
                    _ => name,
                };
                arrays.insert(name, value.clone());
                continue;
            }
            let name = name.strip_prefix("model.visual.").unwrap();
            if name == "patch_embed.proj.weight" {
                vision_arrays.insert(
                    "v.patch_embd.weight".into(),
                    value.try_index_device((.., .., 0, .., ..), stream).unwrap(),
                );
                vision_arrays.insert(
                    "v.patch_embd.weight.1".into(),
                    value.try_index_device((.., .., 1, .., ..), stream).unwrap(),
                );
                continue;
            }
            let name = name
                .replace("pos_embed", "v.position_embd")
                .replace("patch_embed.proj", "v.patch_embd")
                .replace("blocks.", "v.blk.")
                .replace(".attn.qkv.", ".attn_qkv.")
                .replace(".attn.proj.", ".attn_out.")
                .replace(".mlp.linear_fc1.", ".ffn_up.")
                .replace(".mlp.linear_fc2.", ".ffn_down.")
                .replace(".norm1.", ".ln1.")
                .replace(".norm2.", ".ln2.")
                .replace("merger.norm", "v.post_ln")
                .replace("merger.linear_fc1", "mm.0")
                .replace("merger.linear_fc2", "mm.2")
                .replace("deepstack_merger_list.0.norm", "v.deepstack.0.norm")
                .replace("deepstack_merger_list.0.linear_fc1", "v.deepstack.0.fc1")
                .replace("deepstack_merger_list.0.linear_fc2", "v.deepstack.0.fc2");
            vision_arrays.insert(name, value.clone());
        }

        let mut tokens = (0..30)
            .map(|index| format!("token-{index}"))
            .collect::<Vec<_>>();
        tokens.extend(["<|image_pad|>".into(), "<|video_pad|>".into()]);
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3vl".into()),
            ),
            (
                "qwen3vl.embedding_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            ("qwen3vl.block_count".into(), GgufMetadataValue::Uint32(2)),
            (
                "qwen3vl.feed_forward_length".into(),
                GgufMetadataValue::Uint32(24),
            ),
            (
                "qwen3vl.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "qwen3vl.attention.key_length".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "qwen3vl.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-6),
            ),
            (
                "qwen3vl.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "qwen3vl.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "qwen3vl.rope.dimension_sections".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2, 2, 0])),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Uint32(2),
            ),
        ]);
        let vision_metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("clip".into()),
            ),
            (
                "clip.projector_type".into(),
                GgufMetadataValue::String("qwen3vl_merger".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "clip.vision.block_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "clip.vision.attention.head_count".into(),
                GgufMetadataValue::Uint32(4),
            ),
            (
                "clip.vision.patch_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.spatial_merge_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.projection_dim".into(),
                GgufMetadataValue::Uint32(12),
            ),
            (
                "clip.vision.is_deepstack_layers".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true])),
            ),
        ]);

        let fixture = crate::test_utils::SyntheticGguf::dense(&arrays, &metadata);
        let vision_fixture = crate::test_utils::SyntheticGguf::with_packed_tensors(
            &vision_arrays,
            &vision_metadata,
            |name, _| {
                (name.ends_with("attn_qkv.weight")
                    || name.ends_with("attn_out.weight")
                    || name.ends_with("ffn_up.weight")
                    || name.ends_with("ffn_down.weight")
                    || name == "mm.0.weight"
                    || name == "mm.2.weight"
                    || name.ends_with("fc1.weight")
                    || name.ends_with("fc2.weight"))
                .then_some(safemlx_gguf::GgmlType::Q8_0)
            },
        );
        let checkpoint = safemlx::ops::GgufCheckpoint::open(fixture.path()).unwrap();
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let vision_checkpoint = safemlx::ops::GgufCheckpoint::open(vision_fixture.path()).unwrap();
        let vision_metadata = crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
        let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let (mut loaded, eos_token_ids) =
            crate::architectures::qwen::vl::layerwise::load_qwen3_vl_gguf_layerwise_model(
                &checkpoint,
                &metadata,
                &vision_checkpoint,
                &vision_metadata,
                crate::WeightResidency::fully_resident(),
                None,
                stream,
                weights.stream(),
            )
            .unwrap();
        assert_eq!(loaded.args().image_token_id, 30);
        assert_eq!(loaded.args().video_token_id, 31);
        assert_eq!(loaded.args().mrope_section, vec![2, 2, 2]);
        assert_eq!(eos_token_ids, vec![2]);

        let directory = tempfile::tempdir().unwrap();
        let model_path = directory.path().join("qwen3vl-f16.gguf");
        let mmproj_path = directory.path().join("mmproj-Qwen3VL-F16.gguf");
        fs::copy(fixture.path(), &model_path).unwrap();
        fs::copy(vision_fixture.path(), mmproj_path).unwrap();
        let topology = |rank| {
            crate::ParallelTopology::from_rank(
                2,
                rank,
                1,
                2,
                1,
                crate::DeviceAssignment::new(DeviceType::Cpu, 0),
            )
            .unwrap()
        };
        let mut first = crate::architectures::distributed::pipeline::load_pipeline_model(
            &model_path,
            topology(0),
            stream,
            stream,
        )
        .unwrap();
        let mut second = crate::architectures::distributed::pipeline::load_pipeline_model(
            &model_path,
            topology(1),
            stream,
            stream,
        )
        .unwrap();
        let mut first_cache = first.new_cache().unwrap();
        let mut second_cache = second.new_cache().unwrap();
        let before = Array::from_slice(&[1u32], &[1, 1]);
        let after = Array::from_slice(&[2u32], &[1, 1]);
        let pixels = Array::from_slice(&[0.01f32; 96], &[4, 24]);
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&before),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::qwen_grid_thw(&grid),
            ),
            runtime_input::InputPart::text_token_ids(&after),
        ];
        let input = runtime_input::ModelInput::new(&parts);
        let crate::architectures::distributed::pipeline::PipelineStageOutput::Hidden(payload) =
            first
                .prefill_stage(
                    input,
                    crate::architectures::distributed::pipeline::PipelineStep::new(1, 3).unwrap(),
                    None,
                    &mut first_cache,
                    stream,
                )
                .unwrap()
        else {
            panic!("first Qwen3-VL GGUF pipeline stage produced logits")
        };
        let crate::architectures::distributed::pipeline::PipelineStageOutput::Logits(actual) =
            second
                .forward_stage(
                    crate::architectures::distributed::pipeline::PipelineStageInput::Hidden(
                        &payload,
                    ),
                    crate::architectures::distributed::pipeline::PipelineStep::new(1, 3).unwrap(),
                    None,
                    &mut second_cache,
                    stream,
                )
                .unwrap()
        else {
            panic!("last Qwen3-VL GGUF pipeline stage did not produce logits")
        };
        let mut resident_cache = loaded.new_cache();
        let expected = loaded
            .prefill_input_logits(input, &mut resident_cache, stream)
            .unwrap();
        let actual = actual.try_index_device((.., -1, ..), stream).unwrap();
        let actual = actual.evaluated().unwrap();
        let expected = expected.evaluated().unwrap();
        assert!(actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-5));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn parameter_tree_matches_public_qwen3_vl_checkpoint_names() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let model = tiny_model(context.stream());
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.language_model.embed_tokens.weight"));
        assert!(params.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(params.contains_key("model.visual.patch_embed.proj.weight"));
        assert!(params.contains_key("model.visual.deepstack_merger_list.0.norm.weight"));
        assert!(!params.contains_key("lm_head.weight"));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn tiny_image_prefill_runs_deepstack_and_mrope() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut model = tiny_model(stream);
        for (_, parameter) in model.parameters_mut().flatten() {
            *parameter = Array::zeros::<f32>(parameter.shape(), stream).unwrap();
        }
        let text = Array::from_slice(&[1u32, 2], &[1, 2]);
        let pixels = Array::zeros::<f32>(&[4, 24], stream).unwrap();
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let parts = [
            runtime_input::InputPart::text_token_ids(&text),
            runtime_input::InputPart::image_tensor(
                &pixels,
                runtime_input::InputMetadata::qwen_grid_thw(&grid),
            ),
        ];
        let mut cache = model.new_cache();
        let logits = model
            .prefill_input_logits(runtime_input::ModelInput::new(&parts), &mut cache, stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 32]);
        assert_eq!(
            cache.kv[0]
                .as_ref()
                .map(crate::runtime::cache::KeyValueCache::offset),
            Some(3)
        );
    }
}
