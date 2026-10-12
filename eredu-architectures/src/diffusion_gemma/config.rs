use std::num::NonZeroU32;

use eredu_core::InputModalities;
use serde::Deserialize;

use crate::gemma4::{ModelArgs, VisionConfig};

/// Context-encoder visibility selected independently from bidirectional refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderAttention {
    /// Text is causal; a prepared image span has bidirectional visibility.
    CausalWithVisionSpans,
    /// All encoded positions attend bidirectionally within their supplied span.
    Bidirectional,
}

/// Exact DiffusionGemma family geometry. Text parameters serve both invocations.
#[derive(Debug, Clone)]
pub struct FamilyConfig {
    /// Normalized shared text-block geometry, retaining diffusion identity.
    pub text: ModelArgs,
    /// Positive trained canvas length.
    pub canvas_length: NonZeroU32,
    /// Visibility of context encoding; refinement is always bidirectional.
    pub encoder_attention: EncoderAttention,
    /// Optional independently owned vision tower.
    pub vision: Option<VisionConfig>,
    /// Image placeholder token, present exactly when a vision tower is enabled.
    pub image_token_id: Option<u32>,
}

/// Invalid family identity, parameter geometry, or modality policy.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Malformed source configuration.
    #[error("invalid DiffusionGemma JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid shared text geometry.
    #[error(transparent)]
    Text(#[from] crate::gemma4::ConfigError),
    /// Invalid vision geometry.
    #[error(transparent)]
    Vision(#[from] crate::gemma4::VisionConfigError),
    /// The source declares inconsistent or unsupported equations.
    #[error("{0}")]
    Invalid(String),
}

#[derive(Deserialize)]
struct Source {
    model_type: String,
    text_config: serde_json::Value,
    canvas_length: u32,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

fn default_true() -> bool {
    true
}

impl FamilyConfig {
    /// Applies exact selected matrix encodings to shared text and image modules.
    pub fn with_checkpoint_formats(
        &self,
        formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Self {
        let mut config = self.clone();
        config.text.weight_quantization = None;
        config.text.quantized_weights = None;
        config.text.quantized_weight_configs = Some(formats.clone());
        if let Some(vision) = &mut config.vision {
            vision.weight_quantization = None;
            vision.quantized_weights = None;
            vision.quantized_weight_configs = Some(formats);
        }
        config
    }

    /// Whether any shared layer invokes an independently addressed expert bank.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.text.layer_schedule.iter().any(|policy| {
            policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
        })
    }

    /// Parses released family semantics without allocating any native resource.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        let mut source: Source = serde_json::from_slice(bytes)?;
        if source.model_type != "diffusion_gemma" {
            return Err(ConfigError::Invalid(
                "expected diffusion_gemma model_type".into(),
            ));
        }
        let canvas_length = NonZeroU32::new(source.canvas_length).ok_or_else(|| {
            ConfigError::Invalid("DiffusionGemma canvas_length must be positive".into())
        })?;
        let object = source.text_config.as_object_mut().ok_or_else(|| {
            ConfigError::Invalid("DiffusionGemma text_config must be an object".into())
        })?;
        match object.get("hidden_activation").and_then(|x| x.as_str()) {
            None | Some("gelu_pytorch_tanh") => {}
            _ => {
                return Err(ConfigError::Invalid(
                    "DiffusionGemma requires GELU-tanh gated feed-forward equations".into(),
                ))
            }
        }
        let encoder_attention = match object
            .get("use_bidirectional_attention")
            .and_then(|x| x.as_str())
        {
            None | Some("vision") => EncoderAttention::CausalWithVisionSpans,
            Some("all") => EncoderAttention::Bidirectional,
            _ => {
                return Err(ConfigError::Invalid(
                    "unknown DiffusionGemma encoder visibility".into(),
                ))
            }
        };
        object.insert(
            "tie_word_embeddings".into(),
            source.tie_word_embeddings.into(),
        );
        let text = ModelArgs::diffusion_geometry(&serde_json::to_vec(&source.text_config)?)?;
        if canvas_length.get() > text.max_position_embeddings as u32
            || canvas_length.get() > i32::MAX as u32
        {
            return Err(ConfigError::Invalid(
                "DiffusionGemma canvas exceeds positional geometry".into(),
            ));
        }
        if source.vision_config.is_some() != source.image_token_id.is_some() {
            return Err(ConfigError::Invalid(
                "DiffusionGemma vision tower and image token must be declared together".into(),
            ));
        }
        if source
            .image_token_id
            .is_some_and(|id| id >= text.vocab_size as u32)
        {
            return Err(ConfigError::Invalid(
                "DiffusionGemma image token lies outside the vocabulary".into(),
            ));
        }
        if let Some(vision) = &mut source.vision_config {
            vision.input_score_arithmetic = true;
            vision.validate()?;
        }
        Ok(Self {
            text,
            canvas_length,
            encoder_attention,
            vision: source.vision_config,
            image_token_id: source.image_token_id,
        })
    }

    /// Parses the published text GGUF, whose tensor names describe one shared
    /// stack and whose diffusion metadata describes a bidirectional canvas.
    pub fn from_gguf_metadata<C: crate::GgufTensorCatalog + ?Sized>(
        catalog: &C,
        metadata: &std::collections::HashMap<String, eredu_gguf::MetadataValue>,
    ) -> Result<Self, ConfigError> {
        use eredu_gguf::MetadataValue;
        if metadata
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("diffusion-gemma")
        {
            return Err(ConfigError::Invalid(
                "expected diffusion-gemma GGUF architecture".into(),
            ));
        }
        if metadata
            .get("diffusion-gemma.attention.causal")
            .and_then(MetadataValue::as_bool)
            != Some(false)
        {
            return Err(ConfigError::Invalid(
                "diffusion GGUF must declare bidirectional canvas attention".into(),
            ));
        }
        let canvas = metadata
            .get("diffusion.canvas_length")
            .and_then(MetadataValue::as_i64)
            .and_then(|value| u32::try_from(value).ok())
            .and_then(NonZeroU32::new)
            .ok_or_else(|| {
                ConfigError::Invalid("diffusion GGUF requires a positive canvas_length".into())
            })?;
        // Gemma's shared geometry parser validates the same head, sparse bank,
        // rotary and storage equations under this format's metadata namespace.
        let mut shared: std::collections::HashMap<String, MetadataValue> = metadata
            .iter()
            .map(|(key, value)| {
                let key = key
                    .strip_prefix("diffusion-gemma.")
                    .map_or_else(|| key.clone(), |suffix| format!("gemma4.{suffix}"));
                (key, value.clone())
            })
            .collect();
        if !shared.contains_key("gemma4.vocab_size") {
            let count = metadata
                .get("tokenizer.ggml.tokens")
                .and_then(MetadataValue::as_strings)
                .and_then(|tokens| u32::try_from(tokens.len()).ok())
                .filter(|count| *count > 0)
                .ok_or_else(|| {
                    ConfigError::Invalid(
                        "diffusion GGUF requires vocabulary geometry or tokenizer tokens".into(),
                    )
                })?;
            shared.insert("gemma4.vocab_size".into(), MetadataValue::Uint32(count));
        }
        let mut text = ModelArgs::from_gguf_metadata(catalog, &shared)?;
        if text.hidden_size_per_layer_input != 0
            || text.num_experts.is_none_or(|count| count <= 0)
            || text.layer_schedule.iter().any(|policy| {
                policy.key_value == eredu_nn::AttentionStateSource::Shared
                    || (policy.attention == eredu_core::AttentionPolicy::Full
                        && policy.key_value.value()
                            != Some(eredu_nn::AttentionValueSource::ReuseKey))
            })
        {
            return Err(ConfigError::Invalid(
                "diffusion GGUF does not declare the shared sparse stack equations".into(),
            ));
        }
        if canvas.get() > text.max_position_embeddings as u32 || canvas.get() > i32::MAX as u32 {
            return Err(ConfigError::Invalid(
                "diffusion GGUF canvas exceeds positional geometry".into(),
            ));
        }
        text.model_type = "diffusion_gemma_text".into();
        text.apply_diffusion_equations();
        Ok(Self {
            text,
            canvas_length: canvas,
            encoder_attention: EncoderAttention::CausalWithVisionSpans,
            vision: None,
            image_token_id: None,
        })
    }

    /// Cache identity includes invocation visibility and the shared arithmetic.
    pub fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "diffusion_gemma",
            [
                ("text", self.text.architecture_fingerprint()),
                ("canvas_length", self.canvas_length.to_string()),
                ("encoder_attention", format!("{:?}", self.encoder_attention)),
                ("vision", format!("{:?}", self.vision)),
                ("image_token", format!("{:?}", self.image_token_id)),
            ],
        )
    }

    /// Modalities admitted by this exact artifact's declared components.
    pub fn input_modalities(&self) -> InputModalities {
        InputModalities {
            text: true,
            image: self.vision.is_some(),
            audio: false,
            video: self.vision.is_some(),
        }
    }
}
