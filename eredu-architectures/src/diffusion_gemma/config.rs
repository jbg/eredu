use std::num::NonZeroU32;

use eredu_core::InputModalities;
use serde::Deserialize;

use crate::gemma4::{ModelArgs, VisionConfig, VisionOwner, VisionPolicy};

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
pub struct FamilyPolicy {
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

/// Immutable source policy and retained vocabulary, feedback, encoder-scale and media scopes.
#[derive(Debug, Clone)]
pub struct FamilyConfig {
    policy: FamilyPolicy,
    conditioning: std::sync::Arc<super::conditioning::Declaration>,
    encoder_scales: std::sync::Arc<Vec<crate::gemma4::text::specs::ResidualScaleSpec>>,
    routed: std::sync::Arc<Vec<Option<crate::gemma4::text::routed::Declaration>>>,
    attention: std::sync::Arc<Vec<crate::gemma4::text::specs::AttentionSpec>>,
    dense: std::sync::Arc<Vec<crate::gemma4::text::specs::DenseSpec>>,
    blocks: std::sync::Arc<Vec<crate::gemma4::text::specs::BlockSpec>>,
    pinned: std::sync::Arc<crate::gemma4::text::pinned::Declaration>,
    projectors: std::sync::Arc<Vec<crate::gemma4::media::ProjectionSpec>>,
}
impl std::ops::Deref for FamilyConfig {
    type Target = FamilyPolicy;
    fn deref(&self) -> &FamilyPolicy {
        &self.policy
    }
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
    vision_config: Option<VisionPolicy>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
}

fn default_true() -> bool {
    true
}

impl FamilyConfig {
    /// Validates normalized family geometry and prepares independently owned static scopes once.
    pub fn new(policy: FamilyPolicy) -> Result<Self, ConfigError> {
        if policy.vision.is_some() != policy.image_token_id.is_some()
            || policy
                .image_token_id
                .is_some_and(|id| id >= policy.text.vocab_size as u32)
            || policy.canvas_length.get() > policy.text.max_position_embeddings as u32
        {
            return Err(ConfigError::Invalid(
                "incompatible diffusion canvas or image geometry".into(),
            ));
        }
        validate_projector_formats(&policy)?;
        let projectors = policy
            .vision
            .iter()
            .map(|vision| {
                crate::gemma4::media::ProjectionSpec::diffusion_vision(&policy.text, vision)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let blocks = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::BlockSpec::at(
                    &policy.text,
                    layer,
                    "model.decoder.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let dense = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::DenseSpec::at(
                    &policy.text,
                    layer,
                    "model.decoder.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::DenseSpec::validate_scopes(&policy.text, &dense)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let attention = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::AttentionSpec::at(
                    &policy.text,
                    layer,
                    "model.decoder.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::AttentionSpec::validate_scopes(&policy.text, &attention)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let routed = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                (policy.text.layer_policy(layer).is_some_and(|p| {
                    p.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
                }))
                .then(|| {
                    crate::gemma4::text::routed::Declaration::at(
                        &policy.text,
                        layer,
                        "model.decoder.layers",
                    )
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::routed::Declaration::validate_scopes(&policy.text, &routed)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let pinned = crate::gemma4::text::pinned::Declaration::diffusion(&policy.text)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let conditioning = super::conditioning::Declaration::new(&policy.text)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let encoder_scales = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::ResidualScaleSpec::new(format!(
                    "model.encoder.language_model.layers.{layer}.layer_scalar"
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        validate_encoder_formats(&policy)?;
        validate_static_topology(
            &policy,
            &pinned,
            &conditioning,
            &encoder_scales,
            &projectors,
            &blocks,
            &dense,
            &attention,
            &routed,
        )?;
        Ok(Self {
            policy,
            conditioning: conditioning.into(),
            encoder_scales: encoder_scales.into(),
            routed: routed.into(),
            attention: attention.into(),
            dense: dense.into(),
            blocks: blocks.into(),
            pinned: pinned.into(),
            projectors: projectors.into(),
        })
    }
    /// Retained routed selector, expert bank and branch gains for one sparse invocation.
    pub fn routed(
        &self,
        layer: usize,
    ) -> Result<Option<&crate::gemma4::text::routed::Declaration>, eredu_nn::Error> {
        self.routed
            .get(layer)
            .map(|s| s.as_ref())
            .ok_or_else(|| eredu_nn::Error::backend("routed layer outside declared stack"))
    }
    pub(crate) fn routed_scopes(&self) -> &[Option<crate::gemma4::text::routed::Declaration>] {
        &self.routed
    }
    pub(crate) fn routed_layers(
        &self,
    ) -> impl Iterator<Item = (usize, &crate::gemma4::text::routed::Declaration)> {
        self.routed
            .iter()
            .enumerate()
            .filter_map(|(layer, scope)| scope.as_ref().map(|scope| (layer, scope)))
    }
    /// Retained affine/head-gain owners and separate attention invocation/state semantics.
    pub fn attention(
        &self,
        layer: usize,
    ) -> Result<&crate::gemma4::text::specs::AttentionSpec, eredu_nn::Error> {
        self.attention
            .get(layer)
            .ok_or_else(|| eredu_nn::Error::backend("attention outside the declared Gemma stack"))
    }
    pub(crate) fn attention_scopes(&self) -> &[crate::gemma4::text::specs::AttentionSpec] {
        &self.attention
    }
    /// Retained dense gate, value and contraction parameters and their exact GELU equation.
    pub fn dense(
        &self,
        layer: usize,
    ) -> Result<&crate::gemma4::text::specs::DenseSpec, eredu_nn::Error> {
        self.dense
            .get(layer)
            .ok_or_else(|| eredu_nn::Error::backend("dense layer outside the declared Gemma stack"))
    }
    pub(crate) fn dense_scopes(&self) -> &[crate::gemma4::text::specs::DenseSpec] {
        &self.dense
    }
    /// Retained block gain, residual-scale and optional layer-input owners.
    pub fn block(
        &self,
        layer: usize,
    ) -> Result<&crate::gemma4::text::specs::BlockSpec, eredu_nn::Error> {
        self.blocks
            .get(layer)
            .ok_or_else(|| eredu_nn::Error::backend("block outside the declared Gemma stack"))
    }
    pub(crate) fn blocks(&self) -> &[crate::gemma4::text::specs::BlockSpec] {
        &self.blocks
    }
    /// Decoder-only feedback parameters, independent of shared-stack invocation state.
    pub fn conditioning(&self) -> &super::conditioning::Declaration {
        &self.conditioning
    }
    /// One independently owned encoder residual scale for this logical layer.
    pub fn encoder_scalar(
        &self,
        layer: usize,
    ) -> Result<&crate::gemma4::text::specs::ResidualScaleSpec, eredu_nn::Error> {
        self.encoder_scales.get(layer).ok_or_else(|| {
            eredu_nn::Error::backend("encoder scale outside the declared diffusion stack")
        })
    }
    /// Retained shared vocabulary/readout owners, distinct from invocation state.
    pub fn pinned(&self) -> &crate::gemma4::text::pinned::Declaration {
        &self.pinned
    }
    /// Normalized geometry and execution policy, separate from retained edge ownership.
    pub fn policy(&self) -> &FamilyPolicy {
        &self.policy
    }
    /// Declared media-to-text parameter edges consumed by source schemas.
    pub fn projectors(&self) -> &[crate::gemma4::media::ProjectionSpec] {
        &self.projectors
    }
    /// Optional independently owned image-to-text projection.
    pub fn vision_projector(&self) -> Option<&crate::gemma4::media::ProjectionSpec> {
        self.projectors.first()
    }

    /// Applies exact selected matrix encodings to shared text and image modules.
    pub fn with_checkpoint_formats(
        &self,
        formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let mut policy = self.policy.clone();
        policy.text.weight_quantization = None;
        policy.text.quantized_weights = None;
        policy.text.quantized_weight_configs = Some(formats.clone());
        if let Some(vision) = &mut policy.vision {
            let formats = formats
                .into_iter()
                .filter(|(name, _)| name.starts_with("model.encoder.vision_tower."))
                .collect::<std::collections::HashMap<_, _>>();
            *vision = vision.with_formats(None, None, (!formats.is_empty()).then_some(formats))?;
        }
        self.project_storage(policy)
    }
    /// Selects text execution formats while retaining independently admitted tower scopes.
    pub(crate) fn with_text_formats(
        &self,
        formats: std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let mut policy = self.policy.clone();
        policy.text.weight_quantization = None;
        policy.text.quantized_weights = None;
        policy.text.quantized_weight_configs = (!formats.is_empty()).then_some(formats);
        self.project_storage(policy)
    }
    fn project_storage(&self, policy: FamilyPolicy) -> Result<Self, ConfigError> {
        validate_projector_formats(&policy)?;
        let name = crate::gemma4::media::ProjectionSpec::DIFFUSION_VISION_WEIGHT;
        let encoding = policy.text.linear_format_for(name);
        let mut changed = false;
        let projectors = self
            .projectors
            .iter()
            .map(|source| {
                let projected = source
                    .with_format(encoding)
                    .map_err(|e| ConfigError::Invalid(e.to_string()))?;
                changed |= source.projection.format != projected.projection.format;
                Ok(projected)
            })
            .collect::<Result<Vec<_>, ConfigError>>()?;
        let blocks = self
            .blocks
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let dense = self
            .dense
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::DenseSpec::validate_scopes(&policy.text, &dense)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let attention = self
            .attention
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::AttentionSpec::validate_scopes(&policy.text, &attention)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let routed = self
            .routed
            .iter()
            .map(|scope| {
                scope
                    .as_ref()
                    .map(|scope| scope.project_formats(&policy.text))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::routed::Declaration::validate_scopes(&policy.text, &routed)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let pinned = self
            .pinned
            .project_formats(&policy.text)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        validate_encoder_formats(&policy)?;
        let conditioning = self
            .conditioning
            .project_formats(&policy.text)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        validate_static_topology(
            &policy,
            &pinned,
            &conditioning,
            &self.encoder_scales,
            &projectors,
            &blocks,
            &dense,
            &attention,
            &routed,
        )?;
        Ok(Self {
            policy,
            conditioning: conditioning.into(),
            encoder_scales: self.encoder_scales.clone(),
            routed: if self.routed.iter().zip(&routed).all(|(s, t)| match (s, t) {
                (None, None) => true,
                (Some(s), Some(t)) => s.same_formats(t),
                _ => false,
            }) {
                self.routed.clone()
            } else {
                routed.into()
            },
            attention: if self
                .attention
                .iter()
                .zip(&attention)
                .all(|(s, t)| s.same_formats(t))
            {
                self.attention.clone()
            } else {
                attention.into()
            },
            dense: if self
                .dense
                .iter()
                .zip(&dense)
                .all(|(source, target)| source.same_formats(target))
            {
                self.dense.clone()
            } else {
                dense.into()
            },
            blocks: if self
                .blocks
                .iter()
                .zip(&blocks)
                .all(|(source, target)| source.same_formats(target))
            {
                self.blocks.clone()
            } else {
                blocks.into()
            },
            pinned: pinned.into(),
            projectors: if changed {
                projectors.into()
            } else {
                self.projectors.clone()
            },
        })
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
                ));
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
                ));
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
        Self::new(FamilyPolicy {
            text,
            canvas_length,
            encoder_attention,
            vision: source
                .vision_config
                .map(|policy| VisionConfig::new(policy, VisionOwner::DiffusionGemma))
                .transpose()?,
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
        Self::new(FamilyPolicy {
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
                (
                    "vision",
                    format!(
                        "{:?}",
                        self.vision
                            .as_ref()
                            .map(|vision| (vision.owner(), vision.policy()))
                    ),
                ),
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

fn validate_projector_formats(policy: &FamilyPolicy) -> Result<(), ConfigError> {
    for name in policy
        .text
        .quantized_weight_configs
        .iter()
        .flat_map(|f| f.keys())
        .chain(policy.text.quantized_weights.iter().flatten())
    {
        let projector = name.starts_with("model.encoder.embed_vision.");
        if (projector && name != crate::gemma4::media::ProjectionSpec::DIFFUSION_VISION_WEIGHT)
            || (policy.vision.is_none()
                && (projector || name.starts_with("model.encoder.vision_tower.")))
        {
            return Err(ConfigError::Invalid(format!(
                "diffusion media format references undeclared projection {name}"
            )));
        }
    }
    Ok(())
}

fn validate_encoder_formats(policy: &FamilyPolicy) -> Result<(), ConfigError> {
    for name in policy
        .text
        .quantized_weight_configs
        .iter()
        .flat_map(|f| f.keys())
        .chain(policy.text.quantized_weights.iter().flatten())
    {
        if name.starts_with("model.encoder.language_model.layers.") {
            return Err(ConfigError::Invalid(format!(
                "encoder scale has no matrix format extension: {name}"
            )));
        }
    }
    Ok(())
}

fn validate_static_topology(
    policy: &FamilyPolicy,
    pinned: &crate::gemma4::text::pinned::Declaration,
    conditioning: &super::conditioning::Declaration,
    scales: &[crate::gemma4::text::specs::ResidualScaleSpec],
    projectors: &[crate::gemma4::media::ProjectionSpec],
    blocks: &[crate::gemma4::text::specs::BlockSpec],
    dense: &[crate::gemma4::text::specs::DenseSpec],
    attention: &[crate::gemma4::text::specs::AttentionSpec],
    routed: &[Option<crate::gemma4::text::routed::Declaration>],
) -> Result<(), ConfigError> {
    use eredu_nn::DeclareLogicalParameters;
    let declarations = || -> Result<_, eredu_nn::Error> {
        let mut parameters = pinned.logical_parameters()?;
        parameters.extend(conditioning.logical_parameters()?);
        parameters.extend(scales.logical_parameters()?);
        parameters.extend(projectors.logical_parameters()?);
        parameters.extend(blocks.logical_parameters()?);
        parameters.extend(dense.logical_parameters()?);
        parameters.extend(attention.logical_parameters()?);
        for scope in routed.iter().flatten() {
            parameters.extend(scope.logical_parameters()?);
        }
        if let Some(vision) = &policy.vision {
            parameters.extend(vision.logical_parameters()?);
        }
        Ok(parameters)
    };
    let parameters = declarations().map_err(|e| ConfigError::Invalid(e.to_string()))?;
    let references = crate::gemma4::text::specs::format_references(
        policy.text.weight_quantization,
        policy.text.quantized_weights.as_ref(),
        policy.text.quantized_weight_configs.as_ref(),
    )
    .chain(policy.vision.iter().flat_map(|config| {
        crate::gemma4::text::specs::format_references(
            config.weight_quantization,
            config.quantized_weights.as_ref(),
            config.quantized_weight_configs.as_ref(),
        )
    }));
    crate::gemma4::text::specs::validate_format_references(&parameters, routed, references)
        .map_err(|e| ConfigError::Invalid(e.to_string()))?;
    eredu_nn::validate_logical_parameter_topology(&parameters)
        .map_err(|e| ConfigError::Invalid(e.to_string()))
}
