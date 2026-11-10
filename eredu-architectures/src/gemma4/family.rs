//! Composite Gemma 4 decoder and media configuration.

use std::collections::BTreeSet;

use eredu_core::InputModalities;
use serde::Deserialize;

use super::{
    AudioConfig, AudioConfigError, ConfigError, ModelArgs, VisionConfig, VisionConfigError,
};

/// Invalid top-level Gemma configuration.
#[derive(Debug, thiserror::Error)]
pub enum FamilyConfigError {
    /// Top-level JSON decoding failed.
    #[error("invalid Gemma 4 family configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Nested text configuration failed validation.
    #[error(transparent)]
    Text(#[from] ConfigError),
    /// Nested vision configuration failed validation.
    #[error(transparent)]
    Vision(#[from] VisionConfigError),
    /// Nested audio configuration failed validation.
    #[error(transparent)]
    Audio(#[from] AudioConfigError),
    /// Cross-component policy is inconsistent.
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
struct FamilySource {
    #[serde(default = "default_model_type")]
    model_type: String,
    text_config: serde_json::Value,
    #[serde(default)]
    vision_config: Option<serde_json::Value>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
    #[serde(default)]
    audio_config: Option<serde_json::Value>,
    #[serde(default)]
    audio_token_id: Option<i32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    quantization: Option<serde_json::Value>,
}

fn default_model_type() -> String {
    "gemma4".into()
}

const fn default_true() -> bool {
    true
}

/// Normalized Gemma text/vision/audio source policy.
#[derive(Debug, Clone)]
pub struct FamilyPolicy {
    /// Declared outer family-wrapper identity (`gemma4` or `gemma4_unified`).
    pub model_type: String,
    /// Normalized text decoder.
    pub text: ModelArgs,
    /// Optional image/video encoder.
    pub vision: Option<VisionConfig>,
    /// Image placeholder token.
    pub image_token_id: Option<i32>,
    /// Video placeholder token.
    pub video_token_id: Option<i32>,
    /// Optional audio encoder.
    pub audio: Option<AudioConfig>,
    /// Audio placeholder token.
    pub audio_token_id: Option<i32>,
}

/// Immutable family policy with retained complete text and media parameter scopes.
#[derive(Debug, Clone)]
pub struct FamilyConfig {
    policy: FamilyPolicy,
    routed: std::sync::Arc<Vec<Option<crate::gemma4::text::routed::Declaration>>>,
    attention: std::sync::Arc<Vec<crate::gemma4::text::specs::AttentionSpec>>,
    dense: std::sync::Arc<Vec<crate::gemma4::text::specs::DenseSpec>>,
    blocks: std::sync::Arc<Vec<crate::gemma4::text::specs::BlockSpec>>,
    pinned: std::sync::Arc<super::text::pinned::Declaration>,
    projectors: std::sync::Arc<Vec<super::media::ProjectionSpec>>,
}
impl std::ops::Deref for FamilyConfig {
    type Target = FamilyPolicy;
    fn deref(&self) -> &FamilyPolicy {
        &self.policy
    }
}
impl FamilyPolicy {
    /// Validates placeholder ranges, collisions, and modality availability.
    pub fn validate(&self) -> Result<(), FamilyConfigError> {
        if let Some(vision) = &self.vision {
            vision
                .require_root(super::VisionOwner::Gemma4.root())
                .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        }
        for name in self
            .text
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(self.text.quantized_weights.iter().flatten())
        {
            for (kind, present, prefix) in [
                (
                    super::media::ProjectionKind::Vision,
                    self.vision.is_some(),
                    "model.embed_vision.",
                ),
                (
                    super::media::ProjectionKind::Audio,
                    self.audio.is_some(),
                    "model.embed_audio.",
                ),
            ] {
                if name.starts_with(prefix) && (!present || name != kind.gemma_weight()) {
                    return Err(FamilyConfigError::Invalid(format!(
                        "media format references undeclared edge {name}"
                    )));
                }
            }
        }
        let mut tokens = BTreeSet::new();
        for (name, token) in [
            ("image", self.image_token_id),
            ("video", self.video_token_id),
            ("audio", self.audio_token_id),
        ] {
            if let Some(token) = token {
                if token < 0 || token >= self.text.vocab_size {
                    return Err(FamilyConfigError::Invalid(format!(
                        "Gemma 4 {name} placeholder {token} is outside the vocabulary"
                    )));
                }
                if !tokens.insert(token) {
                    return Err(FamilyConfigError::Invalid(format!(
                        "Gemma 4 {name} placeholder {token} collides with another modality"
                    )));
                }
            }
        }
        if self.vision.is_some() && self.image_token_id.is_none() && self.video_token_id.is_none() {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 vision configuration has no image or video placeholder".into(),
            ));
        }
        if self.vision.is_none() && (self.image_token_id.is_some() || self.video_token_id.is_some())
        {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 visual placeholder requires a vision configuration".into(),
            ));
        }
        if self.audio.is_some() != self.audio_token_id.is_some() {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 audio configuration and placeholder must be declared together".into(),
            ));
        }
        Ok(())
    }
}

/// Identifies projector-only media metadata in unified text conversions.
/// Full tower configurations and malformed non-object values remain subject
/// to ordinary geometry validation.
pub(crate) fn is_media_config_stub(value: &serde_json::Value) -> bool {
    value.as_object().is_some_and(|object| {
        !object.contains_key("num_hidden_layers")
            && matches!(
                object.get("model_type").and_then(serde_json::Value::as_str),
                Some("gemma4_unified_vision" | "gemma4_unified_audio")
            )
    })
}

/// Splits a Hugging Face `quantization` object into the default weight
/// format and the per-tensor overrides that mixed-precision conversions carry
/// as nested objects keyed by physical tensor name without the `.weight`
/// suffix (for example `language_model.model.layers.3.mlp.down_proj`). Each
/// override inherits the fields it omits (typically `mode`) from the default
/// and is re-keyed by the canonical `model.*.weight` name the checkpoint plan
/// resolves formats with.
fn split_quantization(
    value: serde_json::Value,
) -> Result<
    (
        serde_json::Value,
        serde_json::Map<String, serde_json::Value>,
    ),
    FamilyConfigError,
> {
    let serde_json::Value::Object(entries) = value else {
        return Ok((value, serde_json::Map::new()));
    };
    let mut default = serde_json::Map::new();
    let mut nested = Vec::new();
    for (key, entry) in entries {
        match entry {
            serde_json::Value::Object(_) => nested.push((key, entry)),
            serde_json::Value::Bool(false) => {
                return Err(FamilyConfigError::Invalid(format!(
                    "Gemma 4 quantization marks {key:?} as unquantized; per-tensor dense overrides are not supported"
                )));
            }
            serde_json::Value::Bool(true) => {}
            other => {
                default.insert(key, other);
            }
        }
    }
    let mut overrides = serde_json::Map::new();
    for (key, entry) in nested {
        let serde_json::Value::Object(fields) = entry else {
            unreachable!("only objects are collected");
        };
        let mut merged = default.clone();
        merged.extend(fields);
        let canonical = crate::gemma4::config::canonical_weight_name(&key);
        if overrides
            .insert(canonical.clone(), serde_json::Value::Object(merged))
            .is_some()
        {
            return Err(FamilyConfigError::Invalid(format!(
                "duplicate Gemma 4 quantization override for {canonical:?}"
            )));
        }
    }
    Ok((serde_json::Value::Object(default), overrides))
}

impl FamilyConfig {
    /// Validates normalized policy and declares cross-component parameters once.
    pub fn new(policy: FamilyPolicy) -> Result<Self, FamilyConfigError> {
        policy.validate()?;
        let mut projectors = Vec::new();
        if let Some(config) = &policy.vision {
            projectors.push(
                super::media::ProjectionSpec::vision(policy.text.hidden_size, config)
                    .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?,
            );
        }
        if let Some(config) = &policy.audio {
            projectors.push(
                super::media::ProjectionSpec::audio(policy.text.hidden_size, config)
                    .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?,
            );
        }
        let blocks = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::BlockSpec::at(
                    &policy.text,
                    layer,
                    "model.language_model.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let dense = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::DenseSpec::at(
                    &policy.text,
                    layer,
                    "model.language_model.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::DenseSpec::validate_scopes(&policy.text, &dense)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let attention = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                crate::gemma4::text::specs::AttentionSpec::at(
                    &policy.text,
                    layer,
                    "model.language_model.layers",
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::AttentionSpec::validate_scopes(&policy.text, &attention)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let routed = (0..policy.text.num_hidden_layers())
            .map(|layer| {
                (policy.text.layer_policy(layer).is_some_and(|p| {
                    p.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe
                }))
                .then(|| {
                    crate::gemma4::text::routed::Declaration::at(
                        &policy.text,
                        layer,
                        "model.language_model.layers",
                    )
                })
                .transpose()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::routed::Declaration::validate_scopes(&policy.text, &routed)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let pinned = super::text::pinned::Declaration::gemma(&policy.text)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        validate_family_scopes(
            &policy,
            &pinned,
            &blocks,
            &dense,
            &attention,
            &routed,
            &projectors,
        )?;
        let exact = Self {
            policy,
            routed: routed.into(),
            attention: attention.into(),
            dense: dense.into(),
            blocks: blocks.into(),
            pinned: pinned.into(),
            projectors: projectors.into(),
        };
        exact.validate()?;
        Ok(exact)
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
    /// Retained vocabulary, final gain and optional per-layer input parameter owners.
    pub fn pinned(&self) -> &super::text::pinned::Declaration {
        &self.pinned
    }
    /// Normalized source policy, independent of retained parameter edges.
    pub fn policy(&self) -> &FamilyPolicy {
        &self.policy
    }
    /// Media-to-text owners used by admission, construction and discovery.
    pub fn projectors(&self) -> &[super::media::ProjectionSpec] {
        &self.projectors
    }
    /// Resolves one optional media edge without rebuilding its parameter geometry.
    pub fn projector(
        &self,
        kind: super::media::ProjectionKind,
    ) -> Option<&super::media::ProjectionSpec> {
        self.projectors.iter().find(|spec| spec.kind == kind)
    }
    pub(in crate::gemma4) fn project_storage(
        &self,
        policy: FamilyPolicy,
    ) -> Result<Self, FamilyConfigError> {
        policy.validate()?;
        let mut projectors = Vec::new();
        let mut changed = false;
        for source in self.projectors.iter() {
            let name = source.projection.weight.id.as_str();
            let input = source.projection.input;
            let (dimensions, epsilon, format) = match source.kind {
                super::media::ProjectionKind::Vision => {
                    let config = policy.vision.as_ref().ok_or_else(|| {
                        FamilyConfigError::Invalid("vision edge has no owning tower".into())
                    })?;
                    (
                        config.hidden_size,
                        config.rms_norm_eps,
                        config.linear_format_for(name, input),
                    )
                }
                super::media::ProjectionKind::Audio => {
                    let config = policy.audio.as_ref().ok_or_else(|| {
                        FamilyConfigError::Invalid("audio edge has no owning tower".into())
                    })?;
                    (
                        config.output_proj_dims,
                        config.rms_norm_eps,
                        config.linear_format_for(name, input),
                    )
                }
            };
            if input != dimensions
                || source.projection.output != policy.text.hidden_size
                || source.epsilon != epsilon
            {
                return Err(FamilyConfigError::Invalid(
                    "storage projection changes media-to-text geometry or normalization".into(),
                ));
            }
            let projected = source
                .with_format(format)
                .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
            changed |= projected.projection.format != source.projection.format;
            projectors.push(projected);
        }
        if usize::from(policy.vision.is_some()) + usize::from(policy.audio.is_some())
            != projectors.len()
        {
            return Err(FamilyConfigError::Invalid(
                "storage projection changes media edge ownership".into(),
            ));
        }
        let blocks = self
            .blocks
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let dense = self
            .dense
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::DenseSpec::validate_scopes(&policy.text, &dense)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let attention = self
            .attention
            .iter()
            .map(|spec| spec.project_formats(&policy.text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::specs::AttentionSpec::validate_scopes(&policy.text, &attention)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
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
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        crate::gemma4::text::routed::Declaration::validate_scopes(&policy.text, &routed)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        let pinned = self
            .pinned
            .project_formats(&policy.text)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        validate_family_scopes(
            &policy,
            &pinned,
            &blocks,
            &dense,
            &attention,
            &routed,
            &projectors,
        )?;
        Ok(Self {
            policy,
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

    /// Returns the nested text implementation identity preserved at admission.
    pub fn effective_model_type(&self) -> &str {
        &self.text.model_type
    }

    /// Whether this configuration uses the unified multimodal family identity.
    pub fn is_unified(&self) -> bool {
        self.model_type == "gemma4_unified"
    }

    /// Returns the input modalities admitted by this exact family variant.
    pub const fn input_modalities(&self) -> InputModalities {
        InputModalities {
            text: true,
            image: self.policy.image_token_id.is_some(),
            audio: self.policy.audio_token_id.is_some(),
            video: self.policy.video_token_id.is_some(),
        }
    }

    /// Retains the actual optional audio projection bias admitted from tensor presence.
    pub fn with_audio_output_projection_bias(&self, present: bool) -> Self {
        let mut exact = self.clone();
        if let Some(audio) = exact.policy.audio.as_mut() {
            *audio = audio.with_output_projection_bias(present);
        }
        exact
    }

    /// Parses and validates one top-level Hugging Face configuration.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, FamilyConfigError> {
        let source: FamilySource = serde_json::from_slice(bytes)?;
        if !matches!(source.model_type.as_str(), "gemma4" | "gemma4_unified") {
            return Err(FamilyConfigError::Invalid(format!(
                "unsupported Gemma 4 family model_type {:?}",
                source.model_type
            )));
        }
        let mut text_value = source.text_config;
        let text_object = text_value.as_object_mut().ok_or_else(|| {
            FamilyConfigError::Invalid("Gemma 4 text_config must be an object".into())
        })?;
        text_object.insert(
            "tie_word_embeddings".into(),
            serde_json::Value::Bool(source.tie_word_embeddings),
        );
        if let Some(quantization) = source.quantization {
            let (default, overrides) = split_quantization(quantization)?;
            text_object.insert("weight_quantization".into(), default);
            if !overrides.is_empty() {
                text_object.insert(
                    "quantized_weight_configs".into(),
                    serde_json::Value::Object(overrides),
                );
            }
        }
        let text = ModelArgs::from_hf_json(&serde_json::to_vec(&text_value)?)?;
        let vision_stub = source
            .vision_config
            .as_ref()
            .is_some_and(is_media_config_stub);
        let audio_stub = source
            .audio_config
            .as_ref()
            .is_some_and(is_media_config_stub);
        let vision = source
            .vision_config
            .filter(|_| !vision_stub)
            .map(|value| -> Result<VisionConfig, FamilyConfigError> {
                let mut config: super::vision::VisionPolicy = serde_json::from_value(value)?;
                // Media-to-text projections use the conversion's decoder encoding.
                // Store it in the media configuration so admission and construction
                // consume the same normalized format.
                let projection = super::media::ProjectionKind::Vision.gemma_weight();
                if let Some(format) = text.linear_format_for(projection).weight_quantization() {
                    config
                        .quantized_weight_configs
                        .get_or_insert_default()
                        .entry(projection.into())
                        .or_insert(format);
                }
                for (name, format) in text.quantized_weight_configs.iter().flatten() {
                    if name.starts_with("model.vision_tower.") || name == projection {
                        config
                            .quantized_weight_configs
                            .get_or_insert_default()
                            .insert(name.clone(), *format);
                    }
                }
                Ok(VisionConfig::new(
                    config,
                    super::vision::VisionOwner::Gemma4,
                )?)
            })
            .transpose()?;
        let audio = source
            .audio_config
            .filter(|_| !audio_stub)
            .map(|value| -> Result<AudioConfig, FamilyConfigError> {
                let mut config: super::audio::AudioPolicy = serde_json::from_value(value)?;
                // Media-to-text projections use the conversion's decoder encoding.
                // Store it in the media configuration so admission and construction
                // consume the same normalized format.
                let projection = super::media::ProjectionKind::Audio.gemma_weight();
                if let Some(format) = text.linear_format_for(projection).weight_quantization() {
                    config
                        .quantized_weight_configs
                        .get_or_insert_default()
                        .entry(projection.into())
                        .or_insert(format);
                }
                for (name, format) in text.quantized_weight_configs.iter().flatten() {
                    if name.starts_with("model.audio_tower.") || name == projection {
                        config
                            .quantized_weight_configs
                            .get_or_insert_default()
                            .insert(name.clone(), *format);
                    }
                }
                Ok(AudioConfig::new(config)?)
            })
            .transpose()?;
        // A text-only conversion keeps the placeholder ids of the towers it
        // reduced to stubs; without an encoder they admit nothing, so they are
        // cleared rather than reported as a modality the artifact cannot serve.
        // A placeholder with no media configuration at all is still an orphan
        // and is rejected by `validate`.
        let (image_token_id, video_token_id) = if vision_stub {
            (None, None)
        } else {
            (source.image_token_id, source.video_token_id)
        };
        let audio_token_id = if audio_stub {
            None
        } else {
            source.audio_token_id
        };
        let policy = FamilyPolicy {
            model_type: source.model_type,
            text,
            vision,
            image_token_id,
            video_token_id,
            audio,
            audio_token_id,
        };
        Self::new(policy)
    }

    /// Validates cross-component policy and retained projector references.
    pub fn validate(&self) -> Result<(), FamilyConfigError> {
        self.policy.validate()?;
        self.pinned
            .validate_for(&self.text)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        use eredu_nn::DeclareLogicalParameters;
        let parameters = self
            .projectors
            .as_slice()
            .logical_parameters()
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
        Ok(())
    }

    /// Stable fingerprint including text schedule and cross-component geometry.
    pub fn architecture_fingerprint(&self) -> String {
        let vision = self.vision.as_ref().map_or_else(
            || "none".into(),
            |config| {
                format!(
                    "{}:{}:{}:{}:{}:{}",
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.num_attention_heads,
                    config.head_dim,
                    config.patch_size,
                    config.pooling_kernel_size
                )
            },
        );
        let audio = self.audio.as_ref().map_or_else(
            || "none".into(),
            |config| {
                format!(
                    "{}:{}:{}:{}:{}:{}:{:?}",
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.num_attention_heads,
                    config.output_proj_dims,
                    config.conv_kernel_size,
                    config.attention_chunk_size,
                    config.subsampling_conv_channels
                )
            },
        );
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "gemma4-family",
            [
                ("model_type", self.model_type.clone()),
                ("text", self.text.architecture_fingerprint()),
                ("vision", vision),
                ("audio", audio),
                (
                    "vision_quantization",
                    self.vision.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            format!(
                                "default={:?};weights={};overrides={}",
                                config.weight_quantization,
                                crate::cache_identity::string_set(
                                    config.quantized_weights.as_ref()
                                ),
                                crate::cache_identity::debug_map(
                                    config.quantized_weight_configs.as_ref()
                                )
                            )
                        },
                    ),
                ),
                (
                    "audio_quantization",
                    self.audio.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            format!(
                                "default={:?};weights={};overrides={}",
                                config.weight_quantization,
                                crate::cache_identity::string_set(
                                    config.quantized_weights.as_ref()
                                ),
                                crate::cache_identity::debug_map(
                                    config.quantized_weight_configs.as_ref()
                                )
                            )
                        },
                    ),
                ),
                (
                    "tokens",
                    format!(
                        "{:?}:{:?}:{:?}",
                        self.image_token_id, self.video_token_id, self.audio_token_id
                    ),
                ),
            ],
        )
    }
}

fn validate_family_scopes(
    policy: &FamilyPolicy,
    pinned: &super::text::pinned::Declaration,
    blocks: &[super::text::specs::BlockSpec],
    dense: &[super::text::specs::DenseSpec],
    attention: &[super::text::specs::AttentionSpec],
    routed: &[Option<super::text::routed::Declaration>],
    projectors: &[super::media::ProjectionSpec],
) -> Result<(), FamilyConfigError> {
    use eredu_nn::DeclareLogicalParameters;
    let declarations = || -> Result<_, eredu_nn::Error> {
        let mut parameters = pinned.logical_parameters()?;
        parameters.extend(blocks.logical_parameters()?);
        parameters.extend(dense.logical_parameters()?);
        parameters.extend(attention.logical_parameters()?);
        for scope in routed.iter().flatten() {
            parameters.extend(scope.logical_parameters()?);
        }
        parameters.extend(projectors.logical_parameters()?);
        if let Some(vision) = &policy.vision {
            parameters.extend(vision.logical_parameters()?);
        }
        if let Some(audio) = &policy.audio {
            parameters.extend(audio.logical_parameters()?);
        }
        Ok(parameters)
    };
    let parameters = declarations().map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
    let references = super::text::specs::format_references(
        policy.text.weight_quantization,
        policy.text.quantized_weights.as_ref(),
        policy.text.quantized_weight_configs.as_ref(),
    )
    .chain(policy.vision.iter().flat_map(|config| {
        super::text::specs::format_references(
            config.weight_quantization,
            config.quantized_weights.as_ref(),
            config.quantized_weight_configs.as_ref(),
        )
    }))
    .chain(policy.audio.iter().flat_map(|config| {
        super::text::specs::format_references(
            config.weight_quantization,
            config.quantized_weights.as_ref(),
            config.quantized_weight_configs.as_ref(),
        )
    }));
    super::text::specs::validate_format_references(&parameters, routed, references)
        .map_err(|e| FamilyConfigError::Invalid(e.to_string()))?;
    eredu_nn::validate_logical_parameter_topology(&parameters)
        .map_err(|e| FamilyConfigError::Invalid(e.to_string()))
}
