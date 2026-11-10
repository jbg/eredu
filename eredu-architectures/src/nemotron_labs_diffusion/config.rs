use std::{collections::HashMap, num::NonZeroU32};

use eredu_checkpoint::WeightQuantization;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{Error, RotarySpec};
use serde::Deserialize;
use serde_json::Value;

use crate::decoder::{AttentionProjection, Config, PartitionedConfig, PositionQueryScaling};

/// Released Pixtral tower and spatial projector geometry.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PixtralPolicy {
    /// Exact tower family identity.
    pub model_type: String,
    /// Patch embedding and residual width.
    pub hidden_size: i32,
    /// SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Number of independently addressable vision layers.
    pub num_hidden_layers: i32,
    /// Number of vision attention heads.
    pub num_attention_heads: i32,
    /// Independent vision head width.
    pub head_dim: i32,
    /// Number of input image channels.
    pub num_channels: i32,
    /// Patch extent in pixels.
    pub patch_size: i32,
    /// Maximum processed image extent.
    pub image_size: i32,
    /// Released pointwise activation.
    pub hidden_act: String,
    /// Vision rotary construction metadata.
    pub rope_parameters: Value,
}

impl PixtralPolicy {
    /// Validates the released tower equations and all dimensions before allocation.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_type != "pixtral"
            || self.hidden_act != "silu"
            || [
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.head_dim,
                self.num_channels,
                self.patch_size,
                self.image_size,
            ]
            .iter()
            .any(|x| *x <= 0)
            || self.num_attention_heads.checked_mul(self.head_dim) != Some(self.hidden_size)
            || self.head_dim % 4 != 0
            || self
                .rope_parameters
                .get("rope_type")
                .and_then(Value::as_str)
                != Some("default")
            || self
                .rope_parameters
                .get("rope_theta")
                .and_then(Value::as_f64)
                .is_none_or(|x| !x.is_finite() || x <= 0.0)
            || self
                .patch_size
                .checked_mul(self.patch_size)
                .and_then(|x| x.checked_mul(self.num_channels))
                .is_none()
        {
            return Err("invalid Pixtral geometry or two-dimensional rotary equations".into());
        }
        Ok(())
    }
}

/// Validated immutable Pixtral policy and its shared patch/block declarations.
#[derive(Clone, Deserialize)]
#[serde(try_from = "PixtralPolicy")]
pub struct PixtralConfig {
    policy: PixtralPolicy,
    tower: std::sync::Arc<super::vision::specs::TowerSpec>,
}
impl PixtralConfig {
    /// Validates source policy and declares every independently acquired tower scope once.
    pub fn new(policy: PixtralPolicy) -> Result<Self, String> {
        policy.validate()?;
        let tower = super::vision::specs::TowerSpec::new(&policy).map_err(|e| e.to_string())?;
        Ok(Self {
            policy,
            tower: std::sync::Arc::new(tower),
        })
    }
    /// Retained global source scopes; local invocations project them independently.
    pub fn tower(&self) -> &std::sync::Arc<super::vision::specs::TowerSpec> {
        &self.tower
    }
    /// Validates the retained scalar policy without rebuilding parameter scopes.
    pub fn validate(&self) -> Result<(), String> {
        self.policy.validate()
    }
}
impl TryFrom<PixtralPolicy> for PixtralConfig {
    type Error = String;
    fn try_from(policy: PixtralPolicy) -> Result<Self, Self::Error> {
        Self::new(policy)
    }
}
impl std::ops::Deref for PixtralConfig {
    type Target = PixtralPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}
impl PartialEq for PixtralConfig {
    fn eq(&self, other: &Self) -> bool {
        self.policy == other.policy
    }
}
// Family cache identity includes policy debug; retained scopes stay outside that identity.
impl std::fmt::Debug for PixtralConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixtralConfig")
            .field("model_type", &self.policy.model_type)
            .field("hidden_size", &self.policy.hidden_size)
            .field("intermediate_size", &self.policy.intermediate_size)
            .field("num_hidden_layers", &self.policy.num_hidden_layers)
            .field("num_attention_heads", &self.policy.num_attention_heads)
            .field("head_dim", &self.policy.head_dim)
            .field("num_channels", &self.policy.num_channels)
            .field("patch_size", &self.policy.patch_size)
            .field("image_size", &self.policy.image_size)
            .field("hidden_act", &self.policy.hidden_act)
            .field("rope_parameters", &self.policy.rope_parameters)
            .finish()
    }
}

/// Architecture-owned shared text geometry and optional Pixtral context ingress.
#[derive(Debug, Clone)]
pub struct FamilyPolicy {
    /// Positive complete invocation geometry, retained for shortened output budgets.
    pub block_size: NonZeroU32,
    /// Token identity reserved for unresolved canvas positions.
    pub mask_token_id: u32,
    /// Optional independently addressable Pixtral tower.
    pub vision: Option<PixtralConfig>,
    /// Patch groups concatenated along each image axis before projection.
    pub spatial_merge_size: u32,
    text: crate::llama::ModelArgs,
    tensor_fp8: bool,
    identity: String,
    query_scaling: PositionQueryScaling,
    draft_adapter: Option<std::sync::Arc<super::adapter::DraftAdapterDeclaration>>,
    text_projection: bool,
}

/// Validated family policy and its independently acquired spatial projector.
#[derive(Debug, Clone)]
pub struct FamilyConfig {
    policy: FamilyPolicy,
    projector: Option<std::sync::Arc<super::vision::specs::ProjectorSpec>>,
    declaration: std::sync::Arc<crate::decoder::DecoderDeclaration>,
}
impl std::ops::Deref for FamilyConfig {
    type Target = FamilyPolicy;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}

/// Invalid family identity, equations, or admitted configuration geometry.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration decoding failed.
    #[error("invalid Nemotron-Labs-Diffusion configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Reusable dense geometry validation failed.
    #[error(transparent)]
    Geometry(#[from] crate::llama::ConfigError),
    /// Source equations are inconsistent with this family.
    #[error("{0}")]
    Invalid(String),
}

impl FamilyConfig {
    /// Validates source policy and retained draft geometry, and declares the spatial projector once.
    pub fn new(policy: FamilyPolicy) -> Result<Self, ConfigError> {
        policy
            .validate(false)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let projector = policy
            .vision
            .as_ref()
            .map(|vision| {
                super::vision::specs::ProjectorSpec::new(
                    vision,
                    policy.spatial_merge_size,
                    policy.text.hidden_size,
                    policy.text.rms_norm_eps,
                )
                .map(std::sync::Arc::new)
            })
            .transpose()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        let declaration = crate::decoder::DecoderDeclaration::new(&policy)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        policy.validate_text_formats(&declaration)?;
        let config = Self {
            policy,
            projector,
            declaration: std::sync::Arc::new(declaration),
        };
        config
            .validate_config()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        Ok(config)
    }
    /// Retained ordinary text scopes used by admission, invocation geometry and materialization.
    pub fn text_declaration(&self) -> &std::sync::Arc<crate::decoder::DecoderDeclaration> {
        &self.declaration
    }
    /// Editable scalar input; selecting geometry requires constructing a validated configuration.
    pub fn policy(&self) -> &FamilyPolicy {
        &self.policy
    }
    /// Retained projector owner, independent of the image tower's invocation state.
    pub fn projector(&self) -> Option<&std::sync::Arc<super::vision::specs::ProjectorSpec>> {
        self.projector.as_ref()
    }
    /// Projects text execution from an admitted source while retaining its artifact identity.
    pub fn text_only(&self) -> Result<Self, ConfigError> {
        let mut text = self.clone();
        text.policy.vision = None;
        text.projector = None;
        text.policy.text_projection = true;
        text.declaration = std::sync::Arc::new(
            self.declaration
                .with_formats(&text.policy)
                .map_err(|e| ConfigError::Invalid(e.to_string()))?,
        );
        Ok(text)
    }
    /// Parses explicit published geometry without a backend resource or inferred head width.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        Self::from_hf_value(&serde_json::from_slice(bytes)?)
    }

    /// Parses the same family contract from an inspected configuration value.
    pub fn from_hf_value(source: &Value) -> Result<Self, ConfigError> {
        let invalid = |message: &str| ConfigError::Invalid(message.into());
        let identity = source
            .get("model_type")
            .and_then(Value::as_str)
            .filter(|id| {
                matches!(
                    *id,
                    "nemotron_labs_diffusion" | "nemotron_labs_diffusion_vlm"
                )
            })
            .ok_or_else(|| invalid("expected an exact Nemotron-Labs-Diffusion model_type"))?
            .to_owned();
        for (key, expected) in [("hidden_act", "silu"), ("dlm_paradigm", "bidirectional")] {
            if source.get(key).and_then(Value::as_str) != Some(expected) {
                return Err(invalid(
                    "Nemotron-Labs-Diffusion requires SwiGLU and bidirectional masked refinement",
                ));
            }
        }
        for key in ["tie_word_embeddings", "attention_bias", "mlp_bias"] {
            if source.get(key).and_then(Value::as_bool) != Some(false) {
                return Err(invalid(
                    "Nemotron-Labs-Diffusion requires untied, bias-free projections",
                ));
            }
        }
        if source.get("sliding_window").is_some_and(|x| !x.is_null()) {
            return Err(invalid(
                "Nemotron-Labs-Diffusion uses complete causal prefix attention",
            ));
        }
        if source
            .get("head_dim")
            .and_then(Value::as_i64)
            .is_none_or(|x| x <= 0)
        {
            return Err(invalid(
                "Nemotron-Labs-Diffusion requires an explicit positive head_dim",
            ));
        }
        let u32_field = |key: &str| {
            source
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|x| u32::try_from(x).ok())
                .ok_or_else(|| invalid("invalid block or mask identity"))
        };
        let block_size = NonZeroU32::new(u32_field("block_size")?)
            .ok_or_else(|| invalid("zero diffusion block size"))?;
        let mask_token_id = u32_field("mask_token_id")?;
        let rope = source
            .get("rope_parameters")
            .filter(|x| x.is_object())
            .ok_or_else(|| {
                invalid("Nemotron-Labs-Diffusion requires explicit rotary parameters")
            })?;
        if source.get("rope_scaling").is_some_and(|x| x != rope) {
            return Err(invalid("conflicting rotary parameter aliases"));
        }
        if rope.get("rope_type").and_then(Value::as_str) != Some("yarn") {
            return Err(invalid(
                "Nemotron-Labs-Diffusion requires YaRN rotary encoding",
            ));
        }
        let beta = rope
            .get("llama_4_scaling_beta")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("missing position-dependent query scaling"))?
            as f32;
        let original_max_positions = rope
            .get("original_max_position_embeddings")
            .and_then(Value::as_i64)
            .and_then(|x| i32::try_from(x).ok())
            .ok_or_else(|| invalid("invalid original rotary extent"))?;
        let query_scaling = PositionQueryScaling {
            beta,
            original_max_positions,
        };
        query_scaling
            .validate()
            .map_err(|error| ConfigError::Invalid(error.to_string()))?;
        let theta = rope
            .get("rope_theta")
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite() && *x > 0.0)
            .ok_or_else(|| invalid("invalid rotary base"))?;
        if source
            .get("rope_theta")
            .is_some_and(|x| x.as_f64() != Some(theta))
        {
            return Err(invalid("conflicting rotary base aliases"));
        }
        // Only dense geometry is reused. Family identity, checkpoint names and
        // invocation semantics remain owned by this configuration.
        let mut dense = source.clone();
        dense["model_type"] = Value::String("mistral".into());
        dense["rope_theta"] = Value::from(theta);
        dense["rope_scaling"] = rope.clone();
        let tensor_fp8 = source
            .get("quantization_config")
            .is_some_and(|v| v["quant_algo"] == "FP8_W8A16");
        if tensor_fp8 {
            let policy =
                eredu_checkpoint::fp8::TensorFp8Metadata::parse(&source["quantization_config"])
                    .map_err(ConfigError::Invalid)?;
            let actual: std::collections::BTreeSet<_> =
                policy.excluded().iter().map(String::as_str).collect();
            let expected = std::collections::BTreeSet::from([
                "diffusion_head",
                "encoder.embed_tokens",
                r"re:encoder\.multi_modal_projector.*",
                r"re:encoder\.vision_tower.*",
            ]);
            if actual != expected {
                return Err(invalid(
                    "tensor FP8 exclusions differ from the shared text projection catalog",
                ));
            }
            dense
                .as_object_mut()
                .expect("validated configuration object")
                .remove("quantization_config");
        }
        let text = crate::llama::model_args_from_config_value(&dense)?;
        if mask_token_id >= text.vocab_size as u32 {
            return Err(invalid("mask token lies outside the admitted vocabulary"));
        }
        let vision: Option<PixtralConfig> = source
            .get("vision_config")
            .filter(|x| {
                !x.is_null()
                    && !(identity == "nemotron_labs_diffusion"
                        && x.as_object().is_some_and(|v| v.is_empty()))
            })
            .map(|x| serde_json::from_value(x.clone()))
            .transpose()?;
        if (identity == "nemotron_labs_diffusion_vlm") != vision.is_some() {
            return Err(invalid(
                "VLM identity requires a Pixtral tower, and text identity excludes it",
            ));
        }
        let spatial_merge_size = if vision.is_some() {
            u32_field("spatial_merge_size")?
        } else {
            1
        };
        if let Some(vision) = &vision {
            vision.validate().map_err(ConfigError::Invalid)?;
            if spatial_merge_size == 0
                || source.get("projector_hidden_act").and_then(Value::as_str) != Some("gelu")
                || source
                    .get("multimodal_projector_bias")
                    .and_then(Value::as_bool)
                    != Some(false)
                || source.get("vision_feature_layer").and_then(Value::as_i64) != Some(-1)
            {
                return Err(invalid(
                    "invalid Pixtral tower or spatial projector equations",
                ));
            }
            vision
                .hidden_size
                .checked_mul(
                    i32::try_from(spatial_merge_size)
                        .map_err(|_| invalid("projector merge width overflow"))?,
                )
                .and_then(|x| x.checked_mul(spatial_merge_size as i32))
                .ok_or_else(|| invalid("projector input width overflow"))?;
        }
        Self::new(FamilyPolicy {
            block_size,
            mask_token_id,
            vision,
            spatial_merge_size,
            text,
            tensor_fp8,
            identity,
            query_scaling,
            draft_adapter: None,
            text_projection: false,
        })
    }

    /// Whether source text projections require scalar E4M3-to-F32 conversion.
    pub fn tensor_fp8(&self) -> bool {
        self.tensor_fp8
    }

    /// Shared dense geometry for exact declarative schema construction.
    pub(crate) fn dense(&self) -> &crate::llama::ModelArgs {
        &self.text
    }
    /// Optional separately sourced draft ancestry and scalar policy.
    pub fn draft_adapter(&self) -> Option<&super::adapter::DraftAdapterConfig> {
        self.draft_adapter.as_ref().map(|d| d.policy())
    }
    /// Source-owned draft scopes, shared by admission and independently placed invocations.
    pub fn draft_parameters(
        &self,
    ) -> Option<&std::sync::Arc<super::adapter::DraftAdapterDeclaration>> {
        self.draft_adapter.as_ref()
    }
    /// Validates and declares an adapter for this backbone before residency selection.
    pub fn with_draft_adapter(
        self,
        adapter: super::adapter::DraftAdapterConfig,
    ) -> Result<Self, ConfigError> {
        let declaration = super::adapter::DraftAdapterDeclaration::new(&self, adapter)
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        self.with_draft_parameters(std::sync::Arc::new(declaration))
            .map_err(ConfigError::Invalid)
    }
    pub(crate) fn with_draft_parameters(
        mut self,
        declaration: std::sync::Arc<super::adapter::DraftAdapterDeclaration>,
    ) -> Result<Self, String> {
        declaration.validate_for(&self).map_err(|e| e.to_string())?;
        self.policy.draft_adapter = Some(declaration);
        Ok(self)
    }

    /// Applies exact canonical matrix encodings without creating another parameter set.
    pub fn with_checkpoint_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, ConfigError> {
        let mut config = self.clone();
        config.policy.text = crate::llama::with_checkpoint_formats(&config.text, formats)
            .map_err(ConfigError::Invalid)?;
        config.declaration = std::sync::Arc::new(
            self.declaration
                .with_formats(&config.policy)
                .map_err(|e| ConfigError::Invalid(e.to_string()))?,
        );
        config.policy.validate_text_formats(&config.declaration)?;
        Ok(config)
    }
}

impl FamilyPolicy {
    fn validate_text_formats(
        &self,
        declaration: &crate::decoder::DecoderDeclaration,
    ) -> Result<(), ConfigError> {
        use eredu_nn::DeclareLogicalParameters;
        let declared = declaration
            .logical_parameters()
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        for name in self
            .text
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(self.text.quantized_weights.iter().flatten())
        {
            let slot = declared
                .iter()
                .find(|p| p.parameter.id.as_str() == name && p.linear_format.is_some())
                .ok_or_else(|| {
                    ConfigError::Invalid(format!("{name} is not a declared shared text matrix"))
                })?;
            eredu_checkpoint::schema::matrix_for_linear_format(
                name,
                std::iter::empty::<String>(),
                slot.shape.clone(),
                self.linear_format(name),
                None,
            )
            .map_err(|e| ConfigError::Invalid(e.to_string()))?;
        }
        Ok(())
    }
    fn policy(&self) -> &Self {
        self
    }
    /// Stable geometry identity shared by causal and transient block invocations.
    pub fn architecture_fingerprint(&self) -> String {
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "nemotron_labs_diffusion",
            [
                (
                    "text",
                    crate::llama::prompt_cache_architecture_fingerprint(&self.text),
                ),
                ("identity", self.identity.clone()),
                ("tensor_fp8", self.tensor_fp8.to_string()),
                ("rotary_arithmetic", "input_products".into()),
                ("attention_arithmetic", "input_scores".into()),
                ("block_size", self.block_size.to_string()),
                ("mask", self.mask_token_id.to_string()),
                ("query_beta", self.query_scaling.beta.to_bits().to_string()),
                (
                    "vision",
                    format!("{:?}/{}", self.vision, self.spatial_merge_size),
                ),
            ],
        )
    }
}

macro_rules! impl_family_config {
    ($ty:ty) => {
        impl Config for $ty {
            fn model_family(&self) -> &'static str {
                "nemotron_labs_diffusion"
            }
            fn model_identity(&self) -> &str {
                &self.identity
            }
            fn architecture_fingerprint(&self) -> String {
                FamilyPolicy::architecture_fingerprint(self.policy())
            }
            fn validate_config(&self) -> Result<(), Error> {
                self.policy().validate(self.policy().text_projection)?;
                if let Some(declaration) = &self.policy().draft_adapter {
                    declaration.validate_for(self)?;
                }
                Ok(())
            }
            fn extend_common_parameters(
                &self,
                common: eredu_runtime::ArchitectureParameterDescription,
            ) -> Result<
                eredu_runtime::ArchitectureParameterDescription,
                eredu_runtime::ParallelPlanError,
            > {
                super::parallel::with_draft_parameters(
                    common,
                    self.policy().draft_adapter.as_deref(),
                )
            }
            fn hidden_size(&self) -> i32 {
                self.text.hidden_size
            }
            fn num_hidden_layers(&self) -> i32 {
                self.text.num_hidden_layers
            }
            fn intermediate_size(&self) -> i32 {
                self.text.intermediate_size
            }
            fn num_attention_heads(&self) -> i32 {
                self.text.num_attention_heads
            }
            fn num_key_value_heads(&self) -> i32 {
                self.text.num_key_value_heads
            }
            fn head_dim(&self) -> i32 {
                self.text.head_dim
            }
            fn rms_norm_epsilon(&self) -> f32 {
                self.text.rms_norm_eps
            }
            fn vocabulary_size(&self) -> i32 {
                self.text.vocab_size
            }
            fn attention_bias(&self, _: AttentionProjection) -> bool {
                false
            }
            fn mlp_bias(&self) -> bool {
                false
            }
            fn tie_word_embeddings(&self) -> bool {
                false
            }
            fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
                &self.text.attention_schedule
            }
            fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
                self.text.weight_quantization_for(name)
            }
            fn rotary_spec(&self, dimensions: i32) -> RotarySpec {
                let mut spec = self.text.rotary_spec(dimensions);
                // Published tables include YaRN amplitude before conversion to the query dtype.
                spec.arithmetic = eredu_nn::RotaryArithmetic::InputProducts;
                spec
            }
            fn attention_arithmetic(&self) -> eredu_nn::AttentionArithmetic {
                // The published eager equation rounds scores and probabilities at input precision.
                eredu_nn::AttentionArithmetic::InputScores
            }
            fn query_position_scaling(&self) -> Option<PositionQueryScaling> {
                Some(self.query_scaling)
            }
        }
    };
}
impl_family_config!(FamilyPolicy);
impl_family_config!(FamilyConfig);

impl PartitionedConfig for FamilyConfig {
    fn restore_global_geometry(&mut self, global: &Self) -> Result<(), Error> {
        match (&self.policy.draft_adapter, &global.policy.draft_adapter) {
            (Some(local), Some(source)) if local.policy() == source.policy() => {}
            (None, None) => {}
            _ => return Err(Error::backend("local draft adapter changes source policy")),
        }
        let mut restored = self.clone();
        restored.policy.text.num_attention_heads = global.num_attention_heads();
        restored.policy.text.num_key_value_heads = global.num_key_value_heads();
        restored.policy.text.intermediate_size = global.intermediate_size();
        restored.policy.draft_adapter = global.policy.draft_adapter.clone();
        restored.declaration =
            std::sync::Arc::new(global.declaration.with_formats(&restored.policy)?);
        restored.validate_config()?;
        *self = restored;
        Ok(())
    }
    fn partition_parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, eredu_runtime::ParallelPlanError>
    {
        super::parallel::parameter_description(self)
    }
    fn set_local_geometry(
        &mut self,
        query_heads: i32,
        key_value_heads: i32,
        intermediate: i32,
    ) -> Result<(), Error> {
        let mut local = self.clone();
        local.policy.text.num_attention_heads = query_heads;
        local.policy.text.num_key_value_heads = key_value_heads;
        local.policy.text.intermediate_size = intermediate;
        local.policy.draft_adapter = self
            .policy
            .draft_adapter
            .as_ref()
            .map(|d| {
                if query_heads == self.num_attention_heads() {
                    Ok(std::sync::Arc::clone(d))
                } else {
                    d.project_heads(query_heads).map(std::sync::Arc::new)
                }
            })
            .transpose()?;
        local.validate_config()?;
        let geometry = (0..self.num_hidden_layers() as usize)
            .map(|layer| {
                let source = self.declaration.block_parameters(layer).attention.geometry;
                Ok((
                    eredu_nn::AttentionHeadGeometry::new(
                        query_heads,
                        key_value_heads,
                        source.key_dimensions(),
                        source.value_dimensions(),
                    )?,
                    intermediate,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        local.declaration = std::sync::Arc::new(self.declaration.local_declaration(
            self,
            &local.policy,
            &geometry,
        )?);
        *self = local;
        Ok(())
    }
}

impl FamilyPolicy {
    fn validate(&self, text_projection: bool) -> Result<(), Error> {
        self.text.validate().map_err(Error::backend)?;
        self.query_scaling.validate()?;
        if self.mask_token_id >= self.text.vocab_size as u32
            || if text_projection {
                self.vision.is_some()
            } else {
                (self.identity == "nemotron_labs_diffusion_vlm") != self.vision.is_some()
            }
        {
            return Err(Error::backend(
                "inconsistent diffusion vocabulary or family identity",
            ));
        }
        if let Some(vision) = &self.vision {
            vision.validate().map_err(Error::backend)?;
            let merge = i32::try_from(self.spatial_merge_size).map_err(Error::backend)?;
            if merge <= 0
                || vision
                    .hidden_size
                    .checked_mul(merge)
                    .and_then(|x| x.checked_mul(merge))
                    .is_none()
            {
                return Err(Error::backend("invalid spatial merge geometry"));
            }
        }
        Ok(())
    }
}
