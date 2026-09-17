//! Normalized K2 Horizon equation and checkpoint policy.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Deref;

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{GroupScoring, RotarySpec, TopKGroupSelectionSpec};
use serde::Deserialize;

use crate::decoder::{AttentionProjection, Config, OutputGateActivation};
use crate::rotary::RopeValue;

/// Invalid K2 Horizon geometry, field type, or equation policy.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Malformed configuration.
    #[error("invalid K2 Horizon configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Inconsistent architecture policy.
    #[error("invalid K2 Horizon configuration: {0}")]
    Invalid(String),
}

/// Published scalar configuration fields. Use the parser to validate them.
#[derive(Clone, Deserialize)]
pub struct Configuration {
    /// Model type, always `k2_horizon`.
    pub model_type: String,
    /// Hidden activation width, independent of projected head width.
    pub hidden_size: i32,
    /// Logical decoder blocks.
    pub num_hidden_layers: i32,
    /// Dense SwiGLU width.
    pub intermediate_size: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Cached key/value head count.
    pub num_key_value_heads: i32,
    /// Explicit per-head width.
    pub head_dim: i32,
    /// Rotary paired width; normalized to head_dim when omitted.
    #[serde(default)]
    pub rope_head_dim: Option<i32>,
    /// RMS stability constant.
    pub rms_norm_eps: f32,
    /// Number of independent block/final normalization groups.
    #[serde(default = "one")]
    pub layernorm_num_groups: i32,
    /// Distinct Q/K normalization scale per head and feature.
    #[serde(default)]
    pub query_key_norm: bool,
    /// Vocabulary rows.
    pub vocab_size: i32,
    /// Configured context limit.
    pub max_position_embeddings: i32,
    /// Rotary frequency base, overridden by rope_parameters.rope_theta.
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    /// External rotary parameters retained in cache identity.
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeValue>>,
    /// Older spelling of rotary parameters.
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Attention projection biases.
    #[serde(default)]
    pub attention_bias: bool,
    /// Separate attention-output gating activation.
    #[serde(default)]
    pub attention_gate_func: Option<String>,
    /// Shared embedding/output matrix.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Every nth layer may use routing, counted from one.
    #[serde(default = "one")]
    pub decoder_sparse_step: i32,
    /// Zero-based dense overrides.
    #[serde(default)]
    pub mlp_only_layers: Option<Vec<usize>>,
    /// Feed-forward expert count.
    #[serde(default)]
    pub num_experts: i32,
    /// Selected feed-forward experts per token.
    #[serde(default)]
    pub num_experts_per_tok: i32,
    /// Per-expert SwiGLU width.
    #[serde(default)]
    pub moe_intermediate_size: i32,
    /// Always-on expert count, stored as one wider SwiGLU.
    #[serde(default)]
    pub num_shared_experts: i32,
    /// Feed-forward selected coefficient normalization.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Selection-only router correction bias.
    #[serde(default)]
    pub moe_gate_bias: bool,
    /// Router score function.
    #[serde(default = "default_scoring")]
    pub router_score_func: String,
    /// Coefficient multiplier; null means one.
    #[serde(default)]
    pub router_scaling_factor: Option<f32>,
    /// Independently routed value expert count.
    #[serde(default)]
    pub mova_num_experts: i32,
    /// Selected value experts per token.
    #[serde(default)]
    pub mova_num_experts_per_tok: i32,
    #[serde(default = "default_activation")]
    hidden_act: String,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    sliding_window: Option<u32>,
}

impl std::fmt::Debug for Configuration {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_identity_fields(output, false)
    }
}
impl Configuration {
    // A single field producer preserves ordinary derived-Debug encoding while
    // the cache fingerprint excludes only the two separately sorted maps.
    fn fmt_identity_fields(
        &self,
        output: &mut std::fmt::Formatter<'_>,
        without_rope: bool,
    ) -> std::fmt::Result {
        output
            .debug_struct("Configuration")
            .field("model_type", &self.model_type)
            .field("hidden_size", &self.hidden_size)
            .field("num_hidden_layers", &self.num_hidden_layers)
            .field("intermediate_size", &self.intermediate_size)
            .field("num_attention_heads", &self.num_attention_heads)
            .field("num_key_value_heads", &self.num_key_value_heads)
            .field("head_dim", &self.head_dim)
            .field("rope_head_dim", &self.rope_head_dim)
            .field("rms_norm_eps", &self.rms_norm_eps)
            .field("layernorm_num_groups", &self.layernorm_num_groups)
            .field("query_key_norm", &self.query_key_norm)
            .field("vocab_size", &self.vocab_size)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .field("rope_theta", &self.rope_theta)
            .field(
                "rope_parameters",
                &if without_rope {
                    None
                } else {
                    self.rope_parameters.as_ref()
                },
            )
            .field(
                "rope_scaling",
                &if without_rope {
                    None
                } else {
                    self.rope_scaling.as_ref()
                },
            )
            .field("attention_bias", &self.attention_bias)
            .field("attention_gate_func", &self.attention_gate_func)
            .field("tie_word_embeddings", &self.tie_word_embeddings)
            .field("decoder_sparse_step", &self.decoder_sparse_step)
            .field("mlp_only_layers", &self.mlp_only_layers)
            .field("num_experts", &self.num_experts)
            .field("num_experts_per_tok", &self.num_experts_per_tok)
            .field("moe_intermediate_size", &self.moe_intermediate_size)
            .field("num_shared_experts", &self.num_shared_experts)
            .field("norm_topk_prob", &self.norm_topk_prob)
            .field("moe_gate_bias", &self.moe_gate_bias)
            .field("router_score_func", &self.router_score_func)
            .field("router_scaling_factor", &self.router_scaling_factor)
            .field("mova_num_experts", &self.mova_num_experts)
            .field("mova_num_experts_per_tok", &self.mova_num_experts_per_tok)
            .field("hidden_act", &self.hidden_act)
            .field("use_sliding_window", &self.use_sliding_window)
            .field("sliding_window", &self.sliding_window)
            .finish()
    }
}

fn one() -> i32 {
    1
}
fn default_theta() -> f32 {
    10_000.0
}
fn default_scoring() -> String {
    "softmax".into()
}
fn default_activation() -> String {
    "silu".into()
}

/// Complete normalized configuration shared by inspection and execution.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    pub(crate) value_output_start: i32,
    // Shared and routed projections can have different encoding alignments and
    // therefore different local widths, even when their global widths agree.
    pub(crate) local_shared_intermediate_size: Option<i32>,
    pub(crate) fields: Configuration,
    pub(crate) schedule: LayerSchedule<AttentionPolicy>,
    pub(crate) dense_layers: BTreeSet<usize>,
    pub(crate) rotary: eredu_nn::RotaryAlgorithm,
    pub(crate) formats: BTreeMap<String, LinearFormat>,
    pub(crate) fp8: Option<eredu_checkpoint::fp8::BlockFp8Metadata>,
}

impl Deref for ModelArgs {
    type Target = Configuration;
    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

/// Which independently routed equation owns a bank inside a logical block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertBank {
    /// Activated value projections, mixed before writing the ordinary KV cache.
    AttentionValue,
    /// Routed SwiGLU plus an independently evaluated shared SwiGLU.
    FeedForward,
}

impl ModelArgs {
    pub(crate) fn value_output_range(&self) -> std::ops::Range<i32> {
        self.value_output_start..self.value_output_start + self.num_key_value_heads * self.head_dim
    }
    /// Exact sparse schedule, including dense overrides and one-based cadence.
    pub fn is_sparse_layer(&self, layer: usize) -> bool {
        layer < self.num_hidden_layers as usize
            && self.num_experts > 0
            && !self.dense_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step as usize)
    }
    /// Whether any layer uses routed feed-forward experts.
    pub fn is_moe(&self) -> bool {
        (0..self.num_hidden_layers as usize).any(|layer| self.is_sparse_layer(layer))
    }
    /// Whether this layer uses routed activated value projections.
    pub fn is_mova_layer(&self, layer: usize) -> bool {
        self.is_sparse_layer(layer) && self.mova_num_experts > 0
    }
    /// Exact distinct router policies. Value top-1 coefficients are not normalized.
    pub fn routing_spec(
        &self,
        bank: ExpertBank,
    ) -> Result<TopKGroupSelectionSpec, eredu_nn::Error> {
        let (count, top_k, normalize) = match bank {
            ExpertBank::AttentionValue => (
                self.mova_num_experts,
                self.mova_num_experts_per_tok,
                self.mova_num_experts_per_tok > 1,
            ),
            ExpertBank::FeedForward => (
                self.num_experts,
                self.num_experts_per_tok,
                self.norm_topk_prob,
            ),
        };
        TopKGroupSelectionSpec::new(
            count,
            top_k,
            match self.router_score_func.as_str() {
                "sigmoid" => GroupScoring::Sigmoid,
                "softmax" => GroupScoring::Softmax,
                _ => unreachable!("validated routing function"),
            },
            normalize,
        )?
        .with_weight_policy(0.0, self.router_scaling_factor.unwrap_or(1.0))
    }
    /// Physical encoding of one exact checkpoint parameter.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.linear_format_for(name).weight_quantization()
    }
    /// Complete operator encoding, including FP8 scale companions.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.formats
            .get(name)
            .copied()
            .unwrap_or(LinearFormat::Dense)
    }
    /// Exact publisher FP8 companion policy, if present.
    pub fn fp8_metadata(&self) -> Option<&eredu_checkpoint::fp8::BlockFp8Metadata> {
        self.fp8.as_ref()
    }
    /// Validates all construction geometry before allocating backend values.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_with_diagnostic(|text| ConfigError::Invalid(text.to_string()))
    }

    pub(crate) fn validate_with_diagnostic<E>(
        &self,
        invalid: impl Fn(std::fmt::Arguments<'_>) -> E,
    ) -> Result<(), E> {
        for value in [
            self.hidden_size,
            self.num_hidden_layers,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.layernorm_num_groups,
            self.decoder_sparse_step,
        ] {
            if value <= 0 {
                return Err(invalid(format_args!(
                    "dimensions, group count and sparse cadence must be positive"
                )));
            }
        }
        if self.model_type != "k2_horizon" || self.hidden_act != "silu" {
            return Err(invalid(format_args!(
                "expected k2_horizon with silu feed-forward activation"
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.hidden_size % self.layernorm_num_groups != 0
        {
            return Err(invalid(format_args!(
                "nonintegral attention or normalization groups"
            )));
        }
        let rotary = self.rope_head_dim.unwrap_or(self.head_dim);
        if self.head_dim % 2 != 0 || rotary <= 0 || rotary > self.head_dim || rotary % 2 != 0 {
            return Err(invalid(format_args!(
                "rotary paired width must be positive, even and no wider than the head"
            )));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err(invalid(format_args!("invalid RMS epsilon or rotary base")));
        }
        if self
            .dense_layers
            .iter()
            .any(|&layer| layer >= self.num_hidden_layers as usize)
        {
            return Err(invalid(format_args!(
                "mlp_only_layers contains an out-of-range layer"
            )));
        }
        if self.schedule.len() != self.num_hidden_layers as usize {
            return Err(invalid(format_args!(
                "attention schedule differs from layer count"
            )));
        }
        if !matches!(self.router_score_func.as_str(), "sigmoid" | "softmax") {
            return Err(invalid(format_args!(
                "router_score_func must be sigmoid or softmax"
            )));
        }
        if self
            .attention_gate_func
            .as_deref()
            .is_some_and(|a| !matches!(a, "silu" | "softplus"))
        {
            return Err(invalid(format_args!("invalid attention gate activation")));
        }
        let scale = self.router_scaling_factor.unwrap_or(1.0);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid(format_args!(
                "router scale must be finite and positive"
            )));
        }
        if self.num_shared_experts < 0 || self.moe_intermediate_size < 0 {
            return Err(invalid(format_args!("negative expert geometry")));
        }
        for (count, top_k) in [
            (self.num_experts, self.num_experts_per_tok),
            (self.mova_num_experts, self.mova_num_experts_per_tok),
        ] {
            if count < 0 || top_k < 0 || top_k > count || (count > 0 && top_k == 0) {
                return Err(invalid(format_args!("invalid expert count or top-k")));
            }
        }
        if (self.num_experts > 0 && self.moe_intermediate_size == 0)
            || (self.num_experts == 0 && (self.mova_num_experts > 0 || self.num_shared_experts > 0))
        {
            return Err(invalid(format_args!(
                "expert banks require routed feed-forward geometry"
            )));
        }
        for (a, b) in [
            (self.num_attention_heads, self.head_dim),
            (self.num_key_value_heads, self.head_dim),
            (self.moe_intermediate_size, self.num_shared_experts),
            (self.moe_intermediate_size, 2),
        ] {
            a.checked_mul(b)
                .ok_or_else(|| invalid(format_args!("projected dimensions exceed i32")))?;
        }
        self.rotary.validate_fixed().map_err(|_| {
            invalid(format_args!(
                "invalid normalized rotary algorithm: {:?}",
                self.rotary
            ))
        })?;
        for format in self.formats.values() {
            format
                .validate_fixed()
                .map_err(|error| invalid(format_args!("{error}")))?;
        }
        Ok(())
    }
}

/// Parses an official dense, MoE, or MoVA configuration without backend access.
pub fn model_args_from_config_value(value: &serde_json::Value) -> Result<ModelArgs, ConfigError> {
    let mut fields: Configuration = serde_json::from_value(value.clone())?;
    if fields.num_hidden_layers <= 0 || fields.num_hidden_layers > 1_000_000 {
        return Err(ConfigError::Invalid("invalid layer count".into()));
    }
    if fields.rope_parameters.is_some()
        && fields.rope_scaling.is_some()
        && fields.rope_parameters != fields.rope_scaling
    {
        return Err(ConfigError::Invalid(
            "conflicting rope_parameters and rope_scaling".into(),
        ));
    }
    let rope = fields
        .rope_parameters
        .as_ref()
        .or(fields.rope_scaling.as_ref());
    if let Some(theta) = rope.and_then(|r| r.get("rope_theta")) {
        fields.rope_theta = match theta {
            RopeValue::Float(theta) => *theta,
            _ => return Err(ConfigError::Invalid("rope_theta must be numeric".into())),
        };
    }
    let rotary = crate::rotary::normalize_algorithm(rope).map_err(ConfigError::Invalid)?;
    let layers = fields.num_hidden_layers as usize;
    let schedule = if fields.use_sliding_window {
        LayerSchedule::all_sliding(
            layers,
            fields
                .sliding_window
                .ok_or_else(|| ConfigError::Invalid("missing sliding window".into()))?,
        )
    } else {
        LayerSchedule::all_full(layers)
    }
    .map_err(|e| ConfigError::Invalid(e.to_string()))?;
    let mut args = ModelArgs {
        value_output_start: 0,
        local_shared_intermediate_size: None,
        dense_layers: fields
            .mlp_only_layers
            .as_deref()
            .unwrap_or_default()
            .iter()
            .copied()
            .collect(),
        fields,
        schedule,
        rotary,
        formats: BTreeMap::new(),
        fp8: value
            .get("quantization_config")
            .filter(|v| !v.is_null())
            .map(eredu_checkpoint::fp8::BlockFp8Metadata::parse)
            .transpose()
            .map_err(ConfigError::Invalid)?,
    };
    args.validate()?;
    super::formats::normalize_source_formats(&mut args)?;
    Ok(args)
}

/// Stable identity binds equations, schedule and exact per-parameter formats.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    prompt_cache_architecture_fingerprint_with_metadata(
        args,
        crate::decoder::identity::Metadata::new(None),
    )
    .expect("ordinary fingerprint formatting is infallible")
}
fn prompt_cache_architecture_fingerprint_with_metadata(
    args: &ModelArgs,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<String, eredu_nn::Error> {
    struct Fields<'a>(&'a Configuration);
    impl std::fmt::Debug for Fields<'_> {
        fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt_identity_fields(output, true)
        }
    }
    struct Rope<'a>(Vec<(&'a String, &'a RopeValue)>);
    impl std::fmt::Debug for Rope<'_> {
        fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            output.debug_map().entries(self.0.iter().copied()).finish()
        }
    }
    let rope = match args.rope_parameters.as_ref().or(args.rope_scaling.as_ref()) {
        None => None,
        Some(source) => {
            let mut rows = metadata.vector(source.len())?;
            rows.extend(source.iter());
            rows.sort_unstable_by_key(|(key, _)| key.as_str());
            Some(Rope(rows))
        }
    };
    metadata.fingerprint("k2_horizon", || {
        Ok([(
            "equation",
            metadata.format(format_args!(
                "input-score-softmax-rounding-sequential-groups-v1|{:?}|{rope:?}|{:?}|{:?}|{:?}",
                Fields(&args.fields),
                args.schedule,
                args.formats,
                args.fp8
            ))?,
        )])
    })
}

impl Config for ModelArgs {
    fn weight_quantization_with_metadata(
        &self,
        name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<WeightQuantization>, eredu_nn::Error> {
        Ok(self.weight_quantization_for(name))
    }

    fn linear_format_with_metadata(
        &self,
        name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<eredu_checkpoint::LinearFormat, eredu_nn::Error> {
        Ok(self.linear_format(name))
    }

    fn parameter_alias_with_metadata(
        &self,
        _name: &str,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, eredu_nn::Error> {
        Ok(None)
    }
    fn block_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, eredu_nn::Error> {
        Ok(None)
    }
    fn attention_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, eredu_nn::Error> {
        Ok(None)
    }
    fn feed_forward_output_normalization_with_metadata(
        &self,
        _layer: usize,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<String>, eredu_nn::Error> {
        Ok(None)
    }
    fn rotary_spec_with_metadata(
        &self,
        dimensions: i32,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RotarySpec, eredu_nn::Error> {
        Ok(self.rotary_spec(dimensions))
    }

    fn attention_arithmetic(&self) -> eredu_nn::AttentionArithmetic {
        eredu_nn::AttentionArithmetic::InputScores
    }
    fn model_family(&self) -> &'static str {
        "k2_horizon"
    }
    fn model_identity(&self) -> &str {
        "k2_horizon"
    }
    fn architecture_fingerprint(&self) -> String {
        prompt_cache_architecture_fingerprint(self)
    }
    fn architecture_fingerprint_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<String, eredu_nn::Error> {
        prompt_cache_architecture_fingerprint_with_metadata(
            self,
            crate::decoder::identity::Metadata::new(Some(context)),
        )
    }
    fn validate_config(&self) -> Result<(), eredu_nn::Error> {
        self.validate().map_err(eredu_nn::Error::backend)
    }
    fn validate_config_with_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        self.validate_with_diagnostic(|text| {
            context.metadata_error(format_args!("invalid K2 Horizon configuration: {text}"))
        })
    }
    fn routed_observation_points(
        &self,
        unit: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoints> {
        if !self.is_sparse_layer(layer) {
            return None;
        }
        let points = eredu_runtime::RoutedObservationPoints::new(
            ExpertBank::FeedForward.id(),
            format!("{unit}.mlp"),
            self.num_experts,
        );
        Some(if self.is_mova_layer(layer) {
            points
                .with_bank(
                    ExpertBank::AttentionValue.id(),
                    format!("{unit}.self_attn.values"),
                    self.mova_num_experts,
                )
                .expect("distinct K2 bank identities")
        } else {
            points
        })
    }
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_key_value_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_eps
    }
    fn normalization_groups(&self) -> Option<i32> {
        Some(self.layernorm_num_groups)
    }
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        self.query_key_norm.then_some(self.rms_norm_eps)
    }
    fn query_key_norm_per_head_weights(&self) -> bool {
        self.query_key_norm
    }
    fn vocabulary_size(&self) -> i32 {
        self.vocab_size
    }
    fn attention_bias(&self, _: AttentionProjection) -> bool {
        self.attention_bias
    }
    fn mlp_bias(&self) -> bool {
        false
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.schedule
    }
    fn linear_format(&self, name: &str) -> LinearFormat {
        self.linear_format_for(name)
    }
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization> {
        self.weight_quantization_for(name)
    }
    fn rotary_spec(&self, _: i32) -> RotarySpec {
        RotarySpec {
            arithmetic: eredu_nn::RotaryArithmetic::InputProducts,
            dimensions: self.rope_head_dim.unwrap_or(self.head_dim),
            base: self.rope_theta,
            traditional: false,
            algorithm: self.rotary,
        }
    }
    fn rotary_pair_dimensions(&self) -> i32 {
        self.rope_head_dim.unwrap_or(self.head_dim)
    }
    fn external_attention_value(&self, layer: usize) -> bool {
        self.is_mova_layer(layer)
    }
    fn attention_value_format(&self, layer: usize) -> LinearFormat {
        self.linear_format_for(&attention_value_name(self, layer).to_string())
    }
    fn attention_value_format_with_metadata(
        &self,
        layer: usize,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<LinearFormat, eredu_nn::Error> {
        let name = attention_value_name(self, layer);
        Ok(self.linear_format_for(&context.metadata_string(format_args!("{name}"))?))
    }
    fn attention_output_gate(&self) -> Option<(&str, OutputGateActivation)> {
        self.attention_gate_func.as_deref().map(|activation| {
            (
                "gate_proj",
                match activation {
                    "silu" => OutputGateActivation::Silu,
                    "softplus" => OutputGateActivation::Softplus(std::f32::consts::LN_2),
                    _ => unreachable!("validated attention gate"),
                },
            )
        })
    }
}

fn attention_value_name(
    args: &ModelArgs,
    layer: usize,
) -> crate::decoder::parameter_metadata::LayerParameterName<'_> {
    let field = if args.is_mova_layer(layer) {
        "v_experts"
    } else {
        "v_proj"
    };
    crate::decoder::parameter_metadata::LayerParameterName {
        root: "model",
        layer,
        module: Some("self_attn"),
        field,
    }
}
