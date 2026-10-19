//! Exact released assistant admission, separate from ordinary target discovery.

mod construction;
pub use construction::{SparkAssistantConstruction, SparkAssistantLoadConfig};

use eredu_checkpoint::{
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    StoredDtype,
};
use serde_json::Value;

use super::super::{DecoderConfig, MuseIdentity};
use super::ProposalLayout;

/// The assistant architecture tag is authoritative even for a `qwen3` config.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SparkAssistantIdentity {
    /// Causal Muse decoder with a Markov/confidence head and seven sampled slots.
    DSpark,
    /// Qwen-style decoder with dynamic block convolution and a sparse selector.
    DFlash2,
}



impl SparkAssistantIdentity {
    /// Published model architecture marker, independent of the target registry.
    pub const fn architecture(self) -> &'static str {
        match self {
            Self::DSpark => "MuseSparkDSparkModel",
            Self::DFlash2 => "DFlash2DraftModel",
        }
    }
}

/// Assistant-specific equations and auxiliary parameter topology.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SparkAssistantEquation {
    /// Muse attention/norm/residual equations with a vanilla Markov head.
    MarkovConfidence {
        /// Rank of both vocabulary tables; the confidence head consumes this latent.
        rank: usize,
    },
    /// Qwen3 GQA with dynamic input/output convolution and conditional selection.
    ConvolutionSelector {
        /// Number of dynamic depthwise taps within one proposal block.
        taps: usize,
        /// Channels sharing a dynamic kernel coefficient.
        group_size: usize,
        /// Rank of the hidden projection and transition codebooks.
        rank: usize,
        /// Candidate count for each draft slot.
        top_k: usize,
    },
}

/// Normalized released geometry without backend tensors or inference policy.
#[derive(Debug, Clone)]
pub struct SparkAssistantConfig {
    identity: SparkAssistantIdentity,
    equation: SparkAssistantEquation,
    proposal_layout: ProposalLayout,
    hidden: usize,
    intermediate: usize,
    layers: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocabulary: usize,
    block_size: usize,
    mask_token: u32,
    capture_layers: Vec<usize>,
    context: usize,
    window: usize,
}

/// Invalid source declaration or incompatible assistant/target pairing.
#[derive(Debug, thiserror::Error)]
#[error("invalid Muse Spark assistant: {0}")]
pub struct SparkAssistantError(String);

impl SparkAssistantConfig {
    /// Recognizes an authoritative assistant declaration before ordinary target discovery.
    /// Complete parsing still validates the exact marker, model type and geometry.
    pub fn declared_by(value: &Value) -> bool {
        value.get("model_type").and_then(Value::as_str) == Some("muse_spark_dspark")
            || value
                .get("architectures")
                .and_then(Value::as_array)
                .is_some_and(|markers| {
                    markers.iter().any(|marker| {
                        matches!(
                            marker.as_str(),
                            Some("MuseSparkDSparkModel" | "DFlash2DraftModel")
                        )
                    })
                })
    }

    /// Selects assistant identity from the complete architecture marker and validates
    /// its released geometry. A DFlash2 config never becomes a Qwen target.
    pub fn from_hf_value(value: &Value) -> Result<Self, SparkAssistantError> {
        let fail = |message: &str| SparkAssistantError(message.into());
        let architectures = value
            .get("architectures")
            .and_then(Value::as_array)
            .ok_or_else(|| fail("architecture marker is missing"))?;
        let identity = match (
            value.get("model_type").and_then(Value::as_str),
            architectures.as_slice(),
        ) {
            (Some("muse_spark_dspark"), [marker]) if marker == "MuseSparkDSparkModel" => {
                SparkAssistantIdentity::DSpark
            }
            (Some("qwen3"), [marker]) if marker == "DFlash2DraftModel" => {
                SparkAssistantIdentity::DFlash2
            }
            _ => return Err(fail("source does not identify a released Spark assistant")),
        };
        let integer = |source: &Value, name: &str| {
            source
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| fail(&format!("{name} must be a positive integer")))
        };
        let exact = |source: &Value, name: &str, expected: Value| {
            if source.get(name).is_some_and(|actual| {
                if expected.is_number() {
                    actual.is_number() && actual.as_f64() == expected.as_f64()
                } else {
                    actual == &expected
                }
            }) {
                Ok(())
            } else {
                Err(fail(&format!(
                    "{name} disagrees with the released equation"
                )))
            }
        };
        let nested = value
            .get("dflash_config")
            .ok_or_else(|| fail("draft policy is absent"))?;
        let (
            equation,
            proposal_layout,
            intermediate,
            query_heads,
            kv_heads,
            head_dim,
            block_size,
            mask_token,
            capture_layers,
        ) = match identity {
            SparkAssistantIdentity::DSpark => {
                let metap = value
                    .get("metap")
                    .ok_or_else(|| fail("DSpark output scaling policy is absent"))?;
                for (name, expected) in [
                    ("base_width", 256.0.into()),
                    ("metap_mode", "sp".into()),
                    ("router_has_one_inf_dim", true.into()),
                    ("use_metap", true.into()),
                ] {
                    exact(metap, name, expected)?;
                }
                for (name, expected) in [
                    ("draft_position_mode", Value::from("rope")),
                    ("norm_zero_centered_gamma", true.into()),
                    ("output_norm_gain_center_type", "zero".into()),
                    ("post_norm_gain_center_type", "one".into()),
                    ("residual_dtype", "fp32".into()),
                    ("post_norm_affine", false.into()),
                    ("use_attn_o_gate", true.into()),
                    ("use_attn_o_norm", true.into()),
                    ("use_qk_norm", true.into()),
                    ("use_ffn_learnable_scales", true.into()),
                    ("use_residual_gate", true.into()),
                    ("use_rope", true.into()),
                    ("use_sqrt_attn_scaling", false.into()),
                    ("markov_head_type", "vanilla".into()),
                    ("markov_rank", 512.into()),
                    ("enable_confidence_head", true.into()),
                    ("confidence_head_with_markov", true.into()),
                    ("sample_from_anchor", true.into()),
                    ("use_aux_hidden_state", true.into()),
                    ("target_hidden_size", 8192.into()),
                    ("target_num_hidden_layers", 62.into()),
                    ("target_layer_ids", serde_json::json!([13, 29, 45, 57, 61])),
                    ("num_target_layers", 5.into()),
                    ("block_size", 7.into()),
                    ("mask_token_id", 200018.into()),
                    ("rope_theta", 500000.0.into()),
                    ("rms_post_norm_eps", 1e-8.into()),
                    ("output_soft_cap_temp", 20.0.into()),
                    ("sliding_window_non_causal", false.into()),
                    ("full_attention_causal", false.into()),
                ] {
                    exact(value, name, expected)?;
                }
                for (name, expected) in [
                    ("causal", true.into()),
                    ("use_aux_hidden_state", true.into()),
                    ("mask_token_id", 200018.into()),
                    ("target_layer_ids", serde_json::json!([13, 29, 45, 57, 61])),
                ] {
                    exact(nested, name, expected)?;
                }
                exact(
                    value
                        .get("moe")
                        .ok_or_else(|| fail("MoE declaration is absent"))?,
                    "is_enabled",
                    false.into(),
                )?;
                (
                    SparkAssistantEquation::MarkovConfidence { rank: 512 },
                    ProposalLayout::SampleFromAnchor,
                    16384,
                    128,
                    16,
                    64,
                    7,
                    200018,
                    vec![13, 29, 45, 57, 61],
                )
            }
            SparkAssistantIdentity::DFlash2 => {
                for (name, expected) in [
                    ("attention_mode", "gqa".into()),
                    ("block_size", 8.into()),
                    ("conv_group_size", 16.into()),
                    ("conv_kernel_size", 2.into()),
                    ("selector_rank", 256.into()),
                    ("selector_top_k", 16.into()),
                    ("mask_token_id", 200002.into()),
                    ("target_layer_ids", serde_json::json!([1, 16, 30, 44, 59])),
                    ("final_logit_softcapping", 20.into()),
                    ("output_multiplier", 0.1767766952966369.into()),
                ] {
                    exact(nested, name, expected)?;
                }
                exact(value, "num_target_layers", 62.into())?;
                exact(value, "is_causal", false.into())?;
                exact(value, "rope_theta", 500000.into())?;
                let rope = value
                    .get("rope_parameters")
                    .ok_or_else(|| fail("rotary declaration is absent"))?;
                exact(rope, "rope_type", "default".into())?;
                exact(rope, "rope_theta", 500000.into())?;
                (
                    SparkAssistantEquation::ConvolutionSelector {
                        taps: 2,
                        group_size: 16,
                        rank: 256,
                        top_k: 16,
                    },
                    ProposalLayout::AnchorThenMasks,
                    28672,
                    64,
                    8,
                    128,
                    8,
                    200002,
                    vec![1, 16, 30, 44, 59],
                )
            }
        };
        for (name, expected) in [
            ("hidden_size", 8192),
            ("intermediate_size", intermediate),
            ("num_hidden_layers", 5),
            ("num_attention_heads", query_heads),
            ("num_key_value_heads", kv_heads),
            ("head_dim", head_dim),
            ("vocab_size", 202048),
            ("draft_vocab_size", 202048),
            ("max_position_embeddings", 1048576),
            ("sliding_window", 2048),
        ] {
            if integer(value, name)? != expected {
                return Err(fail(&format!("{name} differs from the released geometry")));
            }
        }
        for (name, expected) in [
            ("attention_bias", false.into()),
            ("tie_word_embeddings", false.into()),
            ("hidden_act", "silu".into()),
            ("rms_norm_eps", 1e-5.into()),
            (
                "layer_types",
                serde_json::json!(vec!["sliding_attention"; 5]),
            ),
        ] {
            exact(value, name, expected)?;
        }
        if value.get("attention_dropout").and_then(Value::as_f64) != Some(0.0) {
            return Err(fail("inference dropout must be zero"));
        }
        Ok(Self {
            identity,
            equation,
            proposal_layout,
            hidden: 8192,
            intermediate,
            layers: 5,
            query_heads,
            kv_heads,
            head_dim,
            vocabulary: 202048,
            block_size,
            mask_token,
            capture_layers,
            context: 1048576,
            window: 2048,
        })
    }

    /// Distinct published assistant identity.
    pub const fn identity(&self) -> SparkAssistantIdentity {
        self.identity
    }
    /// Typed decoder and auxiliary-head equations.
    pub const fn equation(&self) -> SparkAssistantEquation {
        self.equation
    }
    /// Meaning of the first position in the proposal block.
    pub const fn proposal_layout(&self) -> ProposalLayout {
        self.proposal_layout
    }
    /// Total positions evaluated per proposal block.
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
    /// Maximum number of candidate tokens, excluding any retained anchor.
    pub const fn proposal_capacity(&self) -> usize {
        match self.proposal_layout {
            ProposalLayout::SampleFromAnchor => self.block_size,
            ProposalLayout::AnchorThenMasks => self.block_size - 1,
        }
    }
    /// Checkpoint-local mask token in the shared target vocabulary.
    pub const fn mask_token_id(&self) -> u32 {
        self.mask_token
    }
    /// Ordered zero-based post-block target states, without a layer-hook offset.
    pub fn capture_layers(&self) -> &[usize] {
        &self.capture_layers
    }

    /// Hidden width of the released decoder and target-state encoder.
    pub const fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Global target and released draft vocabulary size.
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }

    /// Number of independent proposal decoder blocks.
    pub const fn num_hidden_layers(&self) -> usize {
        self.layers
    }

    /// Freezes exact ordered target output paths after proving released compatibility.
    pub fn capture_request(
        &self,
        target: &DecoderConfig,
    ) -> Result<crate::composite_execution::ExternalPredictionCaptureRequest, SparkAssistantError>
    {
        self.prove_compatibility(target)?;
        Ok(
            crate::composite_execution::ExternalPredictionCaptureRequest::Muse {
                target_layers: self.capture_layers.clone().into_boxed_slice(),
                target_paths: self
                    .capture_layers
                    .iter()
                    .map(|layer| format!("model.layers.{layer}.output"))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
        )
    }

    /// Proves target geometry and capture availability before reading payloads.
    pub fn prove_compatibility(&self, target: &DecoderConfig) -> Result<(), SparkAssistantError> {
        if target.identity != MuseIdentity::Spark
            || target.hidden_size as usize != self.hidden
            || target.vocab_size as usize != self.vocabulary
            || target.num_hidden_layers != 62
            || target.max_position_embeddings as usize != self.context
            || self
                .capture_layers
                .iter()
                .any(|&layer| layer >= target.num_hidden_layers as usize)
            || self.mask_token as usize >= self.vocabulary
        {
            return Err(SparkAssistantError(
                "target identity, vocabulary, context or capture geometry differs".into(),
            ));
        }
        Ok(())
    }

    /// Strict physical BF16 source schema, including the assistants' distinct
    /// auxiliary heads. DFlash2 borrows its vocabulary modules from the target.
    pub fn safetensors_plan(&self) -> Result<SafetensorsCheckpointPlan, SparkAssistantError> {
        let mut tensors = Vec::new();
        let mut add = |name: String, shape: Vec<usize>| {
            tensors.push(SafetensorsTensorConstraint::required(
                name,
                shape,
                StoredDtypeConstraint::Exact(StoredDtype::BF16),
            ))
        };
        let h = self.hidden;
        add("fc.weight".into(), vec![h, h * self.capture_layers.len()]);
        add("hidden_norm.weight".into(), vec![h]);
        add("norm.weight".into(), vec![h]);
        for layer in 0..self.layers {
            let root = format!("layers.{layer}");
            for (name, output, input) in [
                ("self_attn.q_proj", self.query_heads * self.head_dim, h),
                ("self_attn.k_proj", self.kv_heads * self.head_dim, h),
                ("self_attn.v_proj", self.kv_heads * self.head_dim, h),
                ("self_attn.o_proj", h, self.query_heads * self.head_dim),
            ] {
                add(format!("{root}.{name}.weight"), vec![output, input]);
            }
            let mlp = match self.identity {
                SparkAssistantIdentity::DSpark => "feed_forward",
                SparkAssistantIdentity::DFlash2 => "mlp",
            };
            for (name, output, input) in [
                ("gate_proj", self.intermediate, h),
                ("up_proj", self.intermediate, h),
                ("down_proj", h, self.intermediate),
            ] {
                add(format!("{root}.{mlp}.{name}.weight"), vec![output, input]);
            }
            match self.equation {
                SparkAssistantEquation::MarkovConfidence { .. } => {
                    for local in [
                        "self_attn.pre_qkv_proj_norm.weight",
                        "pre_ffn_norm.weight",
                        "post_ffn_norm.weight",
                        "add_attn.gate",
                        "add_ffn.gate",
                    ] {
                        add(format!("{root}.{local}"), vec![h]);
                    }
                    add(format!("{root}.self_attn.o_gate_proj.weight"), vec![h, h]);
                }
                SparkAssistantEquation::ConvolutionSelector {
                    taps, group_size, ..
                } => {
                    for norm in ["input_layernorm", "post_attention_layernorm"] {
                        add(format!("{root}.{norm}.weight"), vec![h]);
                    }
                    for norm in ["q_norm", "k_norm"] {
                        add(
                            format!("{root}.self_attn.{norm}.weight"),
                            vec![self.head_dim],
                        );
                    }
                    for convolution in ["attention_conv", "mlp_conv"] {
                        add(
                            format!("{root}.{convolution}.base_kernel"),
                            vec![2, taps, h],
                        );
                        add(
                            format!("{root}.{convolution}.kernel_projection.weight"),
                            vec![2 * taps * (h / group_size), h],
                        );
                    }
                }
            }
        }
        match self.equation {
            SparkAssistantEquation::MarkovConfidence { rank } => {
                add("lm_head.weight".into(), vec![self.vocabulary, h]);
                for name in ["markov_w1", "markov_w2"] {
                    add(
                        format!("markov_head.{name}.weight"),
                        vec![self.vocabulary, rank],
                    );
                }
                add("confidence_head.proj.weight".into(), vec![1, h + rank]);
                add("confidence_head.proj.bias".into(), vec![1]);
                tensors.push(SafetensorsTensorConstraint::required(
                    "d2t",
                    vec![self.vocabulary],
                    StoredDtypeConstraint::Exact(StoredDtype::I64),
                ));
            }
            SparkAssistantEquation::ConvolutionSelector { rank, .. } => {
                add(
                    "candidate_selector.hidden_projection.weight".into(),
                    vec![rank, h],
                );
                for name in ["predecessor_codebook", "successor_codebook"] {
                    add(
                        format!("candidate_selector.{name}"),
                        vec![self.vocabulary, rank],
                    );
                }
            }
        }
        SafetensorsCheckpointPlan::new(
            self.identity.architecture(),
            tensors,
            vec![],
            CatalogPolicy::strict(),
        )
        .map_err(|error| SparkAssistantError(error.to_string()))
    }

    /// Accepted-context attention window, distinct from the configured context limit.
    pub const fn attention_window(&self) -> usize {
        self.window
    }

    /// Normalizes DFlash2's body for the shared DFlash construction and context
    /// driver. Its candidate selector and target compatibility are independent.
    pub fn dflash_decoder_config(&self) -> Result<super::DFlashDecoderConfig, SparkAssistantError> {
        let SparkAssistantEquation::ConvolutionSelector {
            taps, group_size, ..
        } = self.equation
        else {
            return Err(SparkAssistantError(
                "DSpark does not use the bidirectional DFlash body".into(),
            ));
        };
        Ok(super::DFlashDecoderConfig {
            equation: super::DFlashDecoderEquation::Float32ResidualNorm,
            hidden_size: self.hidden as i32,
            intermediate_size: self.intermediate as i32,
            num_hidden_layers: self.layers as i32,
            num_attention_heads: self.query_heads as i32,
            num_key_value_heads: self.kv_heads as i32,
            head_dim: self.head_dim as i32,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            sliding_window: self.window as i32,
            block_size: self.block_size,
            target_layer_ids: self.capture_layers.clone(),
            context_projection: super::DFlashContextProjectionLayout::Root,
            sublayer: super::DFlashSublayerPolicy::BlockConvolution {
                taps: taps as i32,
                group_size: group_size as i32,
            },
            quantization: None,
            quantized_weights: Default::default(),
        })
    }

    /// Replicated global candidate-table geometry for the released DFlash2 head.
    pub fn candidate_selector_geometry(
        &self,
    ) -> Result<super::candidate_selector::CandidateSelectorGeometry, SparkAssistantError> {
        let SparkAssistantEquation::ConvolutionSelector { rank, top_k, .. } = self.equation else {
            return Err(SparkAssistantError(
                "DSpark does not use a conditional candidate selector".into(),
            ));
        };
        Ok(super::candidate_selector::CandidateSelectorGeometry {
            hidden: self.hidden as i32,
            vocabulary: self.vocabulary as i32,
            rank: rank as i32,
            top_k: top_k as i32,
        })
    }

    /// Normalizes the released causal dense assistant independently of target
    /// checkpoint admission. Shared decoder modules consume the resulting geometry.
    pub fn dspark_decoder_config(
        &self,
        target: &DecoderConfig,
    ) -> Result<DecoderConfig, SparkAssistantError> {
        self.prove_compatibility(target)?;
        if self.identity != SparkAssistantIdentity::DSpark {
            return Err(SparkAssistantError(
                "DFlash2 does not use the causal Muse assistant body".into(),
            ));
        }
        let mut args = target.clone();
        args.hidden_size = self.hidden as i32;
        args.num_hidden_layers = self.layers as i32;
        args.intermediate_size = self.intermediate as i32;
        args.num_attention_heads = self.query_heads as i32;
        args.num_key_value_heads = self.kv_heads as i32;
        args.head_dim = self.head_dim as i32;
        args.layer_uses_rope = vec![true; self.layers];
        args.rope_theta = 500_000.0;
        args.rope_scaling = None;
        args.attention_schedule = eredu_core::LayerSchedule::new(
            self.layers,
            vec![
                eredu_core::AttentionPolicy::sliding(self.window as u32)
                    .map_err(|e| SparkAssistantError(e.to_string()))?;
                self.layers
            ],
        )
        .map_err(|e| SparkAssistantError(e.to_string()))?;
        args.qk_scale_factor = (43.7840518911_f64 / (self.head_dim as f64).sqrt()) as f32;
        args.weight_convention = super::super::WeightConvention::HuggingFace;
        args.rms_norm_eps = 1e-5;
        args.post_norm_eps = 1e-8;
        args.block_normalization = super::super::BlockNormalization::InputScaledFeedForward;
        args.residual_equation = super::super::ResidualEquation::Learned {
            temperature: 0.3,
            minimum_variance: 1e-3,
        };
        args.output_multiplier = (self.hidden as f64 / 256.0).powf(-0.5) as f32;
        args.final_logit_softcapping = 20.0;
        args.attention_equation = super::super::AttentionEquation::NormalizedHeadsScoreScaled;
        args.moe_intermediate_size = 0;
        args.num_experts = 0;
        args.num_experts_per_tok = 0;
        args.norm_topk_prob = false;
        args.routed_feed_forward = super::super::RoutedFeedForwardEquation::DirectSoftmax;
        args.quantization = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
        args.vision_config = None;
        args.audio_config = None;
        args.validate()
            .map_err(|e| SparkAssistantError(e.to_string()))?;
        Ok(args)
    }

    /// Full-vocabulary transition-head geometry and exact output scaling.
    pub fn dspark_head_geometry(
        &self,
    ) -> Result<super::dspark::DsparkHeadGeometry, SparkAssistantError> {
        let SparkAssistantEquation::MarkovConfidence { rank } = self.equation else {
            return Err(SparkAssistantError(
                "DFlash2 does not use DSpark transition heads".into(),
            ));
        };
        Ok(super::dspark::DsparkHeadGeometry {
            hidden: self.hidden as i32,
            vocabulary: self.vocabulary as i32,
            rank: rank as i32,
            output_multiplier: (self.hidden as f64 / 256.0).powf(-0.5) as f32,
            soft_cap: 20.0,
        })
    }
}
