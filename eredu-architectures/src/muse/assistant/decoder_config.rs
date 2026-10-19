//! Normalized geometry and equations for the shared DFlash decoder body.

use std::collections::HashMap;

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::Error;

/// Normalization, residual and attention arithmetic of a DFlash decoder.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DFlashDecoderEquation {
    /// Eager activation-precision residual addition and backend attention/rotary.
    InputRounded,
    /// FP32 residual sums feed FP32 RMS/affine arithmetic before one output cast.
    /// Residual stores remain in activation precision; rotary products round in
    /// that precision and attention computes scores and value products in FP32.
    Float32ResidualNorm,
}

impl DFlashDecoderEquation {
    pub(super) fn normalization(
        self,
        dimensions: i32,
        epsilon: f32,
        weight: eredu_nn::ParameterSpec,
    ) -> eredu_nn::NormalizationConstructionSpec {
        let mut spec =
            eredu_nn::NormalizationConstructionSpec::learned(dimensions, epsilon, weight);
        if self == Self::Float32ResidualNorm {
            spec.groups = Some(1);
            spec.reduction = eredu_nn::NormalizationReduction::Cascaded;
        }
        spec
    }
}

/// Input/output processing around each attention and feed-forward sublayer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DFlashSublayerPolicy {
    /// Ordinary normalized input and projected output.
    Direct,
    /// Dynamic depthwise convolution confined to each proposal block.
    BlockConvolution {
        /// Number of causal taps, current position first.
        taps: i32,
        /// Channels sharing a dynamic coefficient.
        group_size: i32,
    },
}

/// Canonical namespace of the target-state projection parameters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DFlashContextProjectionLayout {
    /// `encoder.fc` and `encoder.output_norm_enc`.
    Encoder,
    /// `fc` and `hidden_norm`.
    Root,
}

impl DFlashContextProjectionLayout {
    pub(super) const fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Encoder => ("encoder.fc.weight", "encoder.output_norm_enc.weight"),
            Self::Root => ("fc.weight", "hidden_norm.weight"),
        }
    }
}

/// Backend-neutral construction contract for a bidirectional DFlash body.
/// Released checkpoint admission supplies this geometry separately from target
/// compatibility and proposal sampling policy.
#[derive(Debug, Clone)]
pub struct DFlashDecoderConfig {
    /// Released arithmetic equation, independent of storage or residency.
    pub equation: DFlashDecoderEquation,
    /// Residual width.
    pub hidden_size: i32,
    /// SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Decoder depth.
    pub num_hidden_layers: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Key/value head count.
    pub num_key_value_heads: i32,
    /// Channels per head.
    pub head_dim: i32,
    /// Learned RMS epsilon.
    pub rms_norm_eps: f32,
    /// Rotary base.
    pub rope_theta: f32,
    /// Retained committed-context window.
    pub sliding_window: i32,
    /// Anchor plus mask positions per proposal block.
    pub block_size: usize,
    /// Ordered target states projected into each context row.
    pub target_layer_ids: Vec<usize>,
    /// Canonical context parameter namespace.
    pub context_projection: DFlashContextProjectionLayout,
    /// Shared attention and feed-forward wrapper equation.
    pub sublayer: DFlashSublayerPolicy,
    /// Uniform checkpoint encoding when selected.
    pub quantization: Option<WeightQuantization>,
    /// Exact per-weight encoding overrides.
    pub quantized_weights: HashMap<String, WeightQuantization>,
}

impl DFlashDecoderConfig {
    pub(super) fn validate(&self) -> Result<(), Error> {
        let capture_count = i32::try_from(self.target_layer_ids.len())
            .map_err(|_| Error::backend("DFlash target capture count exceeds i32"))?;
        if self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.num_hidden_layers <= 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.head_dim <= 0
            || self.head_dim % 2 != 0
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self.hidden_size.checked_mul(capture_count).is_none()
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || self.sliding_window <= 0
            || self.block_size < 2
            || i32::try_from(self.block_size).is_err()
            || self.target_layer_ids.is_empty()
            || !self
                .target_layer_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err(Error::backend("invalid normalized DFlash decoder geometry"));
        }
        if let DFlashSublayerPolicy::BlockConvolution { taps, group_size } = self.sublayer {
            if taps <= 0 || group_size <= 0 || self.hidden_size % group_size != 0 {
                return Err(Error::backend(
                    "invalid DFlash sublayer convolution geometry",
                ));
            }
        }
        for format in self
            .quantization
            .iter()
            .chain(self.quantized_weights.values())
        {
            format.validate().map_err(Error::backend)?;
        }
        Ok(())
    }

    pub(super) fn linear_format_for(&self, name: &str) -> LinearFormat {
        self.quantized_weights
            .get(name)
            .copied()
            .or(self.quantization)
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }
}

impl From<super::DFlashConfig> for DFlashDecoderConfig {
    fn from(config: super::DFlashConfig) -> Self {
        Self {
            equation: DFlashDecoderEquation::InputRounded,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            sliding_window: config.sliding_window,
            block_size: config.block_size,
            target_layer_ids: config.target_layer_ids,
            context_projection: DFlashContextProjectionLayout::Encoder,
            sublayer: DFlashSublayerPolicy::Direct,
            quantization: config.quantization,
            quantized_weights: config.quantized_weights,
        }
    }
}
