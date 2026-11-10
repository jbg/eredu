//! Exact released vision geometry, independent of a language-model task.

use eredu_nn::{
    multimodal::{MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec},
    Error,
};
use serde::Deserialize;

/// Token-mixing equation selected by one encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum VisionLayerKind {
    /// Gated spatial depthwise convolution, without temporal state.
    #[serde(rename = "conv")]
    Convolution,
    /// Bidirectional grouped-query attention restricted to one patch frame.
    #[serde(rename = "full_attention")]
    Attention,
}

/// Released hybrid vision encoder geometry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VisionConfig {
    /// Exact checkpoint model type.
    pub model_type: String,
    /// Residual width.
    pub hidden_size: i32,
    /// Dense SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Number of independently executable encoder layers.
    pub num_hidden_layers: usize,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Key/value head count.
    pub num_key_value_heads: i32,
    /// Authoritative encoder invocation schedule.
    pub layer_types: Vec<VisionLayerKind>,
    /// Spatial patch side in pixels.
    pub patch_size: i32,
    /// Frames packed into one patch.
    pub temporal_patch_size: i32,
    /// Pixel channels.
    pub in_channels: i32,
    /// Side length of contiguous merge groups.
    pub spatial_merge_size: i32,
    /// Odd square depthwise kernel side.
    pub conv_kernel_size: i32,
    /// Learned RMSNorm epsilon.
    pub norm_eps: f32,
    /// Explicit axial positional policy.
    pub rope_parameters: AxialRope,
}

/// Axial frequency progression stored in vision checkpoint configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxialRope {
    /// Released rotary type (`axial`).
    pub rope_type: String,
    /// Independent H/W frequency base.
    pub rope_theta: f32,
}

impl VisionConfig {
    /// Validates dimensions, schedule and the exact axial rotary equation.
    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "lfm3_vl_vision"
            || self.num_hidden_layers == 0
            || self.num_hidden_layers != self.layer_types.len()
            || [
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.patch_size,
                self.temporal_patch_size,
                self.in_channels,
                self.spatial_merge_size,
                self.conv_kernel_size,
            ]
            .into_iter()
            .any(|n| n <= 0)
            || self.conv_kernel_size % 2 != 1
            || self.hidden_size % self.num_attention_heads != 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || (self.hidden_size / self.num_attention_heads) % 4 != 0
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
            || self.rope_parameters.rope_type != "axial"
        {
            return Err(Error::backend(
                "invalid LFM3 vision geometry or operator policy",
            ));
        }
        self.patch_width()?;
        self.rotary_spec().dimensions()?;
        Ok(())
    }

    /// Flattened channel/time/patch input width.
    pub fn patch_width(&self) -> Result<i32, Error> {
        self.in_channels
            .checked_mul(self.temporal_patch_size)
            .and_then(|n| n.checked_mul(self.patch_size))
            .and_then(|n| n.checked_mul(self.patch_size))
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("LFM3 vision patch width overflow"))
    }

    pub(super) fn rotary_spec(&self) -> MultiAxisRotarySpec {
        let dimensions = self
            .hidden_size
            .checked_div(self.num_attention_heads)
            .unwrap_or(0)
            / 2;
        MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions,
                    position_offset: 0
                };
                2
            ],
            base: self.rope_parameters.rope_theta,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::SplitHalves,
        }
    }
}

/// Spatial merger and language-width projector geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectorConfig {
    /// Side of each contiguous patch block merged into one token.
    pub downsample_factor: i32,
    /// Intermediate GELU projection width.
    pub hidden_size: i32,
    /// Output language-model width.
    pub text_hidden_size: i32,
    /// Whether both projection layers retain a bias.
    pub bias: bool,
    /// Affine LayerNorm epsilon before the merger.
    pub layer_norm_eps: f32,
}

impl ProjectorConfig {
    /// Reads the projector policy from the parent conditional-model config.
    pub fn from_config_value(value: &serde_json::Value) -> Result<Self, Error> {
        #[derive(Deserialize)]
        struct Source {
            downsample_factor: i32,
            projector_hidden_size: i32,
            projector_bias: bool,
            projector_layer_norm_eps: f32,
            projector_hidden_act: String,
            text_config: Text,
        }
        #[derive(Deserialize)]
        struct Text {
            hidden_size: i32,
        }
        let s: Source = serde_json::from_value(value.clone()).map_err(Error::backend_source)?;
        if s.projector_hidden_act != "gelu" {
            return Err(Error::backend("LFM3 vision projector requires exact GELU"));
        }
        Ok(Self {
            downsample_factor: s.downsample_factor,
            hidden_size: s.projector_hidden_size,
            text_hidden_size: s.text_config.hidden_size,
            bias: s.projector_bias,
            layer_norm_eps: s.projector_layer_norm_eps,
        })
    }

    pub(super) fn input_width(&self, vision: &VisionConfig) -> Result<i32, Error> {
        if self.downsample_factor != vision.spatial_merge_size
            || self.downsample_factor <= 0
            || self.hidden_size <= 0
            || self.text_hidden_size <= 0
            || !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
        {
            return Err(Error::backend(
                "invalid LFM3 vision projector or merge policy",
            ));
        }
        vision
            .hidden_size
            .checked_mul(self.downsample_factor)
            .and_then(|n| n.checked_mul(self.downsample_factor))
            .ok_or_else(|| Error::backend("LFM3 vision merger width overflow"))
    }
}
