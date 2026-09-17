//! Backend-neutral Gemma 4 media-ingress geometry and mask plans.

use std::fmt;

use super::VisionConfig;

/// Additive value used to exclude padded vision keys from attention.
pub const VISION_ATTENTION_INVALID_LOGIT: f32 = -1.0e9;

/// Invalid host-known media geometry supplied to a Gemma 4 ingress plan.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IngressPlanError(String);

impl IngressPlanError {
    fn invalid(detail: impl Into<String>) -> Self {
        Self(detail.into())
    }
}

impl fmt::Display for IngressPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for IngressPlanError {}

fn invalid(detail: fmt::Arguments<'_>) -> IngressPlanError {
    IngressPlanError::invalid(detail.to_string())
}

fn positive<E>(
    value: i32,
    name: &str,
    invalid: impl Fn(fmt::Arguments<'_>) -> E,
) -> Result<i32, E> {
    if value <= 0 {
        Err(invalid(format_args!("Gemma 4 {name} must be positive, got {value}")))
    } else {
        Ok(value)
    }
}

fn checked_product<E>(
    left: i32,
    right: i32,
    name: &str,
    invalid: impl Fn(fmt::Arguments<'_>) -> E,
) -> Result<i32, E> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format_args!("Gemma 4 {name} overflowed")))
}

const fn ceil_half(value: i32) -> i32 {
    value / 2 + value % 2
}

/// Architecture-owned ingress geometry for one prepared image or video frame.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VisionIngressPartPlan {
    /// Padded patch width carried by the prepared payload.
    pub padded_patches: i32,
    /// Valid patch prefix before payload padding.
    pub valid_patches: i32,
    /// Decoder placeholder count after square spatial pooling.
    pub decoder_positions: i32,
    /// Unpadded patch-grid height.
    pub grid_height: i32,
    /// Unpadded patch-grid width.
    pub grid_width: i32,
}

impl VisionIngressPartPlan {
    /// Validates one prepared frame and derives its pooled decoder span.
    pub fn new(
        config: &VisionConfig,
        extent: [i32; 3],
        padded_patches: i32,
    ) -> Result<Self, IngressPlanError> {
        Self::with_diagnostic(config, extent, padded_patches, invalid)
    }

    pub(crate) fn with_diagnostic<E>(
        config: &VisionConfig,
        extent: [i32; 3],
        padded_patches: i32,
        invalid: impl Fn(fmt::Arguments<'_>) -> E + Copy,
    ) -> Result<Self, E> {
        let [time, height, width] = extent;
        if time != 1 {
            return Err(invalid(format_args!(
                "Gemma 4 prepared vision parts must contain one frame, got {time}"
            )));
        }
        let height = positive(height, "vision patch-grid height", invalid)?;
        let width = positive(width, "vision patch-grid width", invalid)?;
        let padded_patches = positive(padded_patches, "vision padded patch count", invalid)?;
        let pool = positive(config.pooling_kernel_size, "vision pooling kernel", invalid)?;
        if height % pool != 0 || width % pool != 0 {
            return Err(invalid(format_args!(
                "Gemma 4 vision grid {height}x{width} is not divisible by pooling kernel {pool}"
            )));
        }
        let valid_patches = checked_product(height, width, "valid vision patch count", invalid)?;
        if valid_patches > padded_patches {
            return Err(invalid(format_args!(
                "Gemma 4 vision grid requires {valid_patches} patches but the payload carries {padded_patches}"
            )));
        }
        let decoder_positions = checked_product(
            height / pool,
            width / pool,
            "pooled vision placeholder count",
            invalid,
        )?;
        Ok(Self {
            padded_patches,
            valid_patches,
            decoder_positions,
            grid_height: height,
            grid_width: width,
        })
    }
}

/// Architecture-owned masks and geometry for a padded vision batch.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionIngressBatchPlan {
    /// Common padded patch width.
    pub padded_patches: i32,
    /// Numeric validity values shaped by [`Self::position_valid_shape`].
    pub position_valid_values: Vec<f32>,
    /// Additive padding values shaped by [`Self::key_mask_shape`].
    pub key_mask_values: Vec<f32>,
    /// Unpadded patch-grid extents in batch order.
    pub grid_extents: Vec<(i32, i32)>,
}

impl VisionIngressBatchPlan {
    /// Derives the common padding extent and exact vision masks for a batch.
    pub fn new(parts: &[VisionIngressPartPlan]) -> Result<Self, IngressPlanError> {
        let layout = VisionBatchLayout::new(parts, invalid)?;
        let mut position_valid_values = Vec::with_capacity(layout.values);
        let mut key_mask_values = Vec::with_capacity(layout.values);
        let mut grid_extents = Vec::with_capacity(parts.len());
        for index in 0..layout.values {
            position_valid_values.push(layout.position_valid(index));
            key_mask_values.push(layout.key_mask(index));
        }
        grid_extents.extend(parts.iter().map(|part| (part.grid_height, part.grid_width)));
        Ok(Self {
            padded_patches: layout.padded_patches,
            position_valid_values,
            key_mask_values,
            grid_extents,
        })
    }

    /// Shape of `position_valid_values`.
    pub fn position_valid_shape(&self) -> [i32; 3] {
        [self.grid_extents.len() as i32, self.padded_patches, 1]
    }

    /// Shape of `key_mask_values`.
    pub fn key_mask_shape(&self) -> [i32; 4] {
        [self.grid_extents.len() as i32, 1, 1, self.padded_patches]
    }
}

/// Architecture-owned ingress geometry for one prepared audio item.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AudioIngressPartPlan {
    /// Padded input-frame width carried by the prepared payload.
    pub padded_frames: i32,
    /// Valid input-frame prefix.
    pub valid_frames: i32,
    /// Valid frames after the first stride-two convolution.
    pub valid_first_stage_frames: i32,
    /// Valid frames after both stride-two convolutions.
    pub valid_subsampled_frames: i32,
    /// Decoder placeholder count for projected audio.
    pub decoder_positions: i32,
}

impl AudioIngressPartPlan {
    /// Validates an audio extent and derives both subsampling stages.
    pub fn new(valid_frames: i32, padded_frames: i32) -> Result<Self, IngressPlanError> {
        Self::with_diagnostic(valid_frames, padded_frames, invalid)
    }

    pub(crate) fn with_diagnostic<E>(
        valid_frames: i32,
        padded_frames: i32,
        invalid: impl Fn(fmt::Arguments<'_>) -> E + Copy,
    ) -> Result<Self, E> {
        let padded_frames = positive(padded_frames, "audio padded frame count", invalid)?;
        if valid_frames <= 0 || valid_frames > padded_frames {
            return Err(invalid(format_args!(
                "Gemma 4 valid audio frame count {valid_frames} is outside 1..={padded_frames}"
            )));
        }
        let valid_first_stage_frames = ceil_half(valid_frames);
        let valid_subsampled_frames = ceil_half(valid_first_stage_frames);
        Ok(Self {
            padded_frames,
            valid_frames,
            valid_first_stage_frames,
            valid_subsampled_frames,
            decoder_positions: valid_subsampled_frames,
        })
    }
}

/// Architecture-owned masks and geometry for a padded audio batch.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioIngressBatchPlan {
    /// Common padded input-frame width.
    pub padded_frames: i32,
    /// Common width after the first stride-two convolution.
    pub first_stage_frames: i32,
    /// Numeric first-stage mask values shaped by [`Self::first_stage_mask_shape`].
    pub first_stage_mask_values: Vec<f32>,
    /// Valid output lengths after both stride-two convolutions.
    pub valid_subsampled_frames: Vec<i32>,
}

impl AudioIngressBatchPlan {
    /// Derives the common audio padding extent and first-stage mask for a batch.
    pub fn new(parts: &[AudioIngressPartPlan]) -> Result<Self, IngressPlanError> {
        let layout = AudioBatchLayout::new(parts, invalid)?;
        let mut first_stage_mask_values = Vec::with_capacity(layout.first_stage_values);
        let mut valid_subsampled_frames = Vec::with_capacity(parts.len());
        for index in 0..layout.first_stage_values {
            first_stage_mask_values.push(layout.first_stage_mask(index));
        }
        valid_subsampled_frames.extend(parts.iter().map(|part| part.valid_subsampled_frames));
        Ok(Self {
            padded_frames: layout.padded_frames,
            first_stage_frames: layout.first_stage_frames,
            first_stage_mask_values,
            valid_subsampled_frames,
        })
    }

    /// Shape of `first_stage_mask_values`.
    pub fn first_stage_mask_shape(&self) -> [i32; 4] {
        [
            self.valid_subsampled_frames.len() as i32,
            self.first_stage_frames,
            1,
            1,
        ]
    }
}

/// Borrowed scalar plan shared by ordinary arrays and counted generated inputs.
pub(super) struct VisionBatchLayout<'a> {
    parts: &'a [VisionIngressPartPlan],
    batch: i32,
    pub(super) padded_patches: i32,
    values: usize,
}
impl<'a> VisionBatchLayout<'a> {
    pub(super) fn new<E>(
        parts: &'a [VisionIngressPartPlan],
        invalid: impl Fn(fmt::Arguments<'_>) -> E + Copy,
    ) -> Result<Self, E> {
        if parts.is_empty() {
            return Err(invalid(format_args!("Gemma 4 vision ingress batch must not be empty")));
        }
        let padded_patches = parts.iter().map(|part| part.padded_patches).max().unwrap_or(0);
        let batch = i32::try_from(parts.len())
            .map_err(|_| invalid(format_args!("Gemma 4 vision batch is too large")))?;
        let values = checked_product(batch, padded_patches, "vision mask element count", invalid)?;
        let values = usize::try_from(values)
            .map_err(|_| invalid(format_args!("Gemma 4 vision mask is too large")))?;
        Ok(Self { parts, batch, padded_patches, values })
    }
    pub(super) fn position_valid_shape(&self) -> [i32; 3] {
        [self.batch, self.padded_patches, 1]
    }
    pub(super) fn key_mask_shape(&self) -> [i32; 4] {
        [self.batch, 1, 1, self.padded_patches]
    }
    pub(super) fn position_valid(&self, index: usize) -> f32 {
        if self.patch_valid(index) { 1.0 } else { 0.0 }
    }
    pub(super) fn key_mask(&self, index: usize) -> f32 {
        if self.patch_valid(index) { 0.0 } else { VISION_ATTENTION_INVALID_LOGIT }
    }
    fn patch_valid(&self, index: usize) -> bool {
        let width = self.padded_patches as usize;
        let part = &self.parts[index / width];
        ((index % width) as i32) < part.valid_patches
    }
}

/// Borrowed first-stage and input-mask equations for the same audio plan.
pub(super) struct AudioBatchLayout<'a> {
    parts: &'a [AudioIngressPartPlan],
    batch: i32,
    pub(super) padded_frames: i32,
    first_stage_frames: i32,
    first_stage_values: usize,
}
impl<'a> AudioBatchLayout<'a> {
    pub(super) fn new<E>(
        parts: &'a [AudioIngressPartPlan],
        invalid: impl Fn(fmt::Arguments<'_>) -> E + Copy,
    ) -> Result<Self, E> {
        if parts.is_empty() {
            return Err(invalid(format_args!("Gemma 4 audio ingress batch must not be empty")));
        }
        let padded_frames = parts.iter().map(|part| part.padded_frames).max().unwrap_or(0);
        let first_stage_frames = ceil_half(padded_frames);
        let batch = i32::try_from(parts.len())
            .map_err(|_| invalid(format_args!("Gemma 4 audio batch is too large")))?;
        let values = checked_product(batch, first_stage_frames, "audio mask element count", invalid)?;
        let first_stage_values = usize::try_from(values)
            .map_err(|_| invalid(format_args!("Gemma 4 audio mask is too large")))?;
        Ok(Self { parts, batch, padded_frames, first_stage_frames, first_stage_values })
    }
    pub(super) fn input_mask_shape(&self) -> [i32; 3] {
        [self.batch, self.padded_frames, 1]
    }
    pub(super) fn first_stage_mask_shape(&self) -> [i32; 4] {
        [self.batch, self.first_stage_frames, 1, 1]
    }
    pub(super) fn input_mask(&self, index: usize) -> f32 {
        let width = self.padded_frames as usize;
        if ((index % width) as i32) < self.parts[index / width].valid_frames { 1.0 } else { 0.0 }
    }
    pub(super) fn first_stage_mask(&self, index: usize) -> f32 {
        let width = self.first_stage_frames as usize;
        if ((index % width) as i32) < self.parts[index / width].valid_first_stage_frames { 1.0 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vision_config() -> VisionConfig {
        VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 1,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-5,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    #[test]
    fn vision_plan_owns_placeholder_pooling_and_padding_values() {
        let first = VisionIngressPartPlan::new(&vision_config(), [1, 2, 4], 12).unwrap();
        let second = VisionIngressPartPlan::new(&vision_config(), [1, 2, 2], 8).unwrap();
        assert_eq!(first.decoder_positions, 2);
        assert_eq!(second.decoder_positions, 1);

        let batch = VisionIngressBatchPlan::new(&[first, second]).unwrap();
        assert_eq!(batch.position_valid_shape(), [2, 12, 1]);
        assert_eq!(batch.key_mask_shape(), [2, 1, 1, 12]);
        assert_eq!(
            &batch.position_valid_values[..10],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(batch.key_mask_values[8], VISION_ATTENTION_INVALID_LOGIT);
        assert_eq!(batch.key_mask_values[12 + 3], 0.0);
        assert_eq!(
            batch.key_mask_values[12 + 4],
            VISION_ATTENTION_INVALID_LOGIT
        );
    }

    #[test]
    fn vision_plan_rejects_non_integral_pooling_geometry() {
        let error = VisionIngressPartPlan::new(&vision_config(), [1, 3, 4], 12).unwrap_err();
        assert!(error.to_string().contains("not divisible"));
    }

    #[test]
    fn audio_plan_owns_both_subsampling_stages() {
        let first = AudioIngressPartPlan::new(9, 12).unwrap();
        let second = AudioIngressPartPlan::new(4, 8).unwrap();
        assert_eq!(first.valid_first_stage_frames, 5);
        assert_eq!(first.valid_subsampled_frames, 3);
        assert_eq!(first.decoder_positions, 3);

        let batch = AudioIngressBatchPlan::new(&[first, second]).unwrap();
        assert_eq!(batch.first_stage_mask_shape(), [2, 6, 1, 1]);
        assert_eq!(
            batch.first_stage_mask_values,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(batch.valid_subsampled_frames, vec![3, 1]);
    }
}
