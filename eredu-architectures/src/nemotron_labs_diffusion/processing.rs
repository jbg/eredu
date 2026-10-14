//! Portable image sizing, normalization, patch packing, and token placement policy.
use super::FamilyConfig;
use crate::decoder::Config;
use eredu_media::{
    MediaError,
    image::{RgbImageView, resize_and_normalize_rgb8_cubic},
};

/// Invalid architecture image policy or portable pixel preparation failure.
#[derive(Debug, thiserror::Error)]
pub enum ImagePreparationError {
    /// Invalid image extent or architecture geometry.
    #[error("{0}")]
    Invalid(String),
    /// Portable image transformation failure.
    #[error(transparent)]
    Media(#[from] MediaError),
}
impl ImagePreparationError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

/// Released image delimiter and placeholder identities, checked against tokenizer assets.
pub const IMAGE_TOKENS: [(&str, u32); 4] = [
    ("<|image_start|>", 18),
    ("<|image_pad|>", 19),
    ("<|image_break|>", 20),
    ("<|image_end|>", 21),
];

/// Complete published marker sequence for one aligned unmerged patch grid.
pub fn image_token_ids(grid: (u32, u32), merge: u32) -> Result<Vec<u32>, ImagePreparationError> {
    let (height, width) = grid;
    if merge == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
        return Err(ImagePreparationError::invalid("invalid image marker grid"));
    }
    let rows = height / merge;
    let columns = width / merge;
    let count = (rows as usize)
        .checked_mul(columns as usize + 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| ImagePreparationError::invalid("image marker buffer overflow"))?;
    let mut tokens = Vec::with_capacity(count);
    tokens.push(18);
    for row in 0..rows {
        tokens.extend(std::iter::repeat_n(19, columns as usize));
        tokens.push(if row + 1 == rows { 21 } else { 20 });
    }
    Ok(tokens)
}

/// Validated architecture image policy. It does not allocate a backend tensor.
#[derive(Debug, Clone)]
pub struct ImageProcessor {
    patch: u32,
    merge: u32,
    longest_edge: u32,
    tower_extent: u32,
}

/// Prepared row-major patches and matching complete image marker sequence.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Channel-major patch vectors ordered by image row and column.
    pub patches: Vec<f32>,
    /// Positive unmerged patch grid `(height, width)`.
    pub grid: (u32, u32),
    /// Full start/pad/row-break/end token sequence.
    pub token_ids: Vec<u32>,
}
impl ImageProcessor {
    /// Complete channel-major patch vector width, independent of native tensors.
    pub fn patch_vector_width(&self) -> usize { 3 * self.patch as usize * self.patch as usize }
    /// Spatial merge factor used for marker and decoder geometry.
    pub const fn merge_size(&self) -> u32 { self.merge }
    /// Pixel width of one unmerged patch.
    pub const fn patch_size(&self) -> u32 { self.patch }

    /// Composites transparent RGBA pixels on the published white background
    /// before executing the same RGB resizing and patch policy.
    pub fn prepare_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        row_stride: usize,
    ) -> Result<PreparedImage, ImagePreparationError> {
        let rgb = eredu_media::image::composite_rgba8(pixels, width, height, row_stride, [255; 3])?;
        self.prepare(rgb.as_view())
    }
    /// Selects an explicit processor extent; the published release uses 1400.
    pub fn new(config: &FamilyConfig, longest_edge: u32) -> Result<Self, ImagePreparationError> {
        config
            .validate_config()
            .map_err(|e| ImagePreparationError::invalid(e.to_string()))?;
        let vision = config.vision.as_ref().ok_or_else(|| {
            ImagePreparationError::invalid("text artifact has no image processor")
        })?;
        if vision.num_channels != 3 || longest_edge == 0 || longest_edge > vision.image_size as u32
        {
            return Err(ImagePreparationError::invalid(
                "invalid RGB processor extent",
            ));
        }
        Ok(Self {
            patch: vision.patch_size as u32,
            merge: config.spatial_merge_size,
            longest_edge,
            tower_extent: vision.image_size as u32,
        })
    }

    /// Computes the exact rounded, then ceiling-aligned published image geometry.
    pub fn grid(&self, width: u32, height: u32) -> Result<(u32, u32), ImagePreparationError> {
        if width == 0 || height == 0 {
            return Err(ImagePreparationError::invalid(
                "image dimensions must be positive",
            ));
        }
        let ratio = f64::from(width.max(height)) / f64::from(self.longest_edge);
        let (width, height) = if ratio > 1.0 {
            (
                (f64::from(width) / ratio).round_ties_even() as u32,
                (f64::from(height) / ratio).round_ties_even() as u32,
            )
        } else {
            (width, height)
        };
        if width == 0 || height == 0 {
            return Err(ImagePreparationError::invalid(
                "image aspect ratio rounds a dimension to zero",
            ));
        }
        let extent = self
            .patch
            .checked_mul(self.merge)
            .ok_or_else(|| ImagePreparationError::invalid("image patch geometry overflow"))?;
        let token_width = width.div_ceil(extent);
        let token_height = height.div_ceil(extent);
        let grid = (
            token_height.checked_mul(self.merge),
            token_width.checked_mul(self.merge),
        );
        let (Some(height), Some(width)) = grid else {
            return Err(ImagePreparationError::invalid("image grid overflow"));
        };
        if width > self.tower_extent / self.patch || height > self.tower_extent / self.patch {
            return Err(ImagePreparationError::invalid(
                "aligned image exceeds Pixtral rotary extent",
            ));
        }
        Ok((height, width))
    }

    /// Prepares RGB pixels with floating cubic overshoot and the released CLIP
    /// statistics, then flattens each channel-major convolution patch.
    pub fn prepare(&self, image: RgbImageView<'_>) -> Result<PreparedImage, ImagePreparationError> {
        let (height, width) = self.grid(image.width(), image.height())?;
        let normalized = resize_and_normalize_rgb8_cubic(
            image,
            width * self.patch,
            height * self.patch,
            -0.75,
            1.0 / 255.0,
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        )?;
        let count = (height as usize)
            .checked_mul(width as usize)
            .and_then(|x| x.checked_mul(3))
            .and_then(|x| x.checked_mul(self.patch as usize))
            .and_then(|x| x.checked_mul(self.patch as usize))
            .ok_or_else(|| ImagePreparationError::invalid("prepared patch buffer overflow"))?;
        let mut patches = Vec::with_capacity(count);
        for row in 0..height {
            for column in 0..width {
                for channel in 0..3 {
                    for y in 0..self.patch {
                        for x in 0..self.patch {
                            patches.push(normalized.get(
                                channel,
                                (row * self.patch + y) as usize,
                                (column * self.patch + x) as usize,
                            ));
                        }
                    }
                }
            }
        }
        let token_ids = image_token_ids((height, width), self.merge)?;
        Ok(PreparedImage {
            patches,
            grid: (height, width),
            token_ids,
        })
    }
}
