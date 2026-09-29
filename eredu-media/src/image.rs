//! Shared decoded-image validation and transforms.

#[cfg(feature = "image")]
use image::{imageops::FilterType, ImageBuffer, Rgb};

use crate::{MediaError, ProcessedMediaDtype};

/// Memory order of a normalized RGB output buffer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum NormalizedImageLayout {
    /// Contiguous `[channels, height, width]` order.
    ChannelsHeightWidth,
}

/// Borrowed RGB8 image pixels.
#[derive(Debug, Clone, Copy)]
pub struct RgbImageView<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    row_stride: usize,
}

impl<'a> RgbImageView<'a> {
    /// Creates an RGB8 image view with tightly packed rows.
    pub fn packed(pixels: &'a [u8], width: u32, height: u32) -> Result<Self, MediaError> {
        let row_stride = width as usize * 3;
        Self::with_row_stride(pixels, width, height, row_stride)
    }

    /// Creates an RGB8 image view with an explicit byte stride between rows.
    pub fn with_row_stride(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        row_stride: usize,
    ) -> Result<Self, MediaError> {
        if width == 0 || height == 0 {
            return Err(MediaError::invalid(format!(
                "image dimensions must be positive, got {width}x{height}"
            )));
        }
        let packed_stride = width as usize * 3;
        if row_stride < packed_stride {
            return Err(MediaError::invalid(format!(
                "RGB8 row stride {row_stride} is smaller than packed row size {packed_stride}"
            )));
        }
        let required = row_stride
            .checked_mul(height.saturating_sub(1) as usize)
            .and_then(|prefix| prefix.checked_add(packed_stride))
            .ok_or_else(|| MediaError::invalid("image buffer dimensions overflow"))?;
        if pixels.len() < required {
            return Err(MediaError::invalid(format!(
                "RGB8 image requires at least {required} bytes, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            pixels,
            width,
            height,
            row_stride,
        })
    }

    /// Image width in pixels.
    pub fn width(self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(self) -> u32 {
        self.height
    }

    /// Returns tightly packed row-major RGB pixels without row padding.
    pub fn packed_pixels(self) -> Vec<u8> {
        let packed_stride = self.width as usize * 3;
        if self.row_stride == packed_stride {
            return self.pixels[..packed_stride * self.height as usize].to_vec();
        }
        let mut packed = Vec::with_capacity(packed_stride * self.height as usize);
        for row in 0..self.height as usize {
            let start = row * self.row_stride;
            packed.extend_from_slice(&self.pixels[start..start + packed_stride]);
        }
        packed
    }
}

/// Owned, tightly packed RGB8 image.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl RgbImage {
    /// Borrows this image as an RGB8 view.
    pub fn as_view(&self) -> RgbImageView<'_> {
        RgbImageView {
            pixels: &self.pixels,
            width: self.width,
            height: self.height,
            row_stride: self.width as usize * 3,
        }
    }

    /// Returns tightly packed RGB8 pixels.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Owned normalized image in channel-first `[channels, height, width]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedImage {
    data: Vec<f32>,
    channels: usize,
    width: usize,
    height: usize,
}

impl NormalizedImage {
    /// Scalar dtype of every value.
    pub const fn dtype(&self) -> ProcessedMediaDtype {
        ProcessedMediaDtype::F32
    }

    /// Explicit memory layout.
    pub const fn layout(&self) -> NormalizedImageLayout {
        NormalizedImageLayout::ChannelsHeightWidth
    }

    /// Logical `[channels, height, width]` shape.
    pub const fn shape(&self) -> [usize; 3] {
        [self.channels, self.height, self.width]
    }

    /// Returns normalized channel-first pixels.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns one channel-first pixel.
    pub fn get(&self, channel: usize, y: usize, x: usize) -> f32 {
        self.data[(channel * self.height + y) * self.width + x]
    }
}

/// Additional live data-buffer bound for resize followed by F32 normalization.
/// Excludes the borrowed input, allocator bookkeeping and fixed object headers.
/// The separable resize implementation is pinned in this crate's manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbPreparationBuffers {
    /// Final channel-first normalized data.
    pub normalized_bytes: u64,
    /// Maximum simultaneous additional data buffers, including the final result.
    pub peak_bytes: u64,
}

/// Bounds RGB8 bicubic/Lanczos resize and normalization without allocating pixels.
/// Includes the packed source copy, RGBA32F vertical intermediate, resized RGB8,
/// normalized F32 output, and a conservative interpolation-weight growth allowance.
pub fn rgb_preparation_buffers(
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Result<RgbPreparationBuffers, MediaError> {
    let overflow = || MediaError::invalid("RGB preparation buffer geometry overflow");
    if [source_width, source_height, width, height].contains(&0) {
        return Err(MediaError::invalid(
            "RGB preparation dimensions must be positive",
        ));
    }
    let product = |a: u64, b: u64, c: u64| {
        a.checked_mul(b)
            .and_then(|v| v.checked_mul(c))
            .ok_or_else(overflow)
    };
    let rgb = product(width.into(), height.into(), 3)?;
    let normalized_bytes = rgb.checked_mul(4).ok_or_else(overflow)?;
    let normalization = rgb.checked_add(normalized_bytes).ok_or_else(overflow)?;
    let peak_bytes = if (source_width, source_height) == (width, height) {
        normalization
    } else {
        let source = product(source_width.into(), source_height.into(), 3)?;
        let intermediate = product(source_width.into(), height.into(), 16)?;
        // At most one source axis of F32 weights, including old/new Vec growth
        // and the minimum allocation capacity. Both passes are sequential.
        let weights = u64::from(source_width.max(source_height).max(4))
            .checked_mul(16)
            .ok_or_else(overflow)?;
        source
            .checked_add(intermediate)
            .and_then(|v| v.checked_add(rgb))
            .and_then(|v| v.checked_add(weights))
            .ok_or_else(overflow)?
            .max(normalization)
    };
    if peak_bytes > isize::MAX as u64 {
        return Err(overflow());
    }
    Ok(RgbPreparationBuffers {
        normalized_bytes,
        peak_bytes,
    })
}

/// Resizes an RGB8 image using bicubic interpolation.
pub fn resize_rgb8_bicubic(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
) -> Result<RgbImage, MediaError> {
    #[cfg(feature = "image")]
    {
        resize_rgb8_bicubic_enabled(image, width, height)
    }
    #[cfg(not(feature = "image"))]
    {
        let _ = (image, width, height);
        Err(MediaError::invalid(
            "Catmull-Rom RGB resize requires the `image` feature",
        ))
    }
}

#[cfg(feature = "image")]
fn resize_rgb8_bicubic_enabled(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
) -> Result<RgbImage, MediaError> {
    if width == 0 || height == 0 {
        return Err(MediaError::invalid(format!(
            "resize dimensions must be positive, got {width}x{height}"
        )));
    }
    if width == image.width && height == image.height {
        return Ok(RgbImage {
            pixels: image.packed_pixels(),
            width,
            height,
        });
    }
    let source =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(image.width, image.height, image.packed_pixels())
            .ok_or_else(|| MediaError::invalid("failed to construct RGB8 image buffer"))?;
    let resized = image::imageops::resize(&source, width, height, FilterType::CatmullRom);
    Ok(RgbImage {
        pixels: resized.into_raw(),
        width,
        height,
    })
}

/// Resizes an RGB8 image with a Lanczos3 filter.
pub fn resize_rgb8_lanczos3(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
) -> Result<RgbImage, MediaError> {
    #[cfg(feature = "image")]
    {
        resize_rgb8_lanczos3_enabled(image, width, height)
    }
    #[cfg(not(feature = "image"))]
    {
        let _ = (image, width, height);
        Err(MediaError::invalid(
            "Lanczos3 RGB resize requires the `image` feature",
        ))
    }
}

#[cfg(feature = "image")]
fn resize_rgb8_lanczos3_enabled(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
) -> Result<RgbImage, MediaError> {
    if width == 0 || height == 0 {
        return Err(MediaError::invalid(format!(
            "resize dimensions must be positive, got {width}x{height}"
        )));
    }
    if width == image.width && height == image.height {
        return Ok(RgbImage {
            pixels: image.packed_pixels(),
            width,
            height,
        });
    }
    let source =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(image.width, image.height, image.packed_pixels())
            .ok_or_else(|| MediaError::invalid("failed to construct RGB8 image buffer"))?;
    let resized = image::imageops::resize(&source, width, height, FilterType::Lanczos3);
    Ok(RgbImage {
        pixels: resized.into_raw(),
        width,
        height,
    })
}

/// Rescales and normalizes RGB8 pixels, returning channel-first data.
pub fn rescale_and_normalize_rgb8(
    image: RgbImageView<'_>,
    rescale_factor: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<NormalizedImage, MediaError> {
    if !rescale_factor.is_finite() {
        return Err(MediaError::invalid(format!(
            "image rescale factor must be finite, got {rescale_factor}"
        )));
    }
    if mean.iter().any(|value| !value.is_finite()) {
        return Err(MediaError::invalid(format!(
            "image normalization means must be finite, got {mean:?}"
        )));
    }
    if std.iter().any(|value| *value == 0.0 || !value.is_finite()) {
        return Err(MediaError::invalid(format!(
            "image normalization standard deviations must be finite and nonzero, got {std:?}"
        )));
    }
    let width = image.width as usize;
    let height = image.height as usize;
    let mut data = vec![0.0f32; 3 * width * height];
    for y in 0..height {
        let row = &image.pixels[y * image.row_stride..][..width * 3];
        for x in 0..width {
            for channel in 0..3 {
                let value = row[x * 3 + channel] as f32 * rescale_factor;
                let normalized = (value - mean[channel]) / std[channel];
                if !normalized.is_finite() {
                    return Err(MediaError::invalid(
                        "image normalization produced a non-finite value",
                    ));
                }
                data[(channel * height + y) * width + x] = normalized;
            }
        }
    }
    Ok(NormalizedImage {
        data,
        channels: 3,
        width,
        height,
    })
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::{
        rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3, RgbImageView,
    };

    #[test]
    fn image_view_honors_row_stride() {
        let pixels = [255, 0, 0, 9, 9, 9, 0, 255, 0];
        let view = RgbImageView::with_row_stride(&pixels, 1, 2, 6).unwrap();
        let normalized = rescale_and_normalize_rgb8(view, 1.0 / 255.0, [0.5; 3], [0.5; 3]).unwrap();
        assert_eq!(normalized.data(), &[1.0, -1.0, -1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn no_op_resize_tightly_packs_rows() {
        let pixels = [1, 2, 3, 0, 4, 5, 6];
        let view = RgbImageView::with_row_stride(&pixels, 1, 2, 4).unwrap();
        let resized = resize_rgb8_bicubic(view, 1, 2).unwrap();
        assert_eq!(resized.pixels(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn catmull_rom_and_lanczos3_match_characterized_pixels() {
        let pixels = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let view = RgbImageView::packed(&pixels, 2, 2).unwrap();
        let bicubic = resize_rgb8_bicubic(view, 3, 3).unwrap();
        let lanczos = resize_rgb8_lanczos3(view, 3, 3).unwrap();
        assert_eq!(
            bicubic.pixels(),
            [
                255, 0, 0, 127, 127, 0, 0, 255, 0, 127, 0, 127, 128, 128, 128, 128, 255, 128, 0, 0,
                255, 128, 128, 255, 255, 255, 255,
            ]
        );
        assert_eq!(
            lanczos.pixels(),
            [
                255, 0, 0, 128, 128, 0, 0, 255, 0, 128, 0, 128, 128, 128, 128, 128, 255, 128, 0, 0,
                255, 128, 128, 255, 255, 255, 255,
            ]
        );
    }

    #[test]
    fn rejects_invalid_views_resize_and_normalization() {
        assert!(RgbImageView::packed(&[], 0, 1).is_err());
        assert!(RgbImageView::with_row_stride(&[0; 6], 2, 1, 5).is_err());
        assert!(RgbImageView::with_row_stride(&[0; 5], 2, 1, 6).is_err());
        let view = RgbImageView::packed(&[0, 0, 0], 1, 1).unwrap();
        assert!(resize_rgb8_bicubic(view, 0, 1).is_err());
        assert!(rescale_and_normalize_rgb8(view, f32::NAN, [0.0; 3], [1.0; 3]).is_err());
        assert!(rescale_and_normalize_rgb8(view, 1.0, [f32::NAN; 3], [1.0; 3]).is_err());
        assert!(rescale_and_normalize_rgb8(view, 1.0, [0.0; 3], [0.0; 3]).is_err());
    }
}

#[cfg(test)]
mod preparation_buffer_tests {
    use super::*;
    #[test]
    fn buffer_bounds_include_separable_intermediate_and_identity_copy() {
        assert_eq!(
            rgb_preparation_buffers(10, 20, 5, 4).unwrap(),
            RgbPreparationBuffers {
                normalized_bytes: 240,
                peak_bytes: 1620
            }
        );
        assert_eq!(
            rgb_preparation_buffers(5, 4, 5, 4).unwrap(),
            RgbPreparationBuffers {
                normalized_bytes: 240,
                peak_bytes: 300
            }
        );
        // A small output does not imply small resize workspace for a wide source.
        assert!(
            rgb_preparation_buffers(1_000_000, 1, 1, 1)
                .unwrap()
                .peak_bytes
                > 30_000_000
        );
        assert!(rgb_preparation_buffers(u32::MAX, u32::MAX, u32::MAX, u32::MAX).is_err());
        // A representable output must not admit an unaddressable intermediate Vec.
        assert!(rgb_preparation_buffers(2_000_000_000, 1, 1, 300_000_000).is_err());
        assert!(rgb_preparation_buffers(0, 1, 1, 1).is_err());
    }
}
