//! Shared decoded-image validation and transforms.

#[cfg(feature = "image")]
use image::{ImageBuffer, Rgb, imageops::FilterType};

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

/// Composites straight-alpha RGBA8 samples onto an explicit RGB8 background.
/// Integer rounding follows byte-space image compositing; no alpha is discarded.
pub fn composite_rgba8(
    pixels: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    background: [u8; 3],
) -> Result<RgbImage, MediaError> {
    let packed = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| MediaError::invalid("RGBA row geometry overflow"))?;
    let length = row_stride
        .checked_mul(height.saturating_sub(1) as usize)
        .and_then(|n| n.checked_add(packed))
        .ok_or_else(|| MediaError::invalid("RGBA buffer geometry overflow"))?;
    if width == 0 || height == 0 || row_stride < packed || pixels.len() < length {
        return Err(MediaError::invalid("invalid RGBA buffer geometry"));
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| MediaError::invalid("composited RGB buffer overflow"))?;
    let mut rgb = Vec::with_capacity(count);
    for row in 0..height as usize {
        for pixel in pixels[row * row_stride..][..packed].chunks_exact(4) {
            let alpha = u32::from(pixel[3]);
            for channel in 0..3 {
                rgb.push(
                    ((u32::from(pixel[channel]) * alpha
                        + u32::from(background[channel]) * (255 - alpha)
                        + 127)
                        / 255) as u8,
                );
            }
        }
    }
    Ok(RgbImage {
        pixels: rgb,
        width,
        height,
    })
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

/// Resizes floating RGB samples using a separable four-tap cubic kernel with
/// half-pixel coordinates and replicated borders, then normalizes into CHW.
/// Values retain cubic overshoot. At most four resized source rows are retained.
pub fn resize_and_normalize_rgb8_cubic(
    image: RgbImageView<'_>,
    width: u32,
    height: u32,
    coefficient: f32,
    rescale_factor: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<NormalizedImage, MediaError> {
    if width == 0
        || height == 0
        || !coefficient.is_finite()
        || !rescale_factor.is_finite()
        || mean.iter().any(|x| !x.is_finite())
        || std.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(MediaError::invalid(
            "invalid floating cubic resize or normalization",
        ));
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| MediaError::invalid("cubic resize output overflow"))?;
    let weights = |destination: u32, source: u32, target: u32| {
        let coordinate = ((destination as f64 + 0.5) * source as f64 / target as f64 - 0.5) as f32;
        let origin = coordinate.floor() as i64;
        let x = coordinate - coordinate.floor();
        let a = coefficient;
        let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
        let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        let inverse = 1.0 - x;
        let c2 = ((a + 2.0) * inverse - (a + 3.0)) * inverse * inverse + 1.0;
        let c3 = 1.0 - c0 - c1 - c2;
        let indices = std::array::from_fn::<_, 4, _>(|i| {
            (origin + i as i64 - 1).clamp(0, source as i64 - 1) as usize
        });
        (indices, [c0, c1, c2, c3])
    };
    let columns: Vec<_> = (0..width).map(|x| weights(x, image.width, width)).collect();
    let mut rows = std::collections::BTreeMap::<usize, Vec<f32>>::new();
    let mut data = vec![0.0; count];
    for y in 0..height {
        let (indices, vertical) = weights(y, image.height, height);
        rows.retain(|index, _| indices.contains(index));
        for &row in &indices {
            rows.entry(row).or_insert_with(|| {
                let source = &image.pixels[row * image.row_stride..][..image.width as usize * 3];
                let mut values = vec![0.0; width as usize * 3];
                for (x, (indices, horizontal)) in columns.iter().enumerate() {
                    for channel in 0..3 {
                        values[x * 3 + channel] = (0..4)
                            .map(|tap| source[indices[tap] * 3 + channel] as f32 * horizontal[tap])
                            .sum();
                    }
                }
                values
            });
        }
        for x in 0..width as usize {
            for channel in 0..3 {
                let sample: f32 = (0..4)
                    .map(|tap| rows[&indices[tap]][x * 3 + channel] * vertical[tap])
                    .sum();
                let normalized = (sample * rescale_factor - mean[channel]) / std[channel];
                if !normalized.is_finite() {
                    return Err(MediaError::invalid(
                        "cubic normalization produced a non-finite value",
                    ));
                }
                data[(channel * height as usize + y as usize) * width as usize + x] = normalized;
            }
        }
    }
    Ok(NormalizedImage {
        data,
        channels: 3,
        width: width as usize,
        height: height as usize,
    })
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use super::{
        RgbImageView, rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3,
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
