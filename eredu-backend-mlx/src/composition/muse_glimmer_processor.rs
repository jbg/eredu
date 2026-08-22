//! Hugging Face-compatible Muse-Glimmer image and experimental video host processor.

use std::{fs, path::Path};

use safemlx::{
    ops::{GgufMetadataArray, GgufMetadataValue},
    Array,
};
use serde::Deserialize;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::media::{
        image::{rescale_and_normalize_rgb8, resize_rgb8_lanczos3, NormalizedImage, RgbImageView},
        input::Modality,
        prepared_model_input, push_text_token_ids,
        video::{pad_frame_indices, uniform_sample_indices, validate_rgb_frames},
        MediaInput, MediaPayload, OwnedInputMetadata, PreparedInputPart, PreparedModelInput,
        ProcessorInput, ProcessorPreparationError, VideoFrames, VideoSampling,
    },
};

fn default_true() -> bool {
    true
}

fn default_rescale() -> f32 {
    1.0 / 255.0
}

fn default_mean_std() -> [f32; 3] {
    [0.5; 3]
}

fn default_patch() -> usize {
    14
}

fn default_temporal() -> usize {
    2
}

fn default_merge() -> usize {
    2
}

fn default_image_tokens() -> usize {
    4096
}

fn default_video_tokens() -> usize {
    144
}

fn default_video_frames() -> usize {
    96
}

fn default_video_fps() -> f64 {
    2.0
}

#[derive(Debug, Clone, Deserialize)]
struct VisualConfig {
    #[serde(default = "default_true")]
    do_resize: bool,
    #[serde(default = "default_true")]
    do_rescale: bool,
    #[serde(default = "default_rescale")]
    rescale_factor: f32,
    #[serde(default = "default_true")]
    do_normalize: bool,
    #[serde(default = "default_mean_std")]
    image_mean: [f32; 3],
    #[serde(default = "default_mean_std")]
    image_std: [f32; 3],
    #[serde(default = "default_patch")]
    patch_size: usize,
    #[serde(default = "default_temporal", alias = "temporal_patch_size")]
    temporal_patch_size: usize,
    #[serde(default = "default_merge")]
    merge_size: usize,
    #[serde(default = "default_image_tokens")]
    max_image_tokens: usize,
    #[serde(default = "default_video_tokens")]
    max_video_frame_tokens: usize,
    #[serde(default = "default_video_frames")]
    num_frames: usize,
    #[serde(default = "default_video_fps")]
    fps: f64,
    #[serde(default = "default_true")]
    do_sample_frames: bool,
    #[serde(default = "default_lanczos")]
    resample: u8,
}

fn default_lanczos() -> u8 {
    1
}

#[derive(Debug, Deserialize)]
struct ProcessorConfig {
    image_processor: VisualConfig,
    video_processor: VisualConfig,
}

/// Processor state loaded from the release's `processor_config.json`.
#[derive(Debug, Clone)]
pub struct MuseGlimmerProcessor {
    image: VisualConfig,
    video: VisualConfig,
    gguf_image_only: bool,
}

impl MuseGlimmerProcessor {
    pub fn load(model_dir: &Path) -> Result<Option<Self>, Error> {
        let path = model_dir.join("processor_config.json");
        if !path.exists() {
            return Ok(None);
        }
        let config: ProcessorConfig = serde_json::from_slice(&fs::read(path)?)?;
        validate_config(&config.image_processor, "image")?;
        validate_config(&config.video_processor, "video")?;
        Ok(Some(Self {
            image: config.image_processor,
            video: config.video_processor,
            gguf_image_only: false,
        }))
    }

    pub fn from_gguf(
        metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    ) -> Result<Self, Error> {
        let integer = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_i64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    Error::Processor(format!(
                        "Muse-Glimmer projector GGUF requires positive integer metadata {key:?}"
                    ))
                })
        };
        let rgb = |key: &str| -> Result<[f32; 3], Error> {
            let values = match metadata.get(key) {
                Some(GgufMetadataValue::Array(GgufMetadataArray::Float32(values))) => values,
                _ => {
                    return Err(Error::Processor(format!(
                        "Muse-Glimmer projector GGUF requires three Float32 values in {key:?}"
                    )))
                }
            };
            values.as_slice().try_into().map_err(|_| {
                Error::Processor(format!(
                    "Muse-Glimmer projector GGUF requires three Float32 values in {key:?}"
                ))
            })
        };
        let mut image = VisualConfig {
            do_resize: true,
            do_rescale: true,
            rescale_factor: default_rescale(),
            do_normalize: true,
            image_mean: rgb("clip.vision.image_mean")?,
            image_std: rgb("clip.vision.image_std")?,
            patch_size: integer("clip.vision.patch_size")?,
            // The official projector has temporally collapsed patch weights.
            temporal_patch_size: 1,
            merge_size: integer("clip.vision.spatial_merge_size")?,
            max_image_tokens: default_image_tokens(),
            max_video_frame_tokens: default_video_tokens(),
            num_frames: default_video_frames(),
            fps: default_video_fps(),
            do_sample_frames: true,
            resample: default_lanczos(),
        };
        validate_config(&image, "image")?;
        let video = image.clone();
        image.temporal_patch_size = 1;
        Ok(Self {
            image,
            video,
            gguf_image_only: true,
        })
    }

    pub fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let mut parts = Vec::new();
        for item in input {
            match *item {
                ProcessorInput::TokenIds(ids) => push_text_token_ids(&mut parts, ids),
                ProcessorInput::Media(media) => self.push_media(&mut parts, media, encode_text)?,
            }
        }
        Ok(prepared_model_input(parts)?)
    }

    fn push_media<E>(
        &self,
        parts: &mut Vec<PreparedInputPart>,
        media: MediaInput<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        match (media.modality, media.payload) {
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                push_text_token_ids(
                    parts,
                    &encode_text("<|image_start|>").map_err(ProcessorPreparationError::Text)?,
                );
                parts.push(process_image(image, &self.image)?);
                push_text_token_ids(
                    parts,
                    &encode_text("<|image_end|>").map_err(ProcessorPreparationError::Text)?,
                );
                Ok(())
            }
            (Modality::Video, MediaPayload::VideoFrames(video)) => {
                if self.gguf_image_only {
                    return Err(Error::Processor(
                        "the official Muse-Glimmer GGUF projector is image-only because its temporal patch weights are collapsed"
                            .into(),
                    )
                    .into());
                }
                self.push_video(parts, video, encode_text)
            }
            (modality, _) => Err(Error::Processor(format!(
                "Muse-Glimmer processor does not support {} media",
                modality.as_str()
            ))
            .into()),
        }
    }

    fn push_video<E>(
        &self,
        parts: &mut Vec<PreparedInputPart>,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        let (width, height) = validate_rgb_frames(video.frames)?;
        let source_fps = video.source_fps.unwrap_or(24.0);
        if !source_fps.is_finite() || source_fps <= 0.0 {
            return Err(Error::Processor(format!(
                "Muse-Glimmer video source FPS must be positive, got {source_fps}"
            ))
            .into());
        }
        let total = video.frames.len();
        let requested = match video.sampling {
            VideoSampling::ProcessorDefault if self.video.do_sample_frames => {
                ((total as f64 * self.video.fps / source_fps) as usize)
                    .min(self.video.num_frames)
                    .min(total)
            }
            VideoSampling::ProcessorDefault | VideoSampling::All => total,
            VideoSampling::Fps(fps) => ((total as f64 * fps / source_fps) as usize)
                .min(self.video.num_frames)
                .min(total),
            VideoSampling::FrameCount(count) => count.min(total),
        };
        let requested = requested.max(self.video.temporal_patch_size)
            / self.video.temporal_patch_size
            * self.video.temporal_patch_size;
        let mut indices = uniform_sample_indices(total, requested.min(total).max(1))?;
        let unpadded = indices.clone();
        pad_frame_indices(&mut indices, self.video.temporal_patch_size)?;
        let target = smart_resize(
            height as usize,
            width as usize,
            self.video.patch_size * self.video.merge_size,
            self.video.max_video_frame_tokens,
        )?;
        let mut frames = Vec::with_capacity(indices.len());
        for index in &indices {
            frames.push(normalize(video.frames[*index], target, &self.video)?);
        }
        push_text_token_ids(
            parts,
            &encode_text("<|vid_start|>").map_err(ProcessorPreparationError::Text)?,
        );
        let groups = frames.len() / self.video.temporal_patch_size;
        for group in 0..groups {
            let source_index = unpadded
                .get(group * self.video.temporal_patch_size)
                .copied()
                .or_else(|| unpadded.last().copied())
                .unwrap_or(0);
            push_text_token_ids(
                parts,
                &encode_text(&format!("Time: {:.1}s", source_index as f64 / source_fps))
                    .map_err(ProcessorPreparationError::Text)?,
            );
            let start = group * self.video.temporal_patch_size;
            let end = start + self.video.temporal_patch_size;
            parts.push(process_video_group(&frames[start..end], &self.video)?);
            let boundary = if group + 1 == groups {
                "<|vid_end|>"
            } else {
                "<|vid_frame_separator|>"
            };
            push_text_token_ids(
                parts,
                &encode_text(boundary).map_err(ProcessorPreparationError::Text)?,
            );
        }
        Ok(())
    }
}

fn validate_config(config: &VisualConfig, kind: &str) -> Result<(), Error> {
    if config.patch_size == 0
        || config.temporal_patch_size == 0
        || config.merge_size == 0
        || config.max_image_tokens == 0
        || config.max_video_frame_tokens == 0
    {
        return Err(Error::Processor(format!(
            "Muse-Glimmer {kind} processor dimensions and token limits must be positive"
        )));
    }
    if config.resample != 1 {
        return Err(Error::Processor(format!(
            "Muse-Glimmer {kind} processor requires Lanczos resample mode 1, got {}",
            config.resample
        )));
    }
    Ok(())
}

fn smart_resize(
    height: usize,
    width: usize,
    patch_size: usize,
    max_tokens: usize,
) -> Result<(usize, usize), Error> {
    if height == 0 || width == 0 || patch_size == 0 || max_tokens == 0 {
        return Err(Error::Processor(
            "Muse-Glimmer smart resize requires positive dimensions".into(),
        ));
    }
    let mut ideal_h = height as f64 / patch_size as f64;
    let mut ideal_w = width as f64 / patch_size as f64;
    let ratio = ideal_w / ideal_h;
    if ideal_h * ideal_w > max_tokens as f64 {
        ideal_h = (max_tokens as f64 / ratio).sqrt();
        ideal_w = ideal_h * ratio;
    }
    let mut candidates = Vec::new();
    for h in [ideal_h.floor() as usize, ideal_h.ceil() as usize] {
        for w in [ideal_w.floor() as usize, ideal_w.ceil() as usize] {
            if h > 0 && w > 0 && h.saturating_mul(w) <= max_tokens && !candidates.contains(&(h, w))
            {
                candidates.push((h, w));
            }
        }
    }
    if candidates.is_empty() {
        candidates.push((
            ideal_h.round().max(1.0) as usize,
            ideal_w.round().max(1.0) as usize,
        ));
    }
    let source_ratio = height as f64 / width as f64;
    let (grid_h, grid_w) = candidates
        .into_iter()
        .min_by(|left, right| {
            let left_error = (left.0 as f64 / left.1 as f64 - source_ratio).abs();
            let right_error = (right.0 as f64 / right.1 as f64 - source_ratio).abs();
            left_error.total_cmp(&right_error)
        })
        .expect("candidate list is non-empty");
    Ok((grid_h * patch_size, grid_w * patch_size))
}

fn normalize(
    image: RgbImageView<'_>,
    (height, width): (usize, usize),
    config: &VisualConfig,
) -> Result<NormalizedImage, Error> {
    let resized = if config.do_resize {
        resize_rgb8_lanczos3(image, width as u32, height as u32)?
    } else {
        resize_rgb8_lanczos3(image, image.width(), image.height())?
    };
    let factor = if config.do_rescale {
        config.rescale_factor
    } else {
        1.0
    };
    let (mean, std) = if config.do_normalize {
        (config.image_mean, config.image_std)
    } else {
        ([0.0; 3], [1.0; 3])
    };
    rescale_and_normalize_rgb8(resized.as_view(), factor, mean, std)
}

fn process_image(
    image: RgbImageView<'_>,
    config: &VisualConfig,
) -> Result<PreparedInputPart, Error> {
    let target = smart_resize(
        image.height() as usize,
        image.width() as usize,
        config.patch_size * config.merge_size,
        config.max_image_tokens,
    )?;
    let image = normalize(image, target, config)?;
    let (patches, grid) = pack_patches(std::slice::from_ref(&image), config, true)?;
    Ok(PreparedInputPart::media_tensor(
        Modality::Image,
        patches,
        OwnedInputMetadata::patch_grid(grid),
    ))
}

fn process_video_group(
    frames: &[NormalizedImage],
    config: &VisualConfig,
) -> Result<PreparedInputPart, Error> {
    let (patches, grid) = pack_patches(frames, config, false)?;
    Ok(PreparedInputPart::media_tensor(
        Modality::Video,
        patches,
        OwnedInputMetadata::patch_grid(grid),
    ))
}

fn pack_patches(
    frames: &[NormalizedImage],
    config: &VisualConfig,
    duplicate_image: bool,
) -> Result<(Array, Array), Error> {
    let first = frames
        .first()
        .ok_or_else(|| Error::Processor("Muse-Glimmer patch input is empty".into()))?;
    if first.height() % config.patch_size != 0 || first.width() % config.patch_size != 0 {
        return Err(Error::Processor(format!(
            "Muse-Glimmer processed media {}x{} is not divisible by patch size {}",
            first.width(),
            first.height(),
            config.patch_size
        )));
    }
    if frames.iter().any(|frame| {
        frame.width() != first.width()
            || frame.height() != first.height()
            || frame.channels() != first.channels()
    }) {
        return Err(Error::Processor(
            "Muse-Glimmer video frames must have identical dimensions".into(),
        ));
    }
    let temporal = config.temporal_patch_size;
    if !duplicate_image && frames.len() != temporal {
        return Err(Error::Processor(format!(
            "Muse-Glimmer video group has {} frames, expected {temporal}",
            frames.len()
        )));
    }
    let grid_h = first.height() / config.patch_size;
    let grid_w = first.width() / config.patch_size;
    let patch_width = temporal * first.channels() * config.patch_size * config.patch_size;
    let mut values = Vec::with_capacity(grid_h * grid_w * patch_width);
    for patch_y in 0..grid_h {
        for patch_x in 0..grid_w {
            let mut video_frames = frames.iter();
            for _ in 0..temporal {
                let frame = if duplicate_image {
                    first
                } else {
                    video_frames
                        .next()
                        .expect("video frame count validated above")
                };
                for channel in 0..frame.channels() {
                    for y in 0..config.patch_size {
                        for x in 0..config.patch_size {
                            values.push(frame.get(
                                channel,
                                patch_y * config.patch_size + y,
                                patch_x * config.patch_size + x,
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok((
        Array::from_slice(&values, &[(grid_h * grid_w) as i32, patch_width as i32]),
        Array::from_slice(&[1i32, grid_h as i32, grid_w as i32], &[1, 3]),
    ))
}

#[cfg(test)]
mod tests {
    use super::smart_resize;

    #[test]
    fn smart_resize_preserves_release_merge_divisibility_and_budget() {
        for (height, width) in [(1200, 800), (800, 1200), (28, 4096), (4096, 28)] {
            let (target_h, target_w) = smart_resize(height, width, 28, 4096).unwrap();
            assert_eq!(target_h % 28, 0);
            assert_eq!(target_w % 28, 0);
            assert!(target_h / 28 * (target_w / 28) <= 4096);
        }
    }
}
