//! Hugging Face-compatible Muse-Glimmer image and experimental video host processor.

use eredu_architectures::processor_plan::{
    MuseImagePlan, MusePatchPlan, MuseProcessorPlan, ProcessorPlanError, RgbResample,
    RgbTransformPlan,
};
use eredu_core::{InputMetadataKey, InputModality};
use safemlx::Array;

use crate::{
    backend::error::Error,
    backend::runtime::media::{
        image::{rescale_and_normalize_rgb8, resize_rgb8_lanczos3, NormalizedImage, RgbImageView},
        media_input_part, prepared_model_input, push_text_token_ids,
        video::validate_rgb_frames,
        InputPart, MediaInput, MediaPayload, PreparedModelInput, ProcessorInput,
        ProcessorPreparationError, VideoFrames,
    },
};

/// Processor state loaded from the release's `processor_config.json`.
#[derive(Debug, Clone)]
pub struct MuseGlimmerProcessor {
    plan: MuseProcessorPlan,
}

impl MuseGlimmerProcessor {
    pub fn from_plan(plan: MuseProcessorPlan) -> Self {
        Self { plan }
    }

    pub fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let mut parts = Vec::new();
        for item in input {
            match *item {
                ProcessorInput::TokenIds(ids) => push_text_token_ids(&mut parts, ids)?,
                ProcessorInput::Media(media) => self.push_media(&mut parts, media, encode_text)?,
            }
        }
        Ok(prepared_model_input(parts)?)
    }

    fn push_media<E>(
        &self,
        parts: &mut Vec<InputPart>,
        media: MediaInput<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        match (media.modality, media.payload) {
            (InputModality::Image, MediaPayload::Rgb8(image)) => {
                let plan = self
                    .plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(processor_error)?;
                push_text_token_ids(
                    parts,
                    &encode_text(plan.start_text).map_err(ProcessorPreparationError::Text)?,
                )?;
                parts.push(process_image(image, plan)?);
                push_text_token_ids(
                    parts,
                    &encode_text(plan.end_text).map_err(ProcessorPreparationError::Text)?,
                )?;
                Ok(())
            }
            (InputModality::Video, MediaPayload::VideoFrames(video)) => {
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
        parts: &mut Vec<InputPart>,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        let (width, height) = validate_rgb_frames(video.frames)?;
        let plan = self
            .plan
            .video(
                video.frames.len(),
                height as usize,
                width as usize,
                video.source_fps,
                video.sampling,
            )
            .map_err(processor_error)?;
        push_text_token_ids(
            parts,
            &encode_text(plan.start_text).map_err(ProcessorPreparationError::Text)?,
        )?;
        for group in &plan.groups {
            push_text_token_ids(
                parts,
                &encode_text(&group.timestamp_text).map_err(ProcessorPreparationError::Text)?,
            )?;
            let frames = group
                .source_indices
                .iter()
                .map(|index| normalize(video.frames[*index], plan.transform))
                .collect::<Result<Vec<_>, _>>()?;
            parts.push(process_video_group(&frames, plan.patches)?);
            push_text_token_ids(
                parts,
                &encode_text(group.boundary_text).map_err(ProcessorPreparationError::Text)?,
            )?;
        }
        Ok(())
    }
}

fn processor_error(error: ProcessorPlanError) -> Error {
    Error::Processor(error.to_string())
}

fn normalize(image: RgbImageView<'_>, plan: RgbTransformPlan) -> Result<NormalizedImage, Error> {
    let resized = match plan.resample {
        RgbResample::Lanczos3 => {
            resize_rgb8_lanczos3(image, plan.width as u32, plan.height as u32)?
        }
        RgbResample::Bicubic => {
            return Err(Error::Processor(
                "Muse-Glimmer processor requires Lanczos interpolation".into(),
            ))
        }
    };
    rescale_and_normalize_rgb8(resized.as_view(), plan.rescale_factor, plan.mean, plan.std)
}

fn process_image(image: RgbImageView<'_>, plan: MuseImagePlan) -> Result<InputPart, Error> {
    let image = normalize(image, plan.transform)?;
    let (patches, grid) = pack_patches(std::slice::from_ref(&image), plan.patches, true)?;
    media_input_part(
        InputModality::Image,
        patches,
        [(InputMetadataKey::PatchGrid, grid)],
        [],
    )
}

fn process_video_group(
    frames: &[NormalizedImage],
    plan: MusePatchPlan,
) -> Result<InputPart, Error> {
    let (patches, grid) = pack_patches(frames, plan, false)?;
    media_input_part(
        InputModality::Video,
        patches,
        [(InputMetadataKey::PatchGrid, grid)],
        [],
    )
}

fn pack_patches(
    frames: &[NormalizedImage],
    config: MusePatchPlan,
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
