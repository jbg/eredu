//! Gemma 4 image, video, and audio host preprocessing.

#[cfg(feature = "image")]
use eredu_architectures::processor_plan::{
    Gemma4ImagePlan, Gemma4VideoPlan, RgbResample, RgbTransformPlan,
};
use eredu_architectures::processor_plan::{Gemma4ProcessorPlan, ProcessorPlanError};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{InputExtent, InputMetadataKey};
#[cfg(any(feature = "image", feature = "audio"))]
use safemlx::Array;

use crate::backend::error::Error;
#[cfg(feature = "audio")]
use crate::backend::runtime::media::audio::{extract_log_mel, LogMelConfig};
#[cfg(feature = "image")]
use crate::backend::runtime::media::image::{
    rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3, NormalizedImage,
    RgbImage, RgbImageView,
};
#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::runtime::media::input::Modality;
#[cfg(feature = "image")]
use crate::backend::runtime::media::video::validate_rgb_frames;
#[cfg(any(feature = "image", feature = "audio"))]
use crate::backend::runtime::media::MediaPayload;
#[cfg(feature = "image")]
use crate::backend::runtime::media::VideoFrames;
use crate::backend::runtime::media::{
    media_input_part, prepared_model_input, push_text_token_ids, InputPart, MediaInput,
    PreparedModelInput, ProcessorInput, ProcessorPreparationError,
};

#[derive(Debug, Clone)]
pub struct Gemma4Processor {
    plan: Gemma4ProcessorPlan,
}

impl Gemma4Processor {
    pub fn from_plan(plan: Gemma4ProcessorPlan) -> Option<Self> {
        let supported = cfg!(feature = "image") && plan.has_image()
            || cfg!(feature = "audio") && plan.has_audio();
        supported.then_some(Self { plan })
    }

    pub fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let mut parts = Vec::new();
        for item in input {
            match *item {
                ProcessorInput::TokenIds(token_ids) => {
                    push_text_token_ids(&mut parts, token_ids)?;
                }
                ProcessorInput::Media(media) => {
                    self.push_media_parts(&mut parts, media, encode_text)?;
                }
            }
        }
        Ok(prepared_model_input(parts)?)
    }

    fn push_media_parts<E>(
        &self,
        parts: &mut Vec<InputPart>,
        item: MediaInput<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        #[cfg(not(feature = "image"))]
        let _ = &encode_text;
        #[cfg(not(any(feature = "image", feature = "audio")))]
        let _ = &parts;
        match (item.modality, item.payload) {
            #[cfg(feature = "image")]
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                let plan = self
                    .plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(processor_error)?;
                push_text_token_ids(parts, &[plan.framing.start_token_id])?;
                parts.push(self.process_image(image, &plan)?);
                push_text_token_ids(parts, &[plan.framing.end_token_id])?;
                Ok(())
            }
            #[cfg(feature = "image")]
            (Modality::Video, MediaPayload::VideoFrames(video)) => {
                parts.extend(self.process_video(video, encode_text)?);
                Ok(())
            }
            #[cfg(feature = "audio")]
            (Modality::Audio, MediaPayload::AudioF32(waveform)) => {
                let plan = self.plan.audio().map_err(processor_error)?;
                push_text_token_ids(parts, &[plan.framing.start_token_id])?;
                parts.push(self.process_audio(waveform, &plan)?);
                push_text_token_ids(parts, &[plan.framing.end_token_id])?;
                Ok(())
            }
            _ => Err(Error::Processor(format!(
                "Gemma 4 processor does not support {} media with the enabled features",
                item.modality.as_str()
            ))
            .into()),
        }
    }

    #[cfg(feature = "image")]
    fn process_image(
        &self,
        image: RgbImageView<'_>,
        plan: &Gemma4ImagePlan,
    ) -> Result<InputPart, Error> {
        let resized = resize_rgb8(image, plan.transform)?;
        let normalized = rescale_and_normalize_rgb8(
            resized.as_view(),
            plan.transform.rescale_factor,
            plan.transform.mean,
            plan.transform.std,
        )?;
        let (patches, positions, grid, extent) =
            pack_patches(&normalized, plan.patch_size, plan.max_patches)?;
        media_input_part(
            Modality::Image,
            patches,
            [
                (InputMetadataKey::PatchGrid, grid),
                (InputMetadataKey::PatchPositions, positions),
            ],
            [InputExtent::PatchGrid {
                time: extent[0],
                height: extent[1],
                width: extent[2],
            }],
        )
    }

    #[cfg(feature = "image")]
    fn process_video<E>(
        &self,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Vec<InputPart>, ProcessorPreparationError<E>> {
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
        self.materialize_video(video, &plan, encode_text)
    }

    #[cfg(feature = "image")]
    fn materialize_video<E>(
        &self,
        video: VideoFrames<'_>,
        plan: &Gemma4VideoPlan,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Vec<InputPart>, ProcessorPreparationError<E>> {
        let mut replacement = Vec::with_capacity(plan.frames.len() * 3);
        for frame in &plan.frames {
            let mut prefix =
                encode_text(&frame.timestamp_text).map_err(ProcessorPreparationError::Text)?;
            prefix.push(plan.framing.start_token_id);
            push_text_token_ids(&mut replacement, &prefix)?;

            let resized = resize_rgb8(video.frames[frame.source_index], plan.transform)?;
            let normalized = rescale_and_normalize_rgb8(
                resized.as_view(),
                plan.transform.rescale_factor,
                plan.transform.mean,
                plan.transform.std,
            )?;
            let (patches, positions, grid, extent) =
                pack_patches(&normalized, plan.patch_size, plan.max_patches)?;
            replacement.push(media_input_part(
                Modality::Video,
                patches,
                [
                    (InputMetadataKey::PatchGrid, grid),
                    (InputMetadataKey::PatchPositions, positions),
                ],
                [InputExtent::PatchGrid {
                    time: extent[0],
                    height: extent[1],
                    width: extent[2],
                }],
            )?);
            push_text_token_ids(&mut replacement, &[plan.framing.end_token_id])?;
        }
        Ok(replacement)
    }

    #[cfg(feature = "audio")]
    fn process_audio(
        &self,
        waveform: crate::backend::runtime::media::audio::AudioWaveform<'_>,
        plan: &eredu_architectures::processor_plan::Gemma4AudioPlan,
    ) -> Result<InputPart, Error> {
        let features = extract_log_mel(
            waveform,
            &LogMelConfig {
                sample_rate: plan.sample_rate,
                frame_length: plan.frame_length,
                hop_length: plan.hop_length,
                fft_length: plan.fft_length,
                mel_bins: plan.mel_bins,
                min_frequency: plan.min_frequency,
                max_frequency: plan.max_frequency,
                mel_floor: plan.mel_floor,
                max_samples: plan.max_samples,
                pad_to_multiple: plan.pad_to_multiple,
            },
        )?;
        let tensor = Array::from_slice(
            &features.values,
            &[1, features.frames as i32, features.mel_bins as i32],
        );
        let valid_frames = features.mask.iter().filter(|valid| **valid).count();
        let mask = Array::from_slice(&features.mask, &[1, features.frames as i32]);
        media_input_part(
            Modality::Audio,
            tensor,
            [(InputMetadataKey::AudioMask, mask)],
            [InputExtent::AudioValidFrames(valid_frames)],
        )
    }
}

fn processor_error(error: ProcessorPlanError) -> Error {
    Error::Processor(error.to_string())
}

#[cfg(feature = "image")]
fn resize_rgb8(image: RgbImageView<'_>, plan: RgbTransformPlan) -> Result<RgbImage, Error> {
    match plan.resample {
        RgbResample::Bicubic => resize_rgb8_bicubic(image, plan.width as u32, plan.height as u32),
        RgbResample::Lanczos3 => resize_rgb8_lanczos3(image, plan.width as u32, plan.height as u32),
    }
}

#[cfg(feature = "image")]
fn pack_patches(
    image: &NormalizedImage,
    patch_size: usize,
    max_patches: usize,
) -> Result<(Array, Array, Array, [usize; 3]), Error> {
    if !image.height().is_multiple_of(patch_size) || !image.width().is_multiple_of(patch_size) {
        return Err(Error::Processor(format!(
            "Gemma 4 image dimensions {}x{} are not divisible by patch size {patch_size}",
            image.height(),
            image.width()
        )));
    }
    let patch_height = image.height() / patch_size;
    let patch_width = image.width() / patch_size;
    let patch_count = patch_height * patch_width;
    if patch_count > max_patches {
        return Err(Error::Processor(format!(
            "Gemma 4 image produced {patch_count} patches, exceeding {max_patches}"
        )));
    }
    let patch_dims = image.channels() * patch_size * patch_size;
    let mut patches = vec![0.0f32; max_patches * patch_dims];
    let mut positions = vec![-1i32; max_patches * 2];
    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let patch_index = patch_y * patch_width + patch_x;
            positions[patch_index * 2] = patch_x as i32;
            positions[patch_index * 2 + 1] = patch_y as i32;
            let mut output = patch_index * patch_dims;
            for inner_y in 0..patch_size {
                for inner_x in 0..patch_size {
                    for channel in 0..image.channels() {
                        patches[output] = image.get(
                            channel,
                            patch_y * patch_size + inner_y,
                            patch_x * patch_size + inner_x,
                        );
                        output += 1;
                    }
                }
            }
        }
    }
    Ok((
        Array::from_slice(&patches, &[1, max_patches as i32, patch_dims as i32]),
        Array::from_slice(&positions, &[1, max_patches as i32, 2]),
        Array::from_slice(&[1, patch_height as i32, patch_width as i32], &[1, 3]),
        [1, patch_height, patch_width],
    ))
}

#[cfg(all(test, feature = "image"))]
mod tests {
    use std::collections::HashMap;

    use eredu_architectures::processor_plan::Gemma4ProcessorPlan;
    use eredu_core::{InputMetadataKey, VideoSampling};
    use eredu_gguf::MetadataValue;

    use super::Gemma4Processor;
    use crate::{
        backend::runtime::media::input::{InputPayload, Modality},
        backend::runtime::media::{MediaInput, ProcessorInput, RgbImageView},
    };

    fn tiny_processor() -> Gemma4Processor {
        let model = br#"{
            "boi_token_id":43,"eoi_token_id":44,
            "vision_soft_tokens_per_image":70,
            "vision_config":{"patch_size":2,"pooling_kernel_size":1}
        }"#;
        let video = br#"{
            "patch_size":2,"pooling_kernel_size":1,
            "max_soft_tokens":70,"num_frames":32
        }"#;
        Gemma4Processor {
            plan: Gemma4ProcessorPlan::from_hf_json(model, None, Some(video))
                .unwrap()
                .unwrap(),
        }
    }

    #[test]
    fn gguf_processor_uses_projector_geometry_and_language_boundaries() {
        let model = HashMap::from([
            ("gemma4.boi_token_id".into(), MetadataValue::Uint32(43)),
            ("gemma4.eoi_token_id".into(), MetadataValue::Uint32(44)),
        ]);
        let projector = HashMap::from([
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.pooling_kernel_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.max_soft_tokens".into(),
                MetadataValue::Uint32(280),
            ),
        ]);
        let plan = Gemma4ProcessorPlan::from_gguf_metadata(&model, &projector).unwrap();
        let processor = Gemma4Processor::from_plan(plan).unwrap();
        let plan = processor.plan.image(4, 4).unwrap();
        assert_eq!(plan.patch_size, 2);
        assert_eq!(plan.max_patches, 1120);
        assert_eq!(plan.framing.start_token_id, 43);
        assert_eq!(plan.framing.end_token_id, 44);
    }

    #[test]
    fn processor_wraps_ordered_image_with_boundary_tokens() {
        let processor = tiny_processor();
        let pixels = vec![128u8; 4 * 4 * 3];
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let prepared = processor
            .prepare_input::<std::convert::Infallible>(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality(), Modality::Image);
        assert!(matches!(parts[2].payload(), InputPayload::Tensor(_)));
        assert!(parts[2]
            .metadata_value(InputMetadataKey::PatchPositions)
            .is_some());
    }

    #[test]
    fn processor_interleaves_timestamped_video_frames() {
        let processor = tiny_processor();
        let pixels = vec![128u8; 4 * 4 * 3];
        let frames = [
            RgbImageView::packed(&pixels, 4, 4).unwrap(),
            RgbImageView::packed(&pixels, 4, 4).unwrap(),
        ];
        let mut encoded = Vec::new();
        let prepared = processor
            .prepare_input::<std::convert::Infallible>(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(MediaInput::video_rgb8_with_sampling(
                        &frames,
                        Some(1.0),
                        VideoSampling::ProcessorDefault,
                    )),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |text| {
                    encoded.push(text.to_string());
                    Ok(vec![90])
                },
            )
            .unwrap();
        let parts = prepared.input_parts();
        let video_parts = parts
            .iter()
            .filter(|part| part.modality() == Modality::Video)
            .collect::<Vec<_>>();
        assert_eq!(video_parts.len(), 2);
        assert!(video_parts.iter().all(|part| part
            .metadata_value(InputMetadataKey::PatchPositions)
            .is_some()));
        assert_eq!(encoded, vec!["00:00 ", " 00:01 "]);
    }
}

#[cfg(all(test, feature = "audio"))]
mod audio_tests {
    use eredu_architectures::processor_plan::Gemma4ProcessorPlan;
    use eredu_core::InputMetadataKey;

    use super::Gemma4Processor;
    use crate::{
        backend::runtime::media::input::{InputPayload, Modality},
        backend::runtime::media::{MediaInput, ProcessorInput},
    };

    #[test]
    fn processor_wraps_ordered_audio_with_boundary_tokens() {
        let model = br#"{
            "boa_token_id":43,"eoa_token_id":44,"audio_config":{}
        }"#;
        let processor = Gemma4Processor {
            plan: Gemma4ProcessorPlan::from_hf_json(model, None, None)
                .unwrap()
                .unwrap(),
        };
        let samples = vec![0.0f32; 16_000];
        let audio = MediaInput::audio_f32(&samples, 16_000).unwrap();
        let prepared = processor
            .prepare_input::<std::convert::Infallible>(
                &[
                    ProcessorInput::TokenIds(&[7]),
                    ProcessorInput::Media(audio),
                    ProcessorInput::TokenIds(&[8]),
                ],
                &mut |_| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality(), Modality::Audio);
        assert!(matches!(parts[2].payload(), InputPayload::Tensor(_)));
        assert!(parts[2]
            .metadata_value(InputMetadataKey::AudioMask)
            .is_some());
    }
}
