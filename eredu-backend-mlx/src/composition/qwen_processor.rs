// Qwen image/video protocol preprocessing for neutral prepared inputs.

use std::{fs, path::Path};

use eredu_architectures::processor_plan::{
    ProcessorPlanError, QwenImagePlan, QwenPatchPlan, QwenProcessorPlan, QwenVideoPlan,
    RgbResample, RgbTransformPlan, PROCESSOR_CONFIG_FILENAME, VIDEO_PROCESSOR_CONFIG_FILENAME,
};
use safemlx::Array;

use crate::backend::runtime::media::video::validate_rgb_frames;
use crate::backend::runtime::media::{
    image::{
        rescale_and_normalize_rgb8, resize_rgb8_bicubic, resize_rgb8_lanczos3, NormalizedImage,
        RgbImage, RgbImageView,
    },
    prepared_model_input, push_text_token_ids, MediaInput, MediaPayload, OwnedInputMetadata,
    PreparedInputPart, PreparedModelInput, ProcessorInput, ProcessorPreparationError, VideoFrames,
};
use crate::{backend::error::Error, backend::runtime::media::input::Modality};

#[derive(Debug, Clone)]
pub struct QwenProcessor {
    plan: QwenProcessorPlan,
}

impl QwenProcessor {
    pub fn load_directory(model_dir: &Path) -> Result<Option<Self>, Error> {
        let model = fs::read(model_dir.join("config.json"))?;
        Self::load(model_dir, &model)
    }

    pub fn load(model_dir: &Path, model: &[u8]) -> Result<Option<Self>, Error> {
        let image = read_optional(&model_dir.join(PROCESSOR_CONFIG_FILENAME))?;
        let video = read_optional(&model_dir.join(VIDEO_PROCESSOR_CONFIG_FILENAME))?;
        QwenProcessorPlan::from_hf_json(model, image.as_deref(), video.as_deref())
            .map_err(processor_error)
            .map(|plan| plan.map(|plan| Self { plan }))
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
                    push_text_token_ids(&mut parts, token_ids);
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
        parts: &mut Vec<PreparedInputPart>,
        item: MediaInput<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<(), ProcessorPreparationError<E>> {
        match (item.modality, item.payload) {
            (Modality::Image, MediaPayload::Rgb8(image)) => {
                let plan = self
                    .plan
                    .image(image.height() as usize, image.width() as usize)
                    .map_err(processor_error)?;
                push_text_token_ids(parts, &[plan.framing.start_token_id]);
                parts.push(self.process_image(image, &plan)?);
                push_text_token_ids(parts, &[plan.framing.end_token_id]);
            }
            (Modality::Video, MediaPayload::VideoFrames(video)) => {
                parts.extend(self.process_video(video, encode_text)?);
            }
            (modality, _) => {
                return Err(Error::Processor(format!(
                    "Qwen processor does not support {} media yet",
                    modality.as_str()
                ))
                .into());
            }
        }
        Ok(())
    }

    fn process_image(
        &self,
        image: RgbImageView<'_>,
        plan: &QwenImagePlan,
    ) -> Result<PreparedInputPart, Error> {
        let resized = resize_rgb8(image, plan.transform)?;
        let normalized = rescale_and_normalize_rgb8(
            resized.as_view(),
            plan.transform.rescale_factor,
            plan.transform.mean,
            plan.transform.std,
        )?;
        let (patches, grid_thw) = pack_image_patches(&normalized, plan.patches)?;
        Ok(PreparedInputPart::media_tensor(
            Modality::Image,
            patches,
            OwnedInputMetadata::patch_grid(grid_thw),
        ))
    }

    fn process_video<E>(
        &self,
        video: VideoFrames<'_>,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Vec<PreparedInputPart>, ProcessorPreparationError<E>> {
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

    fn materialize_video<E>(
        &self,
        video: VideoFrames<'_>,
        plan: &QwenVideoPlan,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Vec<PreparedInputPart>, ProcessorPreparationError<E>> {
        let mut parts = Vec::with_capacity(plan.groups.len() * 3);
        for group in &plan.groups {
            let mut prefix = encode_text(&group.timestamp_text)
                .map_err(ProcessorPreparationError::Text)?;
            prefix.push(plan.framing.start_token_id);
            push_text_token_ids(&mut parts, &prefix);
            let mut frames = Vec::with_capacity(group.source_indices.len());
            for &index in &group.source_indices {
                let resized = resize_rgb8(video.frames[index], plan.transform)?;
                frames.push(rescale_and_normalize_rgb8(
                    resized.as_view(),
                    plan.transform.rescale_factor,
                    plan.transform.mean,
                    plan.transform.std,
                )?);
            }
            let (patches, grid_thw) = pack_video_patches(&frames, plan.patches)?;
            parts.push(PreparedInputPart::media_tensor(
                Modality::Video,
                patches,
                OwnedInputMetadata::patch_grid(grid_thw),
            ));
            push_text_token_ids(&mut parts, &[plan.framing.end_token_id]);
        }
        Ok(parts)
    }
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    path.exists().then(|| fs::read(path)).transpose().map_err(Into::into)
}

fn processor_error(error: ProcessorPlanError) -> Error {
    Error::Processor(error.to_string())
}

fn resize_rgb8(image: RgbImageView<'_>, plan: RgbTransformPlan) -> Result<RgbImage, Error> {
    match plan.resample {
        RgbResample::Bicubic => {
            resize_rgb8_bicubic(image, plan.width as u32, plan.height as u32)
        }
        RgbResample::Lanczos3 => {
            resize_rgb8_lanczos3(image, plan.width as u32, plan.height as u32)
        }
    }
}

fn pack_image_patches(
    image: &NormalizedImage,
    config: QwenPatchPlan,
) -> Result<(Array, Array), Error> {
    let grid_h = image.height() / config.patch_size;
    let grid_w = image.width() / config.patch_size;
    if !image.height().is_multiple_of(config.patch_size)
        || !image.width().is_multiple_of(config.patch_size)
    {
        return Err(Error::Processor(format!(
            "processed image {}x{} is not divisible by patch size {}",
            image.width(),
            image.height(),
            config.patch_size
        )));
    }
    if !grid_h.is_multiple_of(config.merge_size) || !grid_w.is_multiple_of(config.merge_size) {
        return Err(Error::Processor(format!(
            "image patch grid {grid_h}x{grid_w} is not divisible by merge size {}",
            config.merge_size
        )));
    }

    let patch_count = grid_h * grid_w;
    let patch_width =
        image.channels() * config.temporal_patch_size * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(patch_count * patch_width);
    for block_y in 0..grid_h / config.merge_size {
        for block_x in 0..grid_w / config.merge_size {
            for merge_y in 0..config.merge_size {
                for merge_x in 0..config.merge_size {
                    let patch_y = (block_y * config.merge_size + merge_y) * config.patch_size;
                    let patch_x = (block_x * config.merge_size + merge_x) * config.patch_size;
                    for channel in 0..image.channels() {
                        for _time in 0..config.temporal_patch_size {
                            for y in 0..config.patch_size {
                                for x in 0..config.patch_size {
                                    patches.push(image.get(channel, patch_y + y, patch_x + x));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let patches = Array::from_slice(&patches, &[patch_count as i32, patch_width as i32]);
    let grid_thw = Array::from_slice(&[1i32, grid_h as i32, grid_w as i32], &[1, 3]);
    Ok((patches, grid_thw))
}

fn pack_video_patches(
    frames: &[NormalizedImage],
    config: QwenPatchPlan,
) -> Result<(Array, Array), Error> {
    let first = frames
        .first()
        .ok_or_else(|| Error::Processor("video must contain processed frames".to_string()))?;
    if !frames.len().is_multiple_of(config.temporal_patch_size) {
        return Err(Error::Processor(format!(
            "{} processed video frames are not divisible by temporal patch size {}",
            frames.len(),
            config.temporal_patch_size
        )));
    }
    if frames.iter().any(|frame| {
        frame.width() != first.width()
            || frame.height() != first.height()
            || frame.channels() != first.channels()
    }) {
        return Err(Error::Processor(
            "processed video frames must have identical dimensions".to_string(),
        ));
    }
    if first.height() % config.patch_size != 0 || first.width() % config.patch_size != 0 {
        return Err(Error::Processor(format!(
            "processed video frame {}x{} is not divisible by patch size {}",
            first.width(),
            first.height(),
            config.patch_size
        )));
    }
    let grid_t = frames.len() / config.temporal_patch_size;
    let grid_h = first.height() / config.patch_size;
    let grid_w = first.width() / config.patch_size;
    if !grid_h.is_multiple_of(config.merge_size) || !grid_w.is_multiple_of(config.merge_size) {
        return Err(Error::Processor(format!(
            "video patch grid {grid_h}x{grid_w} is not divisible by merge size {}",
            config.merge_size
        )));
    }

    let patch_count = grid_t * grid_h * grid_w;
    let patch_width =
        first.channels() * config.temporal_patch_size * config.patch_size * config.patch_size;
    let mut patches = Vec::with_capacity(patch_count * patch_width);
    for temporal_group in 0..grid_t {
        for block_y in 0..grid_h / config.merge_size {
            for block_x in 0..grid_w / config.merge_size {
                for merge_y in 0..config.merge_size {
                    for merge_x in 0..config.merge_size {
                        let patch_y = (block_y * config.merge_size + merge_y) * config.patch_size;
                        let patch_x = (block_x * config.merge_size + merge_x) * config.patch_size;
                        for channel in 0..first.channels() {
                            for time in 0..config.temporal_patch_size {
                                let frame =
                                    &frames[temporal_group * config.temporal_patch_size + time];
                                for y in 0..config.patch_size {
                                    for x in 0..config.patch_size {
                                        patches.push(frame.get(channel, patch_y + y, patch_x + x));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let patches = Array::from_slice(&patches, &[patch_count as i32, patch_width as i32]);
    let grid_thw = Array::from_slice(&[grid_t as i32, grid_h as i32, grid_w as i32], &[1, 3]);
    Ok((patches, grid_thw))
}

#[cfg(test)]
mod tests {
    use eredu_architectures::processor_plan::{QwenPatchPlan, QwenProcessorPlan};

    use super::{pack_image_patches, QwenProcessor};
    use crate::{
        backend::runtime::media::input::{InputPayload, Modality},
        backend::runtime::media::{
            image::{rescale_and_normalize_rgb8, RgbImageView},
            MediaInput, ProcessorInput, VideoSampling,
        },
    };

    fn tiny_patches() -> QwenPatchPlan {
        QwenPatchPlan {
            patch_size: 2,
            temporal_patch_size: 2,
            merge_size: 2,
        }
    }

    fn tiny_processor() -> QwenProcessor {
        let model = br#"{"vision_start_token_id":44,"vision_end_token_id":45}"#;
        let visual = br#"{
            "size":{"shortest_edge":16,"longest_edge":16},
            "patch_size":2,"temporal_patch_size":2,"merge_size":2,
            "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
            "min_frames":1,"max_frames":8
        }"#;
        QwenProcessor {
            plan: QwenProcessorPlan::from_hf_json(model, Some(visual), Some(visual))
                .unwrap()
                .unwrap(),
        }
    }

    #[test]
    fn patch_packing_groups_merge_cells_and_duplicates_time() {
        let mut pixels = Vec::new();
        for value in 0u8..16 {
            pixels.extend_from_slice(&[value, 100 + value, 200 + value]);
        }
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let normalized =
            rescale_and_normalize_rgb8(image, 1.0 / 255.0, [0.0; 3], [1.0; 3]).unwrap();
        let (patches, grid) = pack_image_patches(&normalized, tiny_patches()).unwrap();
        assert_eq!(patches.shape(), &[4, 24]);
        assert_eq!(grid.evaluated().unwrap().as_slice::<i32>(), &[1, 2, 2]);
        let evaluated = patches.evaluated().unwrap();
        let values = evaluated.as_slice::<f32>();
        let first_channel = [0.0, 1.0, 4.0, 5.0].map(|value| value / 255.0);
        assert_eq!(&values[..4], &first_channel);
        assert_eq!(&values[4..8], &first_channel);
    }

    #[test]
    fn processor_wraps_ordered_image_with_vision_boundaries() {
        let processor = tiny_processor();
        let pixels = vec![128u8; 4 * 4 * 3];
        let image = RgbImageView::packed(&pixels, 4, 4).unwrap();
        let prepared = processor
            .prepare_input::<std::convert::Infallible>(
                &[
                    ProcessorInput::TokenIds(&[10]),
                    ProcessorInput::Media(MediaInput::image_rgb8(image)),
                    ProcessorInput::TokenIds(&[11]),
                ],
                &mut |_text| Ok(Vec::new()),
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].modality, Modality::Text);
        assert_eq!(parts[2].modality, Modality::Image);
        assert_eq!(parts[4].modality, Modality::Text);
        let InputPayload::TokenIds(start) = parts[1].payload else {
            panic!("expected vision-start token");
        };
        assert_eq!(start.evaluated().unwrap().as_slice::<u32>(), &[44]);
        let InputPayload::TokenIds(end) = parts[3].payload else {
            panic!("expected vision-end token");
        };
        assert_eq!(end.evaluated().unwrap().as_slice::<u32>(), &[45]);
        let InputPayload::Tensor(patches) = parts[2].payload else {
            panic!("expected processed image tensor");
        };
        assert_eq!(patches.shape(), &[4, 24]);
        assert_eq!(
            parts[2]
                .metadata
                .patch_grid
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2, 2]
        );
    }

    #[test]
    fn processor_expands_video_timestamps_and_packs_temporal_frames() {
        let processor = tiny_processor();
        let frame_pixels = (0..4)
            .map(|frame| vec![frame as u8 * 32; 4 * 4 * 3])
            .collect::<Vec<_>>();
        let frames = frame_pixels
            .iter()
            .map(|pixels| RgbImageView::packed(pixels, 4, 4).unwrap())
            .collect::<Vec<_>>();
        let mut timestamp_text = Vec::new();
        let prepared = processor
            .prepare_input::<std::convert::Infallible>(
                &[
                    ProcessorInput::TokenIds(&[10]),
                    ProcessorInput::Media(MediaInput::video_rgb8_with_sampling(
                        &frames,
                        Some(2.0),
                        VideoSampling::All,
                    )),
                    ProcessorInput::TokenIds(&[11]),
                ],
                &mut |text| {
                    timestamp_text.push(text.to_string());
                    Ok(vec![90 + timestamp_text.len() as u32])
                },
            )
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(timestamp_text, vec!["<0.2 seconds>", "<1.2 seconds>"]);
        assert_eq!(parts.len(), 8);
        assert_eq!(parts[2].modality, Modality::Video);
        assert_eq!(parts[5].modality, Modality::Video);
        let InputPayload::TokenIds(replacement) = parts[1].payload else {
            panic!("expected timestamp replacement tokens");
        };
        assert_eq!(
            replacement.evaluated().unwrap().as_slice::<u32>(),
            &[91, 44]
        );
        let InputPayload::Tensor(first_patches) = parts[2].payload else {
            panic!("expected first processed video tensor");
        };
        assert_eq!(first_patches.shape(), &[4, 24]);
        assert_eq!(
            parts[2]
                .metadata
                .patch_grid
                .unwrap()
                .evaluated()
                .unwrap()
                .as_slice::<i32>(),
            &[1, 2, 2]
        );
        let InputPayload::TokenIds(first_end) = parts[3].payload else {
            panic!("expected first vision-end token");
        };
        assert_eq!(first_end.evaluated().unwrap().as_slice::<u32>(), &[45]);
        let InputPayload::TokenIds(second_prefix) = parts[4].payload else {
            panic!("expected second timestamp prefix tokens");
        };
        assert_eq!(
            second_prefix.evaluated().unwrap().as_slice::<u32>(),
            &[92, 44]
        );
        let InputPayload::Tensor(second_patches) = parts[5].payload else {
            panic!("expected second processed video tensor");
        };
        assert_eq!(second_patches.shape(), &[4, 24]);
    }
}
