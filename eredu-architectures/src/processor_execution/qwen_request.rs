//! Retained Qwen request geometry. Planning performs no resize, normalization or packing.
use super::*;
use std::borrow::Cow;

pub(super) struct Request<'a> {
    steps: Vec<Step<'a>>,
    pub resources: ProcessorRequestResources,
}
enum Step<'a> {
    Tokens(Cow<'a, [u32]>),
    Image(&'a RgbImage, QwenImagePlan),
    Video {
        video: &'a eredu_core::Video,
        indices: Vec<usize>,
        transform: RgbTransformPlan,
        patches: QwenPatchPlan,
    },
}
fn add(a: u64, b: u64) -> Result<u64, ProcessorResourceError> {
    a.checked_add(b).ok_or(ProcessorResourceError::Overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, ProcessorResourceError> {
    a.checked_mul(b).ok_or(ProcessorResourceError::Overflow)
}
fn check(
    budget: Option<ProcessorRequestBudget>,
    report: &ProcessorRequestResources,
) -> Result<(), ProcessorResourceError> {
    if let Some(budget) = budget {
        budget.admit(report)?;
    }
    Ok(())
}
impl<'a> Request<'a> {
    pub fn plan<E: std::fmt::Display>(
        plan: &QwenProcessorPlan,
        request: &'a TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
        budget: Option<ProcessorRequestBudget>,
    ) -> Result<Self, ProcessorExecutionError<E, eredu_media::MediaError>> {
        let mut resources = ProcessorRequestResources::default();
        // Bound planning itself before sampling-index vectors or timestamp callbacks.
        resources.planning_items = mul(request.segments().len() as u64, 24)?;
        let mut step_capacity = mul(request.segments().len() as u64, 3)?;
        for segment in request.segments() {
            match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => {
                    resources.decoded_input_bytes =
                        add(resources.decoded_input_bytes, mul(ids.len() as u64, 4)?)?;
                    resources.decoder_positions =
                        add(resources.decoder_positions, ids.len() as u64)?;
                    resources.planning_items = add(resources.planning_items, ids.len() as u64)?;
                }
                TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                    resources.decoded_input_bytes =
                        add(resources.decoded_input_bytes, image.pixels().len() as u64)?;
                }
                TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                    let frames = plan
                        .video_frame_capacity_bound(video.frames().len())
                        .map_err(plan_error)?;
                    resources.planning_items =
                        add(resources.planning_items, mul(frames as u64, 16)?)?;
                    step_capacity = add(step_capacity, mul(frames as u64, 3)?)?;
                    for frame in video.frames() {
                        resources.decoded_input_bytes =
                            add(resources.decoded_input_bytes, frame.pixels().len() as u64)?;
                    }
                }
                TokenizedMultimodalSegment::Media(Media::Audio(_)) => {
                    return Err(ProcessorExecutionError::Plan(
                        "Qwen visual processor does not accept audio".into(),
                    ))
                }
            }
            check(budget, &resources)?;
        }
        check(budget, &resources)?;
        let step_capacity =
            usize::try_from(step_capacity).map_err(|_| ProcessorResourceError::Overflow)?;
        if step_capacity
            .checked_mul(std::mem::size_of::<Step<'_>>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(ProcessorResourceError::Overflow.into());
        }
        // Count text again through the exact retained steps below.
        resources.decoder_positions = 0;
        let mut result = Self {
            steps: Vec::with_capacity(step_capacity),
            resources,
        };
        let mut scratch = 0;
        let mut retained_tokens = 0;
        for segment in request.segments() {
            match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => result.tokens(Cow::Borrowed(ids))?,
                TokenizedMultimodalSegment::Media(Media::Image(image)) => {
                    let p = plan
                        .image(image.height() as usize, image.width() as usize)
                        .map_err(plan_error)?;
                    scratch =
                        scratch.max(result.media_buffers(image, p.transform, p.patches, 1)?);
                    result.tokens(Cow::Owned(vec![p.framing.start_token_id]))?;
                    result.steps.push(Step::Image(image, p));
                    result.tokens(Cow::Owned(vec![p.framing.end_token_id]))?;
                    retained_tokens = add(retained_tokens, 8)?;
                }
                TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                    let first = consistent_video_frame(video)?;
                    let p = plan
                        .video(
                            video.frames().len(),
                            first.height() as usize,
                            first.width() as usize,
                            video.source_fps(),
                            video.sampling(),
                        )
                        .map_err(plan_error)?;
                    for group in p.groups {
                        let encoded = encode_text(&group.timestamp_text)
                            .map_err(ProcessorExecutionError::Text)?;
                        let prefix_len = add(encoded.len() as u64, 1)?;
                        if mul(prefix_len, 4)? > isize::MAX as u64 {
                            return Err(ProcessorResourceError::Overflow.into());
                        }
                        result.resources.planning_items =
                            add(result.resources.planning_items, prefix_len)?;
                        check(budget, &result.resources)?;
                        let prefix_len = usize::try_from(prefix_len)
                            .map_err(|_| ProcessorResourceError::Overflow)?;
                        let mut prefix = Vec::with_capacity(prefix_len);
                        prefix.extend_from_slice(&encoded);
                        prefix.push(p.framing.start_token_id);
                        drop(encoded);
                        retained_tokens = add(retained_tokens, mul(prefix_len as u64, 4)?)?;
                        retained_tokens = add(retained_tokens, 4)?;
                        result.tokens(Cow::Owned(prefix))?;
                        scratch = scratch.max(result.media_buffers(
                            first,
                            p.transform,
                            p.patches,
                            group.source_indices.len() as u64,
                        )?);
                        result.steps.push(Step::Video {
                            video,
                            indices: group.source_indices,
                            transform: p.transform,
                            patches: p.patches,
                        });
                        result.tokens(Cow::Owned(vec![p.framing.end_token_id]))?;
                    }
                }
                TokenizedMultimodalSegment::Media(Media::Audio(_)) => {
                    unreachable!("rejected before planning")
                }
            }
            result.resources.host_buffer_bytes = add(
                add(result.resources.output_tensor_bytes, scratch)?,
                retained_tokens,
            )?;
            check(budget, &result.resources)?;
        }
        check(budget, &result.resources)?;
        Ok(result)
    }
    fn tokens(&mut self, ids: Cow<'a, [u32]>) -> Result<(), ProcessorResourceError> {
        if !ids.is_empty() {
            self.resources.decoder_positions =
                add(self.resources.decoder_positions, ids.len() as u64)?;
            self.resources.output_tensor_bytes = add(
                self.resources.output_tensor_bytes,
                mul(ids.len() as u64, 4)?,
            )?;
            self.steps.push(Step::Tokens(ids));
        }
        Ok(())
    }
    fn media_buffers<E: std::fmt::Display>(
        &mut self,
        source: &RgbImage,
        transform: RgbTransformPlan,
        patches: QwenPatchPlan,
        frames: u64,
    ) -> Result<u64, ProcessorExecutionError<E, eredu_media::MediaError>> {
        let invalid =
            || ProcessorExecutionError::Plan("invalid processor packing dimensions".into());
        let width = u32::try_from(transform.width).map_err(|_| invalid())?;
        let height = u32::try_from(transform.height).map_err(|_| invalid())?;
        let factor = patches
            .patch_size
            .checked_mul(patches.merge_size)
            .filter(|n| *n > 0)
            .ok_or_else(invalid)?;
        if patches.temporal_patch_size == 0
            || transform.width % factor != 0
            || transform.height % factor != 0
            || frames == 0
        {
            return Err(invalid());
        }
        let positions = mul(
            (transform.width / factor) as u64,
            (transform.height / factor) as u64,
        )?;
        let rows = mul(
            (transform.width / patches.patch_size) as u64,
            (transform.height / patches.patch_size) as u64,
        )?;
        let columns = mul(
            mul(patches.patch_size as u64, patches.patch_size as u64)?,
            mul(3, patches.temporal_patch_size as u64)?,
        )?;
        if rows > i32::MAX as u64 || columns > i32::MAX as u64 || positions == 0 {
            return Err(invalid());
        }
        let payload = mul(mul(rows, columns)?, 4)?;
        if payload > isize::MAX as u64 {
            return Err(ProcessorResourceError::Overflow.into());
        }
        self.resources.decoder_positions = add(self.resources.decoder_positions, positions)?;
        self.resources.output_tensor_bytes =
            add(self.resources.output_tensor_bytes, add(payload, 12)?)?;
        let buffers = eredu_media::image::rgb_preparation_buffers(
            source.width(),
            source.height(),
            width,
            height,
        )
        .map_err(mechanism_error)?;
        // All group frames coexist during packing. tensor_f32 copies the packed Vec
        // into host storage; count both, even though one is also in the final output.
        let normalized = mul(buffers.normalized_bytes, frames)?;
        Ok(add(normalized, buffers.peak_bytes)?.max(add(normalized, mul(payload, 2)?)?))
    }
    pub fn execute<M: ProcessorOperations, E: std::fmt::Display>(
        &self,
        mechanisms: &mut M,
    ) -> Result<PreparedModelInput<M::Tensor>, ProcessorExecutionError<E, M::Error>> {
        let mut parts = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            match step {
                Step::Tokens(ids) => push_tokens(&mut parts, ids, mechanisms)?,
                Step::Image(image, plan) => parts.push(qwen_image(image, plan, mechanisms)?),
                Step::Video {
                    video,
                    indices,
                    transform,
                    patches,
                } => {
                    let frames = indices
                        .iter()
                        .map(|index| {
                            mechanisms
                                .normalize_rgb(&video.frames()[*index], *transform)
                                .map_err(mechanism_error)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let (values, shape, grid) = pack_qwen(&frames, *patches, false)?;
                    let payload = mechanisms
                        .tensor_f32(&values, &shape)
                        .map_err(mechanism_error)?;
                    let metadata = mechanisms
                        .tensor_i32(&grid, &[1, 3])
                        .map_err(mechanism_error)?;
                    parts.push(
                        PreparedInputPart::new(
                            InputModality::Video,
                            PreparedInputPayload::Tensor(payload),
                            [(InputMetadataKey::PatchGrid, metadata)],
                        )
                        .map_err(ProcessorExecutionError::Prepared)?,
                    );
                }
            }
        }
        PreparedModelInput::new(parts, |tensor| mechanisms.identity(tensor))
            .map_err(ProcessorExecutionError::Prepared)
    }
}
