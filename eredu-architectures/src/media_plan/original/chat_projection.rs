//! Render projection of the actual original source. Decoder expansion remains
//! in the existing semantic records; protocol markers are not decoder rows.
use super::*;
use eredu_core::InputExtent;
use eredu_runtime::working_memory::CompositeChatProjection as Projection;
use eredu_runtime::working_memory::CompositeGeneratedText;

fn video_origin(part: &eredu_runtime::input::host::PreparedHostPart<'_>) -> Option<InputExtent> {
    part.extents()
        .iter()
        .copied()
        .find(|e| matches!(e, InputExtent::VideoFrame { .. }))
}
fn framing(
    model: &PreparedModelSources,
    policy: Policy<'_>,
) -> Option<crate::processor_plan::MediaFraming> {
    match policy {
        Policy::Gemma(_) => model.architecture().gemma4()?.framing(InputModality::Video),
        Policy::Vl(_) | Policy::Conditional(_) => model.architecture().qwen()?.framing(),
        Policy::Inkling(_) | Policy::Muse(_) => None,
    }
}
fn timestamp(origin: InputExtent, policy: Policy<'_>) -> Option<CompositeGeneratedText> {
    let InputExtent::VideoFrame {
        index,
        first_source_frame,
        last_source_frame,
        source_fps_bits,
        ..
    } = origin
    else {
        return None;
    };
    let fps = f64::from_bits(source_fps_bits);
    if !fps.is_finite() || fps <= 0.0 {
        return None;
    }
    match policy {
        Policy::Gemma(_) => Some(crate::processor_plan::gemma_timestamp(
            first_source_frame,
            fps,
            index == 0,
        )),
        _ => {
            let seconds = (first_source_frame as f64 / fps + last_source_frame as f64 / fps) / 2.0;
            seconds.is_finite().then(|| {
                crate::processor_plan::qwen_timestamp(first_source_frame, last_source_frame, fps)
            })
        }
    }
}

/// Validate the original source's complete frame grouping before allocating A.
pub(super) fn validate(
    model: &PreparedModelSources,
    source: &OriginalPreparedHostInput,
    policy: Policy<'_>,
) -> Result<(), MediaSemanticError> {
    let bad = || {
        MediaSemanticError::input(
            "video source grouping/timestamp differs from processor projection",
        )
    };
    let mut previous: Option<(usize, usize, usize, usize, usize, u64)> = None;
    for (part_index, part) in source.parts().enumerate() {
        // These processors retain their text framing as original token parts.
        if matches!(policy, Policy::Inkling(_) | Policy::Muse(_)) {
            continue;
        }
        let Some(origin) = video_origin(&part) else {
            continue;
        };
        let InputExtent::VideoFrame {
            group,
            index,
            count,
            first_source_frame,
            last_source_frame,
            source_fps_bits,
        } = origin
        else {
            unreachable!()
        };
        if part.modality() != InputModality::Video
            || count == 0
            || index >= count
            || first_source_frame > last_source_frame
            || timestamp(origin, policy).is_none()
        {
            return Err(bad().at(part_index));
        }
        match previous {
            Some((prior_part, prior_group, prior_index, prior_count, prior_frame, prior_fps))
                if group == prior_group =>
            {
                if index != prior_index + 1
                    || count != prior_count
                    || part_index != prior_part + 3
                    || first_source_frame < prior_frame
                    || source_fps_bits != prior_fps
                {
                    return Err(bad().at(part_index));
                }
            }
            Some((_, prior_group, prior_index, prior_count, _, _)) => {
                if group <= prior_group || prior_index + 1 != prior_count || index != 0 {
                    return Err(bad().at(part_index));
                }
            }
            None if index != 0 => return Err(bad().at(part_index)),
            None => {}
        }
        let framing = framing(model, policy).ok_or_else(bad)?;
        let prefix = part_index
            .checked_sub(1)
            .and_then(|i| source.part(i))
            .ok_or_else(bad)?;
        let suffix = source.part(part_index + 1).ok_or_else(bad)?;
        if prefix.modality() != InputModality::Text
            || suffix.modality() != InputModality::Text
            || last_token(prefix.payload_view().values) != Some(framing.start_token_id)
            || !singleton(suffix.payload_view().values, framing.end_token_id)
        {
            return Err(bad().at(part_index));
        }
        previous = Some((
            part_index,
            group,
            index,
            count,
            last_source_frame,
            source_fps_bits,
        ));
    }
    if previous.is_some_and(|(_, _, index, count, _, _)| index + 1 != count) {
        return Err(bad());
    }
    Ok(())
}

pub(super) fn part(
    model: &PreparedModelSources,
    source: &OriginalPreparedHostInput,
    index: usize,
    policy: Policy<'_>,
    role: CompositeSemanticRole,
) -> Projection {
    let part = source.part(index).expect("original semantic source part");
    match (role, part.modality()) {
        (CompositeSemanticRole::Tokens, InputModality::Text) => {
            if let Some(framing) = framing(model, policy) {
                if let Some(origin) = index
                    .checked_add(1)
                    .and_then(|i| source.part(i))
                    .and_then(|part| video_origin(&part))
                {
                    return Projection::VideoPrefix {
                        origin,
                        text: timestamp(origin, policy).expect("validated timestamp"),
                        framing: framing.start_token_id,
                    };
                }
                if let Some(origin) = index
                    .checked_sub(1)
                    .and_then(|i| source.part(i))
                    .and_then(|part| video_origin(&part))
                {
                    return Projection::VideoSuffix {
                        origin,
                        framing: framing.end_token_id,
                    };
                }
            }
            // Gemma's released template contains a media marker. Its processor
            // emits an opening and closing token around the prepared tensor.
            // Only exact singleton values beside that actual media part qualify.
            if matches!(policy, Policy::Gemma(_)) {
                if let Some(processor) = model.architecture().gemma4() {
                    for (neighbor, opening) in
                        [(index.checked_add(1), true), (index.checked_sub(1), false)]
                    {
                        let Some(media) = neighbor.and_then(|i| source.part(i)) else {
                            continue;
                        };
                        let Some(framing) = processor.framing(media.modality()) else {
                            continue;
                        };
                        let id = if opening {
                            framing.start_token_id
                        } else {
                            framing.end_token_id
                        };
                        if singleton(part.payload_view().values, id) {
                            return Projection::Markers(&[]);
                        }
                    }
                }
            }
            Projection::SourceTokens
        }
        (CompositeSemanticRole::Encoded | CompositeSemanticRole::Projected, modality) => {
            if matches!(policy, Policy::Inkling(_) | Policy::Muse(_)) {
                return Projection::Unavailable;
            }
            if let Some(origin) = video_origin(&part) {
                return match policy {
                    Policy::Gemma(_) => Projection::VideoMarker {
                        origin,
                        compact: &["<|video|>"],
                        expanded: &["<|video|>"],
                    },
                    _ => Projection::VideoMarker {
                        origin,
                        compact: &["<|vision_start|>", "<|video_pad|>", "<|vision_end|>"],
                        expanded: &["<|video_pad|>"],
                    },
                };
            }
            match (policy, modality) {
                (Policy::Vl(_) | Policy::Conditional(_), InputModality::Image) => {
                    Projection::Markers(&["<|image_pad|>"])
                }
                (Policy::Vl(_) | Policy::Conditional(_), InputModality::Video) => {
                    Projection::Markers(&["<|video_pad|>"])
                }
                (Policy::Gemma(_), InputModality::Image) => Projection::Markers(&["<|image|>"]),
                (Policy::Gemma(_), InputModality::Audio) => Projection::Markers(&["<|audio|>"]),
                (Policy::Gemma(_), InputModality::Video) => Projection::Markers(&["<|video|>"]),
                _ => Projection::Unavailable,
            }
        }
        _ => Projection::Unavailable,
    }
}

fn singleton(values: HostTensorValues<'_>, id: u32) -> bool {
    match values {
        HostTensorValues::U32(values) => values == [id],
        HostTensorValues::I32(values) => values.len() == 1 && u32::try_from(values[0]) == Ok(id),
        _ => false,
    }
}
fn last_token(values: HostTensorValues<'_>) -> Option<u32> {
    match values {
        HostTensorValues::U32(values) => values.last().copied(),
        HostTensorValues::I32(values) => values.last().and_then(|v| u32::try_from(*v).ok()),
        _ => None,
    }
}
