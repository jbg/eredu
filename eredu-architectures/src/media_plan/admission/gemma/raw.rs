//! The ordinary Gemma ingress equations with an explicit diagnostic destination.
use super::*;
use super::view::{Input, Shape, Values};

fn nonzero_positive<D: Destination>(value: i32, field: &'static str, destination: D) -> Result<u64, D::Error> {
    let value = destination.positive(value, field)?;
    if value == 0 {
        return Err(destination.configuration(field, format_args!("expected a positive value, got zero")));
    }
    Ok(value)
}

pub(in crate::media_plan) fn part_view<D: Destination>(
    args: &FamilyConfig,
    inspected: Input<'_>,
    destination: D,
) -> Result<Gemma4InputPartPlan, D::Error> {
    destination.controls::<(
        &FamilyConfig, Input<'_>, D, &InputPartDescriptor,
        InputModality, InputPayloadKind, Shape<'_>, [usize; 3], [i32; 3],
        [Result<i32, D::Error>; 3], Result<i32, D::Error>,
        Values<'_, i32>, Values<'_, bool>, &crate::gemma4::VisionConfig,
        crate::gemma4::VisionIngressPartPlan, crate::gemma4::AudioIngressPartPlan,
        Result<crate::gemma4::VisionIngressPartPlan, D::Error>,
        Result<crate::gemma4::AudioIngressPartPlan, D::Error>,
        MediaShapePlan, Gemma4InputPartPlan, Result<Gemma4InputPartPlan, D::Error>,
        Option<Result<Gemma4InputPartPlan, D::Error>>, (D, &FamilyConfig),
        i32, i32, i32, u64, usize, usize, std::fmt::Arguments<'_>
    )>()?;
    if let Some(result) = inspected.projected(args, destination) {
        return result;
    }
    let modality = inspected.modality;
    let payload = inspected.kind;
    let shape = inspected.shape;
    match (modality, payload) {
        (modality @ (InputModality::Image | InputModality::Video), InputPayloadKind::Tensor) => {
            if inspected.patch_grid.is_none() {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "prepared Gemma {} input has no patch grid",
                        modality.as_str()
                    ),
                ));
            }
            let extent = inspected.patch_extent.ok_or_else(|| {
                destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "prepared Gemma {} input has no host-known patch extent",
                        modality.as_str()
                    ),
                )
            })?;
            let grid = inspected.patch_grid.expect("checked above");
            let extent_i32 = extent.map(|value| {
                i32::try_from(value).map_err(|_| destination.overflow("Gemma patch extent"))
            });
            let [time, height, width] = extent_i32;
            let extent_i32 = [time?, height?, width?];
            if !grid.shape.matches(&[1, 3])
                || grid.values != extent_i32
                || grid.values.len() != 3
            {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "prepared Gemma {} patch grid must be one row matching extent {extent:?}",
                        modality.as_str()
                    ),
                ));
            }
            let vision = args.vision.as_ref().ok_or_else(|| {
                destination.unsupported(&args.model_type, "loaded Gemma model has no vision tower")
            })?;
            let media_shape = media_view(args, inspected, destination)?;
            let padded_patches =
                i32::try_from(shape.at(1)).map_err(|_| destination.overflow("Gemma padded vision patch count"))?;
            let ingress =
                crate::gemma4::VisionIngressPartPlan::with_diagnostic(
                    vision, extent_i32, padded_patches,
                    |detail| destination.unsupported(&args.model_type, detail),
                )?;
            let valid_patches = patch_count(
                inspected
                    .patch_positions
                    .expect("Gemma media shape validation requires patch positions"),
                &args.model_type,
                destination,
            )?;
            if valid_patches != ingress.valid_patches as u64
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(destination.unsupported(
                    &args.model_type,
                    "prepared Gemma patch extent and position metadata disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Vision {
                placeholder_token_id: placeholder(args, modality, destination)?,
                ingress,
                shape: media_shape,
            })
        }
        (InputModality::Audio, InputPayloadKind::Tensor) => {
            let valid_frames = inspected.audio_valid_frames.ok_or_else(|| {
                destination.unsupported(
                    &args.model_type,
                    "prepared Gemma audio has no host-known valid-frame extent",
                )
            })?;
            let media_shape = media_view(args, inspected, destination)?;
            let padded_frames =
                i32::try_from(shape.at(1)).map_err(|_| destination.overflow("Gemma padded audio frame count"))?;
            let valid_frames_i32 =
                i32::try_from(valid_frames).map_err(|_| destination.overflow("Gemma valid audio frame count"))?;
            let ingress = crate::gemma4::AudioIngressPartPlan::with_diagnostic(
                valid_frames_i32, padded_frames,
                |detail| destination.unsupported(&args.model_type, detail),
            )?;
            let valid_mask_frames = inspected
                .audio_mask
                .expect("Gemma media shape validation requires an audio mask")
                .values
                .iter()
                .filter(|value| **value)
                .count();
            if valid_mask_frames != valid_frames
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(destination.unsupported(
                    &args.model_type,
                    "prepared Gemma audio valid-frame extent and mask disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Audio {
                placeholder_token_id: placeholder(args, InputModality::Audio, destination)?,
                ingress,
                shape: media_shape,
            })
        }
        (modality, payload) => Err(destination.unsupported(
            &args.model_type,
            format_args!(
                "Gemma 4 does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}

fn patch_count<D: Destination>(
    positions: Values<'_, i32>,
    architecture: &str,
    destination: D,
) -> Result<u64, D::Error> {
    destination.controls::<(
        Values<'_, i32>, &str, D, u64, usize,
        std::slice::Iter<'_, [i32; 2]>, &[i32], Option<u64>, Result<u64, D::Error>,
        std::fmt::Arguments<'_>
    )>()?;
    if positions.shape.len() != 3 || positions.shape.at(0) != 1 || positions.shape.at(2) != 2 {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma patch positions must be [1, patches, 2], got {:?}",
                positions.shape
            ),
        ));
    }
    let expected = destination.checked_mul(positions.shape.at(1), 2, "Gemma patch-position scalar count")?;
    if u64::try_from(positions.values.len()).ok() != Some(expected) {
        return Err(destination.unsupported(
            architecture,
            "Gemma patch positions do not match their declared shape",
        ));
    }
    u64::try_from(
        positions
            .values
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|pair| pair[0] >= 0 && pair[1] >= 0)
            .count(),
    )
    .map_err(|_| destination.overflow("Gemma valid patch count"))
}

fn vision_view<D: Destination>(
    config: &crate::gemma4::VisionConfig,
    text_hidden: u64,
    input: Input<'_>,
    architecture: &str,
    destination: D,
) -> Result<MediaShapePlan, D::Error> {
    destination.controls::<(
        &crate::gemma4::VisionConfig, u64, Input<'_>, &str, D,
        Values<'_, i32>,
        u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64,
        MediaShapePlan, Result<MediaShapePlan, D::Error>,
        Result<u64, D::Error>, Result<u64, D::Error>, Result<u64, D::Error>,
        std::fmt::Arguments<'_>
    )>()?;
    if input.shape.len() != 3 || input.shape.at(0) != 1 {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma prepared vision tensor must be [1, patches, patch_dims], got {:?}",
                input.shape
            ),
        ));
    }
    let patch = nonzero_positive(config.patch_size, "Gemma vision patch size", destination)?;
    let expected_patch_dims = destination.checked_mul(
        3,
        destination.checked_mul(patch, patch, "Gemma vision patch area")?,
        "Gemma vision patch dimensions",
    )?;
    if input.shape.at(2) != expected_patch_dims {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma prepared patches have width {}, expected {expected_patch_dims}",
                input.shape.at(2)
            ),
        ));
    }
    let position_ids = input
        .patch_positions
        .ok_or_else(|| destination.unsupported(architecture, "prepared Gemma media has no patch positions"))?;
    let valid_patches = patch_count(position_ids, architecture, destination)?;
    let pool = nonzero_positive(config.pooling_kernel_size, "Gemma pooling kernel", destination)?;
    let pool_area = destination.checked_mul(pool, pool, "Gemma pooling area")?;
    if valid_patches % pool_area != 0 {
        return Err(destination.unsupported(
            architecture,
            format_args!("Gemma valid patch count {valid_patches} is not divisible by {pool_area}"),
        ));
    }
    let positions = valid_patches / pool_area;
    let padded_patches = input.shape.at(1);
    if position_ids.shape.at(1) != padded_patches {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma patch positions {:?} do not match prepared vision payload {:?}",
                position_ids.shape, input.shape
            ),
        ));
    }
    let hidden = destination.positive(config.hidden_size, "Gemma vision hidden size")?;
    let intermediate = destination.positive(config.intermediate_size, "Gemma vision intermediate size")?;
    let depth = destination.positive(config.num_hidden_layers, "Gemma vision depth")?;
    let patch_hidden = destination.checked_mul(padded_patches, hidden, "Gemma vision patch hidden elements")?;
    let per_layer = destination.checked_add(
        destination.checked_mul(48, patch_hidden, "Gemma vision layer hidden workspace")?,
        destination.checked_mul(
            8,
            destination.checked_mul(
                padded_patches,
                intermediate,
                "Gemma vision intermediate elements",
            )?,
            "Gemma vision MLP workspace",
        )?,
        "Gemma vision layer workspace",
    )?;
    let output_workspace = destination.checked_mul(
        positions,
        destination.checked_add(
            destination.checked_mul(8, hidden, "Gemma pooled vision workspace")?,
            destination.checked_mul(8, text_hidden, "Gemma projected vision workspace")?,
            "Gemma vision output workspace per position",
        )?,
        "Gemma vision output workspace",
    )?;
    let graph_scalars = destination.checked_add(
        destination.checked_mul(20, patch_hidden, "Gemma vision setup workspace")?,
        destination.checked_add(
            destination.checked_mul(depth, per_layer, "Gemma all vision layers")?,
            output_workspace,
            "Gemma layers plus output workspace",
        )?,
        "Gemma vision graph workspace",
    )?;
    Ok(MediaShapePlan {
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

fn audio_view<D: Destination>(
    config: &crate::gemma4::AudioConfig,
    text_hidden: u64,
    input: Input<'_>,
    architecture: &str,
    destination: D,
) -> Result<MediaShapePlan, D::Error> {
    destination.controls::<(
        &crate::gemma4::AudioConfig, u64, Input<'_>, &str, D,
        Values<'_, bool>, std::slice::Iter<'_, bool>, usize,
        u64, u64, u64, u64, u64, u64, u64, u64, u64, u64,
        u64, u64, u64, u64, u64, u64, u64, u64, u64, u64,
        MediaShapePlan, Result<MediaShapePlan, D::Error>,
        Result<u64, D::Error>, Result<u64, D::Error>, Result<u64, D::Error>,
        std::fmt::Arguments<'_>
    )>()?;
    if input.shape.len() != 3
        || input.shape.at(0) != 1
        || input.shape.at(2) != 128
    {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma prepared audio tensor must be [1, frames, 128], got {:?}",
                input.shape
            ),
        ));
    }
    let mask = input
        .audio_mask
        .ok_or_else(|| destination.unsupported(architecture, "prepared Gemma audio has no frame mask"))?;
    let frames = input.shape.at(1);
    if !mask.shape.matches(&[1, frames]) || u64::try_from(mask.values.len()).ok() != Some(frames) {
        return Err(destination.unsupported(
            architecture,
            format_args!(
                "Gemma audio mask must be [1, {frames}], got {:?}",
                mask.shape
            ),
        ));
    }
    let valid_frames =
        u64::try_from(mask.values.iter().filter(|value| **value).count()).map_err(|_| {
            destination.overflow("Gemma valid audio frame count")
        })?;
    let positions = valid_frames.div_ceil(4);
    let sequence = frames.div_ceil(4);
    let hidden = destination.positive(config.hidden_size, "Gemma audio hidden size")?;
    let depth = destination.positive(config.num_hidden_layers, "Gemma audio depth")?;
    let heads = destination.positive(config.num_attention_heads, "Gemma audio heads")?;
    let chunk = nonzero_positive(config.attention_chunk_size, "Gemma audio attention chunk", destination)?;
    let past = nonzero_positive(
        config.attention_context_left.checked_sub(1).ok_or(
            destination.overflow("Gemma audio left context"),
        )?,
        "Gemma audio left context",
        destination,
    )?;
    let padded_sequence = destination.checked_mul(
        sequence.div_ceil(chunk),
        chunk,
        "Gemma padded audio sequence",
    )?;
    let chunks = padded_sequence / chunk;
    let attention_elements = destination.checked_mul(
        destination.checked_mul(
            destination.checked_mul(
                destination.checked_mul(chunks, heads, "Gemma audio attention chunk heads")?,
                chunk,
                "Gemma audio attention queries",
            )?,
            destination.checked_add(chunk, past, "Gemma audio attention key bound")?,
            "Gemma audio attention scores",
        )?,
        4,
        "Gemma audio logits/relative/mask/probability workspace",
    )?;
    let layer_workspace = destination.checked_add(
        destination.checked_mul(
            80,
            destination.checked_mul(sequence, hidden, "Gemma audio hidden elements")?,
            "Gemma audio layer hidden workspace",
        )?,
        attention_elements,
        "Gemma audio layer workspace",
    )?;
    let first_frames = frames.div_ceil(2);
    let first_channels = destination.positive(
        *config.subsampling_conv_channels.first().ok_or_else(|| {
            destination.configuration("subsampling_conv_channels", format_args!("Gemma audio has no first convolution channel count"))
        })?,
        "Gemma audio first convolution channels",
    )?;
    let second_channels = destination.positive(
        *config.subsampling_conv_channels.get(1).ok_or_else(|| {
            destination.configuration("subsampling_conv_channels", format_args!("Gemma audio has no second convolution channel count"))
        })?,
        "Gemma audio second convolution channels",
    )?;
    let conv_workspace = destination.checked_add(
        destination.checked_mul(
            6,
            destination.checked_mul(
                destination.checked_mul(first_frames, 64, "Gemma first convolution grid")?,
                first_channels,
                "Gemma first convolution elements",
            )?,
            "Gemma first convolution workspace",
        )?,
        destination.checked_mul(
            6,
            destination.checked_mul(
                destination.checked_mul(sequence, 32, "Gemma second convolution grid")?,
                second_channels,
                "Gemma second convolution elements",
            )?,
            "Gemma second convolution workspace",
        )?,
        "Gemma convolution workspace",
    )?;
    let output = destination.positive(config.output_proj_dims, "Gemma audio output size")?;
    let output_workspace = destination.checked_mul(
        positions,
        destination.checked_add(
            destination.checked_mul(8, output, "Gemma audio output workspace")?,
            destination.checked_mul(8, text_hidden, "Gemma audio text projection workspace")?,
            "Gemma audio output workspace per position",
        )?,
        "Gemma audio projected output workspace",
    )?;
    let graph_scalars = destination.checked_add(
        conv_workspace,
        destination.checked_add(
            destination.checked_mul(depth, layer_workspace, "Gemma all audio layers")?,
            output_workspace,
            "Gemma audio layers plus output",
        )?,
        "Gemma audio graph workspace",
    )?;
    Ok(MediaShapePlan {
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

/// Derives prepared Gemma 4 media geometry from normalized family policy.
pub(in crate::media_plan) fn media_view<D: Destination>(
    args: &crate::gemma4::FamilyConfig,
    input: Input<'_>,
    destination: D,
) -> Result<MediaShapePlan, D::Error> {
    destination.controls::<(
        &FamilyConfig, Input<'_>, D, u64,
        &crate::gemma4::VisionConfig, &crate::gemma4::AudioConfig,
        (D, &FamilyConfig), (D, &FamilyConfig, Input<'_>, u64),
        MediaShapePlan, Result<MediaShapePlan, D::Error>, InputModality
    )>()?;
    let text_hidden = destination.positive(args.text.hidden_size, "Gemma text hidden size")?;
    match input.modality {
        InputModality::Image | InputModality::Video => args
            .vision
            .as_ref()
            .ok_or_else(|| destination.unsupported(&args.model_type, "loaded model has no vision tower"))
            .and_then(|config| vision_view(config, text_hidden, input, &args.model_type, destination)),
        InputModality::Audio => args
            .audio
            .as_ref()
            .ok_or_else(|| destination.unsupported(&args.model_type, "loaded model has no audio tower"))
            .and_then(|config| audio_view(config, text_hidden, input, &args.model_type, destination)),
        InputModality::Text => Err(destination.unsupported(
            &args.model_type,
            "text is not a Gemma media modality",
        )),
        _ => Err(destination.unsupported(
            &args.model_type,
            "unknown modality is not supported by Gemma",
        )),
    }
}

// Existing entry points borrow their owned descriptors; original compilation
// supplies the same views directly from I's immutable slots.
pub(in crate::media_plan) fn part<D: Destination>(args: &FamilyConfig, input: &MediaAdmissionInput, destination: D) -> Result<Gemma4InputPartPlan, D::Error> {
    part_view(args, Input::legacy(input), destination)
}
pub(in crate::media_plan) fn gemma4<D: Destination>(args: &FamilyConfig, input: &MediaAdmissionInput, destination: D) -> Result<MediaShapePlan, D::Error> {
    media_view(args, Input::legacy(input), destination)
}
#[cfg(test)]
pub(in crate::media_plan) fn gemma_valid_patch_count<D: Destination>(positions: &MetadataValues<i32>, architecture: &str, destination: D) -> Result<u64, D::Error> {
    patch_count(Values::legacy(positions), architecture, destination)
}
#[cfg(test)]
pub(in crate::media_plan) fn gemma_vision<D: Destination>(config: &crate::gemma4::VisionConfig, text_hidden: u64, input: &MediaAdmissionInput, architecture: &str, destination: D) -> Result<MediaShapePlan, D::Error> {
    vision_view(config, text_hidden, Input::legacy(input), architecture, destination)
}
#[cfg(test)]
pub(in crate::media_plan) fn gemma_audio<D: Destination>(config: &crate::gemma4::AudioConfig, text_hidden: u64, input: &MediaAdmissionInput, architecture: &str, destination: D) -> Result<MediaShapePlan, D::Error> {
    audio_view(config, text_hidden, Input::legacy(input), architecture, destination)
}

pub(in crate::media_plan) fn original_part(
    args: &FamilyConfig,
    part: &eredu_runtime::input::host::PreparedHostPart<'_>,
) -> Result<Gemma4InputPartPlan, crate::media_plan::MediaSemanticError> {
    part_view(args, super::view::Input::original(part)?, super::view::SourceDestination)
}
