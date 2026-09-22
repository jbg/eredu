//! One borrowed Muse geometry worker for ordinary and original admission.
use super::gemma::view::Input;
use super::inkling::Destination;
use super::*;
pub(in crate::media_plan) fn media<D: Destination>(
    args: &crate::muse_glimmer::DecoderConfig,
    input: Input<'_>,
    destination: D,
) -> Result<MediaShapePlan, D::Error> {
    if input.modality == InputModality::Audio
        || (input.modality == InputModality::Video
            && args.weight_convention == crate::muse_glimmer::WeightConvention::Gguf)
    {
        return Err(destination.unsupported(
            &args.model_type,
            format_args!(
                "loaded Muse-Glimmer artifact does not support {}",
                input.modality.as_str()
            ),
        ));
    }
    let vision = args.vision_config.as_ref().ok_or_else(|| {
        destination.unsupported(
            &args.model_type,
            "loaded Muse-Glimmer artifact has no vision projector",
        )
    })?;
    let grid = input.patch_grid.as_ref().ok_or_else(|| {
        destination.unsupported(
            &args.model_type,
            "Muse-Glimmer media requires patch_grid metadata",
        )
    })?;
    if grid.shape.len() != 2 || grid.shape.at(0) == 0 || grid.shape.at(1) != 3 {
        return Err(destination.unsupported(
            &args.model_type,
            format_args!(
                "Muse-Glimmer patch_grid must be [items, 3], got {:?}",
                grid.shape
            ),
        ));
    }
    let expected_values =
        destination.checked_mul(grid.shape.at(0), 3, "Muse patch-grid scalar count")?;
    if u64::try_from(grid.values.len()).ok() != Some(expected_values) {
        return Err(destination.unsupported(
            &args.model_type,
            "Muse-Glimmer patch_grid has an incomplete row",
        ));
    }
    let merge = destination.positive(vision.merge_size, "Muse vision merge size")?;
    if merge == 0 {
        return Err(destination.configuration(
            "Muse vision merge size",
            format_args!("expected a positive value"),
        ));
    }
    let mut patches = 0u64;
    let mut positions = 0u64;
    for entry in grid.values.as_chunks::<3>().0 {
        if entry.iter().any(|value| *value <= 0)
            || u64::try_from(entry[1]).unwrap_or_default() % merge != 0
            || u64::try_from(entry[2]).unwrap_or_default() % merge != 0
        {
            return Err(destination.unsupported(
                &args.model_type,
                "Muse-Glimmer vision grids must be positive and merge-divisible",
            ));
        }
        let t = entry[0] as u64;
        let h = entry[1] as u64;
        let w = entry[2] as u64;
        patches = destination.checked_add(
            patches,
            destination.checked_mul(
                destination.checked_mul(t, h, "Muse vision t*h")?,
                w,
                "Muse vision patches",
            )?,
            "Muse vision patch total",
        )?;
        positions = destination.checked_add(
            positions,
            destination.checked_mul(
                destination.checked_mul(t, h / merge, "Muse merged t*h")?,
                w / merge,
                "Muse merged positions",
            )?,
            "Muse merged position total",
        )?;
    }
    if (input.shape.len() > 0).then(|| input.shape.at(0)) != Some(patches) {
        return Err(destination.unsupported(
            &args.model_type,
            format_args!(
                "Muse-Glimmer payload has {} patches but metadata describes {patches}",
                (input.shape.len() > 0)
                    .then(|| input.shape.at(0))
                    .unwrap_or_default()
            ),
        ));
    }
    let graph = destination.checked_mul(
        patches,
        destination.positive(vision.hidden_size, "Muse vision hidden size")?,
        "Muse vision activation scalars",
    )?;
    Ok(MediaShapePlan {
        decoder_positions: positions,
        execution_workspace_scalars: destination.checked_mul(
            graph,
            8,
            "Muse vision graph multiplier",
        )?,
    })
}
