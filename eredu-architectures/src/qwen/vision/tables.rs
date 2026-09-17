//! Shared tower policy projected into fixed neutral table equations.
use super::{VisionConfig, VisionMode};
use eredu_nn::{
    multimodal::{MultiAxisRotaryLayout, RotaryAxisSpec},
    sequence_layout::{
        InterpolationMode, PatchAttentionWindows, PatchEncoderTableError, PatchEncoderTableSpec,
        PatchPositionTableSpec, PatchTraversal,
    },
};

pub(crate) fn encoder_table_spec(
    config: &VisionConfig,
) -> Result<PatchEncoderTableSpec, PatchEncoderTableError> {
    if config.num_position_embeddings <= 0
        || config.num_heads <= 0
        || config.hidden_size <= 0
        || config.hidden_size % config.num_heads != 0
    {
        return Err(PatchEncoderTableError::SourceGeometry);
    }
    let side = (config.num_position_embeddings as f64).sqrt() as i32;
    if side.checked_mul(side) != Some(config.num_position_embeddings) {
        return Err(PatchEncoderTableError::SourceGeometry);
    }
    let dimensions = (config.hidden_size / config.num_heads) / 2;
    Ok(PatchEncoderTableSpec {
        merge: config.spatial_merge_size,
        positions: match config.mode {
            VisionMode::DeepStack => PatchPositionTableSpec::Interpolated {
                height: side,
                width: side,
                mode: InterpolationMode::AlignCorners,
                traversal: PatchTraversal::MergeMajor(config.spatial_merge_size),
            },
            VisionMode::WindowScheduled => PatchPositionTableSpec::Gathered {
                height: side,
                width: side,
            },
        },
        windows: match config.mode {
            VisionMode::DeepStack => PatchAttentionWindows::Full,
            VisionMode::WindowScheduled => PatchAttentionWindows::Windowed {
                window_size: config.window_size,
                patch_size: config.patch_size,
            },
        },
        rotary_axes: [RotaryAxisSpec {
            dimensions,
            position_offset: 0,
        }; 2],
        rotary_base: 10000.,
        rotary_minimum: 0,
        rotary_layout: MultiAxisRotaryLayout::SplitHalves,
    })
}
