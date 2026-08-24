//! Backend-neutral prepared-media admission and workspace plans.
//!
//! Concrete backends extract shapes and small metadata values from their
//! native arrays. Architecture policy validates those values and returns the
//! decoder positions and scalar workspace implied by the family equations.

use eredu_core::CapabilityError;

use crate::qwen::{
    hybrid::ParsedHybridConfig,
    vision::{VisionAttentionPolicy, VisionConfig},
    vl::ModelArgs as QwenVlModelArgs,
};

/// A non-text input modality presented to an architecture media tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaModality {
    /// Still-image input.
    Image,
    /// Audio input.
    Audio,
    /// Video input.
    Video,
}

impl MediaModality {
    fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Typed values and shape extracted from a small native metadata array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaMetadata<T> {
    /// Logical array shape.
    pub shape: Vec<u64>,
    /// Row-major scalar values.
    pub values: Vec<T>,
}

/// Backend-neutral description of one prepared media tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedMediaInput {
    /// Input modality.
    pub modality: MediaModality,
    /// Logical payload-array shape.
    pub payload_shape: Vec<u64>,
    /// Optional `(time, height, width)` rows.
    pub patch_grid: Option<MediaMetadata<i32>>,
    /// Optional two-axis patch positions, with negative padding entries.
    pub patch_positions: Option<MediaMetadata<i32>>,
    /// Optional valid-frame mask.
    pub audio_mask: Option<MediaMetadata<bool>>,
}

/// Architecture-derived geometry and conservative execution workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaShapePlan {
    /// Decoder positions occupied by the projected media.
    pub decoder_positions: u64,
    /// Conservative count of temporary execution scalars.
    pub execution_workspace_scalars: u64,
}

/// Architecture-owned Qwen image/video ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenVisionIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after spatial merging.
    pub placeholder_count: u64,
    /// Validated `(time, height, width)` rows consumed by Qwen position policy.
    pub patch_grid: Vec<(i32, i32, i32)>,
}

/// Architecture-owned Inkling ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after architecture-specific projection.
    pub placeholder_count: u64,
}

/// Architecture-owned Muse-Glimmer ingress policy for one prepared tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseGlimmerIngressPlan {
    /// Placeholder token selected for the input modality.
    pub placeholder_token_id: u32,
    /// Decoder placeholder span after spatial merging.
    pub placeholder_count: u64,
    /// Validated `(time, height, width)` rows consumed by vision execution.
    pub patch_grid: Vec<(i32, i32, i32)>,
}

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

fn nonzero_positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    let value = positive(value, field)?;
    if value == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field,
            detail: "expected a positive value, got zero".into(),
        });
    }
    Ok(value)
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn dimension(shape: &[u64], axis: usize, field: &'static str) -> Result<u64, CapabilityError> {
    shape
        .get(axis)
        .copied()
        .ok_or(CapabilityError::InvalidConfiguration {
            field,
            detail: format!("shape {shape:?} has no axis {axis}"),
        })
}

fn unsupported(architecture: &str, reason: impl Into<String>) -> CapabilityError {
    CapabilityError::UnsupportedInput {
        architecture: architecture.into(),
        reason: reason.into(),
    }
}

fn qwen_attention_chunk_squares(
    grid: &[(i32, i32, i32)],
    merge: u64,
    window_size: i32,
    patch_size: i32,
) -> Result<(u64, u64), CapabilityError> {
    let patch = nonzero_positive(patch_size, "Qwen vision patch size")?;
    let window_pixels = nonzero_positive(window_size, "Qwen vision window size")?;
    let merger_window = window_pixels / merge / patch;
    if merger_window == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "window_size",
            detail: format!(
                "Qwen vision window {window_pixels} is too small for merge {merge} and patch {patch}"
            ),
        });
    }
    let merge_area = checked_mul(merge, merge, "Qwen attention merge area")?;
    let merge_area_square =
        checked_mul(merge_area, merge_area, "Qwen attention merge-area square")?;
    let mut full_squares = 0u64;
    let mut window_squares = 0u64;
    for (time, height, width) in grid {
        let time = nonzero_positive(*time, "Qwen grid time")?;
        let height = nonzero_positive(*height, "Qwen grid height")?;
        let width = nonzero_positive(*width, "Qwen grid width")?;
        if height % merge != 0 || width % merge != 0 {
            return Err(CapabilityError::InvalidConfiguration {
                field: "patch_grid",
                detail: format!(
                    "Qwen grid ({height}, {width}) is not divisible by spatial merge {merge}"
                ),
            });
        }
        let full_length = checked_mul(height, width, "Qwen full-attention chunk length")?;
        full_squares = checked_add(
            full_squares,
            checked_mul(
                time,
                checked_mul(full_length, full_length, "Qwen full-attention chunk square")?,
                "Qwen full-attention temporal chunks",
            )?,
            "Qwen full-attention chunk-square total",
        )?;

        let merged_height = height / merge;
        let merged_width = width / merge;
        let height_full = merged_height / merger_window;
        let height_remainder = merged_height % merger_window;
        let width_full = merged_width / merger_window;
        let width_remainder = merged_width % merger_window;
        let window_square = checked_mul(merger_window, merger_window, "Qwen merger-window square")?;
        let height_square_sum = checked_add(
            checked_mul(
                height_full,
                window_square,
                "Qwen full height-window squares",
            )?,
            checked_mul(
                height_remainder,
                height_remainder,
                "Qwen remainder height-window square",
            )?,
            "Qwen height-window square sum",
        )?;
        let width_square_sum = checked_add(
            checked_mul(width_full, window_square, "Qwen full width-window squares")?,
            checked_mul(
                width_remainder,
                width_remainder,
                "Qwen remainder width-window square",
            )?,
            "Qwen width-window square sum",
        )?;
        let item_window_squares = checked_mul(
            checked_mul(
                height_square_sum,
                width_square_sum,
                "Qwen merged window-area squares",
            )?,
            merge_area_square,
            "Qwen patch window-area squares",
        )?;
        window_squares = checked_add(
            window_squares,
            checked_mul(time, item_window_squares, "Qwen temporal window chunks")?,
            "Qwen window-attention chunk-square total",
        )?;
    }
    Ok((full_squares, window_squares))
}

/// Derives prepared Qwen image/video geometry from normalized vision policy.
pub fn qwen_vision(
    config: &VisionConfig,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if !matches!(input.modality, MediaModality::Image | MediaModality::Video) {
        return Err(unsupported(
            architecture,
            format!("{} is not a Qwen vision modality", input.modality.as_str()),
        ));
    }
    if input.payload_shape.len() != 2 {
        return Err(unsupported(
            architecture,
            format!(
                "Qwen prepared vision tensor must be [patches, patch_dims], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let patches = dimension(&input.payload_shape, 0, "Qwen prepared patch count")?;
    let merge = nonzero_positive(config.spatial_merge_size, "spatial_merge_size")?;
    let patch = nonzero_positive(config.patch_size, "Qwen vision patch size")?;
    let expected_patch_dims = checked_mul(
        checked_mul(
            nonzero_positive(config.in_channels, "Qwen vision input channels")?,
            nonzero_positive(
                config.temporal_patch_size,
                "Qwen vision temporal patch size",
            )?,
            "Qwen temporal input channels",
        )?,
        checked_mul(patch, patch, "Qwen vision patch area")?,
        "Qwen vision patch dimensions",
    )?;
    if dimension(&input.payload_shape, 1, "Qwen prepared patch dimensions")? != expected_patch_dims
    {
        return Err(unsupported(
            architecture,
            format!(
                "Qwen prepared patches have width {}, expected {expected_patch_dims}",
                input.payload_shape[1]
            ),
        ));
    }
    let merge_area = checked_mul(merge, merge, "Qwen spatial merge area")?;
    if patches % merge_area != 0 {
        return Err(unsupported(
            architecture,
            format!("Qwen patch count {patches} is not divisible by {merge_area}"),
        ));
    }
    let positions = patches / merge_area;
    let metadata = input
        .patch_grid
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Qwen media has no grid_thw metadata"))?;
    if metadata.shape.len() != 2 || metadata.shape[1] != 3 || metadata.shape[0] == 0 {
        return Err(unsupported(
            architecture,
            format!(
                "patch grid must be shaped [items, 3], got {:?}",
                metadata.shape
            ),
        ));
    }
    let expected_values = checked_mul(metadata.shape[0], 3, "Qwen patch-grid scalar count")?;
    if u64::try_from(metadata.values.len()).ok() != Some(expected_values) {
        return Err(unsupported(
            architecture,
            "Qwen patch grid has an incomplete row",
        ));
    }
    let grid = metadata
        .values
        .chunks_exact(3)
        .map(|row| (row[0], row[1], row[2]))
        .collect::<Vec<_>>();
    let described_patches = grid.iter().try_fold(0u64, |total, (time, height, width)| {
        let item_patches = checked_mul(
            checked_mul(
                nonzero_positive(*time, "Qwen grid time")?,
                nonzero_positive(*height, "Qwen grid height")?,
                "Qwen grid time-height",
            )?,
            nonzero_positive(*width, "Qwen grid width")?,
            "Qwen grid item patches",
        )?;
        checked_add(total, item_patches, "Qwen described patch total")
    })?;
    if described_patches != patches {
        return Err(unsupported(
            architecture,
            format!("Qwen grid describes {described_patches} patches but payload has {patches}"),
        ));
    }
    let (full_chunk_squares, window_chunk_squares) =
        qwen_attention_chunk_squares(&grid, merge, config.window_size, config.patch_size)?;
    let depth =
        u64::try_from(config.layer_count()).map_err(|_| CapabilityError::ArithmeticOverflow {
            operation: "Qwen vision depth",
        })?;
    let full_blocks = u64::try_from(
        config
            .layer_schedule
            .iter()
            .filter(|policy| matches!(policy.attention, VisionAttentionPolicy::Full))
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Qwen full-attention block count",
    })?;
    let window_blocks = depth - full_blocks;
    let heads = positive(config.num_heads, "Qwen vision heads")?;
    let hidden = positive(config.hidden_size, "Qwen vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Qwen vision intermediate size")?;
    let out_hidden = positive(config.out_hidden_size, "Qwen vision output size")?;
    let patch_hidden = checked_mul(patches, hidden, "Qwen patch hidden elements")?;
    let patch_intermediate =
        checked_mul(patches, intermediate, "Qwen patch intermediate elements")?;
    let per_block = checked_add(
        checked_mul(32, patch_hidden, "Qwen block hidden workspace")?,
        checked_mul(6, patch_intermediate, "Qwen block intermediate workspace")?,
        "Qwen block workspace",
    )?;
    let block_workspace = checked_mul(depth, per_block, "Qwen all-block workspace")?;
    let full_attention = checked_mul(
        checked_mul(
            checked_mul(
                full_blocks,
                full_chunk_squares,
                "Qwen full-attention blocks",
            )?,
            heads,
            "Qwen full-attention heads",
        )?,
        2,
        "Qwen full-attention score/probability bound",
    )?;
    let window_attention = checked_mul(
        checked_mul(
            checked_mul(
                window_blocks,
                window_chunk_squares,
                "Qwen window-attention blocks",
            )?,
            heads,
            "Qwen window-attention heads",
        )?,
        2,
        "Qwen window-attention score/probability bound",
    )?;
    let merge_width = checked_mul(hidden, merge_area, "Qwen merger width")?;
    let merger_output = checked_mul(
        positions,
        checked_add(
            checked_mul(12, merge_width, "Qwen merger hidden workspace")?,
            checked_mul(6, out_hidden, "Qwen merger output workspace")?,
            "Qwen merger per-position workspace",
        )?,
        "Qwen merger workspace",
    )?;
    let mergers = checked_add(
        1,
        u64::try_from(config.deepstack_layer_count()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "Qwen deepstack merger count",
            }
        })?,
        "Qwen merger count",
    )?;
    let graph_scalars = checked_add(
        checked_add(
            checked_mul(16, patch_hidden, "Qwen vision setup workspace")?,
            block_workspace,
            "Qwen setup plus blocks",
        )?,
        checked_add(
            checked_add(full_attention, window_attention, "Qwen attention workspace")?,
            checked_mul(mergers, merger_output, "Qwen all-merger workspace")?,
            "Qwen attention plus mergers",
        )?,
        "Qwen vision graph workspace",
    )?;
    Ok(MediaShapePlan {
        decoder_positions: positions,
        execution_workspace_scalars: graph_scalars,
    })
}

/// Derives Qwen media geometry when the normalized hybrid policy may omit its
/// vision tower.
pub fn qwen_hybrid_vision(
    config: Option<&VisionConfig>,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    config
        .ok_or_else(|| unsupported(architecture, "loaded model has no vision configuration"))
        .and_then(|config| qwen_vision(config, input, architecture))
}

fn qwen_vision_ingress(
    config: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<QwenVisionIngressPlan, CapabilityError> {
    let shape = qwen_hybrid_vision(config, input, architecture)?;
    let token_id = match input.modality {
        MediaModality::Image => image_token_id,
        MediaModality::Video => video_token_id,
        MediaModality::Audio => None,
    }
    .ok_or_else(|| unsupported(architecture, "prepared media placeholder token is absent"))?;
    let placeholder_token_id =
        u32::try_from(token_id).map_err(|_| CapabilityError::InvalidConfiguration {
            field: "Qwen media placeholder token",
            detail: format!("expected a non-negative token ID, got {token_id}"),
        })?;
    let metadata = input
        .patch_grid
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Qwen media has no grid_thw metadata"))?;
    let patch_grid = metadata
        .values
        .chunks_exact(3)
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    Ok(QwenVisionIngressPlan {
        placeholder_token_id,
        placeholder_count: shape.decoder_positions,
        patch_grid,
    })
}

/// Validates prepared Qwen3-VL media and derives its execution ingress policy.
pub fn qwen_vl_ingress(
    args: &QwenVlModelArgs,
    input: &PreparedMediaInput,
) -> Result<QwenVisionIngressPlan, CapabilityError> {
    qwen_vision_ingress(
        Some(&args.vision),
        Some(args.image_token_id),
        Some(args.video_token_id),
        input,
        &args.model_type,
    )
}

/// Validates prepared conditional Qwen3.5 media and derives its execution
/// ingress policy.
pub fn qwen_hybrid_ingress(
    args: &ParsedHybridConfig,
    input: &PreparedMediaInput,
) -> Result<QwenVisionIngressPlan, CapabilityError> {
    qwen_vision_ingress(
        args.vision.as_ref(),
        args.image_token_id,
        args.video_token_id,
        input,
        &args.text.model_type,
    )
}

fn gemma_valid_patch_count(
    positions: &MediaMetadata<i32>,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if positions.shape.len() != 3 || positions.shape[0] != 1 || positions.shape[2] != 2 {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma patch positions must be [1, patches, 2], got {:?}",
                positions.shape
            ),
        ));
    }
    let expected = checked_mul(positions.shape[1], 2, "Gemma patch-position scalar count")?;
    if u64::try_from(positions.values.len()).ok() != Some(expected) {
        return Err(unsupported(
            architecture,
            "Gemma patch positions do not match their declared shape",
        ));
    }
    u64::try_from(
        positions
            .values
            .chunks_exact(2)
            .filter(|pair| pair[0] >= 0 && pair[1] >= 0)
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Gemma valid patch count",
    })
}

fn gemma_vision(
    config: &crate::gemma4::VisionConfig,
    text_hidden: u64,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.payload_shape.len() != 3 || input.payload_shape[0] != 1 {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared vision tensor must be [1, patches, patch_dims], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let patch = nonzero_positive(config.patch_size, "Gemma vision patch size")?;
    let expected_patch_dims = checked_mul(
        3,
        checked_mul(patch, patch, "Gemma vision patch area")?,
        "Gemma vision patch dimensions",
    )?;
    if input.payload_shape[2] != expected_patch_dims {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared patches have width {}, expected {expected_patch_dims}",
                input.payload_shape[2]
            ),
        ));
    }
    let position_ids = input
        .patch_positions
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Gemma media has no patch positions"))?;
    let valid_patches = gemma_valid_patch_count(position_ids, architecture)?;
    let pool = nonzero_positive(config.pooling_kernel_size, "Gemma pooling kernel")?;
    let pool_area = checked_mul(pool, pool, "Gemma pooling area")?;
    if valid_patches % pool_area != 0 {
        return Err(unsupported(
            architecture,
            format!("Gemma valid patch count {valid_patches} is not divisible by {pool_area}"),
        ));
    }
    let positions = valid_patches / pool_area;
    let padded_patches = input.payload_shape[1];
    if position_ids.shape[1] != padded_patches {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma patch positions {:?} do not match prepared vision payload {:?}",
                position_ids.shape, input.payload_shape
            ),
        ));
    }
    let hidden = positive(config.hidden_size, "Gemma vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Gemma vision intermediate size")?;
    let depth = positive(config.num_hidden_layers, "Gemma vision depth")?;
    let patch_hidden = checked_mul(padded_patches, hidden, "Gemma vision patch hidden elements")?;
    let per_layer = checked_add(
        checked_mul(48, patch_hidden, "Gemma vision layer hidden workspace")?,
        checked_mul(
            8,
            checked_mul(
                padded_patches,
                intermediate,
                "Gemma vision intermediate elements",
            )?,
            "Gemma vision MLP workspace",
        )?,
        "Gemma vision layer workspace",
    )?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, hidden, "Gemma pooled vision workspace")?,
            checked_mul(8, text_hidden, "Gemma projected vision workspace")?,
            "Gemma vision output workspace per position",
        )?,
        "Gemma vision output workspace",
    )?;
    let graph_scalars = checked_add(
        checked_mul(20, patch_hidden, "Gemma vision setup workspace")?,
        checked_add(
            checked_mul(depth, per_layer, "Gemma all vision layers")?,
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

fn gemma_audio(
    config: &crate::gemma4::AudioConfig,
    text_hidden: u64,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.payload_shape.len() != 3
        || input.payload_shape[0] != 1
        || input.payload_shape[2] != 128
    {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma prepared audio tensor must be [1, frames, 128], got {:?}",
                input.payload_shape
            ),
        ));
    }
    let mask = input
        .audio_mask
        .as_ref()
        .ok_or_else(|| unsupported(architecture, "prepared Gemma audio has no frame mask"))?;
    let frames = input.payload_shape[1];
    if mask.shape != [1, frames] || u64::try_from(mask.values.len()).ok() != Some(frames) {
        return Err(unsupported(
            architecture,
            format!(
                "Gemma audio mask must be [1, {frames}], got {:?}",
                mask.shape
            ),
        ));
    }
    let valid_frames =
        u64::try_from(mask.values.iter().filter(|value| **value).count()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "Gemma valid audio frame count",
            }
        })?;
    let positions = valid_frames.div_ceil(4);
    let sequence = frames.div_ceil(4);
    let hidden = positive(config.hidden_size, "Gemma audio hidden size")?;
    let depth = positive(config.num_hidden_layers, "Gemma audio depth")?;
    let heads = positive(config.num_attention_heads, "Gemma audio heads")?;
    let chunk = nonzero_positive(config.attention_chunk_size, "Gemma audio attention chunk")?;
    let past = nonzero_positive(
        config.attention_context_left.checked_sub(1).ok_or(
            CapabilityError::ArithmeticOverflow {
                operation: "Gemma audio left context",
            },
        )?,
        "Gemma audio left context",
    )?;
    let padded_sequence = checked_mul(
        sequence.div_ceil(chunk),
        chunk,
        "Gemma padded audio sequence",
    )?;
    let chunks = padded_sequence / chunk;
    let attention_elements = checked_mul(
        checked_mul(
            checked_mul(
                checked_mul(chunks, heads, "Gemma audio attention chunk heads")?,
                chunk,
                "Gemma audio attention queries",
            )?,
            checked_add(chunk, past, "Gemma audio attention key bound")?,
            "Gemma audio attention scores",
        )?,
        4,
        "Gemma audio logits/relative/mask/probability workspace",
    )?;
    let layer_workspace = checked_add(
        checked_mul(
            80,
            checked_mul(sequence, hidden, "Gemma audio hidden elements")?,
            "Gemma audio layer hidden workspace",
        )?,
        attention_elements,
        "Gemma audio layer workspace",
    )?;
    let first_frames = frames.div_ceil(2);
    let first_channels = positive(
        *config.subsampling_conv_channels.first().ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no first convolution channel count".into(),
            }
        })?,
        "Gemma audio first convolution channels",
    )?;
    let second_channels = positive(
        *config.subsampling_conv_channels.get(1).ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no second convolution channel count".into(),
            }
        })?,
        "Gemma audio second convolution channels",
    )?;
    let conv_workspace = checked_add(
        checked_mul(
            6,
            checked_mul(
                checked_mul(first_frames, 64, "Gemma first convolution grid")?,
                first_channels,
                "Gemma first convolution elements",
            )?,
            "Gemma first convolution workspace",
        )?,
        checked_mul(
            6,
            checked_mul(
                checked_mul(sequence, 32, "Gemma second convolution grid")?,
                second_channels,
                "Gemma second convolution elements",
            )?,
            "Gemma second convolution workspace",
        )?,
        "Gemma convolution workspace",
    )?;
    let output = positive(config.output_proj_dims, "Gemma audio output size")?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, output, "Gemma audio output workspace")?,
            checked_mul(8, text_hidden, "Gemma audio text projection workspace")?,
            "Gemma audio output workspace per position",
        )?,
        "Gemma audio projected output workspace",
    )?;
    let graph_scalars = checked_add(
        conv_workspace,
        checked_add(
            checked_mul(depth, layer_workspace, "Gemma all audio layers")?,
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
pub fn gemma4(
    args: &crate::gemma4::FamilyConfig,
    input: &PreparedMediaInput,
) -> Result<MediaShapePlan, CapabilityError> {
    let text_hidden = positive(args.text.hidden_size, "Gemma text hidden size")?;
    match input.modality {
        MediaModality::Image | MediaModality::Video => args
            .vision
            .as_ref()
            .ok_or_else(|| unsupported(&args.model_type, "loaded model has no vision tower"))
            .and_then(|config| gemma_vision(config, text_hidden, input, &args.model_type)),
        MediaModality::Audio => args
            .audio
            .as_ref()
            .ok_or_else(|| unsupported(&args.model_type, "loaded model has no audio tower"))
            .and_then(|config| gemma_audio(config, text_hidden, input, &args.model_type)),
    }
}

/// Derives prepared Inkling media geometry from normalized family policy.
pub fn inkling(
    args: &crate::inkling::ModelArgs,
    input: &PreparedMediaInput,
) -> Result<MediaShapePlan, CapabilityError> {
    match input.modality {
        MediaModality::Image => {
            let config = args.vision_config.as_ref().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "loaded Inkling model has no vision configuration",
                )
            })?;
            if input.payload_shape.len() != 5 || input.payload_shape[1..] != [2, 40, 40, 3] {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                        input.payload_shape
                    ),
                ));
            }
            let patches = input.payload_shape[0];
            let text_hidden = positive(config.text_hidden_size, "Inkling vision output size")?;
            let layer_outputs = [
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        8 * 8,
                        "Inkling vision grid",
                    )?,
                    128,
                    "Inkling vision layer 1",
                )?,
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        4 * 4,
                        "Inkling vision grid",
                    )?,
                    512,
                    "Inkling vision layer 2",
                )?,
                checked_mul(
                    checked_mul(patches, 2, "Inkling vision time")?,
                    4_800,
                    "Inkling vision layer 3",
                )?,
                checked_mul(patches, text_hidden, "Inkling vision layer 4")?,
            ];
            let graph_scalars = layer_outputs.iter().try_fold(0u64, |total, value| {
                checked_add(
                    total,
                    checked_mul(12, *value, "Inkling vision layer workspace")?,
                    "Inkling vision graph workspace",
                )
            })?;
            Ok(MediaShapePlan {
                decoder_positions: patches,
                execution_workspace_scalars: graph_scalars,
            })
        }
        MediaModality::Audio => {
            let config = args.audio_config.as_ref().ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "loaded Inkling model has no audio configuration",
                )
            })?;
            let [1, padded_frames, payload_codebooks] = input.payload_shape.as_slice() else {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling audio tokens must be [1, frames, codebooks], got {:?}",
                        input.payload_shape
                    ),
                ));
            };
            let (padded_frames, payload_codebooks) = (*padded_frames, *payload_codebooks);
            let codebooks = positive(config.num_codebooks, "Inkling audio codebooks")?;
            if payload_codebooks != codebooks {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "Inkling audio payload has {payload_codebooks} codebooks, expected {codebooks}"
                    ),
                ));
            }
            let frames = if let Some(mask) = &input.audio_mask {
                if mask.shape != [1, padded_frames]
                    || u64::try_from(mask.values.len()).ok() != Some(padded_frames)
                {
                    return Err(unsupported(
                        &args.model_type,
                        format!(
                            "Inkling audio mask must be [1, {padded_frames}], got {:?}",
                            mask.shape
                        ),
                    ));
                }
                let frames = mask.values.iter().take_while(|value| **value).count();
                if mask.values[frames..].iter().any(|value| *value) {
                    return Err(unsupported(
                        &args.model_type,
                        "Inkling audio mask must describe one valid prefix",
                    ));
                }
                u64::try_from(frames).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Inkling valid audio frame count",
                })?
            } else {
                padded_frames
            };
            let hidden = positive(config.text_hidden_size, "Inkling audio hidden size")?;
            let embedded = checked_mul(
                checked_mul(padded_frames, codebooks, "Inkling audio frame codebooks")?,
                hidden,
                "Inkling audio embedding elements",
            )?;
            let reduced = checked_mul(padded_frames, hidden, "Inkling audio reduced elements")?;
            let graph_scalars = checked_add(
                checked_mul(4, embedded, "Inkling audio embedding workspace")?,
                checked_mul(12, reduced, "Inkling audio reduction/norm workspace")?,
                "Inkling audio graph workspace",
            )?;
            Ok(MediaShapePlan {
                decoder_positions: frames,
                execution_workspace_scalars: graph_scalars,
            })
        }
        MediaModality::Video => Err(unsupported(
            &args.model_type,
            "video is not a supported Inkling modality",
        )),
    }
}

/// Derives the authoritative Inkling placeholder token and span for execution.
pub fn inkling_ingress(
    args: &crate::inkling::ModelArgs,
    input: &PreparedMediaInput,
) -> Result<InklingIngressPlan, CapabilityError> {
    let shape = inkling(args, input)?;
    let placeholder_token_id = match input.modality {
        MediaModality::Image => args.image_token_id,
        MediaModality::Audio => args.audio_token_id,
        MediaModality::Video => {
            return Err(unsupported(
                &args.model_type,
                "video is not a supported Inkling modality",
            ))
        }
    };
    Ok(InklingIngressPlan {
        placeholder_token_id,
        placeholder_count: shape.decoder_positions,
    })
}

/// Derives prepared Muse-Glimmer media geometry and artifact modality policy.
pub fn muse_glimmer(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &PreparedMediaInput,
) -> Result<MediaShapePlan, CapabilityError> {
    if input.modality == MediaModality::Audio
        || (input.modality == MediaModality::Video
            && args.weight_convention == crate::muse_glimmer::WeightConvention::Gguf)
    {
        return Err(unsupported(
            &args.model_type,
            format!(
                "loaded Muse-Glimmer artifact does not support {}",
                input.modality.as_str()
            ),
        ));
    }
    let grid = input.patch_grid.as_ref().ok_or_else(|| {
        unsupported(
            &args.model_type,
            "Muse-Glimmer media requires patch_grid metadata",
        )
    })?;
    if grid.shape.len() != 2 || grid.shape[0] == 0 || grid.shape[1] != 3 {
        return Err(unsupported(
            &args.model_type,
            format!(
                "Muse-Glimmer patch_grid must be [items, 3], got {:?}",
                grid.shape
            ),
        ));
    }
    let expected_values = checked_mul(grid.shape[0], 3, "Muse patch-grid scalar count")?;
    if u64::try_from(grid.values.len()).ok() != Some(expected_values) {
        return Err(unsupported(
            &args.model_type,
            "Muse-Glimmer patch_grid has an incomplete row",
        ));
    }
    let merge = nonzero_positive(args.vision_config.merge_size, "Muse vision merge size")?;
    let mut patches = 0u64;
    let mut positions = 0u64;
    for entry in grid.values.chunks_exact(3) {
        if entry.iter().any(|value| *value <= 0)
            || u64::try_from(entry[1]).unwrap_or_default() % merge != 0
            || u64::try_from(entry[2]).unwrap_or_default() % merge != 0
        {
            return Err(unsupported(
                &args.model_type,
                "Muse-Glimmer vision grids must be positive and merge-divisible",
            ));
        }
        let t = entry[0] as u64;
        let h = entry[1] as u64;
        let w = entry[2] as u64;
        patches = checked_add(
            patches,
            checked_mul(
                checked_mul(t, h, "Muse vision t*h")?,
                w,
                "Muse vision patches",
            )?,
            "Muse vision patch total",
        )?;
        positions = checked_add(
            positions,
            checked_mul(
                checked_mul(t, h / merge, "Muse merged t*h")?,
                w / merge,
                "Muse merged positions",
            )?,
            "Muse merged position total",
        )?;
    }
    if input.payload_shape.first().copied() != Some(patches) {
        return Err(unsupported(
            &args.model_type,
            format!(
                "Muse-Glimmer payload has {} patches but metadata describes {patches}",
                input.payload_shape.first().copied().unwrap_or_default()
            ),
        ));
    }
    let graph = checked_mul(
        patches,
        positive(args.vision_config.hidden_size, "Muse vision hidden size")?,
        "Muse vision activation scalars",
    )?;
    Ok(MediaShapePlan {
        decoder_positions: positions,
        execution_workspace_scalars: checked_mul(graph, 8, "Muse vision graph multiplier")?,
    })
}

/// Derives the authoritative Muse-Glimmer placeholder span and vision grid.
pub fn muse_glimmer_ingress(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &PreparedMediaInput,
) -> Result<MuseGlimmerIngressPlan, CapabilityError> {
    let shape = muse_glimmer(args, input)?;
    let placeholder_token_id = match input.modality {
        MediaModality::Image => args.image_token_id,
        MediaModality::Video => args.video_token_id,
        MediaModality::Audio => {
            return Err(unsupported(
                &args.model_type,
                "loaded Muse-Glimmer artifact does not support audio",
            ))
        }
    };
    let patch_grid = input
        .patch_grid
        .as_ref()
        .ok_or_else(|| {
            unsupported(
                &args.model_type,
                "Muse-Glimmer media requires patch_grid metadata",
            )
        })?
        .values
        .chunks_exact(3)
        .map(|entry| (entry[0], entry[1], entry[2]))
        .collect();
    Ok(MuseGlimmerIngressPlan {
        placeholder_token_id,
        placeholder_count: shape.decoder_positions,
        patch_grid,
    })
}

/// Rejects prepared media for a text-only architecture.
pub fn text_only(
    architecture: &str,
    input: &PreparedMediaInput,
) -> Result<MediaShapePlan, CapabilityError> {
    Err(unsupported(
        architecture,
        format!("{} media is not supported", input.modality.as_str()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn input(modality: MediaModality, payload_shape: &[u64]) -> PreparedMediaInput {
        PreparedMediaInput {
            modality,
            payload_shape: payload_shape.to_vec(),
            patch_grid: None,
            patch_positions: None,
            audio_mask: None,
        }
    }

    #[test]
    fn gemma_patch_positions_ignore_padding() {
        let positions = MediaMetadata {
            shape: vec![1, 5, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1, -1, -1],
        };
        assert_eq!(gemma_valid_patch_count(&positions, "gemma4").unwrap(), 4);
    }

    #[test]
    fn qwen_plan_owns_window_geometry_and_grid_validation() {
        let config = VisionConfig {
            layer_schedule: eredu_core::attention::LayerSchedule::new(
                2,
                vec![
                    crate::qwen::vision::VisionLayerPolicy {
                        attention: VisionAttentionPolicy::Windowed,
                        deepstack_merger: Some(0),
                    },
                    crate::qwen::vision::VisionLayerPolicy {
                        attention: VisionAttentionPolicy::Full,
                        deepstack_merger: None,
                    },
                ],
            )
            .unwrap(),
            hidden_size: 8,
            hidden_act: "silu".into(),
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 4,
            in_channels: 3,
            patch_size: 1,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            window_size: 4,
            out_hidden_size: 8,
            linear_formats: Default::default(),
        };
        let mut prepared = input(MediaModality::Image, &[4, 3]);
        prepared.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        let plan = qwen_vision(&config, &prepared, "qwen3_vl").unwrap();
        assert_eq!(plan.decoder_positions, 1);
        assert!(plan.execution_workspace_scalars > 4 * 8);

        let ingress =
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl").unwrap();
        assert_eq!(ingress.placeholder_token_id, 22);
        assert_eq!(ingress.placeholder_count, 1);
        assert_eq!(ingress.patch_grid, vec![(1, 2, 2)]);

        prepared.modality = MediaModality::Video;
        assert_eq!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",)
                .unwrap()
                .placeholder_token_id,
            23
        );

        prepared.patch_grid.as_mut().unwrap().values = vec![1, 2, 4];
        assert!(matches!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        prepared.patch_grid.as_mut().unwrap().values = vec![i32::MAX, i32::MAX, i32::MAX];
        assert!(matches!(
            qwen_vision_ingress(Some(&config), Some(22), Some(23), &prepared, "qwen3_vl",),
            Err(CapabilityError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn gemma_plan_owns_patch_pooling_and_audio_mask_geometry() {
        let vision_config = crate::gemma4::VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 1,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-5,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        let mut vision = input(MediaModality::Image, &[1, 5, 3]);
        vision.patch_positions = Some(MediaMetadata {
            shape: vec![1, 5, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1, -1, -1],
        });
        let vision_plan = gemma_vision(&vision_config, 8, &vision, "gemma4").unwrap();
        assert_eq!(vision_plan.decoder_positions, 1);

        let audio_config = crate::gemma4::AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            output_proj_dims: 8,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 3,
            attention_context_right: 0,
            attention_invalid_logits_value: -1e9,
            attention_logit_cap: 50.0,
            residual_weight: 1.0,
            rms_norm_eps: 1e-5,
            subsampling_conv_channels: vec![4, 8],
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        let mut audio = input(MediaModality::Audio, &[1, 8, 128]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 8],
            values: vec![true, true, true, true, true, true, false, false],
        });
        let audio_plan = gemma_audio(&audio_config, 8, &audio, "gemma4").unwrap();
        assert_eq!(audio_plan.decoder_positions, 2);
    }

    fn tiny_inkling() -> crate::inkling::ModelArgs {
        crate::inkling::ModelArgs::from_hf_json(
            &serde_json::to_vec(&json!({
                "model_type":"inkling_mm_model",
                "text_config":{
                    "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                    "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                    "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                    "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                    "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                    "intermediate_size":24,"dense_intermediate_size":48,
                    "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                    "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                    "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                    "use_global_scale":true,"gate_activation":"sigmoid"
                },
                "audio_config":{"decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16},
                "vision_config":{"decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                    "n_channels":3,"n_layers":4}
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn inkling_plan_owns_hmlp_and_dmel_shapes() {
        let args = tiny_inkling();
        let image = input(MediaModality::Image, &[1, 2, 40, 40, 3]);
        let image_plan = inkling(&args, &image).unwrap();
        assert_eq!(image_plan.decoder_positions, 1);
        assert_eq!(
            inkling_ingress(&args, &image).unwrap().placeholder_token_id,
            args.image_token_id
        );

        let mut audio = input(MediaModality::Audio, &[1, 3, 80]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![true, true, false],
        });
        let audio_plan = inkling(&args, &audio).unwrap();
        assert_eq!(audio_plan.decoder_positions, 2);
        let ingress = inkling_ingress(&args, &audio).unwrap();
        assert_eq!(ingress.placeholder_token_id, args.audio_token_id);
        assert_eq!(ingress.placeholder_count, 2);

        audio.audio_mask.as_mut().unwrap().values = vec![true, false, true];
        assert!(matches!(
            inkling_ingress(&args, &audio),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
        assert!(matches!(
            inkling(&args, &input(MediaModality::Audio, &[3, 80])),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
        assert!(matches!(
            inkling(&args, &input(MediaModality::Video, &[1, 2])),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    fn tiny_muse() -> crate::muse_glimmer::DecoderConfig {
        crate::muse_glimmer::DecoderConfig::from_hf_value(&json!({
          "architectures":["MuseGlimmerForConditionalGeneration"],
          "model_type":"muse_glimmer",
          "image_token_id":22,"video_token_id":23,"out_hidden_size":32,"projector_hidden_size":16,
          "text_config":{"model_type":"muse_glimmer_text","hidden_size":16,"num_hidden_layers":1,
            "intermediate_size":0,"moe_intermediate_size":12,"num_experts":4,"num_experts_per_tok":2,
            "norm_topk_prob":true,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "rms_norm_eps":0.00001,"post_norm_eps":0.00001,"vocab_size":24,"max_position_embeddings":64,
            "rope_theta":10000.0,"layer_types":["sliding_attention"],"layer_rope_theta":[10000.0],
            "sliding_window":8,"tie_word_embeddings":false,"hidden_act":"silu","attention_dropout":0.0,
            "qk_scale_factor":1.0,"output_multiplier":1.0,"final_logit_softcapping":30.0},
          "vision_config":{"model_type":"muse_glimmer_vision","hidden_size":8,"intermediate_size":12,
            "num_attention_heads":2,"num_hidden_layers":1,"patch_size":2,"patch_temporal":1,"merge_size":2,
            "pos_emb_height":2,"pos_emb_width":2,"max_position_embeddings":4,"layer_norm_eps":0.00001,
            "hidden_act":"gelu","layer_types":["full_attention"],
            "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"}}
        }))
        .unwrap()
    }

    #[test]
    fn muse_plan_owns_grid_geometry_and_artifact_video_policy() {
        let mut prepared = input(MediaModality::Video, &[4, 12]);
        prepared.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        let mut args = tiny_muse();
        let plan = muse_glimmer(&args, &prepared).unwrap();
        assert_eq!(plan.decoder_positions, 1);
        let ingress = muse_glimmer_ingress(&args, &prepared).unwrap();
        assert_eq!(ingress.placeholder_token_id, 23);
        assert_eq!(ingress.placeholder_count, 1);
        assert_eq!(ingress.patch_grid, vec![(1, 2, 2)]);

        args.weight_convention = crate::muse_glimmer::WeightConvention::Gguf;
        assert!(matches!(
            muse_glimmer(&args, &prepared),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }
}
