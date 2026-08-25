//! Backend-neutral prepared-media admission and workspace plans.
//!
//! Concrete backends extract shapes and small metadata values from their
//! native arrays. Architecture policy validates those values and returns the
//! decoder positions and scalar workspace implied by the family equations.

use eredu_core::CapabilityError;

use crate::qwen::{
    hybrid::{HybridConfig, ParsedHybridConfig},
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
    /// Host-known `(time, height, width)` extent for one vision part.
    pub patch_extent: Option<[i32; 3]>,
    /// Host-known valid prefix of an audio feature sequence.
    pub audio_valid_frames: Option<i32>,
}

/// Modality of one prepared model-input part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedInputModality {
    /// Token or projected text input.
    Text,
    /// Still-image input.
    Image,
    /// Audio input.
    Audio,
    /// Video input.
    Video,
}

impl PreparedInputModality {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Backend-neutral payload facts for one prepared model-input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedInputPayload {
    /// Token IDs with their logical shape.
    TokenIds(Vec<u64>),
    /// Already-projected embeddings with their logical shape.
    Embeddings(Vec<u64>),
    /// A model-native tensor and any available media metadata.
    Tensor {
        /// Logical tensor shape.
        shape: Vec<u64>,
        /// Media geometry for a non-text modality.
        media: Option<PreparedMediaInput>,
    },
}

impl PreparedInputPayload {
    fn as_str(&self) -> &'static str {
        match self {
            Self::TokenIds(_) => "token-ID",
            Self::Embeddings(_) => "embedding",
            Self::Tensor { .. } => "tensor",
        }
    }
}

/// Backend-neutral description of one prepared model-input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedInputPart {
    /// Semantic modality attached to the part.
    pub modality: PreparedInputModality,
    /// Payload representation and logical shape or metadata.
    pub payload: PreparedInputPayload,
}

/// Architecture-owned admission and accounting plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedInputPartPlan {
    /// Text decoder positions supplied as token IDs or accepted embeddings.
    Text {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained by decoder input assembly.
        modality: PreparedInputModality,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native media requiring tower execution.
    Media {
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
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

/// Qwen3-VL admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenVlInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width text embeddings supplied directly by the caller.
    ProjectedText {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Media {
        /// Validated placeholder and patch-grid ingress.
        ingress: QwenVisionIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

/// Qwen3.5/Qwen3-Next admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenHybridInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained for capability accounting.
        modality: PreparedInputModality,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Media {
        /// Validated placeholder and patch-grid ingress.
        ingress: QwenVisionIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

/// Gemma 4 admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4InputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width embeddings supplied directly by the caller.
    Projected {
        /// Semantic modality retained by the decoder input.
        modality: PreparedInputModality,
        /// Architecture-selected placeholder token repeated under the embeddings.
        placeholder_token_id: u32,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Vision {
        /// Architecture-selected placeholder token.
        placeholder_token_id: u32,
        /// Validated execution ingress geometry.
        ingress: crate::gemma4::VisionIngressPartPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
    /// Model-native audio input.
    Audio {
        /// Architecture-selected placeholder token.
        placeholder_token_id: u32,
        /// Validated execution ingress geometry.
        ingress: crate::gemma4::AudioIngressPartPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<Gemma4InputPartPlan> for PreparedInputPartPlan {
    fn from(plan: Gemma4InputPartPlan) -> Self {
        match plan {
            Gemma4InputPartPlan::TextTokens { positions } => Self::Text { positions },
            Gemma4InputPartPlan::Projected {
                modality,
                positions,
                ..
            } => Self::Projected {
                modality,
                positions,
            },
            Gemma4InputPartPlan::Vision { shape, .. }
            | Gemma4InputPartPlan::Audio { shape, .. } => Self::Media { shape },
        }
    }
}

impl From<QwenVlInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: QwenVlInputPartPlan) -> Self {
        match plan {
            QwenVlInputPartPlan::TextTokens { positions }
            | QwenVlInputPartPlan::ProjectedText { positions } => Self::Text { positions },
            QwenVlInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

impl From<QwenHybridInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: QwenHybridInputPartPlan) -> Self {
        match plan {
            QwenHybridInputPartPlan::TextTokens { positions } => Self::Text { positions },
            QwenHybridInputPartPlan::Projected {
                modality,
                positions,
            } => Self::Projected {
                modality,
                positions,
            },
            QwenHybridInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

/// Inkling admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InklingInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Decoder-width image or audio embeddings supplied by the caller.
    Projected {
        /// Semantic modality retained by decoder input assembly.
        modality: PreparedInputModality,
        /// Architecture-selected placeholder token repeated under the embeddings.
        placeholder_token_id: u32,
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or audio input.
    Media {
        /// Semantic modality consumed by the tower.
        modality: PreparedInputModality,
        /// Architecture-selected placeholder and span.
        ingress: InklingIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<InklingInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: InklingInputPartPlan) -> Self {
        match plan {
            InklingInputPartPlan::TextTokens { positions } => Self::Text { positions },
            InklingInputPartPlan::Projected {
                modality,
                positions,
                ..
            } => Self::Projected {
                modality,
                positions,
            },
            InklingInputPartPlan::Media { shape, .. } => Self::Media { shape },
        }
    }
}

/// Muse-Glimmer admission and execution plan for one prepared input part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuseGlimmerInputPartPlan {
    /// Ordinary text token IDs.
    TextTokens {
        /// Decoder positions occupied by the part.
        positions: u64,
    },
    /// Model-native image or video input.
    Vision {
        /// Semantic modality consumed by the vision tower.
        modality: PreparedInputModality,
        /// Architecture-selected placeholder span and patch grid.
        ingress: MuseGlimmerIngressPlan,
        /// Decoder and workspace accounting for tower execution.
        shape: MediaShapePlan,
    },
}

impl From<MuseGlimmerInputPartPlan> for PreparedInputPartPlan {
    fn from(plan: MuseGlimmerInputPartPlan) -> Self {
        match plan {
            MuseGlimmerInputPartPlan::TextTokens { positions } => Self::Text { positions },
            MuseGlimmerInputPartPlan::Vision { shape, .. } => Self::Media { shape },
        }
    }
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

fn qwen_vision_ingress_with_shape(
    config: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<(QwenVisionIngressPlan, MediaShapePlan), CapabilityError> {
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
    Ok((
        QwenVisionIngressPlan {
            placeholder_token_id,
            placeholder_count: shape.decoder_positions,
            patch_grid,
        },
        shape,
    ))
}

fn qwen_vision_ingress(
    config: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedMediaInput,
    architecture: &str,
) -> Result<QwenVisionIngressPlan, CapabilityError> {
    qwen_vision_ingress_with_shape(config, image_token_id, video_token_id, input, architecture)
        .map(|(ingress, _)| ingress)
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

fn batch_one_sequence(
    shape: &[u64],
    rank: usize,
    name: &str,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if shape.len() != rank || shape.first() != Some(&1) || shape.get(1).copied().unwrap_or(0) == 0 {
        return Err(unsupported(
            architecture,
            format!("prepared {name} must be batch-one with rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape[1])
}

/// Validates one prepared Qwen3-VL part and derives the exact prefill and
/// capability-accounting plan consumed by a concrete backend.
pub fn qwen_vl_input_part(
    args: &QwenVlModelArgs,
    input: &PreparedInputPart,
) -> Result<QwenVlInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(QwenVlInputPartPlan::TextTokens {
                positions: batch_one_sequence(shape, 2, "text token IDs", &args.model_type)?,
            })
        }
        (PreparedInputModality::Text, PreparedInputPayload::Embeddings(shape)) => {
            let positions = batch_one_sequence(shape, 3, "text embeddings", &args.model_type)?;
            let hidden = positive(args.text.hidden_size, "Qwen3-VL hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared text embeddings must have hidden width {hidden}, got {:?}",
                        shape
                    ),
                ));
            }
            Ok(QwenVlInputPartPlan::ProjectedText { positions })
        }
        (
            PreparedInputModality::Image | PreparedInputModality::Video,
            PreparedInputPayload::Tensor {
                media: Some(media), ..
            },
        ) => {
            let expected = if input.modality == PreparedInputModality::Image {
                MediaModality::Image
            } else {
                MediaModality::Video
            };
            if media.modality != expected {
                return Err(unsupported(
                    &args.model_type,
                    "prepared input modality disagrees with its media metadata",
                ));
            }
            let (ingress, shape) = qwen_vision_ingress_with_shape(
                Some(&args.vision),
                Some(args.image_token_id),
                Some(args.video_token_id),
                media,
                &args.model_type,
            )?;
            Ok(QwenVlInputPartPlan::Media { ingress, shape })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Qwen3-VL does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
}

fn qwen_hybrid_input_part_with_policy(
    text: &HybridConfig,
    vision: Option<&VisionConfig>,
    image_token_id: Option<i32>,
    video_token_id: Option<i32>,
    input: &PreparedInputPart,
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(QwenHybridInputPartPlan::TextTokens {
                positions: batch_one_sequence(shape, 2, "text token IDs", &text.model_type)?,
            })
        }
        (
            modality @ (PreparedInputModality::Text
            | PreparedInputModality::Image
            | PreparedInputModality::Video),
            PreparedInputPayload::Embeddings(shape),
        ) => {
            let positions = batch_one_sequence(
                shape,
                3,
                &format!("{} embeddings", modality.as_str()),
                &text.model_type,
            )?;
            let hidden = positive(text.hidden_size, "Qwen hybrid hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &text.model_type,
                    format!(
                        "prepared {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            Ok(QwenHybridInputPartPlan::Projected {
                modality: *modality,
                positions,
            })
        }
        (
            PreparedInputModality::Image | PreparedInputModality::Video,
            PreparedInputPayload::Tensor {
                media: Some(media), ..
            },
        ) => {
            let expected = if input.modality == PreparedInputModality::Image {
                MediaModality::Image
            } else {
                MediaModality::Video
            };
            if media.modality != expected {
                return Err(unsupported(
                    &text.model_type,
                    "prepared input modality disagrees with its media metadata",
                ));
            }
            let (ingress, shape) = qwen_vision_ingress_with_shape(
                vision,
                image_token_id,
                video_token_id,
                media,
                &text.model_type,
            )?;
            Ok(QwenHybridInputPartPlan::Media { ingress, shape })
        }
        (modality, payload) => Err(unsupported(
            &text.model_type,
            format!(
                "Qwen hybrid does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
}

/// Validates one prepared conditional Qwen3.5 input part and derives the exact
/// prefill and capability-accounting plan consumed by a concrete backend.
pub fn qwen_hybrid_input_part(
    args: &ParsedHybridConfig,
    input: &PreparedInputPart,
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    qwen_hybrid_input_part_with_policy(
        &args.text,
        args.vision.as_ref(),
        args.image_token_id,
        args.video_token_id,
        input,
    )
}

/// Validates one prepared text-only Qwen3.5/Qwen3-Next input part.
pub fn qwen_hybrid_text_input_part(
    args: &HybridConfig,
    input: &PreparedInputPart,
) -> Result<QwenHybridInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(QwenHybridInputPartPlan::TextTokens {
                positions: batch_one_sequence(shape, 2, "text token IDs", &args.model_type)?,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "text-only Qwen hybrid does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
}

fn gemma_batch_one_sequence(
    shape: &[u64],
    rank: usize,
    name: &str,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if shape.len() != rank || shape.first() != Some(&1) || shape.get(1).copied().unwrap_or(0) == 0 {
        return Err(unsupported(
            architecture,
            format!("prepared Gemma {name} must be batch-one with rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape[1])
}

fn gemma_placeholder_token(
    args: &crate::gemma4::FamilyConfig,
    modality: PreparedInputModality,
) -> Result<u32, CapabilityError> {
    let token = match modality {
        PreparedInputModality::Text => Some(args.text.pad_token_id),
        PreparedInputModality::Image => args.image_token_id,
        PreparedInputModality::Video => args.video_token_id,
        PreparedInputModality::Audio => args.audio_token_id,
    };
    token
        .and_then(|token| u32::try_from(token).ok())
        .ok_or_else(|| {
            unsupported(
                &args.model_type,
                format!("Gemma 4 has no valid {} placeholder", modality.as_str()),
            )
        })
}

/// Validates one prepared Gemma 4 part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn gemma4_input_part(
    args: &crate::gemma4::FamilyConfig,
    input: &PreparedInputPart,
) -> Result<Gemma4InputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(Gemma4InputPartPlan::TextTokens {
                positions: gemma_batch_one_sequence(shape, 2, "text token IDs", &args.model_type)?,
            })
        }
        (modality, PreparedInputPayload::Embeddings(shape)) => {
            let positions = gemma_batch_one_sequence(
                shape,
                3,
                &format!("{} embeddings", modality.as_str()),
                &args.model_type,
            )?;
            let hidden = positive(args.text.hidden_size, "Gemma text hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            Ok(Gemma4InputPartPlan::Projected {
                modality: *modality,
                placeholder_token_id: gemma_placeholder_token(args, *modality)?,
                positions,
            })
        }
        (
            modality @ (PreparedInputModality::Image | PreparedInputModality::Video),
            PreparedInputPayload::Tensor {
                shape,
                media: Some(media),
            },
        ) => {
            let expected = if *modality == PreparedInputModality::Image {
                MediaModality::Image
            } else {
                MediaModality::Video
            };
            if media.modality != expected || media.payload_shape != *shape {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma input modality or shape disagrees with its media metadata",
                ));
            }
            if media.patch_grid.is_none() {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} input has no patch grid",
                        modality.as_str()
                    ),
                ));
            }
            let extent = media.patch_extent.ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} input has no host-known patch extent",
                        modality.as_str()
                    ),
                )
            })?;
            let grid = media.patch_grid.as_ref().expect("checked above");
            if grid.shape != [1, 3] || grid.values.as_slice() != extent || grid.values.len() != 3 {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Gemma {} patch grid must be one row matching extent {extent:?}",
                        modality.as_str()
                    ),
                ));
            }
            let vision = args.vision.as_ref().ok_or_else(|| {
                unsupported(&args.model_type, "loaded Gemma model has no vision tower")
            })?;
            let padded_patches =
                i32::try_from(shape[1]).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma padded vision patch count",
                })?;
            let ingress = crate::gemma4::VisionIngressPartPlan::new(vision, extent, padded_patches)
                .map_err(|error| unsupported(&args.model_type, error.to_string()))?;
            let media_shape = gemma4(args, media)?;
            let valid_patches = gemma_valid_patch_count(
                media
                    .patch_positions
                    .as_ref()
                    .expect("Gemma media shape validation requires patch positions"),
                &args.model_type,
            )?;
            if valid_patches != ingress.valid_patches as u64
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma patch extent and position metadata disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Vision {
                placeholder_token_id: gemma_placeholder_token(args, *modality)?,
                ingress,
                shape: media_shape,
            })
        }
        (
            PreparedInputModality::Audio,
            PreparedInputPayload::Tensor {
                shape,
                media: Some(media),
            },
        ) => {
            if media.modality != MediaModality::Audio || media.payload_shape != *shape {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma audio shape disagrees with its media metadata",
                ));
            }
            let valid_frames = media.audio_valid_frames.ok_or_else(|| {
                unsupported(
                    &args.model_type,
                    "prepared Gemma audio has no host-known valid-frame extent",
                )
            })?;
            let padded_frames =
                i32::try_from(shape[1]).map_err(|_| CapabilityError::ArithmeticOverflow {
                    operation: "Gemma padded audio frame count",
                })?;
            let ingress = crate::gemma4::AudioIngressPartPlan::new(valid_frames, padded_frames)
                .map_err(|error| unsupported(&args.model_type, error.to_string()))?;
            let media_shape = gemma4(args, media)?;
            let valid_mask_frames = media
                .audio_mask
                .as_ref()
                .expect("Gemma media shape validation requires an audio mask")
                .values
                .iter()
                .filter(|value| **value)
                .count();
            if valid_mask_frames != usize::try_from(valid_frames).unwrap_or_default()
                || media_shape.decoder_positions != ingress.decoder_positions as u64
            {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Gemma audio valid-frame extent and mask disagree",
                ));
            }
            Ok(Gemma4InputPartPlan::Audio {
                placeholder_token_id: gemma_placeholder_token(args, PreparedInputModality::Audio)?,
                ingress,
                shape: media_shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Gemma 4 does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
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

/// Validates one prepared Inkling part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn inkling_input_part(
    args: &crate::inkling::ModelArgs,
    input: &PreparedInputPart,
) -> Result<InklingInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(InklingInputPartPlan::TextTokens {
                positions: batch_one_sequence(
                    shape,
                    2,
                    "Inkling text token IDs",
                    &args.model_type,
                )?,
            })
        }
        (
            modality @ (PreparedInputModality::Image | PreparedInputModality::Audio),
            PreparedInputPayload::Embeddings(shape),
        ) => {
            let positions = batch_one_sequence(
                shape,
                3,
                &format!("Inkling {} embeddings", modality.as_str()),
                &args.model_type,
            )?;
            let hidden = positive(args.text_config.hidden_size, "Inkling text hidden size")?;
            if shape[2] != hidden {
                return Err(unsupported(
                    &args.model_type,
                    format!(
                        "prepared Inkling {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            let placeholder_token_id = match modality {
                PreparedInputModality::Image => args.image_token_id,
                PreparedInputModality::Audio => args.audio_token_id,
                PreparedInputModality::Text | PreparedInputModality::Video => unreachable!(),
            };
            Ok(InklingInputPartPlan::Projected {
                modality: *modality,
                placeholder_token_id,
                positions,
            })
        }
        (
            modality @ (PreparedInputModality::Image | PreparedInputModality::Audio),
            PreparedInputPayload::Tensor {
                shape,
                media: Some(media),
            },
        ) => {
            let expected = if *modality == PreparedInputModality::Image {
                MediaModality::Image
            } else {
                MediaModality::Audio
            };
            if media.modality != expected || media.payload_shape != *shape {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Inkling input modality or shape disagrees with its media metadata",
                ));
            }
            let shape = inkling(args, media)?;
            let ingress = InklingIngressPlan {
                placeholder_token_id: if *modality == PreparedInputModality::Image {
                    args.image_token_id
                } else {
                    args.audio_token_id
                },
                placeholder_count: shape.decoder_positions,
            };
            Ok(InklingInputPartPlan::Media {
                modality: *modality,
                ingress,
                shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Inkling does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
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

/// Validates one prepared Muse-Glimmer part and derives the exact placeholder,
/// ingress geometry, and capability-accounting plan consumed by a backend.
pub fn muse_glimmer_input_part(
    args: &crate::muse_glimmer::DecoderConfig,
    input: &PreparedInputPart,
) -> Result<MuseGlimmerInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(MuseGlimmerInputPartPlan::TextTokens {
                positions: batch_one_sequence(
                    shape,
                    2,
                    "Muse-Glimmer text token IDs",
                    &args.model_type,
                )?,
            })
        }
        (
            modality @ (PreparedInputModality::Image | PreparedInputModality::Video),
            PreparedInputPayload::Tensor {
                shape,
                media: Some(media),
            },
        ) => {
            let expected = if *modality == PreparedInputModality::Image {
                MediaModality::Image
            } else {
                MediaModality::Video
            };
            if media.modality != expected || media.payload_shape != *shape {
                return Err(unsupported(
                    &args.model_type,
                    "prepared Muse-Glimmer input modality or shape disagrees with its media metadata",
                ));
            }
            let shape = muse_glimmer(args, media)?;
            let placeholder_token_id = if *modality == PreparedInputModality::Image {
                args.image_token_id
            } else {
                args.video_token_id
            };
            let patch_grid = media
                .patch_grid
                .as_ref()
                .expect("Muse-Glimmer shape validation requires a patch grid")
                .values
                .chunks_exact(3)
                .map(|entry| (entry[0], entry[1], entry[2]))
                .collect();
            Ok(MuseGlimmerInputPartPlan::Vision {
                modality: *modality,
                ingress: MuseGlimmerIngressPlan {
                    placeholder_token_id,
                    placeholder_count: shape.decoder_positions,
                    patch_grid,
                },
                shape,
            })
        }
        (modality, payload) => Err(unsupported(
            &args.model_type,
            format!(
                "Muse-Glimmer does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
}

/// Validates one prepared input part for a text-only architecture.
pub fn text_only_input_part(
    architecture: &str,
    input: &PreparedInputPart,
) -> Result<PreparedInputPartPlan, CapabilityError> {
    match (&input.modality, &input.payload) {
        (PreparedInputModality::Text, PreparedInputPayload::TokenIds(shape)) => {
            Ok(PreparedInputPartPlan::Text {
                positions: batch_one_sequence(shape, 2, "text token IDs", architecture)?,
            })
        }
        (modality, payload) => Err(unsupported(
            architecture,
            format!(
                "text-only architecture does not support a {} {} payload",
                modality.as_str(),
                payload.as_str()
            ),
        )),
    }
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
            patch_extent: None,
            audio_valid_frames: None,
        }
    }

    fn qwen_vl_args() -> QwenVlModelArgs {
        crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap()
    }

    fn qwen_hybrid_args() -> ParsedHybridConfig {
        crate::qwen::hybrid::model_args_from_config_value(&json!({
            "model_type":"qwen3_5","image_token_id":30,"video_token_id":31,
            "text_config":{
                "model_type":"qwen3_5_text","vocab_size":32,"hidden_size":16,
                "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"max_position_embeddings":64,"intermediate_size":32,
                "linear_conv_kernel_dim":2,"linear_key_head_dim":4,
                "linear_value_head_dim":4,"linear_num_key_heads":2,
                "linear_num_value_heads":4,
                "layer_types":["linear_attention","full_attention"],
                "tie_word_embeddings":false
            },
            "vision_config":{
                "depth":1,"hidden_size":8,"intermediate_size":16,"num_heads":2,
                "num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":16
            }
        }))
        .unwrap()
    }

    fn gemma_family() -> crate::gemma4::FamilyConfig {
        crate::gemma4::FamilyConfig::from_hf_json(
            &serde_json::to_vec(&json!({
                "model_type":"gemma4_unified",
                "tie_word_embeddings":false,
                "image_token_id":60,
                "audio_token_id":61,
                "text_config":{
                    "model_type":"gemma4_text","hidden_size":16,"num_hidden_layers":2,
                    "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                    "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":64,
                    "max_position_embeddings":128,"layer_types":["full_attention","full_attention"]
                },
                "vision_config":{
                    "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                    "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
                    "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                    "rms_norm_eps":0.000001
                },
                "audio_config":{
                    "hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                    "output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,
                    "attention_context_left":5,"attention_context_right":0,
                    "attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,
                    "residual_weight":0.5,"rms_norm_eps":0.000001,
                    "subsampling_conv_channels":[4,8]
                }
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn qwen_vl_admission_accepts_only_text_projected_embeddings() {
        let args = qwen_vl_args();
        let projected = PreparedInputPart {
            modality: PreparedInputModality::Text,
            payload: PreparedInputPayload::Embeddings(vec![1, 4, 32]),
        };
        assert_eq!(
            qwen_vl_input_part(&args, &projected).unwrap(),
            QwenVlInputPartPlan::ProjectedText { positions: 4 }
        );

        let non_text = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Embeddings(vec![1, 4, 32]),
        };
        assert!(matches!(
            qwen_vl_input_part(&args, &non_text),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn qwen_hybrid_admission_accepts_projected_text_image_and_video() {
        let args = qwen_hybrid_args();
        for modality in [
            PreparedInputModality::Text,
            PreparedInputModality::Image,
            PreparedInputModality::Video,
        ] {
            let projected = PreparedInputPart {
                modality,
                payload: PreparedInputPayload::Embeddings(vec![1, 4, 16]),
            };
            assert_eq!(
                qwen_hybrid_input_part(&args, &projected).unwrap(),
                QwenHybridInputPartPlan::Projected {
                    modality,
                    positions: 4,
                }
            );
        }

        let wrong_width = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Embeddings(vec![1, 4, 8]),
        };
        assert!(matches!(
            qwen_hybrid_input_part(&args, &wrong_width),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let audio = PreparedInputPart {
            modality: PreparedInputModality::Audio,
            payload: PreparedInputPayload::Embeddings(vec![1, 4, 16]),
        };
        assert!(matches!(
            qwen_hybrid_input_part(&args, &audio),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let projected_text = PreparedInputPart {
            modality: PreparedInputModality::Text,
            payload: PreparedInputPayload::Embeddings(vec![1, 4, 16]),
        };
        assert!(matches!(
            qwen_hybrid_text_input_part(&args.text, &projected_text),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn text_only_input_plan_rejects_every_non_token_payload() {
        let tokens = PreparedInputPart {
            modality: PreparedInputModality::Text,
            payload: PreparedInputPayload::TokenIds(vec![1, 4]),
        };
        assert_eq!(
            text_only_input_part("llama", &tokens).unwrap(),
            PreparedInputPartPlan::Text { positions: 4 }
        );

        for input in [
            PreparedInputPart {
                modality: PreparedInputModality::Text,
                payload: PreparedInputPayload::Embeddings(vec![1, 4, 16]),
            },
            PreparedInputPart {
                modality: PreparedInputModality::Image,
                payload: PreparedInputPayload::Embeddings(vec![1, 4, 16]),
            },
        ] {
            assert!(matches!(
                text_only_input_part("llama", &input),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
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
    fn gemma_input_plan_owns_payload_placeholder_and_ingress_policy() {
        let args = gemma_family();
        assert_eq!(
            crate::capability::gemma4(&args)
                .unwrap()
                .mtp_checkpoint_kind(),
            Some(eredu_core::MtpCheckpointKind::Separate)
        );

        let projected = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Embeddings(vec![1, 3, 16]),
        };
        assert_eq!(
            gemma4_input_part(&args, &projected).unwrap(),
            Gemma4InputPartPlan::Projected {
                modality: PreparedInputModality::Image,
                placeholder_token_id: 60,
                positions: 3,
            }
        );

        let mut vision = input(MediaModality::Image, &[1, 4, 48]);
        vision.patch_grid = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![1, 2, 2],
        });
        vision.patch_positions = Some(MediaMetadata {
            shape: vec![1, 4, 2],
            values: vec![0, 0, 1, 0, 0, 1, 1, 1],
        });
        vision.patch_extent = Some([1, 2, 2]);
        let vision = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Tensor {
                shape: vision.payload_shape.clone(),
                media: Some(vision),
            },
        };
        assert!(matches!(
            gemma4_input_part(&args, &vision).unwrap(),
            Gemma4InputPartPlan::Vision {
                placeholder_token_id: 60,
                ingress: crate::gemma4::VisionIngressPartPlan {
                    decoder_positions: 1,
                    ..
                },
                ..
            }
        ));
        let mut inconsistent_vision = vision.clone();
        let PreparedInputPayload::Tensor {
            media: Some(media), ..
        } = &mut inconsistent_vision.payload
        else {
            unreachable!()
        };
        media.patch_extent = Some([1, 4, 1]);
        assert!(matches!(
            gemma4_input_part(&args, &inconsistent_vision),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let mut audio = input(MediaModality::Audio, &[1, 8, 128]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 8],
            values: vec![true, true, true, true, true, true, false, false],
        });
        audio.audio_valid_frames = Some(6);
        let audio = PreparedInputPart {
            modality: PreparedInputModality::Audio,
            payload: PreparedInputPayload::Tensor {
                shape: audio.payload_shape.clone(),
                media: Some(audio),
            },
        };
        assert!(matches!(
            gemma4_input_part(&args, &audio).unwrap(),
            Gemma4InputPartPlan::Audio {
                placeholder_token_id: 61,
                ingress: crate::gemma4::AudioIngressPartPlan {
                    decoder_positions: 2,
                    ..
                },
                ..
            }
        ));
        let mut inconsistent_audio = audio.clone();
        let PreparedInputPayload::Tensor {
            media: Some(media), ..
        } = &mut inconsistent_audio.payload
        else {
            unreachable!()
        };
        media.audio_valid_frames = Some(5);
        assert!(matches!(
            gemma4_input_part(&args, &inconsistent_audio),
            Err(CapabilityError::UnsupportedInput { .. })
        ));

        let video = PreparedInputPart {
            modality: PreparedInputModality::Video,
            payload: PreparedInputPayload::Embeddings(vec![1, 1, 16]),
        };
        assert!(matches!(
            gemma4_input_part(&args, &video),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn capability_estimate_derives_embedded_mtp_from_exact_configuration() {
        let mut args = qwen_hybrid_args();
        assert_eq!(
            crate::capability::qwen_hybrid(&args)
                .unwrap()
                .mtp_checkpoint_kind(),
            None
        );
        args.text.mtp_num_hidden_layers = 2;
        assert_eq!(
            crate::capability::qwen_hybrid(&args)
                .unwrap()
                .mtp_checkpoint_kind(),
            Some(eredu_core::MtpCheckpointKind::Embedded)
        );
    }

    #[test]
    fn qwen_plan_owns_window_geometry_and_grid_validation() {
        let config = VisionConfig {
            mode: crate::qwen::vision::VisionMode::WindowScheduled,
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
        let image_part = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Tensor {
                shape: image.payload_shape.clone(),
                media: Some(image),
            },
        };
        assert!(matches!(
            inkling_input_part(&args, &image_part).unwrap(),
            InklingInputPartPlan::Media {
                ingress: InklingIngressPlan {
                    placeholder_token_id,
                    placeholder_count: 1,
                },
                ..
            } if placeholder_token_id == args.image_token_id
        ));

        let mut audio = input(MediaModality::Audio, &[1, 3, 80]);
        audio.audio_mask = Some(MediaMetadata {
            shape: vec![1, 3],
            values: vec![true, true, false],
        });
        let audio_plan = inkling(&args, &audio).unwrap();
        assert_eq!(audio_plan.decoder_positions, 2);
        let audio_part = |media: PreparedMediaInput| PreparedInputPart {
            modality: PreparedInputModality::Audio,
            payload: PreparedInputPayload::Tensor {
                shape: media.payload_shape.clone(),
                media: Some(media),
            },
        };
        assert!(matches!(
            inkling_input_part(&args, &audio_part(audio.clone())).unwrap(),
            InklingInputPartPlan::Media {
                ingress: InklingIngressPlan {
                    placeholder_token_id,
                    placeholder_count: 2,
                },
                ..
            } if placeholder_token_id == args.audio_token_id
        ));

        audio.audio_mask.as_mut().unwrap().values = vec![true, false, true];
        assert!(matches!(
            inkling_input_part(&args, &audio_part(audio)),
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

    #[test]
    fn inkling_input_plan_matches_projected_execution_policy() {
        let args = tiny_inkling();
        let projected = PreparedInputPart {
            modality: PreparedInputModality::Image,
            payload: PreparedInputPayload::Embeddings(vec![1, 3, 32]),
        };
        assert_eq!(
            inkling_input_part(&args, &projected).unwrap(),
            InklingInputPartPlan::Projected {
                modality: PreparedInputModality::Image,
                placeholder_token_id: args.image_token_id,
                positions: 3,
            }
        );

        for modality in [PreparedInputModality::Text, PreparedInputModality::Video] {
            let rejected = PreparedInputPart {
                modality,
                payload: PreparedInputPayload::Embeddings(vec![1, 3, 32]),
            };
            assert!(matches!(
                inkling_input_part(&args, &rejected),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
        }
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
        let prepared_part = |media: PreparedMediaInput| PreparedInputPart {
            modality: PreparedInputModality::Video,
            payload: PreparedInputPayload::Tensor {
                shape: media.payload_shape.clone(),
                media: Some(media),
            },
        };
        assert!(matches!(
            muse_glimmer_input_part(&args, &prepared_part(prepared.clone())).unwrap(),
            MuseGlimmerInputPartPlan::Vision {
                ingress: MuseGlimmerIngressPlan {
                    placeholder_token_id: 23,
                    placeholder_count: 1,
                    patch_grid,
                },
                ..
            } if patch_grid == vec![(1, 2, 2)]
        ));

        args.weight_convention = crate::muse_glimmer::WeightConvention::Gguf;
        assert!(matches!(
            muse_glimmer_input_part(&args, &prepared_part(prepared)),
            Err(CapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn muse_input_plan_rejects_projected_and_audio_payloads() {
        let args = tiny_muse();
        for input in [
            PreparedInputPart {
                modality: PreparedInputModality::Image,
                payload: PreparedInputPayload::Embeddings(vec![1, 1, 16]),
            },
            PreparedInputPart {
                modality: PreparedInputModality::Audio,
                payload: PreparedInputPayload::Tensor {
                    shape: vec![1, 2, 3],
                    media: Some(input(MediaModality::Audio, &[1, 2, 3])),
                },
            },
        ] {
            assert!(matches!(
                muse_glimmer_input_part(&args, &input),
                Err(CapabilityError::UnsupportedInput { .. })
            ));
        }
    }
}
