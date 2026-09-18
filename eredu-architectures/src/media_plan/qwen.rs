//! Shared borrowed Qwen admission equations. No inspection, allocation or authority.
use super::MediaShapePlan;
use crate::qwen::vision::{VisionAttentionPolicy, VisionConfig};
use eredu_core::{CapabilityError, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::input::host::{HostInputPartView, HostTensorValues};
use std::fmt;

/// Allocation-free semantic failure. Scalar diagnostics carry no source authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSemanticError {
    /// Actual source part, or None for whole-input/configuration validation.
    pub part: Option<usize>,
    /// Metadata role when the failure concerns one.
    pub key: Option<InputMetadataKey>,
    /// Stable equation/check name.
    pub operation: &'static str,
    /// Expected scalar when applicable.
    pub expected: Option<u64>,
    /// Observed scalar when applicable.
    pub actual: Option<i128>,
    fault: Fault,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    Input,
    Configuration,
    Overflow,
}
impl MediaSemanticError {
    /// Whether fixed checked arithmetic, rather than source semantics, failed.
    pub const fn is_overflow(&self) -> bool {
        matches!(self.fault, Fault::Overflow)
    }

    pub(crate) fn diagnostic(self) -> eredu_runtime::working_memory::CompositeSemanticDiagnostic {
        eredu_runtime::working_memory::CompositeSemanticDiagnostic {
            operation: self.operation,
            part: self.part,
            key: self.key,
            expected: self.expected,
            actual: self.actual,
        }
    }
    pub(crate) const fn input(operation: &'static str) -> Self {
        Self {
            part: None,
            key: None,
            operation,
            expected: None,
            actual: None,
            fault: Fault::Input,
        }
    }
    pub(crate) const fn overflow(operation: &'static str) -> Self {
        Self {
            fault: Fault::Overflow,
            ..Self::input(operation)
        }
    }
    pub(crate) const fn at(mut self, part: usize) -> Self {
        self.part = Some(part);
        self
    }
    pub(crate) const fn scalars(mut self, expected: u64, actual: i128) -> Self {
        self.expected = Some(expected);
        self.actual = Some(actual);
        self
    }
    fn config(operation: &'static str, actual: i32) -> Self {
        Self {
            fault: Fault::Configuration,
            actual: Some(i128::from(actual)),
            ..Self::input(operation)
        }
    }
    pub(crate) fn legacy(self, architecture: &str) -> CapabilityError {
        self.legacy_with(architecture, |message| Ok::<_, std::convert::Infallible>(message.to_string()))
            .unwrap_or_else(|never| match never {})
    }

    pub(crate) fn legacy_with<E>(
        self, architecture: &str,
        mut text: impl FnMut(fmt::Arguments<'_>) -> Result<String, E>,
    ) -> Result<CapabilityError, E> {
        Ok(match self.fault {
            Fault::Overflow => CapabilityError::ArithmeticOverflow {
                operation: self.operation,
            },
            Fault::Configuration => CapabilityError::InvalidConfiguration {
                field: self.operation,
                detail: text(format_args!("{self}"))?,
            },
            Fault::Input => CapabilityError::UnsupportedInput {
                architecture: text(format_args!("{architecture}"))?,
                reason: text(format_args!("{self}"))?,
            },
        })
    }

}
impl fmt::Display for MediaSemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (part {:?}, expected {:?}, actual {:?})",
            self.operation, self.part, self.expected, self.actual
        )
    }
}
impl std::error::Error for MediaSemanticError {}

#[derive(Clone, Copy)]
pub(crate) enum ShapeRef<'a> {
    Host(&'a [usize]),
    Dimensions(&'a [u64]),
}
impl ShapeRef<'_> {
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Host(s) => s.len(),
            Self::Dimensions(s) => s.len(),
        }
    }
    pub(crate) fn get(self, axis: usize) -> Option<u64> {
        match self {
            Self::Host(s) => s.get(axis).and_then(|n| u64::try_from(*n).ok()),
            Self::Dimensions(s) => s.get(axis).copied(),
        }
    }
    fn dimension(self, axis: usize) -> Result<u64, MediaSemanticError> {
        self.get(axis)
            .ok_or(MediaSemanticError::input("Qwen input shape axis"))
    }
}
#[derive(Clone, Copy)]
pub(crate) struct GridRef<'a> {
    pub shape: ShapeRef<'a>,
    pub values: &'a [i32],
}
impl<'a> GridRef<'a> {
    pub(crate) fn rows(self) -> Result<&'a [[i32; 3]], MediaSemanticError> {
        if self.shape.len() != 2 || self.shape.get(1) != Some(3) || self.shape.get(0) == Some(0) {
            return Err(MediaSemanticError::input(
                "Qwen patch grid must be [items, 3]",
            ));
        }
        let expected = self
            .shape
            .dimension(0)?
            .checked_mul(3)
            .ok_or(MediaSemanticError::overflow("Qwen patch-grid scalar count"))?;
        if u64::try_from(self.values.len()).ok() != Some(expected) {
            return Err(MediaSemanticError::input(
                "Qwen patch grid has an incomplete row",
            ));
        }
        let (rows, remainder) = self.values.as_chunks::<3>();
        if !remainder.is_empty() {
            return Err(MediaSemanticError::input(
                "Qwen patch grid has an incomplete row",
            ));
        }
        Ok(rows)
    }
}
#[derive(Clone, Copy)]
pub(crate) struct InspectedPartRef<'a> {
    pub modality: InputModality,
    pub kind: InputPayloadKind,
    pub shape: ShapeRef<'a>,
    pub grid: Option<GridRef<'a>>,
}
impl<'a> InspectedPartRef<'a> {
    pub(crate) fn original(
        part: eredu_runtime::input::host::PreparedHostPart<'a>,
    ) -> Result<Self, MediaSemanticError> {
        let payload = part.payload_view();
        let inspect_metadata =
            part.kind() == InputPayloadKind::Tensor && part.modality() != InputModality::Text;
        if inspect_metadata {
            // Match the legacy inspector's actual metadata reads, including
            // present but equation-unused positions. Original source encodings
            // are concrete; no native cast/read or owned descriptor is required.
            for (key, value) in part.metadata() {
                let valid = match key {
                    InputMetadataKey::PatchGrid | InputMetadataKey::PatchPositions => {
                        matches!(value.values, HostTensorValues::I32(_))
                    }
                    InputMetadataKey::AudioMask => {
                        matches!(value.values, HostTensorValues::Bool(_))
                    }
                    _ => true,
                };
                if !valid {
                    return Err(MediaSemanticError {
                        key: Some(key),
                        ..MediaSemanticError::input(
                            "Qwen metadata encoding differs from legacy read",
                        )
                    });
                }
            }
        }
        let grid = inspect_metadata
            .then(|| part.metadata_view(InputMetadataKey::PatchGrid))
            .flatten()
            .map(|(_, slot)| {
                let HostTensorValues::I32(values) = slot.values else {
                    return Err(MediaSemanticError::input(
                        "Qwen patch grid requires i32 values",
                    ));
                };
                Ok(GridRef {
                    shape: ShapeRef::Host(slot.shape),
                    values,
                })
            })
            .transpose()?;
        Ok(Self {
            modality: part.modality(),
            kind: part.kind(),
            shape: ShapeRef::Host(payload.shape),
            grid,
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenPartRole {
    Tokens,
    Projected,
    Encoded,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenPartRef<'a> {
    pub role: QwenPartRole,
    pub modality: InputModality,
    pub positions: u64,
    pub placeholder: u32,
    pub grid: &'a [[i32; 3]],
    pub workspace_scalars: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct QwenPolicy<'a> {
    pub hidden: i32,
    pub vision: Option<&'a VisionConfig>,
    pub image: Option<i32>,
    pub video: Option<i32>,
    pub projected_media: bool,
}
fn positive(value: i32, field: &'static str) -> Result<u64, MediaSemanticError> {
    u64::try_from(value).map_err(|_| MediaSemanticError::config(field, value))
}
fn nonzero_positive(value: i32, field: &'static str) -> Result<u64, MediaSemanticError> {
    let result = positive(value, field)?;
    if result == 0 {
        Err(MediaSemanticError::config(field, value))
    } else {
        Ok(result)
    }
}
fn checked_add(a: u64, b: u64, operation: &'static str) -> Result<u64, MediaSemanticError> {
    a.checked_add(b)
        .ok_or(MediaSemanticError::overflow(operation))
}
fn checked_mul(a: u64, b: u64, operation: &'static str) -> Result<u64, MediaSemanticError> {
    a.checked_mul(b)
        .ok_or(MediaSemanticError::overflow(operation))
}

pub(crate) fn qwen_part<'a>(
    policy: QwenPolicy<'_>,
    input: InspectedPartRef<'a>,
) -> Result<QwenPartRef<'a>, MediaSemanticError> {
    let role = match (input.modality, input.kind) {
        (InputModality::Text, InputPayloadKind::TokenIds) => QwenPartRole::Tokens,
        (InputModality::Text, InputPayloadKind::Embeddings) => QwenPartRole::Projected,
        (InputModality::Image | InputModality::Video, InputPayloadKind::Embeddings)
            if policy.projected_media =>
        {
            QwenPartRole::Projected
        }
        (InputModality::Image | InputModality::Video, InputPayloadKind::Tensor) => {
            QwenPartRole::Encoded
        }
        _ => {
            return Err(MediaSemanticError::input(
                "Qwen unsupported modality/payload role",
            ));
        }
    };
    if role != QwenPartRole::Encoded {
        let rank = if role == QwenPartRole::Tokens { 2 } else { 3 };
        if input.shape.len() != rank
            || input.shape.get(0) != Some(1)
            || input.shape.get(1).is_none_or(|n| n == 0)
        {
            return Err(MediaSemanticError::input(
                "Qwen prepared sequence must be positive and batch-one",
            ));
        }
        if role == QwenPartRole::Projected
            && input.shape.get(2) != Some(positive(policy.hidden, "Qwen decoder hidden size")?)
        {
            return Err(MediaSemanticError::input(
                "Qwen projected hidden width differs from decoder",
            ));
        }
        return Ok(QwenPartRef {
            role,
            modality: input.modality,
            positions: input.shape.dimension(1)?,
            placeholder: 0,
            grid: &[],
            workspace_scalars: 0,
        });
    }
    let vision = policy.vision.ok_or(MediaSemanticError::input(
        "loaded Qwen model has no vision configuration",
    ))?;
    let shape = qwen_vision(vision, input)?;
    let token = match input.modality {
        InputModality::Image => policy.image,
        InputModality::Video => policy.video,
        _ => None,
    }
    .ok_or(MediaSemanticError::input(
        "prepared media placeholder token is absent",
    ))?;
    let placeholder = u32::try_from(token)
        .map_err(|_| MediaSemanticError::config("Qwen media placeholder token", token))?;
    Ok(QwenPartRef {
        role,
        modality: input.modality,
        positions: shape.decoder_positions,
        placeholder,
        grid: input
            .grid
            .ok_or(MediaSemanticError::input(
                "prepared Qwen media has no grid_thw metadata",
            ))?
            .rows()?,
        workspace_scalars: shape.execution_workspace_scalars,
    })
}

fn attention_squares(
    grid: &[[i32; 3]],
    merge: u64,
    window_size: i32,
    patch_size: i32,
) -> Result<(u64, u64), MediaSemanticError> {
    let patch = nonzero_positive(patch_size, "Qwen vision patch size")?;
    let window = nonzero_positive(window_size, "Qwen vision window size")?;
    let merger_window = window / merge / patch;
    if merger_window == 0 {
        return Err(MediaSemanticError::config("window_size", window_size));
    }
    let area = checked_mul(merge, merge, "Qwen attention merge area")?;
    let area_squared = checked_mul(area, area, "Qwen attention merge-area square")?;
    let (mut full, mut windows) = (0, 0);
    for &[t, h, w] in grid {
        let t = nonzero_positive(t, "Qwen grid time")?;
        let h = nonzero_positive(h, "Qwen grid height")?;
        let w = nonzero_positive(w, "Qwen grid width")?;
        if h % merge != 0 || w % merge != 0 {
            return Err(MediaSemanticError::config(
                "patch_grid",
                i32::try_from(h).expect("source i32 height"),
            ));
        }
        let length = checked_mul(h, w, "Qwen full-attention chunk length")?;
        full = checked_add(
            full,
            checked_mul(
                t,
                checked_mul(length, length, "Qwen full-attention chunk square")?,
                "Qwen full-attention temporal chunks",
            )?,
            "Qwen full-attention chunk-square total",
        )?;
        let square = checked_mul(merger_window, merger_window, "Qwen merger-window square")?;
        let h = h / merge;
        let w = w / merge;
        let hs = checked_add(
            checked_mul(h / merger_window, square, "Qwen full height-window squares")?,
            checked_mul(
                h % merger_window,
                h % merger_window,
                "Qwen remainder height-window square",
            )?,
            "Qwen height-window square sum",
        )?;
        let ws = checked_add(
            checked_mul(w / merger_window, square, "Qwen full width-window squares")?,
            checked_mul(
                w % merger_window,
                w % merger_window,
                "Qwen remainder width-window square",
            )?,
            "Qwen width-window square sum",
        )?;
        let item = checked_mul(
            checked_mul(hs, ws, "Qwen merged window-area squares")?,
            area_squared,
            "Qwen patch window-area squares",
        )?;
        windows = checked_add(
            windows,
            checked_mul(t, item, "Qwen temporal window chunks")?,
            "Qwen window-attention chunk-square total",
        )?;
    }
    Ok((full, windows))
}
pub(crate) fn qwen_vision(
    config: &VisionConfig,
    input: InspectedPartRef<'_>,
) -> Result<MediaShapePlan, MediaSemanticError> {
    if !matches!(input.modality, InputModality::Image | InputModality::Video)
        || input.shape.len() != 2
    {
        return Err(MediaSemanticError::input(
            "Qwen vision requires image/video [patches, patch_dims]",
        ));
    }
    let patches = input.shape.dimension(0)?;
    let merge = nonzero_positive(config.spatial_merge_size, "spatial_merge_size")?;
    let patch = nonzero_positive(config.patch_size, "Qwen vision patch size")?;
    let expected = checked_mul(
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
    let actual = input.shape.dimension(1)?;
    if actual != expected {
        return Err(MediaSemanticError::input("Qwen prepared patch dimensions")
            .scalars(expected, i128::from(actual)));
    }
    let merge_area = checked_mul(merge, merge, "Qwen spatial merge area")?;
    if patches % merge_area != 0 {
        return Err(MediaSemanticError::input(
            "Qwen patch count not divisible by merge area",
        ));
    }
    let positions = patches / merge_area;
    let grid = input
        .grid
        .ok_or(MediaSemanticError::input(
            "prepared Qwen media has no grid_thw metadata",
        ))?
        .rows()?;
    let described = grid.iter().try_fold(0, |total, &[time, height, width]| {
        let item = checked_mul(
            checked_mul(
                nonzero_positive(time, "Qwen grid time")?,
                nonzero_positive(height, "Qwen grid height")?,
                "Qwen grid time-height",
            )?,
            nonzero_positive(width, "Qwen grid width")?,
            "Qwen grid item patches",
        )?;
        checked_add(total, item, "Qwen described patch total")
    })?;
    if described != patches {
        return Err(
            MediaSemanticError::input("Qwen grid differs from payload patch count")
                .scalars(patches, i128::from(described)),
        );
    }
    let (full_chunk_squares, window_chunk_squares) =
        attention_squares(grid, merge, config.window_size, config.patch_size)?;
    let depth = u64::try_from(config.layer_count())
        .map_err(|_| MediaSemanticError::overflow("Qwen vision depth"))?;
    let full_blocks = u64::try_from(
        config
            .layer_schedule
            .iter()
            .filter(|policy| matches!(policy.attention, VisionAttentionPolicy::Full))
            .count(),
    )
    .map_err(|_| MediaSemanticError::overflow("Qwen full-attention block count"))?;
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
        u64::try_from(config.deepstack_layer_count())
            .map_err(|_| MediaSemanticError::overflow("Qwen deepstack merger count"))?,
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
