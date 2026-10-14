//! Exact prepared image admission and semantic decoder positions.
use super::{FamilyConfig, processing::image_token_ids};
use crate::{
    decoder::Config,
    media_plan::{
        self, AdmittedCompositeInput, CompositePartPlan, MediaShapePlan, PreparedInputPartPlan,
    },
};
use eredu_core::checkpoint::TensorDtype;
use eredu_core::{CapabilityError, InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedModelInput};

/// Ordered decoder parts, including image delimiters and row separators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputPartPlan {
    /// Vocabulary IDs supplied by the tokenizer.
    TextTokens {
        /// Complete sequence extent.
        positions: u64,
    },
    /// Caller-supplied text embeddings with retained original token identity.
    TextEmbeddings {
        /// Complete sequence extent.
        positions: u64,
    },
    /// One image, whose projected patches replace only pad markers.
    Image {
        /// Unmerged patch grid.
        grid: (i32, i32),
        /// Complete published marker sequence.
        token_ids: Vec<u32>,
        /// True when patches require the tower; false for projected embeddings.
        native: bool,
        /// Decoder positions and invocation workspace.
        shape: MediaShapePlan,
    },
}
impl InputPartPlan {
    pub(crate) fn token_ids<T: eredu_nn::Tensor>(
        &self,
        part: &PreparedInputPart<T>,
        context: &T::Context,
    ) -> Result<T, eredu_nn::Error> {
        match self {
            Self::TextTokens { .. } => Ok(part.payload().value().clone()),
            Self::TextEmbeddings { .. } => part
                .metadata_value(InputMetadataKey::OriginalTokenIds)
                .cloned()
                .ok_or_else(|| {
                    eredu_nn::Error::backend("admitted text embeddings lost token identity")
                }),
            Self::Image { token_ids, .. } => {
                let ids = token_ids
                    .iter()
                    .map(|id| i32::try_from(*id).map_err(eredu_nn::Error::backend))
                    .collect::<Result<Vec<_>, _>>()?;
                T::from_i32_slice(
                    &ids,
                    &[
                        1,
                        i32::try_from(ids.len()).map_err(eredu_nn::Error::backend)?,
                    ],
                    context,
                )
            }
        }
    }
}
impl CompositePartPlan for InputPartPlan {
    fn decoder_positions(&self) -> u64 {
        match self {
            Self::TextTokens { positions } | Self::TextEmbeddings { positions } => *positions,
            Self::Image { shape, .. } => shape.decoder_positions,
        }
    }
}
impl From<InputPartPlan> for PreparedInputPartPlan {
    fn from(value: InputPartPlan) -> Self {
        match value {
            InputPartPlan::TextTokens { positions }
            | InputPartPlan::TextEmbeddings { positions } => Self::Text { positions },
            InputPartPlan::Image {
                shape,
                native: true,
                ..
            } => Self::Media { shape },
            InputPartPlan::Image {
                shape,
                native: false,
                ..
            } => Self::Projected {
                modality: InputModality::Image,
                positions: shape.decoder_positions,
            },
        }
    }
}
fn reject(reason: impl Into<String>) -> CapabilityError {
    CapabilityError::UnsupportedInput {
        architecture: "Nemotron-Labs-Diffusion".into(),
        reason: reason.into(),
    }
}
fn overflow() -> CapabilityError {
    CapabilityError::ArithmeticOverflow {
        operation: "Pixtral prepared input geometry",
    }
}

/// Validates native payload identity and bounded integer metadata before execution.
pub fn admit_input<T>(
    config: &FamilyConfig,
    input: &PreparedModelInput<T>,
    inspector: &impl PreparedInputInspector<T>,
) -> Result<AdmittedCompositeInput<InputPartPlan>, CapabilityError> {
    config
        .validate_config()
        .map_err(|e| reject(e.to_string()))?;
    media_plan::admit_composite_input(input, inspector, |part| admit_part(config, part, inspector))
}
fn admit_part<T>(
    config: &FamilyConfig,
    part: &PreparedInputPart<T>,
    inspector: &impl PreparedInputInspector<T>,
) -> Result<InputPartPlan, CapabilityError> {
    let descriptor = part
        .descriptor(&|tensor| inspector.identity(tensor))
        .map_err(|e| reject(e.to_string()))?;
    let payload = descriptor.payload();
    let shape = payload.shape();
    let floating = matches!(
        payload.dtype(),
        TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
    );
    match (part.modality(), descriptor.payload_kind()) {
        (InputModality::Text, InputPayloadKind::TokenIds)
            if shape.len() == 2
                && shape[0] == 1
                && matches!(payload.dtype(), TensorDtype::I32 | TensorDtype::U32)
                && part.metadata().is_empty() =>
        {
            Ok(InputPartPlan::TextTokens {
                positions: u64::try_from(shape[1]).map_err(|_| overflow())?,
            })
        }
        (InputModality::Text, InputPayloadKind::Embeddings)
            if shape.len() == 3
                && shape[0] == 1
                && shape[2] == config.hidden_size() as usize
                && floating =>
        {
            if part.metadata().len() != 1 {
                return Err(reject("text embeddings require only original token IDs"));
            }
            let original = descriptor
                .metadata_value(InputMetadataKey::OriginalTokenIds)
                .ok_or_else(|| reject("text embeddings have no original token IDs"))?;
            if original.shape() != &shape[..2]
                || !matches!(original.dtype(), TensorDtype::I32 | TensorDtype::U32)
            {
                return Err(reject("text embeddings and original token IDs disagree"));
            }
            Ok(InputPartPlan::TextEmbeddings {
                positions: u64::try_from(shape[1]).map_err(|_| overflow())?,
            })
        }
        (
            InputModality::Image,
            kind @ (InputPayloadKind::Tensor | InputPayloadKind::Embeddings),
        ) if floating => {
            let vision = config
                .vision
                .as_ref()
                .ok_or_else(|| reject("text artifact has no image tower"))?;
            if part.metadata().len() != 1 || part.extents().len() != 1 {
                return Err(reject(
                    "image input requires exactly one patch grid and host extent",
                ));
            }
            let metadata = descriptor
                .metadata_value(InputMetadataKey::PatchGrid)
                .ok_or_else(|| reject("image input has no patch grid"))?;
            if metadata.shape() != [1, 3] || *metadata.dtype() != TensorDtype::I32 {
                return Err(reject(
                    "image patch grid must be one int32 (time,height,width) row",
                ));
            }
            let grid =
                inspector.i32_values(part.metadata_value(InputMetadataKey::PatchGrid).unwrap())?;
            let [1, height, width] = grid.as_slice() else {
                return Err(reject("image patch grid requires time one"));
            };
            let merge = i32::try_from(config.spatial_merge_size).map_err(|_| overflow())?;
            if *height <= 0
                || *width <= 0
                || height % merge != 0
                || width % merge != 0
                || *height > vision.image_size / vision.patch_size
                || *width > vision.image_size / vision.patch_size
            {
                return Err(reject("invalid aligned image grid"));
            }
            if part.extents()
                != [InputExtent::PatchGrid {
                    time: 1,
                    height: *height as usize,
                    width: *width as usize,
                }]
            {
                return Err(reject("image metadata and host extent disagree"));
            }
            let patches = (*height as u64)
                .checked_mul(*width as u64)
                .ok_or_else(overflow)?;
            let projected = patches / u64::from(config.spatial_merge_size).pow(2);
            let native = kind == InputPayloadKind::Tensor;
            let expected = if native {
                vec![
                    usize::try_from(patches).map_err(|_| overflow())?,
                    (vision.num_channels * vision.patch_size * vision.patch_size) as usize,
                ]
            } else {
                vec![
                    1,
                    usize::try_from(projected).map_err(|_| overflow())?,
                    config.hidden_size() as usize,
                ]
            };
            if shape != expected {
                return Err(reject(format!(
                    "image payload must have shape {expected:?}, got {shape:?}"
                )));
            }
            let token_ids =
                image_token_ids((*height as u32, *width as u32), config.spatial_merge_size)
                    .map_err(|e| reject(e.to_string()))?;
            let workspace = if native {
                let attention = patches
                    .checked_mul(patches)
                    .and_then(|v| v.checked_mul(vision.num_attention_heads as u64))
                    .and_then(|v| v.checked_mul(2))
                    .ok_or_else(overflow)?;
                let linear = patches
                    .checked_mul(
                        6 * vision.hidden_size as u64 + 2 * vision.intermediate_size as u64,
                    )
                    .ok_or_else(overflow)?;
                attention
                    .checked_add(linear)
                    .and_then(|v| {
                        projected
                            .checked_mul(config.hidden_size() as u64)
                            .and_then(|projected| v.checked_add(projected))
                    })
                    .ok_or_else(overflow)?
            } else {
                0
            };
            Ok(InputPartPlan::Image {
                grid: (*height, *width),
                shape: MediaShapePlan {
                    capture_media: native
                        .then_some(eredu_core::capture::CaptureMediaShape { patches, projected }),
                    decoder_positions: token_ids.len() as u64,
                    execution_workspace_scalars: workspace,
                },
                token_ids,
                native,
            })
        }
        _ => Err(reject("unsupported image/text payload or precision")),
    }
}
