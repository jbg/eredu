//! Shared prepared token/media assembly; family policy supplies projected token identity.
use super::vl::InputPart;
use crate::{composite_execution::PreparedCompositeInput, media_plan::QwenInputPartPlan};
use eredu_nn::{Error, Tensor};

#[derive(Clone)]
enum PreparedInputKind {
    Text(usize),
    Projected(usize, usize),
    Image(usize, usize),
    Video(usize, usize),
}

/// Semantic roles of handles retained by shared Qwen request assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedInputTensorRole {
    /// Original or architecture-declared token IDs for an ordered input part.
    Tokens(usize),
    /// Concatenated flattened patches for all encoder input parts.
    Pixels,
    /// Already projected features for an ordered input part.
    Projected(usize),
}

/// Architecture-owned tensor assembly for one admitted Qwen request.
#[derive(Clone)]
pub struct PreparedInput<T> {
    tokens: Vec<T>,
    grids: Vec<Vec<(i32, i32, i32)>>,
    pixels: Option<T>,
    kinds: Vec<PreparedInputKind>,
    projected: Vec<Option<T>>,
}

impl<T: Tensor> PreparedInput<T> {
    /// All native handles retained through deferred ingress and encoder completion.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.storage_values().map(|(_, value)| value)
    }
    /// Named borrowed roots for one combined storage survey. This excludes the
    /// original processor input, which its enclosing request may also retain.
    pub fn storage_values(&self) -> impl Iterator<Item = (PreparedInputTensorRole, &T)> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(part, value)| (PreparedInputTensorRole::Tokens(part), value))
            .chain(
                self.pixels
                    .iter()
                    .map(|value| (PreparedInputTensorRole::Pixels, value)),
            )
            .chain(
                self.projected
                    .iter()
                    .enumerate()
                    .filter_map(|(part, value)| {
                        value
                            .as_ref()
                            .map(|value| (PreparedInputTensorRole::Projected(part), value))
                    }),
            )
    }
    /// Borrows the assembled request through the shared ordered text/media vocabulary.
    pub fn with_parts<R>(&self, apply: impl FnOnce(&[InputPart<'_, T>], Option<&T>) -> R) -> R {
        let parts = self
            .kinds
            .iter()
            .map(|kind| match *kind {
                PreparedInputKind::Text(token) => InputPart::Text(&self.tokens[token]),
                PreparedInputKind::Projected(token, original) => InputPart::Projected {
                    tokens: &self.tokens[token],
                    embeddings: self.projected[original]
                        .as_ref()
                        .expect("projected input retains its embeddings"),
                },
                PreparedInputKind::Image(token, grid) => InputPart::Image {
                    tokens: &self.tokens[token],
                    grid: &self.grids[grid],
                },
                PreparedInputKind::Video(token, grid) => InputPart::Video {
                    tokens: &self.tokens[token],
                    grid: &self.grids[grid],
                },
            })
            .collect::<Vec<_>>();
        apply(&parts, self.pixels.as_ref())
    }

    /// Concatenates semantic token identity in decoder order.
    pub fn token_ids(&self, context: &T::Context) -> Result<T, Error> {
        T::concatenate(&self.tokens, 1, context)
    }
}

/// Policy for embeddings without supplied original token metadata.
#[derive(Debug, Clone, Copy)]
pub enum ProjectedTokenPolicy {
    /// Architectures whose equations require original vocabulary identity.
    Required,
    /// Architectures that explicitly declare a semantic placeholder.
    Placeholder(u32),
}

/// Returns exact token handles, or architecture-declared media placeholders.
pub fn prepared_semantic_tokens<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenInputPartPlan>,
    policy: ProjectedTokenPolicy,
    context: &T::Context,
) -> Result<Vec<T>, Error> {
    input
        .prepared()
        .parts()
        .iter()
        .zip(input.admitted().parts())
        .enumerate()
        .map(|(index, (part, plan))| {
            let placeholder = |id: u32, positions: u64| {
                T::full_i32(
                    i32::try_from(id).map_err(Error::backend)?,
                    &[1, i32::try_from(positions).map_err(Error::backend)?],
                    context,
                )
            };
            match plan {
                QwenInputPartPlan::TextTokens { .. } => match part.payload() {
                    eredu_runtime::PreparedInputPayload::TokenIds(ids) => Ok(ids.clone()),
                    _ => Err(Error::backend("admitted text part lost token IDs")),
                },
                QwenInputPartPlan::Projected { positions, .. } => {
                    if let Some(ids) =
                        part.metadata_value(eredu_core::InputMetadataKey::OriginalTokenIds)
                    {
                        if ids.shape() != [1, i32::try_from(*positions).map_err(Error::backend)?] {
                            return Err(Error::backend_source(
                                eredu_runtime::TokenInputError::Geometry,
                            ));
                        }
                        if !matches!(
                            ids.element_type(),
                            Some(
                                eredu_nn::TensorElementType::I32 | eredu_nn::TensorElementType::U32
                            )
                        ) {
                            return Err(Error::backend_source(
                                eredu_runtime::TokenInputError::ScalarType,
                            ));
                        }
                        return Ok(ids.clone());
                    }
                    match policy {
                        ProjectedTokenPolicy::Placeholder(id) => placeholder(id, *positions),
                        ProjectedTokenPolicy::Required => Err(Error::backend_source(
                            eredu_core::PreparedInputError::MissingMetadata {
                                part: index,
                                modality: part.modality(),
                                key: eredu_core::InputMetadataKey::OriginalTokenIds,
                            },
                        )),
                    }
                }
                QwenInputPartPlan::Media { ingress, .. } => {
                    placeholder(ingress.placeholder_token_id, ingress.placeholder_count)
                }
            }
        })
        .collect()
}

/// Materializes Qwen placeholder IDs, patch grids, and ordered target segments
/// from an architecture admission.
pub fn prepare_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, QwenInputPartPlan>,
    policy: ProjectedTokenPolicy,
    context: &T::Context,
) -> Result<PreparedInput<T>, Error> {
    let prepared = input.prepared();
    let admitted = input.admitted();
    if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
        return Err(Error::backend(
            "Qwen prepared input no longer matches its admission",
        ));
    }
    let tokens = prepared_semantic_tokens(input, policy, context)?;
    let mut grids = Vec::new();
    let mut pixels = Vec::new();
    let mut kinds = Vec::with_capacity(prepared.len());
    let mut projected = Vec::with_capacity(prepared.len());
    for (original, (part, plan)) in prepared.parts().iter().zip(admitted.parts()).enumerate() {
        match plan {
            QwenInputPartPlan::TextTokens { .. } => {
                kinds.push(PreparedInputKind::Text(original));
                projected.push(None);
            }
            QwenInputPartPlan::Projected { .. } => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(Error::backend("Qwen projected part lost its embeddings"));
                };

                kinds.push(PreparedInputKind::Projected(original, original));
                projected.push(Some(value.clone()));
            }
            QwenInputPartPlan::Media { ingress, .. } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Qwen admitted media lost its tensor payload",
                    ));
                };

                grids.push(ingress.patch_grid.clone());
                pixels.push(value.clone());
                let token_index = original;
                let grid_index = grids.len() - 1;
                kinds.push(match part.modality() {
                    eredu_core::InputModality::Image => {
                        PreparedInputKind::Image(token_index, grid_index)
                    }
                    eredu_core::InputModality::Video => {
                        PreparedInputKind::Video(token_index, grid_index)
                    }
                    _ => {
                        return Err(Error::backend(
                            "Qwen admission contains an unsupported modality",
                        ));
                    }
                });
                projected.push(None);
            }
        }
    }
    let pixels = match pixels.len() {
        0 => None,
        1 => pixels.pop(),
        _ => Some(T::concatenate(&pixels, 0, context)?),
    };
    Ok(PreparedInput {
        tokens,
        grids,
        pixels,
        kinds,
        projected,
    })
}
