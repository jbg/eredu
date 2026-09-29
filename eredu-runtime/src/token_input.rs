//! Exact original token provenance and padding for architecture-owned requests.
use eredu_nn::{Tensor, TensorElementType};
use std::borrow::Cow;

/// Explicit original IDs before any embedding/media replacement.
#[derive(Debug)]
pub enum OriginalTokenIds<'a, T> {
    /// Retained row-major host IDs; resolving these performs no tensor transfer.
    Host(&'a [u64]),
    /// An explicit I32/U32 token tensor, never an embedding or floating proxy.
    Tensor(&'a T),
}
/// Per-token zero/one visibility, independent of additive attention masks.
#[derive(Debug)]
pub enum TokenVisibility<'a, T> {
    /// Retained row-major visibility.
    Host(&'a [bool]),
    /// A `[batch,tokens]` zero/one tensor. Causal masking belongs to the architecture.
    Tensor(&'a T),
}
/// Invalid exact-ID or visibility input, rejected before architecture state changes.
#[derive(Debug, thiserror::Error)]
pub enum TokenInputError {
    /// Shape, host length, or request admission differs from the declared geometry.
    #[error("token input geometry exceeds or differs from its admitted shape")]
    Geometry,
    /// Token tensors cannot use a floating or wider unchecked representation.
    #[error("original token IDs require an explicit I32 or U32 tensor")]
    ScalarType,
    /// IDs must fit the declared vocabulary before any embedding/table operation.
    #[error("original token ID is outside the declared vocabulary")]
    Vocabulary,
    /// Visibility must consist of exact zero/one values.
    #[error("token visibility must contain only zero and one")]
    Visibility,
    /// Tensor transfer or extraction failed.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
}
fn count(batch: i32, tokens: i32, limit: usize) -> Result<usize, TokenInputError> {
    if batch <= 0 || tokens <= 0 {
        return Err(TokenInputError::Geometry);
    }
    (batch as usize)
        .checked_mul(tokens as usize)
        .filter(|n| *n <= limit)
        .ok_or(TokenInputError::Geometry)
}
impl<'a, T: Tensor> OriginalTokenIds<'a, T> {
    /// Resolves exact integers under a pre-transfer element bound. Tensor IDs are
    /// limited to a positive I32 vocabulary; U32 values above that range reject.
    pub fn resolve(
        &self,
        batch: i32,
        tokens: i32,
        vocabulary: i32,
        max_tokens: usize,
        context: &T::Context,
    ) -> Result<Cow<'a, [u64]>, TokenInputError> {
        let count = count(batch, tokens, max_tokens)?;
        if vocabulary <= 0 {
            return Err(TokenInputError::Vocabulary);
        }
        let ids = match self {
            Self::Host(ids) if ids.len() == count => Cow::Borrowed(*ids),
            Self::Host(_) => return Err(TokenInputError::Geometry),
            Self::Tensor(value) => {
                if value.shape() != [batch, tokens] {
                    return Err(TokenInputError::Geometry);
                }
                if !matches!(
                    value.element_type(),
                    Some(TensorElementType::I32 | TensorElementType::U32)
                ) {
                    return Err(TokenInputError::ScalarType);
                }
                let ids = value.to_i32_vec(context)?;
                if ids.len() != count {
                    return Err(TokenInputError::Geometry);
                }
                Cow::Owned(
                    ids.into_iter()
                        .map(|id| u64::try_from(id).map_err(|_| TokenInputError::Vocabulary))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };
        if ids.iter().any(|id| *id >= vocabulary as u64) {
            return Err(TokenInputError::Vocabulary);
        }
        Ok(ids)
    }
}
impl<'a, T: Tensor> TokenVisibility<'a, T> {
    /// Resolves bounded row-major visibility without accepting additive masks or
    /// rounding fractional/NaN values to a boolean.
    pub fn resolve(
        &self,
        batch: i32,
        tokens: i32,
        max_tokens: usize,
        context: &T::Context,
    ) -> Result<Cow<'a, [bool]>, TokenInputError> {
        let count = count(batch, tokens, max_tokens)?;
        match self {
            Self::Host(values) if values.len() == count => Ok(Cow::Borrowed(*values)),
            Self::Host(_) => Err(TokenInputError::Geometry),
            Self::Tensor(values) => {
                if values.shape() != [batch, tokens] {
                    return Err(TokenInputError::Geometry);
                }
                if !matches!(
                    values.element_type(),
                    Some(
                        TensorElementType::Bool
                            | TensorElementType::I8
                            | TensorElementType::I16
                            | TensorElementType::I32
                            | TensorElementType::U8
                            | TensorElementType::U16
                            | TensorElementType::U32
                            | TensorElementType::F16
                            | TensorElementType::Bf16
                            | TensorElementType::F32
                    )
                ) {
                    return Err(TokenInputError::Visibility);
                }
                let values = values.to_f32_vec(context)?;
                if values.len() != count {
                    return Err(TokenInputError::Geometry);
                }
                Ok(Cow::Owned(
                    values
                        .into_iter()
                        .map(|value| match value {
                            0. => Ok(false),
                            1. => Ok(true),
                            _ => Err(TokenInputError::Visibility),
                        })
                        .collect::<Result<_, _>>()?,
                ))
            }
        }
    }
}
