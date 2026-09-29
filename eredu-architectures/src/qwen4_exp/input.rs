//! Original request provenance retained independently from evolving embeddings.
use super::ngram::NGramError;
use eredu_nn::residual_streams::ResidualStreamGeometry;
use eredu_nn::{Error, RotaryPosition, Tensor, TensorElementType};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureBoundaryError, ArchitectureBoundaryValue,
    BoundaryTensorDimension as D, BoundaryTensorDtype as K, BoundaryTensorSpec, OriginalTokenIds,
    TokenInputError, TokenVisibility,
};
use std::sync::Arc;

/// Exact request metadata transported with every residual-stream boundary.
#[derive(Debug, Clone)]
pub struct RequestBoundary<T> {
    /// Original pre-media IDs, exact I32 `[batch,tokens]`.
    pub ids: T,
    /// Exact zero/one key visibility, I32 `[batch,tokens]`.
    pub visible: T,
    /// I32 `[prefix,rotary_mode,rotary_offset,position_delta]`; mode one selects supplied products.
    pub control: T,
    /// FP32 current-token products `[batch,tokens,rotary_width]`, zero in offset mode.
    pub cosine: T,
    /// FP32 current-token products in the same geometry.
    pub sine: T,
}
/// Cold transport declaration; original IDs never use floating activation precision.
#[derive(Debug, Clone, Copy)]
pub struct RequestBoundarySchema {
    geometry: ResidualStreamGeometry,
    rotary_width: i32,
}
impl RequestBoundarySchema {
    /// Retains checked residual and rotary geometry without a native allocation.
    pub fn new(geometry: ResidualStreamGeometry, rotary_width: i32) -> Result<Self, Error> {
        if rotary_width <= 0 || rotary_width % 2 != 0 {
            return Err(Error::backend("invalid request rotary width"));
        }
        Ok(Self {
            geometry,
            rotary_width,
        })
    }
}
impl ArchitectureBoundary for RequestBoundarySchema {
    type Boundary<T> = RequestBoundary<T>;
    const IDENTITY: &'static str = "qwen4_exp.request.v2";
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::new(
            "hidden",
            [
                D::Batch,
                D::Sequence,
                D::Fixed(self.geometry.streams()),
                D::Fixed(self.geometry.hidden_size()),
            ],
            K::Activation,
        )
    }
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        vec![
            BoundaryTensorSpec::new("original_ids", [D::Batch, D::Sequence], K::Int32),
            BoundaryTensorSpec::new("visibility", [D::Batch, D::Sequence], K::Int32),
            BoundaryTensorSpec::new("position", [D::Fixed(4)], K::Int32),
            BoundaryTensorSpec::new(
                "rotary_cosine",
                [D::Batch, D::Sequence, D::Fixed(self.rotary_width)],
                K::Float32,
            ),
            BoundaryTensorSpec::new(
                "rotary_sine",
                [D::Batch, D::Sequence, D::Fixed(self.rotary_width)],
                K::Float32,
            ),
        ]
    }
    fn encode<T>(
        &self,
        b: RequestBoundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        [
            ("original_ids", b.ids),
            ("visibility", b.visible),
            ("position", b.control),
            ("rotary_cosine", b.cosine),
            ("rotary_sine", b.sine),
        ]
        .into_iter()
        .map(|(role, t)| ArchitectureBoundaryValue::new(role, t))
        .collect()
    }
    fn decode<T>(&self, tensors: Vec<T>) -> Result<RequestBoundary<T>, ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(RequestBoundary {
            ids: tensors.next().unwrap(),
            visible: tensors.next().unwrap(),
            control: tensors.next().unwrap(),
            cosine: tensors.next().unwrap(),
            sine: tensors.next().unwrap(),
        })
    }
}
/// Typed request validation failures, preserved as sources by the layered adapter.
#[derive(Debug, thiserror::Error)]
pub enum RequestError {
    /// Exact integer or visibility carrier failed validation before execution.
    #[error(transparent)]
    Input(#[from] TokenInputError),
    /// Required original IDs or their vocabulary/geometry are invalid.
    #[error(transparent)]
    Tokens(#[from] NGramError),
    /// The declared wire geometry, scalar kind or position is malformed.
    #[error("invalid qwen4_exp request boundary or causal position")]
    Boundary,
    /// The requested cumulative prefix exceeds the retained execution policy.
    #[error("qwen4_exp history needs {required} tokens, limit is {limit}")]
    History {
        /// Prefix plus current invocation.
        required: u64,
        /// Admitted cumulative per-lane token capacity.
        limit: u64,
    },
    /// Tensor construction, extraction or transfer failed.
    #[error(transparent)]
    Tensor(#[from] Error),
}
/// One bounded pass's host IDs and visibility plus exact transport tensors.
#[derive(Debug, Clone)]
pub struct RequestContext<T> {
    boundary: RequestBoundary<T>,
    ids: Arc<[u64]>,
    visible: Arc<[bool]>,
    offset: i32,
    rotary_offset: Option<i32>,
    position_delta: i32,
}
impl<T: Tensor> RequestContext<T> {
    /// Requires explicit original IDs even when embeddings have already been supplied.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ids: Option<OriginalTokenIds<'_, T>>,
        visible: Option<TokenVisibility<'_, T>>,
        batch: i32,
        tokens: i32,
        offset: i32,
        vocabulary: i32,
        max_tokens: usize,
        rotary_width: i32,
        rotary: Option<RotaryPosition<'_, T>>,
        position_delta: i32,
        context: &T::Context,
    ) -> Result<Self, RequestError> {
        let ids = ids.ok_or(NGramError::MissingTokenIds)?;
        let ids = ids.resolve(batch, tokens, vocabulary, max_tokens, context)?;
        let visible = visible
            .as_ref()
            .map(|v| v.resolve(batch, tokens, max_tokens, context))
            .transpose()?;
        let visible = visible.as_deref();
        let count = usize::try_from(batch)
            .ok()
            .and_then(|b| usize::try_from(tokens).ok().and_then(|t| b.checked_mul(t)))
            .filter(|n| *n > 0 && *n <= max_tokens)
            .ok_or(RequestError::Boundary)?;
        if ids.len() != count
            || visible.is_some_and(|v| v.len() != count)
            || offset < 0
            || offset.checked_add(tokens).is_none()
            || rotary_width <= 0
            || rotary_width % 2 != 0
            || vocabulary <= 0
        {
            return Err(RequestError::Boundary);
        }
        let ids = ids
            .iter()
            .map(|id| {
                if *id < vocabulary as u64 {
                    Ok(*id as i32)
                } else {
                    Err(NGramError::Token(*id))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let visibility: Vec<_> = (0..count)
            .map(|i| i32::from(visible.is_none_or(|v| v[i])))
            .collect();
        let shape = [batch, tokens, rotary_width];
        let rotary = match rotary {
            Some(rotary) => rotary,
            None => RotaryPosition::Offset(
                offset
                    .checked_add(position_delta)
                    .filter(|v| *v >= 0)
                    .ok_or(RequestError::Boundary)?,
            ),
        };
        let (mode, rotary_offset, cosine, sine) = match rotary {
            RotaryPosition::Offset(position)
                if position >= 0
                    && position.checked_add(tokens).is_some()
                    && offset.checked_add(position_delta) == Some(position) =>
            {
                (
                    0,
                    position,
                    T::full_f32(0., &shape, context)?,
                    T::full_f32(0., &shape, context)?,
                )
            }
            RotaryPosition::Embeddings { cosine, sine } if cosine.shape() == sine.shape() => {
                let normalize = |t: &T| -> Result<T, RequestError> {
                    let shape_in = t.shape();
                    let valid = match shape_in {
                        [b, n, d] | [b, 1, n, d] => {
                            (*b == 1 || *b == batch) && *n == tokens && *d == rotary_width
                        }
                        _ => false,
                    };
                    if !valid {
                        return Err(RequestError::Boundary);
                    }
                    Ok(t.reshape(&[shape_in[0], tokens, rotary_width], context)?
                        .cast_float(TensorElementType::F32, context)?
                        .broadcast_to(&shape, context)?
                        .compact(context)?)
                };
                (1, 0, normalize(cosine)?, normalize(sine)?)
            }
            _ => return Err(RequestError::Boundary),
        };
        let boundary = RequestBoundary {
            ids: T::from_i32_slice(&ids, &[batch, tokens], context)?,
            visible: T::from_i32_slice(&visibility, &[batch, tokens], context)?,
            control: T::from_i32_slice(
                &[offset, mode, rotary_offset, position_delta],
                &[4],
                context,
            )?,
            cosine,
            sine,
        };
        Ok(Self {
            boundary,
            ids: ids.into_iter().map(|id| id as u64).collect(),
            visible: visibility.into_iter().map(|v| v == 1).collect(),
            offset,
            position_delta,
            rotary_offset: (mode == 0).then_some(rotary_offset),
        })
    }
    /// Revalidates transported provenance before any lookup, projection or state mutation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_boundary(
        boundary: RequestBoundary<T>,
        batch: i32,
        tokens: i32,
        expected_offset: i32,
        vocabulary: i32,
        max_tokens: usize,
        rotary_width: i32,
        context: &T::Context,
    ) -> Result<Self, RequestError> {
        let count = usize::try_from(batch)
            .ok()
            .and_then(|b| usize::try_from(tokens).ok().and_then(|t| b.checked_mul(t)))
            .filter(|n| *n > 0 && *n <= max_tokens)
            .ok_or(RequestError::Boundary)?;
        if vocabulary <= 0
            || rotary_width <= 0
            || rotary_width % 2 != 0
            || boundary.ids.shape() != [batch, tokens]
            || boundary.visible.shape() != [batch, tokens]
            || boundary.control.shape() != [4]
            || [&boundary.ids, &boundary.visible, &boundary.control]
                .iter()
                .any(|t| t.element_type() != Some(TensorElementType::I32))
            || [&boundary.cosine, &boundary.sine].iter().any(|t| {
                t.shape() != [batch, tokens, rotary_width]
                    || t.element_type() != Some(TensorElementType::F32)
            })
        {
            return Err(RequestError::Boundary);
        }
        let control = boundary.control.to_i32_vec(context)?;
        if control.len() != 4
            || control[0] < 0
            || control[0] != expected_offset
            || control[0].checked_add(tokens).is_none()
            || !(0..=1).contains(&control[1])
            || control[2] < 0
            || control[2].checked_add(tokens).is_none()
            || (control[1] == 1 && control[2] != 0)
            || (control[1] == 0 && control[0].checked_add(control[3]) != Some(control[2]))
        {
            return Err(RequestError::Boundary);
        }
        let raw = boundary.ids.to_i32_vec(context)?;
        if raw.len() != count || raw.iter().any(|id| *id < 0 || *id >= vocabulary) {
            return Err(NGramError::Shape.into());
        }
        let visible = boundary.visible.to_i32_vec(context)?;
        if visible.len() != count || visible.iter().any(|v| *v != 0 && *v != 1) {
            return Err(RequestError::Boundary);
        }
        Ok(Self {
            offset: control[0],
            position_delta: control[3],
            rotary_offset: (control[1] == 0).then_some(control[2]),
            ids: raw.into_iter().map(|id| id as u64).collect(),
            visible: visible.into_iter().map(|v| v == 1).collect(),
            boundary,
        })
    }
    /// Exact original IDs remain distinct from padding-induced lexical reset values.
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }
    /// Key visibility for QSA and recurrent/lexical padding.
    pub fn visible(&self) -> &[bool] {
        &self.visible
    }
    /// Absolute prefix before this pass begins.
    pub fn offset(&self) -> i32 {
        self.offset
    }
    /// Current-token text or media rotary values.
    pub fn rotary(&self) -> RotaryPosition<'_, T> {
        match self.rotary_offset {
            Some(offset) => RotaryPosition::Offset(offset),
            None => RotaryPosition::Embeddings {
                cosine: &self.boundary.cosine,
                sine: &self.boundary.sine,
            },
        }
    }
    /// Exact media delta transported with this invocation and committed by ingress owner.
    pub fn position_delta(&self) -> i32 {
        self.position_delta
    }
    /// Zero/one mask in the current residual scalar kind, avoiding dtype promotion.
    pub fn padding(&self, like: &T, context: &T::Context) -> Result<T, Error> {
        T::from_f32_slice(
            &self
                .visible
                .iter()
                .map(|v| u8::from(*v) as f32)
                .collect::<Vec<_>>(),
            self.boundary.ids.shape(),
            context,
        )?
        .cast_float(
            like.element_type()
                .ok_or_else(|| Error::backend("missing residual scalar kind"))?,
            context,
        )
    }
    /// Cheap retained handles for transport; host IDs are reconstructed only from exact I32.
    pub fn boundary(&self) -> RequestBoundary<T> {
        self.boundary.clone()
    }
    /// Every native value required through execution/transport completion.
    pub fn retained(&self) -> std::array::IntoIter<&T, 5> {
        [
            &self.boundary.ids,
            &self.boundary.visible,
            &self.boundary.control,
            &self.boundary.cosine,
            &self.boundary.sine,
        ]
        .into_iter()
    }
}
