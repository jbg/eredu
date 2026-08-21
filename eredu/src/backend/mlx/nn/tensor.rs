//! Shared tensor helpers used by model implementations.

use safemlx::{
    arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::{
        expand_dims,
        indexing::{NewAxis, TryIndexOp},
        quantized_matmul, reshape, softmax_axis,
    },
    Array, Dtype, Stream,
};
use std::cell::RefCell;

pub(crate) use crate::backend::mlx::nn::rope;
use crate::backend::mlx::runtime::cache::KeyValueCache;

thread_local! {
    static TOKEN_VALIDATION_SCOPE: RefCell<Option<Vec<TokenValidation>>> = const {
        RefCell::new(None)
    };
}

/// One lazy device-side token-domain assertion.
pub(crate) struct TokenValidation {
    invalid: Array,
    message: String,
}

/// Assertions collected while constructing one asynchronous submission.
#[derive(Default)]
pub(crate) struct TokenValidationBatch {
    validations: Vec<TokenValidation>,
}

impl TokenValidationBatch {
    /// Device reductions that must be included in the submission event.
    pub(crate) fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.validations
            .iter()
            .map(|validation| &validation.invalid)
    }

    /// Checks completed one-bit reductions without submitting or waiting for work.
    pub(crate) fn validate_completed(&self) -> Result<(), Exception> {
        for validation in &self.validations {
            let invalid = validation.invalid.evaluated()?;
            if invalid.as_slice::<bool>().first().copied() == Some(true) {
                return Err(Exception::custom(validation.message.clone()));
            }
        }
        Ok(())
    }
}

/// RAII collector for token assertions belonging to one semantic submission.
pub(crate) struct TokenValidationScope {
    active: bool,
}

impl TokenValidationScope {
    /// Starts one non-nestable submission scope.
    pub(crate) fn begin() -> Result<Self, Exception> {
        TOKEN_VALIDATION_SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(Exception::custom(
                    "token validation submission scopes cannot be nested",
                ));
            }
            *slot = Some(Vec::new());
            Ok(Self { active: true })
        })
    }

    /// Seals the collected lazy assertions for completion ownership.
    pub(crate) fn finish(mut self) -> TokenValidationBatch {
        let validations = TOKEN_VALIDATION_SCOPE.with(|slot| {
            slot.borrow_mut()
                .take()
                .expect("active token validation scope must own a collector")
        });
        self.active = false;
        TokenValidationBatch { validations }
    }
}

impl Drop for TokenValidationScope {
    fn drop(&mut self) {
        if self.active {
            TOKEN_VALIDATION_SCOPE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

fn register_token_validation(validation: TokenValidation) -> Result<(), Exception> {
    TOKEN_VALIDATION_SCOPE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let validations = slot.as_mut().ok_or_else(|| {
            Exception::custom("device token validation requires an asynchronous submission scope")
        })?;
        validations.push(validation);
        Ok(())
    })
}

/// Registers a lazy device-side token-domain assertion and normalizes IDs to
/// `int32` for embedding and sequential-decision handoff.
pub(crate) fn validate_token_domain(
    tokens: &Array,
    cardinality: i32,
    sentinel: Option<i32>,
    stream: &Stream,
) -> Result<Array, Exception> {
    if cardinality <= 0 {
        return Err(Exception::custom("token domain must be non-empty"));
    }
    if !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32) {
        return Err(Exception::custom(format!(
            "token IDs must use int32 or uint32 storage, got {:?}",
            tokens.dtype()
        )));
    }
    let tokens = tokens.as_type::<i32>(stream)?;
    if tokens.size() == 0 {
        return Ok(tokens);
    }
    let ordinary = tokens
        .ge(Array::from_int(0), stream)?
        .logical_and(&tokens.lt(Array::from_int(cardinality), stream)?, stream)?;
    let valid = match sentinel {
        Some(sentinel) => {
            ordinary.logical_or(&tokens.eq(Array::from_int(sentinel), stream)?, stream)?
        }
        None => ordinary,
    };
    let invalid = valid.logical_not(stream)?.any(false, stream)?;
    register_token_validation(TokenValidation {
        invalid,
        message: format!(
            "token ID is outside 0..{cardinality}{}",
            sentinel.map_or_else(String::new, |sentinel| format!(" and sentinel {sentinel}"))
        ),
    })?;
    Ok(tokens)
}

#[allow(unused_macros)]
macro_rules! try_unwrap {
    ($expr:expr) => {
        match $expr {
            core::result::Result::Ok(val) => val,
            core::result::Result::Err(e) => return Some(Err(e.into())),
        }
    };
}

// def quantized_scaled_dot_product_attention(
//     queries: mx.array,
//     q_keys: tuple[mx.array, mx.array, mx.array],
//     q_values: tuple[mx.array, mx.array, mx.array],
//     scale: float,
//     mask: Optional[mx.array],
//     group_size: int = 64,
//     bits: int = 8,
// ) -> mx.array:
//     B, n_q_heads, L, D = queries.shape
//     n_kv_heads = q_keys[0].shape[-3]
//     n_repeats = n_q_heads // n_kv_heads

//     queries *= scale

//     if n_repeats > 1:
//         queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
//         q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
//         q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

//     scores = mx.quantized_matmul(
//         queries, *q_keys, transpose=True, group_size=group_size, bits=bits
//     )
//     if mask is not None:
//         if isinstance(mask, str):
//             qL, kL = scores.shape[-2:]
//             q_indices = mx.arange(kL - qL, kL)
//             k_indices = mx.arange(kL)
//             mask = q_indices[:, None] >= k_indices[None]
//         if mask.dtype == mx.bool_:
//             scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
//         else:
//             scores += mask
//     scores = mx.softmax(scores, axis=-1, precise=True)
//     out = mx.quantized_matmul(
//         scores, *q_values, transpose=False, group_size=group_size, bits=bits
//     )

//     if n_repeats > 1:
//         out = mx.reshape(out, (B, n_q_heads, L, D))

//     return out

fn index_out_of_bound_exception() -> Exception {
    Exception::custom("index out of bound")
}

#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn quantized_scaled_dot_product_attention(
    queries: Array,
    mut q_keys: QuantizedKeys,
    mut q_values: QuantizedValues,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let q_shape = queries.shape();
    let B = *q_shape.first().ok_or_else(index_out_of_bound_exception)?;
    let n_q_heads = *q_shape.get(1).ok_or_else(index_out_of_bound_exception)?;
    let L = *q_shape.get(2).ok_or_else(index_out_of_bound_exception)?;
    let D = *q_shape.get(3).ok_or_else(index_out_of_bound_exception)?;

    let q_keys_shape = q_keys.keys.shape();
    let n_kv_heads = q_keys_shape[q_keys_shape.len() - 3];
    let n_repeats = n_q_heads / n_kv_heads;

    let mut queries = queries.multiply(Array::from_f32(scale), stream)?;

    if n_repeats > 1 {
        queries = reshape(&queries, &[B, n_kv_heads, n_repeats, L, D], stream)?;

        q_keys.keys = expand_dims(q_keys.keys, -3, stream)?;
        q_keys.scales = expand_dims(q_keys.scales, -3, stream)?;
        q_keys.biases = expand_dims(q_keys.biases, -3, stream)?;

        q_values.values = expand_dims(q_values.values, -3, stream)?;
        q_values.scales = expand_dims(q_values.scales, -3, stream)?;
        q_values.biases = expand_dims(q_values.biases, -3, stream)?;
    }

    let mut scores = quantized_matmul(
        &queries,
        q_keys.keys,
        q_keys.scales,
        &q_keys.biases,
        true,
        group_size,
        bits,
        stream,
    )?;

    if let Some(mask) = mask {
        // TODO: handle str type mask

        if mask.dtype() == Dtype::Bool {
            let finfo_min = scores.dtype().finfo_min()?;
            scores =
                safemlx::ops::r#where(mask, scores, Array::from_f32(finfo_min as f32), stream)?;
        } else {
            scores = scores.add(mask, stream)?;
        }
    }
    scores = softmax_axis(scores, -1, true, stream)?;
    let mut out = quantized_matmul(
        scores,
        q_values.values,
        q_values.scales,
        &q_values.biases,
        false,
        group_size,
        bits,
        stream,
    )?;

    if n_repeats > 1 {
        out = reshape(out, &[B, n_q_heads, L, D], stream)?;
    }

    Ok(out)
}

/// Quantized key tensor and its dequantization parameters.
pub struct QuantizedKeys {
    /// Packed quantized keys.
    pub keys: Array,
    /// Per-group quantization scales.
    pub scales: Array,
    /// Per-group quantization biases.
    pub biases: Array,
}

/// Quantized value tensor and its dequantization parameters.
pub struct QuantizedValues {
    /// Packed quantized values.
    pub values: Array,
    /// Per-group quantization scales.
    pub scales: Array,
    /// Per-group quantization biases.
    pub biases: Array,
}

/// Either original or quantized attention keys.
pub enum MaybeQuantizedKeys {
    /// Floating-point keys.
    Original(Array),
    /// Quantized keys plus scale metadata.
    Quantized(QuantizedKeys),
}

impl From<Array> for MaybeQuantizedKeys {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedKeys> for MaybeQuantizedKeys {
    fn from(value: QuantizedKeys) -> Self {
        Self::Quantized(value)
    }
}

/// Either original or quantized attention values.
pub enum MaybeQuantizedValues {
    /// Floating-point values.
    Original(Array),
    /// Quantized values plus scale metadata.
    Quantized(QuantizedValues),
}

impl From<Array> for MaybeQuantizedValues {
    fn from(value: Array) -> Self {
        Self::Original(value)
    }
}

impl From<QuantizedValues> for MaybeQuantizedValues {
    fn from(value: QuantizedValues) -> Self {
        Self::Quantized(value)
    }
}

pub(crate) fn scaled_dot_product_attention<C>(
    queries: Array,
    keys: impl Into<MaybeQuantizedKeys>,
    values: impl Into<MaybeQuantizedValues>,
    cache: Option<C>,
    scale: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    let keys = keys.into();
    let values = values.into();

    if let Some(cache) = cache {
        if cache.is_quantized() {
            let group_size = cache
                .group_size()
                .ok_or_else(|| Exception::custom("Cache is quantized but group size is not set"))?;
            let bits = cache
                .bits()
                .ok_or_else(|| Exception::custom("Cache is quantized but bits are not set"))?;

            let (keys, values) = match (keys, values) {
                (MaybeQuantizedKeys::Quantized(keys), MaybeQuantizedValues::Quantized(values)) => {
                    (keys, values)
                }
                _ => {
                    return Err(Exception::custom(
                        "Both keys and values must be quantized when KV cache is quantized",
                    ));
                }
            };

            return quantized_scaled_dot_product_attention(
                queries, keys, values, scale, mask, group_size, bits, stream,
            );
        }
    }

    let (keys, values) = match (keys, values) {
        (MaybeQuantizedKeys::Original(keys), MaybeQuantizedValues::Original(values)) => {
            (keys, values)
        }
        _ => {
            return Err(Exception::custom(
                "Both keys and values must NOT be quantized when KV cache is NOT quantized",
            ));
        }
    };

    safemlx::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
        None,
        stream,
    )
}

#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    lengths: Option<Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N, stream = stream)?;
    let linds = arange!(start = offset, stop = offset + N, stream = stream)?;
    let linds = linds.try_index_device((.., NewAxis), stream)?;
    let rinds = rinds.try_index_device(NewAxis, stream)?;

    let mut mask = linds.ge(&rinds, stream)?;
    if let Some(window_size) = window_size {
        let rinds_window = rinds.add(Array::from_int(window_size), stream)?;
        mask = mask.logical_and(&linds.le(&rinds_window, stream)?, stream)?;
    }

    if let Some(lengths) = lengths {
        let lengths = lengths.try_index_device((.., NewAxis, NewAxis, NewAxis), stream)?;
        mask = mask.logical_and(&linds.lt(&lengths, stream)?, stream)?;
    }

    Ok(mask)
}
