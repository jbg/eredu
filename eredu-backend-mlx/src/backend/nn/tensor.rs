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

pub use crate::backend::nn::rope;
use crate::backend::runtime::cache::KeyValueCache;

thread_local! {
    static TOKEN_VALIDATION_SCOPE: RefCell<Option<Vec<TokenValidation>>> = const {
        RefCell::new(None)
    };
}

/// One lazy device-side token-domain assertion.
pub struct TokenValidation {
    invalid: Array,
    message: String,
}

/// Assertions collected while constructing one asynchronous submission.
#[derive(Default)]
pub struct TokenValidationBatch {
    validations: Vec<TokenValidation>,
}

impl TokenValidationBatch {
    /// Device reductions that must be included in the submission event.
    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.validations
            .iter()
            .map(|validation| &validation.invalid)
    }

    /// Checks completed one-bit reductions without submitting or waiting for work.
    pub fn validate_completed(&self) -> Result<(), Exception> {
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
pub struct TokenValidationScope {
    active: bool,
}

impl TokenValidationScope {
    /// Starts one non-nestable submission scope.
    pub fn begin() -> Result<Self, Exception> {
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
    pub fn finish(mut self) -> TokenValidationBatch {
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
pub fn validate_token_domain(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedAttentionGeometry {
    batch: i32,
    query_heads: i32,
    kv_heads: i32,
    query_length: i32,
    head_dimension: i32,
}

fn invalid_quantized_attention(message: impl Into<String>) -> Exception {
    Exception::custom(format!(
        "invalid quantized attention geometry: {}",
        message.into()
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_quantized_attention_geometry(
    queries: &[i32],
    keys: &[i32],
    key_scales: &[i32],
    key_biases: &[i32],
    values: &[i32],
    value_scales: &[i32],
    value_biases: &[i32],
    group_size: i32,
    bits: i32,
) -> Result<QuantizedAttentionGeometry, Exception> {
    let &[batch, query_heads, query_length, head_dimension] = queries else {
        return Err(invalid_quantized_attention(format!(
            "queries must have rank 4, got shape {queries:?}"
        )));
    };
    let &[_, kv_heads, key_length, _] = keys else {
        return Err(invalid_quantized_attention(format!(
            "packed keys must have rank 4, got shape {keys:?}"
        )));
    };
    if key_scales.len() != 4
        || key_biases.len() != 4
        || values.len() != 4
        || value_scales.len() != 4
        || value_biases.len() != 4
    {
        return Err(invalid_quantized_attention(format!(
            "packed keys, values, scales, and biases must all have rank 4; got key scales {key_scales:?}, key biases {key_biases:?}, values {values:?}, value scales {value_scales:?}, and value biases {value_biases:?}"
        )));
    }

    if group_size != 16 && (group_size <= 0 || group_size % 32 != 0) {
        return Err(invalid_quantized_attention(format!(
            "group size must be 16 or a positive multiple of 32, got {group_size}"
        )));
    }
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(invalid_quantized_attention(format!(
            "bit width must be one of 2, 3, 4, 5, 6, or 8, got {bits}"
        )));
    }
    if query_heads <= 0 || kv_heads <= 0 {
        return Err(invalid_quantized_attention(format!(
            "query and key/value head counts must be positive, got {query_heads} and {kv_heads}"
        )));
    }
    if query_heads % kv_heads != 0 {
        return Err(invalid_quantized_attention(format!(
            "query head count {query_heads} must be divisible by key/value head count {kv_heads}"
        )));
    }
    if head_dimension <= 0 || head_dimension % group_size != 0 || head_dimension % 32 != 0 {
        return Err(invalid_quantized_attention(format!(
            "query head dimension {head_dimension} must be positive and divisible by group size {group_size} and 32"
        )));
    }
    let packed_bits = head_dimension.checked_mul(bits).ok_or_else(|| {
        invalid_quantized_attention(format!(
            "query head dimension {head_dimension} overflows at {bits} bits"
        ))
    })?;
    if packed_bits % 32 != 0 {
        return Err(invalid_quantized_attention(format!(
            "query head dimension {head_dimension} cannot be packed into 32-bit words at {bits} bits"
        )));
    }
    let expected_packed_dimension = packed_bits / 32;
    let expected_group_dimension = head_dimension / group_size;
    let expected_packed = [batch, kv_heads, key_length, expected_packed_dimension];
    let expected_groups = [batch, kv_heads, key_length, expected_group_dimension];

    if keys != expected_packed || values != expected_packed {
        return Err(invalid_quantized_attention(format!(
            "packed key and value shapes must both be {expected_packed:?}, got keys {keys:?} and values {values:?}"
        )));
    }
    if key_scales != expected_groups
        || key_biases != expected_groups
        || value_scales != expected_groups
        || value_biases != expected_groups
    {
        return Err(invalid_quantized_attention(format!(
            "key and value scale and bias shapes must all be {expected_groups:?}, got key scales {key_scales:?}, key biases {key_biases:?}, value scales {value_scales:?}, and value biases {value_biases:?}"
        )));
    }

    Ok(QuantizedAttentionGeometry {
        batch,
        query_heads,
        kv_heads,
        query_length,
        head_dimension,
    })
}

#[allow(non_snake_case, clippy::too_many_arguments)]
/// Computes scaled dot-product attention over affine-quantized keys and values.
///
/// Keys and values use `[batch, kv_heads, sequence, packed_head_dimension]`;
/// their scales and biases replace the last dimension with
/// `head_dimension / group_size`.
///
/// # Errors
///
/// Returns an error when tensor ranks or paired geometry are inconsistent,
/// head counts are not compatible, quantization parameters are unsupported,
/// or an MLX operation fails.
pub fn quantized_scaled_dot_product_attention(
    queries: Array,
    mut q_keys: QuantizedKeys,
    mut q_values: QuantizedValues,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let geometry = validate_quantized_attention_geometry(
        queries.shape(),
        q_keys.keys.shape(),
        q_keys.scales.shape(),
        q_keys.biases.shape(),
        q_values.values.shape(),
        q_values.scales.shape(),
        q_values.biases.shape(),
        group_size,
        bits,
    )?;
    let B = geometry.batch;
    let n_q_heads = geometry.query_heads;
    let n_kv_heads = geometry.kv_heads;
    let L = geometry.query_length;
    let D = geometry.head_dimension;
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
    /// Packed quantized keys shaped `[batch, kv_heads, sequence, packed_head_dimension]`.
    pub keys: Array,
    /// Per-group quantization scales with the same leading geometry as `keys`.
    pub scales: Array,
    /// Per-group quantization biases with the same shape as `scales`.
    pub biases: Array,
}

/// Quantized value tensor and its dequantization parameters.
pub struct QuantizedValues {
    /// Packed quantized values shaped `[batch, kv_heads, sequence, packed_head_dimension]`.
    pub values: Array,
    /// Per-group quantization scales with the same leading geometry as `values`.
    pub scales: Array,
    /// Per-group quantization biases with the same shape as `scales`.
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

pub fn scaled_dot_product_attention<C>(
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
pub fn create_causal_mask(
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

#[cfg(test)]
mod tests {
    use super::{
        quantized_scaled_dot_product_attention, validate_quantized_attention_geometry,
        QuantizedKeys, QuantizedValues,
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::backend::runtime::checkpoint::quantization::quantize_tensor;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use eredu_checkpoint::AffineQuantization;
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

    const QUERIES: &[i32] = &[2, 8, 3, 64];
    const PACKED: &[i32] = &[2, 2, 5, 8];
    const GROUPS: &[i32] = &[2, 2, 5, 2];

    fn validate(
        queries: &[i32],
        keys: &[i32],
        key_scales: &[i32],
        key_biases: &[i32],
        values: &[i32],
        value_scales: &[i32],
        value_biases: &[i32],
        group_size: i32,
        bits: i32,
    ) -> String {
        validate_quantized_attention_geometry(
            queries,
            keys,
            key_scales,
            key_biases,
            values,
            value_scales,
            value_biases,
            group_size,
            bits,
        )
        .unwrap_err()
        .to_string()
    }

    #[test]
    fn quantized_attention_accepts_consistent_geometry() {
        let geometry = validate_quantized_attention_geometry(
            QUERIES, PACKED, GROUPS, GROUPS, PACKED, GROUPS, GROUPS, 32, 4,
        )
        .unwrap();

        assert_eq!(geometry.batch, 2);
        assert_eq!(geometry.query_heads, 8);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.query_length, 3);
        assert_eq!(geometry.head_dimension, 64);
    }

    #[test]
    fn quantized_attention_rejects_malformed_ranks() {
        let query_error = validate(
            &[2, 8, 3],
            PACKED,
            GROUPS,
            GROUPS,
            PACKED,
            GROUPS,
            GROUPS,
            32,
            4,
        );
        assert!(query_error.contains("queries must have rank 4"));

        let field_error = validate(
            QUERIES,
            PACKED,
            &[2, 2, 5],
            GROUPS,
            PACKED,
            GROUPS,
            GROUPS,
            32,
            4,
        );
        assert!(field_error.contains("must all have rank 4"));
    }

    #[test]
    fn public_quantized_attention_returns_error_for_short_key_rank() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let scalar = Array::from_int(0);
        let result = quantized_scaled_dot_product_attention(
            Array::from_slice(&[0.0f32; 32], &[1, 1, 1, 32]),
            QuantizedKeys {
                keys: scalar.clone(),
                scales: scalar.clone(),
                biases: scalar.clone(),
            },
            QuantizedValues {
                values: scalar.clone(),
                scales: scalar.clone(),
                biases: scalar,
            },
            1.0,
            None,
            32,
            4,
            context.stream(),
        );
        let error = match result {
            Ok(_) => panic!("rank-zero packed keys must return an error"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("packed keys must have rank 4"));
    }

    #[test]
    fn quantized_attention_rejects_invalid_head_geometry() {
        let zero_heads = validate(
            QUERIES,
            &[2, 0, 5, 8],
            &[2, 0, 5, 2],
            &[2, 0, 5, 2],
            &[2, 0, 5, 8],
            &[2, 0, 5, 2],
            &[2, 0, 5, 2],
            32,
            4,
        );
        assert!(zero_heads.contains("head counts must be positive"));

        let indivisible_heads = validate(
            &[2, 6, 3, 64],
            &[2, 4, 5, 8],
            &[2, 4, 5, 2],
            &[2, 4, 5, 2],
            &[2, 4, 5, 8],
            &[2, 4, 5, 2],
            &[2, 4, 5, 2],
            32,
            4,
        );
        assert!(indivisible_heads.contains("must be divisible"));
    }

    #[test]
    fn quantized_attention_rejects_invalid_quantization_parameters() {
        for group_size in [0, 24] {
            let error = validate(
                QUERIES, PACKED, GROUPS, GROUPS, PACKED, GROUPS, GROUPS, group_size, 4,
            );
            assert!(error.contains("group size must be 16 or a positive multiple of 32"));
        }

        let bits_error = validate(
            QUERIES, PACKED, GROUPS, GROUPS, PACKED, GROUPS, GROUPS, 32, 7,
        );
        assert!(bits_error.contains("bit width must be one of"));

        let group_alignment_error = validate(
            &[2, 8, 3, 48],
            PACKED,
            GROUPS,
            GROUPS,
            PACKED,
            GROUPS,
            GROUPS,
            32,
            4,
        );
        assert!(group_alignment_error.contains("head dimension 48"));
    }

    #[test]
    fn quantized_attention_rejects_inconsistent_paired_geometry() {
        let packed_error = validate(
            QUERIES,
            PACKED,
            GROUPS,
            GROUPS,
            &[2, 2, 4, 8],
            GROUPS,
            GROUPS,
            32,
            4,
        );
        assert!(packed_error.contains("packed key and value shapes"));

        let metadata_error = validate(
            QUERIES,
            PACKED,
            &[2, 2, 5, 3],
            GROUPS,
            PACKED,
            GROUPS,
            GROUPS,
            32,
            4,
        );
        assert!(metadata_error.contains("scale and bias shapes"));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn quantized_attention_executes_valid_metal_geometry() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let keys = Array::from_slice(&[0.25f32; 2 * 32], &[2, 32]);
        let values = Array::from_slice(&[0.5f32; 2 * 32], &[2, 32]);
        let config = AffineQuantization::new(32, 4).unwrap();
        let keys = quantize_tensor(&keys, config, stream).unwrap();
        let values = quantize_tensor(&values, config, stream).unwrap();
        let output = quantized_scaled_dot_product_attention(
            Array::from_slice(&[0.125f32; 2 * 32], &[1, 2, 1, 32]),
            QuantizedKeys {
                keys: keys.weight.reshape(&[1, 1, 2, 4], stream).unwrap(),
                scales: keys.scales.reshape(&[1, 1, 2, 1], stream).unwrap(),
                biases: keys.biases.unwrap().reshape(&[1, 1, 2, 1], stream).unwrap(),
            },
            QuantizedValues {
                values: values.weight.reshape(&[1, 1, 2, 4], stream).unwrap(),
                scales: values.scales.reshape(&[1, 1, 2, 1], stream).unwrap(),
                biases: values
                    .biases
                    .unwrap()
                    .reshape(&[1, 1, 2, 1], stream)
                    .unwrap(),
            },
            32.0f32.sqrt().recip(),
            None,
            32,
            4,
            stream,
        )
        .unwrap();

        assert_eq!(output.shape(), &[1, 2, 1, 32]);
        output.evaluated().unwrap();
    }
}
