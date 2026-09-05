// Key/value cache behavior tests.

use std::fs;

use super::{
    BlockwiseAttentionAccumulator, CompressedLatentCache, ConcatKeyValueCache,
    KeyValueAttentionBlock, KeyValueCache, PagedKeyValueCache, PoolingCache,
};
use crate::backend::runtime::cache::residency::{
    open_prompt_cache, CacheBlockArrays, CacheResidencyManager,
};
use crate::backend::ExecutionContext;
use eredu_core::cache::{
    CacheRankIdentity, CacheRepresentation, PromptCacheDescriptor, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheTopology,
};
use eredu_runtime::{inspect_prompt_cache, resolve_prompt_cache_root, PagedCacheOptions};
use safemlx::{
    fast::ScaledDotProductAttentionMask,
    ops::{indexing::TryIndexOp, zeros_dtype},
    transforms::eval,
    Array, Device, DeviceType, Dtype,
};

fn paged_options(full_attention: bool) -> PagedCacheOptions {
    PagedCacheOptions::new(2, 96, 4096, 1)
        .unwrap()
        .with_full_attention(full_attention)
}

fn fully_masked_blockwise_output(mask: &Array, stream: &safemlx::Stream) -> f32 {
    let queries = Array::zeros::<f32>(&[1, 1, 1, 1], stream).unwrap();
    let first = KeyValueAttentionBlock::unleased(
        0,
        2,
        Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap(),
        Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]),
    );
    let second = KeyValueAttentionBlock::unleased(
        2,
        4,
        Array::zeros::<f32>(&[1, 1, 2, 1], stream).unwrap(),
        Array::from_slice(&[3.0f32, 4.0], &[1, 1, 2, 1]),
    );
    let mut accumulator =
        BlockwiseAttentionAccumulator::new(&queries, 1.0, Some(mask), 3, None, 0, None, 4, stream)
            .unwrap();
    accumulator.accumulate(&first, stream).unwrap();
    accumulator.accumulate(&second, stream).unwrap();
    let output = accumulator.finish(stream).unwrap();
    eval([&output]).unwrap();
    output.evaluated().unwrap().as_slice::<f32>()[0]
}

#[derive(Clone, Copy, Debug)]
struct BlockwiseAttentionCase {
    batch: i32,
    query_heads: i32,
    key_value_heads: i32,
    query_len: i32,
    context_len: i32,
    head_dim: i32,
    block_size: i32,
    dtype: Dtype,
    logit_magnitude: f32,
    tolerance: f64,
}

fn patterned_values(length: usize, stride: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let centered = ((index * stride + stride / 2) % modulus) as f32 - (modulus / 2) as f32;
            centered * scale
        })
        .collect()
}

fn assert_blockwise_attention_matches_reference(
    case: BlockwiseAttentionCase,
    stream: &safemlx::Stream,
) -> Array {
    assert!(case.query_len <= case.context_len);
    assert_eq!(case.query_heads % case.key_value_heads, 0);
    let query_elements = (case.batch * case.query_heads * case.query_len * case.head_dim) as usize;
    let key_elements =
        (case.batch * case.key_value_heads * case.context_len * case.head_dim) as usize;
    let query = Array::from_slice(
        &patterned_values(query_elements, 7, 23, 0.125 * case.logit_magnitude),
        &[case.batch, case.query_heads, case.query_len, case.head_dim],
    )
    .as_dtype(case.dtype, stream)
    .unwrap();
    let keys = Array::from_slice(
        &patterned_values(key_elements, 11, 29, 0.1 * case.logit_magnitude),
        &[
            case.batch,
            case.key_value_heads,
            case.context_len,
            case.head_dim,
        ],
    )
    .as_dtype(case.dtype, stream)
    .unwrap();
    let values = Array::from_slice(
        &patterned_values(key_elements, 5, 19, 0.05),
        &[
            case.batch,
            case.key_value_heads,
            case.context_len,
            case.head_dim,
        ],
    )
    .as_dtype(case.dtype, stream)
    .unwrap();
    let scale = (case.head_dim as f32).sqrt().recip();
    let query_start = (case.context_len - case.query_len) as i64;
    let mut accumulator = BlockwiseAttentionAccumulator::new(
        &query,
        scale,
        None,
        query_start,
        None,
        0,
        None,
        case.context_len as i64,
        stream,
    )
    .unwrap();
    let mut block_start = 0;
    while block_start < case.context_len {
        let block_end = (block_start + case.block_size).min(case.context_len);
        let block = KeyValueAttentionBlock::unleased(
            block_start as i64,
            block_end as i64,
            keys.try_index_device((.., .., block_start..block_end, ..), stream)
                .unwrap(),
            values
                .try_index_device((.., .., block_start..block_end, ..), stream)
                .unwrap(),
        );
        accumulator.accumulate(&block, stream).unwrap();
        block_start = block_end;
    }
    let actual = accumulator.finish(stream).unwrap();
    assert_eq!(actual.dtype(), case.dtype);

    // Compare in FP32 after quantizing the inputs to the requested runtime
    // dtype. This isolates the online reduction from input quantization and
    // verifies that lower-precision caches still accumulate in FP32.
    let query_reference = query.as_dtype(Dtype::Float32, stream).unwrap();
    let key_reference = keys.as_dtype(Dtype::Float32, stream).unwrap();
    let value_reference = values.as_dtype(Dtype::Float32, stream).unwrap();
    let allowed = super::absolute_attention_mask(
        query_start,
        case.query_len,
        0,
        case.context_len as i64,
        None,
        0,
    );
    let mask = Array::from_slice(&allowed, &[case.query_len, case.context_len]);
    let reference = safemlx::fast::scaled_dot_product_attention(
        query_reference,
        key_reference,
        value_reference,
        scale,
        Some(ScaledDotProductAttentionMask::Array(&mask)),
        None,
        stream,
    )
    .unwrap();
    let actual = actual.as_dtype(Dtype::Float32, stream).unwrap();
    eval([&actual, &reference]).unwrap();
    assert!(
        actual
            .all_close(&reference, case.tolerance, case.tolerance, None, stream)
            .unwrap()
            .item::<bool>(stream),
        "blockwise attention case did not match its contiguous reference: {case:?}"
    );
    actual
}
