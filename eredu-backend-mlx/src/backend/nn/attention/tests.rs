use super::{indexed_sparse_attention, sliding_window_prefill_attention};
use safemlx::{fast::ScaledDotProductAttentionMask, Array, Device, DeviceType};

use crate::backend::{nn::tensor::create_causal_mask, ExecutionContext};

#[test]
#[ignore = "requires MLX runtime execution"]
fn chunked_sliding_prefill_matches_full_masked_gqa_attention() {
    let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let queries = Array::from_slice(
        &[
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
            0.3, 0.2, 0.1,
        ],
        &[1, 2, 5, 2],
    );
    let keys = Array::from_slice(
        &[0.2f32, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
        &[1, 1, 5, 2],
    );
    let values = Array::from_slice(
        &[1.0f32, 0.0, 0.8, 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8],
        &[1, 1, 5, 2],
    );
    let mask = create_causal_mask(5, None, Some(2), None, stream).unwrap();
    let reference = safemlx::fast::scaled_dot_product_attention(
        queries.clone(),
        keys.clone(),
        values.clone(),
        2.0f32.sqrt().recip(),
        Some(ScaledDotProductAttentionMask::Array(&mask)),
        None,
        stream,
    )
    .unwrap()
    .transpose_axes(&[0, 2, 1, 3], stream)
    .unwrap()
    .reshape(&[1, 5, 4], stream)
    .unwrap();

    let chunked = sliding_window_prefill_attention(
        queries,
        keys,
        values,
        2.0f32.sqrt().recip(),
        3,
        0,
        1,
        5,
        None,
        stream,
    )
    .unwrap();

    assert!(chunked
        .all_close(&reference, 1e-5, 1e-5, None, stream)
        .unwrap()
        .item::<bool>(stream));
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn chunked_sliding_prefill_preserves_sink_softmax_semantics() {
    let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let queries = Array::from_slice(&[0.0f32; 4], &[1, 1, 4, 1]);
    let keys = Array::from_slice(&[0.0f32; 4], &[1, 1, 4, 1]);
    let values = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 1, 4, 1]);
    let sinks = Array::from_slice(&[0.0f32], &[1]);
    let actual = sliding_window_prefill_attention(
        queries,
        keys,
        values,
        1.0,
        2,
        0,
        1,
        4,
        Some(&sinks),
        stream,
    )
    .unwrap();
    let actual = actual.evaluated().unwrap();
    let expected = [5.0f32, 10.0, 50.0 / 3.0, 70.0 / 3.0];
    for (actual, expected) in actual.as_slice::<f32>().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn sliding_prefill_matches_independent_scalar_softmax_reference() {
    let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let query_values = [1.0f32, 2.0, 3.0, 4.0];
    let key_values = [0.5f32, 1.0, 1.5, 2.0];
    let value_values = [10.0f32, 20.0, 30.0, 40.0];
    let actual = sliding_window_prefill_attention(
        Array::from_slice(&query_values, &[1, 1, 4, 1]),
        Array::from_slice(&key_values, &[1, 1, 4, 1]),
        Array::from_slice(&value_values, &[1, 1, 4, 1]),
        1.0,
        2,
        0,
        1,
        4,
        None,
        stream,
    )
    .unwrap();
    let actual = actual.evaluated().unwrap();

    let expected = (0..query_values.len())
        .map(|query| {
            let first = query.saturating_sub(1);
            let logits = (first..=query)
                .map(|key| query_values[query] * key_values[key])
                .collect::<Vec<_>>();
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = logits
                .iter()
                .map(|logit| (logit - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            weights
                .iter()
                .zip(&value_values[first..=query])
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                / denominator
        })
        .collect::<Vec<_>>();
    for (actual, expected) in actual.as_slice::<f32>().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn indexed_attention_shares_softmax_with_sink() {
    let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let queries = Array::from_slice(&[1.0f32], &[1, 1, 1, 1]);
    let local = Array::from_slice(&[1.0f32], &[1, 1, 1]);
    let pooled = Array::from_slice(&[2.0f32, 3.0], &[1, 2, 1]);
    let indices = Array::from_slice(&[1i32], &[1, 1, 1]);
    let sinks = Array::from_slice(&[0.0f32], &[1]);
    let output = indexed_sparse_attention(
        &queries,
        &local,
        &local,
        &pooled,
        &pooled,
        &indices,
        1.0,
        None,
        None,
        Some(&sinks),
        stream,
    )
    .unwrap();
    let output = output.evaluated().unwrap();
    let denominator = 1.0f32.exp() + 3.0f32.exp() + 1.0;
    let expected = (1.0f32.exp() + 3.0 * 3.0f32.exp()) / denominator;
    assert!((output.as_slice::<f32>()[0] - expected).abs() < 1e-5);
}
