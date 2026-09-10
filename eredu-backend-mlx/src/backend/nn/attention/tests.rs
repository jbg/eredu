use super::{indexed_sparse_attention, sliding_window_prefill_attention};
use safemlx::{fast::ScaledDotProductAttentionMask, Array, Device, DeviceType};

use crate::backend::{nn::tensor::create_causal_mask, ExecutionContext};

#[test]
#[ignore = "requires MLX runtime execution"]
fn bounded_input_score_attention_matches_pytorch_with_broadcast_noncausal_mask() {
    use safemlx::Dtype;
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../validation/attention_input_scores.json"
    ))
    .unwrap();
    let tensor = |name: &str| {
        let value = &fixture["bounded"][name];
        let shape = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let data = value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        Array::from_slice(&data, &shape)
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap()
    };
    let mask = Array::from_slice(&(0..257).map(|x| x % 7 != 0).collect::<Vec<_>>(), &[257]);
    let actual = super::attention_with_softcap(
        &tensor("queries"),
        &tensor("keys"),
        &tensor("values"),
        fixture["scale"].as_f64().unwrap() as f32,
        Some(&mask),
        None,
        None,
        eredu_nn::AttentionArithmetic::InputScores,
        stream,
    )
    .unwrap();
    assert!(actual
        .all_close(&tensor("output"), 0.0, 0.0, None, stream)
        .unwrap()
        .item::<bool>(stream));
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn input_score_rounding_matches_pytorch_for_contiguous_and_bounded_paged_attention() {
    use crate::backend::runtime::cache::{
        kv::{KeyValueCache, PagedKeyValueCache},
        residency::CacheResidencyManager,
    };
    use eredu_nn::AttentionArithmetic;
    use safemlx::Dtype;
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../validation/attention_input_scores.json"
    ))
    .unwrap();
    let tensor = |name: &str| {
        let value = &fixture[name];
        let shape = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let values = value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        Array::from_slice(&values, &shape)
            .as_dtype(Dtype::Bfloat16, stream)
            .unwrap()
    };
    let q = tensor("queries");
    let k = tensor("keys");
    let v = tensor("values");
    let expected = tensor("output");
    let mask = create_causal_mask(3, Some(2), None, None, stream).unwrap();
    let contiguous = super::attention_with_softcap(
        &q,
        &k,
        &v,
        fixture["scale"].as_f64().unwrap() as f32,
        Some(&mask),
        None,
        None,
        AttentionArithmetic::InputScores,
        stream,
    )
    .unwrap();
    assert!(contiguous
        .all_close(&expected, 0.0, 0.0, None, stream)
        .unwrap()
        .item::<bool>(stream));
    for page_size in [1, 2, 3, 5] {
        let options =
            eredu_runtime::PagedCacheOptions::new(page_size, 80 * page_size as u64, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true);
        let manager = CacheResidencyManager::new(options).unwrap();
        let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
        use safemlx::ops::indexing::TryIndexOp;
        let cached_values = v.try_index_device((.., .., .., ..4), stream).unwrap();
        let cached_expected = expected
            .try_index_device((.., .., .., ..4), stream)
            .unwrap();
        cache
            .update_for_attention(k.clone(), cached_values, stream)
            .unwrap();
        let actual = cache
            .paged_attention(
                &q,
                fixture["scale"].as_f64().unwrap() as f32,
                None,
                None,
                None,
                AttentionArithmetic::InputScores,
                stream,
            )
            .unwrap()
            .unwrap();
        assert_eq!(actual.dtype(), Dtype::Bfloat16);
        assert!(
            actual
                .all_close(&cached_expected, 0.0, 0.0, None, stream)
                .unwrap()
                .item::<bool>(stream),
            "page size {page_size}"
        );
        let report = manager.report().unwrap();
        assert_eq!(
            report.prefill_full_attention_blocks,
            2 * 5u64.div_ceil(page_size as u64)
        );
        assert!(report.peak_device_bytes <= manager.options().device_budget_bytes());
    }
}

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

#[test]
#[ignore = "requires MLX runtime execution"]
fn score_softcap_precedes_boolean_and_additive_masks_and_preserves_sinks() {
    let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = ctx.stream();
    let queries = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1]);
    let keys = Array::from_slice(&[0.0f32, 3.0, 100.0], &[1, 1, 3, 1]);
    let values = Array::from_slice(&[2.0f32, 4.0, 10000.0], &[1, 1, 3, 1]);
    let sinks = Array::from_slice(&[0.5f32, 0.5], &[2]);
    let expected = (0..2)
        .map(|head| {
            let p = ((head + 1) as f32 * 3.0).tanh().exp();
            (2.0 + 4.0 * p) / (1.0 + p + 0.5f32.exp())
        })
        .collect::<Vec<_>>();
    let expected = Array::from_slice(&expected, &[1, 2, 1, 1]);
    for mask in [
        Array::from_slice(&[true, true, false], &[1, 3]),
        Array::from_slice(&[0.0f32, 0.0, f32::NEG_INFINITY], &[1, 3]),
    ] {
        let actual = super::attention_with_softcap(
            &queries,
            &keys,
            &values,
            1.0,
            Some(&mask),
            Some(&sinks),
            Some(1.0),
            eredu_nn::AttentionArithmetic::Fused,
            stream,
        )
        .unwrap();
        assert!(actual
            .all_close(&expected, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
fn native_bf16_attention_query_tiles_match_independent_reference() {
    use super::attention_with_softcap;
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let fixture = Array::load_safetensors(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/validation/bf16_attention.safetensors"
        ),
        &weights,
    )
    .unwrap();
    for width in [6, 7, 8, 9, 10, 93, 193, 513] {
        let allowed: Vec<bool> = (0..width)
            .flat_map(|q| (0..width).map(move |k| k <= q))
            .collect();
        let mask = Array::from_slice(&allowed, &[width, width]);
        let output = attention_with_softcap(
            &fixture[&format!("{width}.queries")],
            &fixture[&format!("{width}.keys")],
            &fixture[&format!("{width}.values")],
            0.125,
            Some(&mask),
            None,
            None,
            eredu_nn::AttentionArithmetic::InputScores,
            &stream,
        )
        .unwrap();
        let actual = output
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let expected = fixture[&format!("{width}.output")]
            .as_dtype(Dtype::Float32, &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let mismatches: Vec<_> = actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "width {width}: {} mismatches; first {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(12)]
        );
    }
}
