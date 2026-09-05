#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_returns_zero_for_fully_false_boolean_mask() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let mask = Array::from_slice(&[false, false, false, false], &[1, 4]);
    let output = fully_masked_blockwise_output(&mask, context.stream());
    assert_eq!(output, 0.0);
    assert!(output.is_finite());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_returns_zero_for_all_negative_infinity_mask() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let mask = Array::from_slice(&[f32::NEG_INFINITY; 4], &[1, 4]);
    let output = fully_masked_blockwise_output(&mask, context.stream());
    assert_eq!(output, 0.0);
    assert!(output.is_finite());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_matches_multiple_block_sizes_and_batched_queries() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    for block_size in [1, 2, 3, 5, 8] {
        assert_blockwise_attention_matches_reference(
            BlockwiseAttentionCase {
                batch: 2,
                query_heads: 4,
                key_value_heads: 4,
                query_len: 3,
                context_len: 8,
                head_dim: 4,
                block_size,
                dtype: Dtype::Float32,
                logit_magnitude: 1.0,
                tolerance: 1e-5,
            },
            context.stream(),
        );
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_matches_grouped_and_multi_query_attention() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    for key_value_heads in [2, 1] {
        assert_blockwise_attention_matches_reference(
            BlockwiseAttentionCase {
                batch: 2,
                query_heads: 4,
                key_value_heads,
                query_len: 2,
                context_len: 7,
                head_dim: 4,
                block_size: 3,
                dtype: Dtype::Float32,
                logit_magnitude: 1.0,
                tolerance: 1e-5,
            },
            context.stream(),
        );
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_accumulates_lower_precision_inputs_in_float32() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    for (dtype, tolerance) in [(Dtype::Float16, 2e-3), (Dtype::Bfloat16, 1e-2)] {
        assert_blockwise_attention_matches_reference(
            BlockwiseAttentionCase {
                batch: 2,
                query_heads: 4,
                key_value_heads: 2,
                query_len: 3,
                context_len: 7,
                head_dim: 4,
                block_size: 3,
                dtype,
                logit_magnitude: 1.0,
                tolerance,
            },
            context.stream(),
        );
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn blockwise_attention_remains_finite_with_large_logits() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let output = assert_blockwise_attention_matches_reference(
        BlockwiseAttentionCase {
            batch: 2,
            query_heads: 4,
            key_value_heads: 1,
            query_len: 3,
            context_len: 9,
            head_dim: 8,
            block_size: 4,
            dtype: Dtype::Float32,
            logit_magnitude: 1_000.0,
            tolerance: 1e-5,
        },
        context.stream(),
    );
    assert!(output
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_full_attention_matches_contiguous_causal_attention() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let queries = Array::from_slice(
        &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        &[1, 1, 5, 2],
    );
    let keys = Array::from_slice(
        &[1.0f32, 0.0, 0.8, 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8],
        &[1, 1, 5, 2],
    );
    let values = Array::from_slice(
        &[0.2f32, 1.0, 0.4, 0.8, 0.6, 0.6, 0.8, 0.4, 1.0, 0.2],
        &[1, 1, 5, 2],
    );
    cache
        .update_and_fetch(keys.clone(), values.clone(), stream)
        .unwrap();
    let paged = cache
        .paged_attention(&queries, 2.0f32.sqrt().recip(), None, None, stream)
        .unwrap()
        .unwrap();
    let reference = safemlx::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        2.0f32.sqrt().recip(),
        Some(ScaledDotProductAttentionMask::Causal),
        None,
        stream,
    )
    .unwrap();
    eval([&paged, &reference]).unwrap();
    assert!(paged
        .all_close(&reference, 1e-5, 1e-5, None, stream)
        .unwrap()
        .item::<bool>(stream));
    let report = manager.report().unwrap();
    assert_eq!(report.logical_cached_tokens, 5);
    assert_eq!(report.block_seals, 2);
    assert!(report.peak_device_bytes <= manager.options().device_budget_bytes());
    assert_eq!(report.prefill_full_attention_blocks, 3);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_attention_preserves_learned_sink_normalization() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
    let mut cache = PagedKeyValueCache::new(manager, 0, None).unwrap();
    let queries = Array::from_slice(&[0.25f32, 0.5, 0.75, 1.0], &[1, 2, 2, 1]);
    let keys = Array::from_slice(&[0.2f32, 0.4, 0.6, 0.8], &[1, 2, 2, 1]);
    let values = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    let sinks = Array::from_slice(&[0.1f32, -0.2], &[2]);
    cache
        .update_and_fetch(keys.clone(), values.clone(), stream)
        .unwrap();
    let paged = cache
        .paged_attention(&queries, 1.0, None, Some(&sinks), stream)
        .unwrap()
        .unwrap();
    let reference = safemlx::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        1.0,
        Some(ScaledDotProductAttentionMask::Causal),
        Some(&sinks),
        stream,
    )
    .unwrap();
    eval([&paged, &reference]).unwrap();
    assert!(paged
        .all_close(&reference, 1e-5, 1e-5, None, stream)
        .unwrap()
        .item::<bool>(stream));
}
