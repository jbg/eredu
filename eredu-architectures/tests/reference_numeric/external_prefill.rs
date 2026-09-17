//! The production raw-context equation, without a tiny checkpoint exception.
use super::*;

fn config() -> eredu_architectures::muse_glimmer::DFlashConfig {
    eredu_architectures::muse_glimmer::DFlashConfig {
        model_type: "muse_glimmer_assistant".into(),
        hidden_size: 2,
        intermediate_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        max_position_embeddings: 16,
        sliding_window: 3,
        block_size: 16,
        mask_token_id: 1,
        target_layer_ids: vec![2, 0],
        quantization: None,
        quantized_weights: Default::default(),
    }
}

#[test]
fn dflash_raw_span_suffix_preserves_numeric_tap_order_without_model_construction() {
    let config = config();
    let context = NumericContext::default();
    assert!(
        config.validate_released().is_err(),
        "semantic helper does not admit a tiny checkpoint"
    );
    let taps = [
        NumericTensor::new(vec![1, 5, 2], (0..10).map(|x| x as f32 + 10.0).collect()),
        NumericTensor::new(vec![1, 5, 2], (0..10).map(|x| x as f32 + 100.0).collect()),
    ];
    let full = config.assemble_target_states(&taps, &context).unwrap();
    assert_eq!(&full.data[..4], &[10.0, 11.0, 100.0, 101.0]);
    let mut pending = None;
    for (start, end) in [(0, 2), (2, 4), (4, 5)] {
        let local = taps
            .iter()
            .map(|tap| tap.axis_slice(1, start, end))
            .collect::<Vec<_>>();
        let next = config
            .prepare_raw_context_span(pending.as_ref(), &local, &context)
            .unwrap();
        let expected = full.axis_slice(1, end.saturating_sub(3), end);
        assert_eq!(next.shape, expected.shape);
        assert_eq!(next.data, expected.data);
        pending = Some(next);
    }
    let saved = pending.unwrap();
    let bad = [
        NumericTensor::new(vec![1, 1, 3], vec![1.0, 2.0, 3.0]),
        taps[1].axis_slice(1, 4, 5),
    ];
    assert!(config
        .prepare_raw_context_span(Some(&saved), &bad, &context)
        .is_err());
    assert_eq!(saved.data, full.axis_slice(1, 2, 5).data);
    let wide = config
        .prepare_raw_context_span(Some(&saved), &taps, &context)
        .unwrap();
    assert_eq!(wide.data, full.axis_slice(1, 2, 5).data);
    let reversed = config
        .assemble_target_states(&[taps[1].clone(), taps[0].clone()], &context)
        .unwrap();
    assert_ne!(
        reversed.data, full.data,
        "tap order must not be normalized by layer id"
    );
}
