use super::*;

#[test]
fn grouped_positions_masks_duplicates_and_origin_match_scalar_equations() {
    use eredu_core::checkpoint::TensorDtype;
    let queries = NumericTensor::new(
        vec![1, 4, 2, 1],
        vec![0.5, 1., 1.5, 2., -0.5, -1., -1.5, -2.],
    );
    let keys = NumericTensor::new(vec![1, 2, 3, 1], vec![1., 2., 3., -1., -2., -3.]);
    let values = NumericTensor::new(
        vec![1, 2, 3, 2],
        vec![2., 4., 8., 16., 32., 64., -2., -4., -8., -16., -32., -64.],
    );
    let positions = NumericTensor::new(
        vec![1, 2, 5],
        vec![12., 10., 12., -1., 999., 12., 10., 12., -1., 999.],
    )
    .with_dtype(TensorDtype::I32);
    let valid = NumericTensor::new(vec![1, 2, 5], vec![1., 1., 1., 1., 0., 1., 1., 1., 1., 0.])
        .with_dtype(TensorDtype::Bool);
    let mask = NumericTensor::new(
        vec![2, 5],
        vec![
            0.2,
            -0.3,
            0.4,
            0.,
            0.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.,
            0.,
        ],
    );
    let request = IndexedAttentionInput {
        queries: &queries,
        keys: &keys,
        values: &values,
        key_position_offset: 10,
        selected_positions: &positions,
        validity: Some(&valid),
        mask: Some(&mask),
        local: None,
        scale: 0.7,
        arithmetic: eredu_nn::AttentionArithmetic::Fused,
        sinks: None,
    };
    let actual = indexed_attention(request).unwrap();
    for head in 0..4 {
        let kv = head / 2;
        let scores = [2, 0, 2]
            .into_iter()
            .zip([0.2, -0.3, 0.4])
            .map(|(p, bias)| (queries.data[head * 2] * keys.data[kv * 3 + p] * 0.7 + bias).exp())
            .collect::<Vec<_>>();
        for d in 0..2 {
            let expected = [2, 0, 2]
                .into_iter()
                .zip(&scores)
                .map(|(p, s)| s * values.data[(kv * 3 + p) * 2 + d])
                .sum::<f32>()
                / scores.iter().sum::<f32>();
            assert!((actual.data[head * 4 + d] - expected).abs() < 1e-5);
            assert_eq!(actual.data[head * 4 + 2 + d], 0.0);
        }
    }
    let bad_positions = positions.clone().with_dtype(TensorDtype::F32);
    assert!(indexed_attention(IndexedAttentionInput {
        selected_positions: &bad_positions,
        ..request
    })
    .is_err());
    let bad_validity = valid.clone().with_dtype(TensorDtype::F32);
    assert!(indexed_attention(IndexedAttentionInput {
        validity: Some(&bad_validity),
        ..request
    })
    .is_err());
    assert!(indexed_attention(IndexedAttentionInput {
        key_position_offset: 11,
        ..request
    })
    .is_err());
}

#[test]
fn selected_attention_cache_owned_history_matches_chunked_resident_history() {
    use eredu_core::checkpoint::TensorDtype;
    for owned in [false, true] {
        let context = NumericContext {
            cache_owned_attention: owned,
            ..Default::default()
        };
        let mut cache = NumericCache::new(None);
        for position in 0..5 {
            let token = NumericTensor::new(vec![1, 1, 1, 1], vec![(position + 1) as f32]);
            let (keys, values) = cache
                .update_for_attention(token.clone(), token.clone(), &context)
                .unwrap();
            let positions = NumericTensor::new(vec![1, 1, 2], vec![0., position as f32])
                .with_dtype(TensorDtype::I32);
            let output = cache
                .indexed_attention::<NumericBackend>(
                    IndexedAttentionInput {
                        queries: &token,
                        keys: &keys,
                        values: &values,
                        key_position_offset: 0,
                        selected_positions: &positions,
                        validity: None,
                        mask: None,
                        local: None,
                        scale: 0.1,
                        arithmetic: eredu_nn::AttentionArithmetic::Fused,
                        sinks: None,
                    },
                    &context,
                )
                .unwrap();
            let x = (position + 1) as f32;
            let expected = ((0.1 * x).exp() + x * (0.1 * x * x).exp())
                / ((0.1 * x).exp() + (0.1 * x * x).exp());
            assert!((output.data[0] - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn shared_decoder_selected_attention_preserves_gqa_gate_and_cached_history() {
    use eredu_architectures::decoder::{Attention, AttentionInput};
    let input = NumericTensor::new([1, 6, 2], (0..12).map(|i| (i as f32 - 5.) / 7.).collect());
    let selected = NumericTensor::from_i32_slice(
        &[0, -1, -1, 0, 1, -1, 0, 2, -1, 1, 3, -1, 0, 2, 4, 1, 3, 5],
        &[1, 6, 3],
        &NumericContext::default(),
    )
    .unwrap();
    let mut mask = vec![f32::NEG_INFINITY; 36];
    for q in 0..6 {
        for &position in &selected.data[q * 3..q * 3 + 3] {
            if position >= 0. {
                mask[q * 6 + position as usize] = 0.;
            }
        }
    }
    let mask = NumericTensor::new([1, 1, 6, 6], mask);
    let build = |context: &NumericContext| {
        let linear = |id, input, output| {
            NumericBackend::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(id).unwrap(),
                    bias: None,
                    format: dense_linear_format(),
                },
                context,
            )
            .unwrap()
        };
        let mut result = Attention::<NumericBackend>::from_gated_parts(
            2,
            1,
            2,
            linear("selected.q", 2, 8),
            linear("selected.k", 2, 2),
            linear("selected.v", 2, 2),
            linear("selected.o", 4, 2),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        struct Load;
        impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
            fn visit_mut(&mut self, m: ParameterMetadata, v: &'a mut NumericTensor) {
                let sign = if m.id.as_str().ends_with('v') {
                    -1.
                } else {
                    1.
                };
                for (i, value) in v.data.iter_mut().enumerate() {
                    *value = sign * ((i * 7 % 13) as f32 - 6.) / 11.;
                }
            }
        }
        result.visit_parameters_mut(&mut Load);
        result
    };
    let context = NumericContext::default();
    let dense = build(&context)
        .forward(
            AttentionInput::<_, NumericCache> {
                selected_positions: None,
                hidden: &input,
                mask: Some(&mask),
                cache: None,
                allow_sliding_prefill: false,
                rotary_position: None,
            },
            &context,
        )
        .unwrap();
    let sparse = build(&context)
        .forward(
            AttentionInput::<_, NumericCache> {
                selected_positions: Some(&selected),
                hidden: &input,
                mask: None,
                cache: None,
                allow_sliding_prefill: false,
                rotary_position: None,
            },
            &context,
        )
        .unwrap();
    assert!(sparse.data.iter().any(|x| x.abs() > 1e-4));
    assert_tensor_close(&sparse, &dense, "shared decoder selected attention");
    for cache_owned_attention in [false, true] {
        let context = NumericContext {
            cache_owned_attention,
            ..Default::default()
        };
        let mut cache = NumericCache::new(None);
        let mut layer = build(&context);
        let mut pieces = vec![];
        for token in 0..6 {
            pieces.push(
                layer
                    .forward(
                        AttentionInput {
                            selected_positions: Some(&selected.axis_slice(1, token, token + 1)),
                            hidden: &input.axis_slice(1, token, token + 1),
                            mask: None,
                            cache: Some(&mut cache),
                            allow_sliding_prefill: false,
                            rotary_position: None,
                        },
                        &context,
                    )
                    .unwrap(),
            );
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
            &dense,
            "selected cached decoder",
        );
        assert_eq!(cache.offset(), 6);
    }
}
