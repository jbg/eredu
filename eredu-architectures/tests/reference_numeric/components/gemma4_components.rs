//! Gemma4's exact dense GELU, branch norms, and whole-residual learned scale.
use super::*;

#[test]
fn gemma4_shared_attention_infers_the_publication_start_for_rotary_queries() {
    let args = gemma4::ModelArgs::from_hf_json(
        br#"{
        "model_type":"gemma4_text", "hidden_size":8, "num_hidden_layers":2,
        "intermediate_size":12, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":19,
        "max_position_embeddings":64, "num_kv_shared_layers":1,
        "layer_types":["full_attention","full_attention"]
    }"#,
    )
    .unwrap();
    for cache_owned_attention in [false, true] {
        let context = NumericContext {
            cache_owned_attention,
            ..Default::default()
        };
        let mut publisher = gemma4::Attention::<NumericBackend>::new(
            &args,
            0,
            args.layer_policy(0).unwrap(),
            &context,
        )
        .unwrap();
        let mut inferred = gemma4::Attention::<NumericBackend>::new(
            &args,
            1,
            args.layer_policy(1).unwrap(),
            &context,
        )
        .unwrap();
        let mut explicit = inferred.clone();
        let mut cache = NumericCache::new(None);
        let mut shared = gemma4::SharedAttentionStates::new();
        for sequence in [3, 1, 2] {
            let start = cache.offset();
            let hidden = NumericTensor::new(
                [1, sequence, 8],
                (0..sequence * 8)
                    .map(|i| (i as f32 * 0.31 + start as f32 * 0.7).sin())
                    .collect(),
            );
            publisher
                .forward(
                    gemma4::AttentionInput {
                        hidden: &hidden,
                        mask: None,
                        cache: Some(&mut cache),
                        shared: &mut shared,
                        rotary_position: None,
                    },
                    &context,
                )
                .unwrap();
            let actual = inferred
                .forward(
                    gemma4::AttentionInput {
                        hidden: &hidden,
                        mask: None,
                        cache: Some(&mut cache),
                        shared: &mut shared,
                        rotary_position: None,
                    },
                    &context,
                )
                .unwrap();
            let expected = explicit
                .forward(
                    gemma4::AttentionInput {
                        hidden: &hidden,
                        mask: None,
                        cache: Some(&mut cache),
                        shared: &mut shared,
                        rotary_position: Some(RotaryPosition::Offset(start)),
                    },
                    &context,
                )
                .unwrap();
            assert_tensor_exact(
                &actual,
                &expected,
                "shared Gemma publication rotary frontier",
            );
            assert!(actual.data.iter().any(|value| value.abs() > 1e-5));
        }
    }
}

fn args(sparse: bool, media: bool) -> gemma4::ModelArgs {
    gemma4::ModelArgs::from_hf_json(
        &serde_json::to_vec(&serde_json::json!({
            "model_type":"gemma4_unified","hidden_size":8,"num_hidden_layers":1,
            "intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,
            "head_dim":4,"rms_norm_eps":1e-5,"vocab_size":19,"max_position_embeddings":64,
            "layer_types":["full_attention"],"attention_bias":true,"attention_k_eq_v":true,
            "enable_moe_block":sparse,"num_experts":sparse.then_some(3),"top_k_experts":sparse.then_some(2),"moe_intermediate_size":sparse.then_some(5),
            "hidden_size_per_layer_input":if media {4} else {0},"vocab_size_per_layer_input":media.then_some(19),
            "final_logit_softcapping":7.0
        }))
        .unwrap(),
    )
    .unwrap()
}

fn run(
    block: &mut gemma4::DenseBlock<NumericBackend>,
    input: &NumericTensor,
    media: Option<&NumericTensor>,
    capture: Option<&mut Components>,
) -> NumericTensor {
    let mut shared = gemma4::SharedAttentionStates::new();
    let request = gemma4::BlockInput {
        hidden: input,
        mask: None,
        cache: None::<&mut NumericCache>,
        shared: &mut shared,
        per_layer_input: media,
        rotary_position: None,
    };
    let context = NumericContext::default();
    match capture {
        Some(capture) => block
            .forward_observed(
                request,
                &context,
                &mut ComponentInstrumentation::new("model.language_model.layers.0", capture),
            )
            .unwrap(),
        None => block.forward(request, &context).unwrap(),
    }
}

fn normalized(input: &NumericTensor, norm: &NumericNorm) -> NumericTensor {
    let mut output = input.clone();
    let width = *input.shape.last().unwrap() as usize;
    for row in output.data.chunks_mut(width) {
        let denominator = (row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            / width as f64
            + f64::from(norm.epsilon))
        .sqrt();
        for (i, value) in row.iter_mut().enumerate() {
            *value = (f64::from(*value) * f64::from(norm.weight.data[i] + norm.offset)
                / denominator) as f32;
        }
    }
    output
}

#[test]
fn gemma4_components_reconstruct_branch_writes_masks_and_learned_residual_scaling() {
    for sparse in [false, true] {
        for media_enabled in [false, true] {
            let args = args(sparse, media_enabled);
            let context = NumericContext::default();
            let mut block = gemma4::DenseBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
            let mut norms = vec![
                &mut block.input_norm,
                &mut block.post_attention_norm,
                &mut block.pre_feed_forward_norm,
                &mut block.post_feed_forward_norm,
            ];
            norms.extend(block.post_feed_forward_norm_1.iter_mut());
            norms.extend(block.pre_feed_forward_norm_2.iter_mut());
            norms.extend(block.post_feed_forward_norm_2.iter_mut());
            norms.extend(block.per_layer_norm.iter_mut());
            for norm in norms {
                norm.weight =
                    NumericTensor::new([8], (0..8).map(|i| 0.8 + i as f32 * 0.07).collect());
            }
            block
                .layer_scalar
                .replace(NumericTensor::new([1], vec![1.3]));
            let input = NumericTensor::new(
                [1, 3, 8],
                (0..24).map(|i| (i as f32 * 0.37 + 0.3).sin()).collect(),
            );
            let media = media_enabled.then(|| {
                NumericTensor::new(
                    [1, 3, 4],
                    (0..12)
                        .map(|i| (i as f32 * 0.31 + 0.8).cos() * 0.4)
                        .collect(),
                )
            });
            let ordinary = run(&mut block.clone(), &input, media.as_ref(), None);
            let mut baseline = Components::strict();
            let observed = run(
                &mut block.clone(),
                &input,
                media.as_ref(),
                Some(&mut baseline),
            );
            assert_tensor_exact(&ordinary, &observed, "Gemma4 no-op component hooks");
            let key = |suffix: &str| format!("model.language_model.layers.0.{suffix}");
            let dense_scope = if sparse {
                "dense_feed_forward"
            } else {
                "feed_forward"
            };
            for (input_scope, write_scope, projection, norm) in [
                (
                    "attention",
                    "attention",
                    &block.attention.output,
                    &block.post_attention_norm,
                ),
                (
                    "dense_feed_forward",
                    dense_scope,
                    &block.mlp.down,
                    block
                        .post_feed_forward_norm_1
                        .as_ref()
                        .unwrap_or(&block.post_feed_forward_norm),
                ),
            ] {
                let values = &baseline.values[&key(&format!("{input_scope}.write_input"))];
                assert!(values.data.iter().any(|value| value.abs() > 1e-4));
                let components = values.shape[2] as usize;
                let mut writes = vec![0.0; 24];
                for token in 0..3 {
                    for out in 0..8 {
                        let bias = projection
                            .bias
                            .as_ref()
                            .map_or(0.0, |(value, _)| value.data[out]);
                        writes[token * 8 + out] = ((0..components)
                            .map(|component| {
                                f64::from(values.data[token * components + component])
                                    * f64::from(
                                        projection.weight.data[out * components + component],
                                    )
                            })
                            .sum::<f64>()
                            + f64::from(bias))
                            as f32;
                    }
                }
                let write = NumericTensor::new([1, 3, 8], writes);
                assert_tensor_close(
                    &write,
                    &baseline.values[&key(&format!("{write_scope}.write"))],
                    "Gemma4 signed component sum plus separate projection bias",
                );
                assert_tensor_close(
                    &normalized(&write, norm),
                    &baseline.values[&key(&format!("{write_scope}.output"))],
                    "Gemma4 post-projection RMS normalization",
                );
            }
            if sparse {
                let routed = &baseline.values[&key("routed_feed_forward.write")];
                assert!(routed.data.iter().any(|value| value.abs() > 1e-5));
                assert_tensor_close(
                    &normalized(routed, block.post_feed_forward_norm_2.as_ref().unwrap()),
                    &baseline.values[&key("routed_feed_forward.output")],
                    "Gemma4 postnorm of complete routed sum",
                );
                let summed = baseline.values[&key("dense_feed_forward.output")]
                    .add(
                        &baseline.values[&key("routed_feed_forward.output")],
                        &context,
                    )
                    .unwrap();
                assert_tensor_exact(
                    &summed,
                    &baseline.values[&key("feed_forward.write")],
                    "Gemma4 normalized branch sum",
                );
                assert_tensor_close(
                    &normalized(&summed, &block.post_feed_forward_norm),
                    &baseline.values[&key("feed_forward.output")],
                    "Gemma4 common branch normalization",
                );
            }
            let mut reconstructed = input
                .add(&baseline.values[&key("attention.output")], &context)
                .unwrap()
                .add(&baseline.values[&key("feed_forward.output")], &context)
                .unwrap();
            if media_enabled {
                reconstructed = reconstructed
                    .add(&baseline.values[&key("per_layer.output")], &context)
                    .unwrap();
            }
            assert_tensor_exact(
                &reconstructed,
                &baseline.values[&key("residual.before_scale")],
                "Gemma4 entire residual before learned scaling",
            );
            reconstructed = reconstructed.multiply_scalar(1.3, &context).unwrap();
            assert_tensor_exact(
                &reconstructed,
                &ordinary,
                "Gemma4 scale includes embedding and every write",
            );
            for target in [
                "model.language_model.layers.0.attention.channels",
                "model.language_model.layers.0.dense_feed_forward.units",
            ] {
                let mut trial = Components {
                    zero: Some((target, 1, 2)),
                    reject_duplicates: true,
                    ..Default::default()
                };
                let changed = run(&mut block.clone(), &input, media.as_ref(), Some(&mut trial));
                let before = &trial.values[target];
                let after = &trial.values[&format!("{target}.effective")];
                let width = before.shape[2] as usize;
                for i in 0..before.data.len() {
                    assert_eq!(
                        after.data[i],
                        if i == width + 2 { 0.0 } else { before.data[i] }
                    );
                }
                assert_eq!(&changed.data[..8], &ordinary.data[..8]);
                assert_eq!(&changed.data[16..], &ordinary.data[16..]);
                assert!(changed.data[8..16]
                    .iter()
                    .zip(&ordinary.data[8..16])
                    .any(|(a, b)| (a - b).abs() > 1e-5));
            }
            let mut keep = Components {
                zero: Some(("model.language_model.layers.0.attention.channels", 1, 2)),
                keep: Some((
                    "model.language_model.layers.0.dense_feed_forward.units",
                    1,
                    2,
                )),
                reject_duplicates: true,
                ..Default::default()
            };
            run(&mut block.clone(), &input, media.as_ref(), Some(&mut keep));
            let units = key("dense_feed_forward.units");
            assert!(
                (keep.values[&units].data[14] - baseline.values[&units].data[14]).abs() > 1e-6,
                "surviving GELU unit must recompute from changed residual"
            );
            let mut warmed = block.clone();
            run(&mut warmed, &input, media.as_ref(), None);
            warmed
                .layer_scalar
                .replace(NumericTensor::new([1], vec![-0.4]));
            let mut independently_edited = block.clone();
            independently_edited
                .layer_scalar
                .replace(NumericTensor::new([1], vec![-0.4]));
            assert_tensor_exact(
                &run(&mut warmed, &input, media.as_ref(), None),
                &run(&mut independently_edited, &input, media.as_ref(), None),
                "Gemma4 warmed scale replacement",
            );
            warmed
                .layer_scalar
                .replace(NumericTensor::new([1], vec![1.3]));
            assert_tensor_exact(
                &run(&mut warmed, &input, media.as_ref(), None),
                &ordinary,
                "Gemma4 learned scale restoration",
            );
        }
    }
}
