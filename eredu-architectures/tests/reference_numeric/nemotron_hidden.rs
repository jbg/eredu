//! Supplied embeddings and normalized readout share ordinary target execution.
use super::*;
use nemotron_h::TargetReadout;

pub(super) fn supplied(tokens: &NumericTensor, width: i32) -> NumericTensor {
    NumericTensor::new(
        vec![tokens.dim(0), tokens.dim(1), width],
        (0..tokens.dim(0) * tokens.dim(1) * width)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03125)
            .collect(),
    )
}

#[test]
fn nemotron_supplied_hidden_readout_matches_logits_and_streamed_state() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":7, "hidden_size":8,
        "intermediate_size":12, "num_hidden_layers":2,
        "hybrid_override_pattern":"M*", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true
    }))
    .unwrap();
    let context = NumericContext::default();
    let make_model =
        || nemotron_h::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
    let make_state = || {
        DeviceState::<NumericBackend, _>::create(
            nemotron_h::state_layout(&args).unwrap(),
            |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
        )
        .unwrap()
    };
    let mut ordinary = ResidentRuntime::new(make_model(), &context).unwrap();
    let mut hidden = ResidentRuntime::new(make_model(), &context).unwrap();
    let mut streamed = LayerwiseRuntime::new(make_model(), RebuildingUnitPolicy::default());
    let mut ordinary_state = make_state();
    let mut hidden_state = make_state();
    let mut streamed_state = make_state();
    let inputs = [
        NumericTensor::token_ids(&[1, 2, 3]),
        NumericTensor::token_ids(&[4]),
        NumericTensor::token_ids(&[5]),
    ];
    for (step, tokens) in inputs.iter().enumerate() {
        let embedded = supplied(tokens, args.hidden_size);
        let expected = ordinary
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    tokens,
                    &embedded,
                    None,
                    TargetReadout::Logits,
                ),
                &mut ordinary_state,
                &context,
            )
            .unwrap();
        let normalized = hidden
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    tokens,
                    &embedded,
                    None,
                    TargetReadout::NormalizedHidden,
                ),
                &mut hidden_state,
                &context,
            )
            .unwrap();
        assert_eq!(normalized.shape(), &[1, tokens.dim(1), args.hidden_size]);
        let projected = hidden
            .architecture_mut()
            .static_modules_mut()
            .project_normalized(&normalized, &context)
            .unwrap();
        assert_tensor_close(
            &projected,
            &expected,
            "single-pass normalized state projects to ordinary logits",
        );
        let bounded = streamed
            .forward(
                nemotron_h::EmbeddedInput::target_embedded(
                    tokens,
                    &embedded,
                    None,
                    TargetReadout::NormalizedHidden,
                ),
                &mut streamed_state,
                &context,
            )
            .unwrap();
        assert_tensor_close(&bounded, &normalized, "bounded normalized hidden output");
        assert_state_exact(
            &ordinary_state,
            &hidden_state,
            2,
            &format!("same target state after readout {step}"),
        );
        assert_state_exact(
            &ordinary_state,
            &streamed_state,
            2,
            &format!("same bounded target state {step}"),
        );
    }
    assert_eq!(
        context.embedding_forward_calls.get(),
        0,
        "supplied ingress must never re-embed token identities"
    );
    assert_eq!(
        streamed.policy().events.len(),
        3 * expected_stream_events(&[(0, 0), (0, 1)]).len()
    );
}

#[test]
fn nemotron_hidden_readout_crosses_tp2_pp2_boundaries() {
    assert_dense_nemotron_partitioned_readout(TargetReadout::NormalizedHidden);
}

#[test]
fn nemotron_readout_wire_rejects_malformed_or_inexact_controls() {
    let context = NumericContext::default();
    for mode in [TargetReadout::Logits, TargetReadout::NormalizedHidden] {
        let encoded = mode.encode::<NumericTensor>(&context).unwrap();
        assert_eq!(TargetReadout::decode(&encoded, &context).unwrap(), mode);
    }
    for control in [
        NumericTensor::from_i32_slice(&[2], &[1], &context).unwrap(),
        NumericTensor::from_i32_slice(&[0, 1], &[2], &context).unwrap(),
        NumericTensor::new(vec![1], vec![1.0]),
    ] {
        assert!(TargetReadout::decode(&control, &context).is_err());
    }
}
