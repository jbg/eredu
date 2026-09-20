use super::*;

#[test]
fn inkling_wider_histories_preserve_each_typed_missing_component_error() {
    use std::error::Error as _;
    let context = NumericContext::default();
    for kernel in [3, 4] {
        let f = fixture(configuration(false, kernel, false));
        for slot in 0..4 {
            let mut layer =
                inkling::DecoderLayer::<NumericBackend>::new(&f.args.text_config, 0, &context)
                    .unwrap();
            layer.visit_parameters_mut(&mut Populate(&f.parameters, BTreeMap::new()));
            let mut state = state(&f);
            let role = StateTensorRole::Convolution { slot };
            assert!(state.as_mut()[0].fixed.remove(&role).is_some());
            let hidden =
                NumericTensor::new([1, 2, 8], (0..16).map(|i| 0.2 + i as f32 * 0.03).collect());
            let error = layer
                .forward(&hidden, Some(&mut state.as_mut()[0]), &context)
                .unwrap_err();
            let mut cause = error.source();
            let mut found = false;
            while let Some(current) = cause {
                if let Some(StateError::UnknownComponent { role: actual }) =
                    current.downcast_ref::<StateError>()
                {
                    assert_eq!(*actual, role);
                    found = true;
                    break;
                }
                cause = current.source();
            }
            assert!(found, "original typed role failure must survive: {error}");
            // Earlier operators may already have changed their own state. No
            // rollback or all-or-nothing layer execution is claimed here.
        }
    }
}

fn hidden(ids: &[usize]) -> NumericTensor {
    NumericTensor::new(
        [1, ids.len() as i32, 8],
        ids.iter()
            .flat_map(|id| {
                (0..8).map(move |feature| 0.2 + *id as f32 * 0.04 + feature as f32 * 0.015)
            })
            .collect(),
    )
}
fn predict(
    model: &mut Architecture,
    ids: &[usize],
    depth: usize,
    state: &mut State,
    context: &NumericContext,
) -> inkling::PartitionMtpOutput<NumericTensor> {
    let result = model
        .forward_partition_mtp(
            &hidden(ids),
            &NumericTensor::token_ids(ids),
            depth,
            state.as_mut(),
            None,
            context,
        )
        .unwrap();
    nonzero(&result.hidden);
    nonzero(&result.logits);
    assert_tensor_exact(
        &result.tokens,
        &NumericTensor::token_ids(ids),
        "actual prediction tokens",
    );
    result
}

#[test]
fn inkling_width_one_prediction_keeps_kv_without_granting_prediction_rows() {
    let mut config = configuration(false, 3, false);
    config["mtp_config"] = serde_json::json!({"num_nextn_predict_layers":2,"local_layer_ids":[1],"chain_hidden_post_norm":true,"sconv_kernel_size":1});
    let f = fixture(config);
    let context = NumericContext::default();
    let layout = inkling::mtp_state_layout(&f.args).unwrap().unwrap();
    assert_eq!(layout.len(), 2);
    for policy in layout.layers().iter() {
        assert!(policy.attention().is_some());
        assert!(policy.fixed_state().is_empty());
    }
    for policy in inkling::state_layout(&f.args).unwrap().layers().iter() {
        assert_eq!(policy.fixed_state().len(), 4);
    }
    let mut model = Architecture::new(f.args.clone(), &context).unwrap();
    <Architecture as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut model)
        .visit_parameters_mut(&mut Populate(&f.parameters, BTreeMap::new()));
    let declarations=<Architecture as LayeredArchitecture<NumericBackend,State>>::prefill_observation_declarations(&model, None).unwrap();
    assert!(declarations.len() >= 19);
    assert!(declarations
        .iter()
        .all(|d| !d.path().starts_with("model.mtp.")));
    let complete = ArchitectureParameters::state_layout(&model, None).unwrap();
    assert_eq!(complete.len(), 4);
    assert_eq!(complete.segments().len(), 2);
    assert_eq!(
        complete.segments()[0].id().as_str(),
        inkling::TARGET_STATE_SEGMENT
    );
    assert_eq!(
        complete.segments()[1].id().as_str(),
        inkling::PREDICTION_STATE_SEGMENT
    );
    let mut prefix = State::create(layout, |_, policy| {
        Ok::<_, Error>(NumericHybridLayerState::new(policy))
    })
    .unwrap();
    for depth in 0..2 {
        predict(&mut model, &[4, 2], depth, &mut prefix, &context);
    }
    let mut full = prefix.clone();
    let mut split = prefix.clone();
    for depth in 0..2 {
        let expected = predict(&mut model, &[1, 3, 5, 2, 6], depth, &mut full, &context);
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        for ids in [&[1, 3][..], &[5][..], &[2, 6][..]] {
            let output = predict(&mut model, ids, depth, &mut split, &context);
            logits.push(output.logits);
            hidden.push(output.hidden);
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&logits, 1, &context).unwrap(),
            &expected.logits,
            "same prediction convolution equation logits",
        );
        assert_tensor_close(
            &NumericTensor::concatenate(&hidden, 1, &context).unwrap(),
            &expected.hidden,
            "same prediction convolution equation hidden",
        );
        compare_state(&split, &full);
    }
    for id in [7, 8, 9] {
        for depth in 0..2 {
            let a = predict(&mut model, &[id], depth + 2, &mut split, &context);
            let b = predict(&mut model, &[id], depth + 2, &mut full, &context);
            assert_tensor_close(&a.logits, &b.logits, "continued cyclic prediction logits");
            assert_tensor_close(&a.hidden, &b.hidden, "continued cyclic prediction hidden");
            compare_state(&split, &full);
        }
    }
    for layer in split.as_ref() {
        assert!(layer.fixed.is_empty());
        assert_eq!(layer.fixed_offset, 0);
        assert_eq!(layer.position(), 10);
        let cache = layer.attention.as_ref().unwrap();
        nonzero(cache.keys.as_ref().unwrap());
        nonzero(cache.values.as_ref().unwrap());
    }
}
