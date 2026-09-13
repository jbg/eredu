//! Derived normalization values must follow parameter replacement and restoration.
use super::*;

#[test]
fn muse_centered_normalizations_follow_replacements_after_warm_execution() {
    let args = muse_glimmer::DecoderConfig::from_hf_value(&dense_muse_partition_fixture()).unwrap();
    let context = NumericContext::default();
    let template =
        muse_glimmer::TransformerBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
    let inputs = [0.3, 1.7].map(|phase| {
        NumericTensor::new(
            [1, 3, 8],
            (0..24).map(|i| (i as f32 * 0.37 + phase).sin()).collect(),
        )
    });
    for target in 0..4 {
        let mut warmed = template.clone();
        let mut fresh = template.clone();
        warmed
            .forward(&inputs[0], None, None::<&mut NumericCache>, &context)
            .unwrap();
        let slot = |block: &mut muse_glimmer::TransformerBlock<NumericBackend>,
                    values: NumericTensor| {
            match target {
                0 => &mut block.input_norm.weight,
                1 => &mut block.post_attention_norm.weight,
                2 => &mut block.pre_feed_forward_norm.weight,
                3 => &mut block.post_feed_forward_norm.weight,
                _ => unreachable!(),
            }
            .replace(values);
        };
        let original = match target {
            0 => &template.input_norm.weight,
            1 => &template.post_attention_norm.weight,
            2 => &template.pre_feed_forward_norm.weight,
            3 => &template.post_feed_forward_norm.weight,
            _ => unreachable!(),
        }
        .as_ref()
        .clone();
        let edited = NumericTensor::new([8], (0..8).map(|i| (i as f32 - 2.5) * 0.23).collect());
        slot(&mut warmed, edited.clone());
        slot(&mut fresh, edited);
        for input in &inputs {
            let mut baseline = template.clone();
            let baseline = baseline
                .forward(input, None, None::<&mut NumericCache>, &context)
                .unwrap();
            let expected = fresh
                .forward(input, None, None::<&mut NumericCache>, &context)
                .unwrap();
            assert!(expected
                .data
                .iter()
                .zip(&baseline.data)
                .any(|(a, b)| (a - b).abs() > 1e-4));
            let actual = warmed
                .forward(input, None, None::<&mut NumericCache>, &context)
                .unwrap();
            assert_tensor_close(
                &actual,
                &expected,
                &format!("Muse warmed normalization edit {target}"),
            );
        }
        slot(&mut warmed, original);
        for input in &inputs {
            let mut baseline = template.clone();
            let expected = baseline
                .forward(input, None, None::<&mut NumericCache>, &context)
                .unwrap();
            let actual = warmed
                .forward(input, None, None::<&mut NumericCache>, &context)
                .unwrap();
            assert_tensor_close(
                &actual,
                &expected,
                &format!("Muse normalization restoration {target}"),
            );
        }
    }
}

fn observed_block(
    block: &mut muse_glimmer::TransformerBlock<NumericBackend>,
    input: &NumericTensor,
    capture: &mut Components,
) -> NumericTensor {
    block
        .forward_observed(
            input,
            None,
            None::<&mut NumericCache>,
            &NumericContext::default(),
            &mut ComponentInstrumentation::new("model.layers.0", capture),
        )
        .unwrap()
}

#[test]
fn muse_components_reconstruct_post_normalized_writes_and_recompute_masked_survivors() {
    for convention in [
        muse_glimmer::WeightConvention::HuggingFace,
        muse_glimmer::WeightConvention::Gguf,
    ] {
        let mut args =
            muse_glimmer::DecoderConfig::from_hf_value(&dense_muse_partition_fixture()).unwrap();
        args.weight_convention = convention;
        let context = NumericContext::default();
        let mut block =
            muse_glimmer::TransformerBlock::<NumericBackend>::new(&args, 0, &context).unwrap();
        // Store identical nonzero effective gains under both released conventions.
        let offset = if convention == muse_glimmer::WeightConvention::HuggingFace {
            1.0
        } else {
            0.0
        };
        for norm in [
            &mut block.input_norm.weight,
            &mut block.post_attention_norm.weight,
            &mut block.pre_feed_forward_norm.weight,
            &mut block.post_feed_forward_norm.weight,
        ] {
            norm.replace(NumericTensor::new(
                [8],
                (0..8).map(|i| 0.9 + i as f32 * 0.04 - offset).collect(),
            ));
        }
        let input = NumericTensor::new(
            [1, 3, 8],
            (0..24).map(|i| (i as f32 * 0.37 + 0.3).sin()).collect(),
        );
        let ordinary = block
            .clone()
            .forward(&input, None, None::<&mut NumericCache>, &context)
            .unwrap();
        let mut baseline = Components::strict();
        let observed = observed_block(&mut block.clone(), &input, &mut baseline);
        assert_tensor_exact(&ordinary, &observed, "Muse no-op components");
        let muse_glimmer::FeedForward::Dense(mlp) = &block.feed_forward else {
            unreachable!()
        };
        for (boundary, projection, norm) in [
            (
                "attention",
                &block.attention.output,
                &block.post_attention_norm.weight,
            ),
            (
                "feed_forward",
                &mlp.down,
                &block.post_feed_forward_norm.weight,
            ),
        ] {
            let key = |suffix: &str| format!("model.layers.0.{boundary}.{suffix}");
            let values = &baseline.values[&key("write_input")];
            assert!(values.data.iter().any(|x| x.abs() > 1e-4));
            let width = values.shape[2] as usize;
            let mut writes = vec![0.0; 24];
            for token in 0..3 {
                for out in 0..8 {
                    writes[token * 8 + out] = (0..width)
                        .map(|i| {
                            f64::from(values.data[token * width + i])
                                * f64::from(projection.weight.data[out * width + i])
                        })
                        .sum::<f64>() as f32;
                }
            }
            assert_tensor_close(
                &NumericTensor::new([1, 3, 8], writes.clone()),
                &baseline.values[&key("write")],
                "Muse component sum before postnorm",
            );
            for row in writes.chunks_mut(8) {
                let denominator = (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 8.0
                    + f64::from(args.post_norm_eps))
                .sqrt();
                for (i, value) in row.iter_mut().enumerate() {
                    let gain = norm.as_ref().data[i]
                        + if convention == muse_glimmer::WeightConvention::HuggingFace {
                            1.0
                        } else {
                            0.0
                        };
                    *value = (f64::from(*value) * f64::from(gain) / denominator) as f32;
                }
            }
            assert_tensor_close(
                &NumericTensor::new([1, 3, 8], writes),
                &baseline.values[&key("output")],
                "Muse gain and epsilon after projection",
            );
        }
        let reconstructed = input
            .add(
                &baseline.values["model.layers.0.attention.output"],
                &context,
            )
            .unwrap()
            .add(
                &baseline.values["model.layers.0.feed_forward.output"],
                &context,
            )
            .unwrap();
        assert_tensor_exact(&reconstructed, &observed, "Muse residual reconstruction");
        for target in [
            "model.layers.0.attention.channels",
            "model.layers.0.feed_forward.units",
        ] {
            let mut trial = Components {
                zero: Some((target, 1, 2)),
                reject_duplicates: true,
                ..Default::default()
            };
            let changed = observed_block(&mut block.clone(), &input, &mut trial);
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
        let mut trial = Components {
            zero: Some(("model.layers.0.attention.channels", 2, 1)),
            keep: Some(("model.layers.0.feed_forward.units", 2, 3)),
            reject_duplicates: true,
            ..Default::default()
        };
        observed_block(&mut block.clone(), &input, &mut trial);
        let units = &trial.values["model.layers.0.feed_forward.units"];
        let effective = &trial.values["model.layers.0.feed_forward.units.effective"];
        let width = units.shape[2] as usize;
        assert_ne!(
            units.data[2 * width + 3],
            baseline.values["model.layers.0.feed_forward.units"].data[2 * width + 3]
        );
        for i in 0..width {
            assert_eq!(
                effective.data[2 * width + i],
                if i == 3 {
                    units.data[2 * width + i]
                } else {
                    0.0
                }
            );
        }
    }
}

#[test]
fn muse_readout_preserves_nonunit_multiplier_after_the_actual_projection() {
    for tied in [false, true] {
        let mut config = dense_muse_partition_fixture();
        config["text_config"]["output_multiplier"] = 1.9.into();
        config["text_config"]["tie_word_embeddings"] = tied.into();
        let args = muse_glimmer::DecoderConfig::from_hf_value(&config).unwrap();
        let context = NumericContext::default();
        type Model = muse_glimmer::LayeredModel<NumericBackend>;
        type State = DeviceState<NumericBackend, NumericHybridLayerState>;
        let model = Model::new(args.clone(), &context).unwrap();
        for hooks in [
            <Model as LayeredArchitecture<NumericBackend, State>>::observation_hooks(&model),
            <Model as ParallelLayeredArchitecture<NumericBackend, State>>::parallel_observation_hooks(&model),
            <Model as eredu_runtime::PartitionedLayeredArchitecture<NumericBackend, State>>::partition_observation_hooks(&model, true),
        ] {
            for site in [eredu_runtime::inspection::ObservationHookSite::Input,
                         eredu_runtime::inspection::ObservationHookSite::Unit,
                         eredu_runtime::inspection::ObservationHookSite::Readout] {
                assert!(hooks.supports(site));
            }
        }

        let tokens = NumericTensor::token_ids(&[1, 4, 2]);
        let parts = [muse_glimmer::DecoderInputPart::Text(&tokens)];
        let run = |capture: &mut Components| {
            let architecture =
                muse_glimmer::LayeredModel::<NumericBackend>::new(args.clone(), &context).unwrap();
            let mut runtime = LayerwiseRuntime::new(architecture, RebuildingUnitPolicy::default());
            let mut state = DeviceState::<NumericBackend, _>::create(
                muse_glimmer::state_layout(&args).unwrap(),
                |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
            )
            .unwrap();
            runtime
                .forward_with_observer(
                    muse_glimmer::ModelInput {
                        parts: &parts,
                        vision: None,
                        mask: None,
                    },
                    &mut state,
                    &context,
                    capture,
                )
                .unwrap()
        };
        let mut baseline = Components::strict();
        let logits = run(&mut baseline);
        assert_tensor_exact(
            &baseline.values["readout.embedding.effective"],
            &baseline.values["model.layers.0.input"],
            "Muse assembled ingress",
        );
        assert_tensor_exact(
            &baseline.values["readout.normalized.effective"],
            &baseline.values["readout.projection_input"],
            "Muse actual F32 head input",
        );
        let expected = baseline.values["readout.linear"]
            .clone()
            .multiply_scalar(
                args.output_multiplier / args.final_logit_softcapping,
                &context,
            )
            .unwrap()
            .tanh(&context)
            .unwrap()
            .multiply_scalar(args.final_logit_softcapping, &context)
            .unwrap();
        assert_tensor_exact(&expected, &logits, "Muse readout scale then softcap");
        let mut trial = Components {
            keep: Some(("readout.normalized", 1, usize::MAX)),
            reject_duplicates: true,
            ..Default::default()
        };
        let changed = run(&mut trial);
        assert_tensor_exact(
            &baseline.values["readout.normalized"],
            &trial.values["readout.normalized"],
            "Muse original readout evidence",
        );
        let width = logits.shape[2] as usize;
        assert!(logits.data[width..2 * width].iter().any(|x| x.abs() > 1e-5));
        assert!(changed.data[width..2 * width].iter().all(|x| *x == 0.0));
        assert_eq!(&changed.data[..width], &logits.data[..width]);
        assert_eq!(&changed.data[2 * width..], &logits.data[2 * width..]);
    }
}
