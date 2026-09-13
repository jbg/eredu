//! Prediction fusion and scheduled operators consume the effective observations.
use super::*;

#[derive(Default)]
struct Observer {
    values: BTreeMap<String, NumericTensor>,
    zero: Option<String>,
}

impl ActivationObserver<NumericTensor, Error> for Observer {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        assert!(
            self.values.insert(path.into(), value.clone()).is_none(),
            "duplicate {path}"
        );
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        Ok((self.zero.as_deref() == Some(path)).then(|| {
            let mut changed = value.clone();
            let width = *value.shape.last().unwrap() as usize;
            // Change only component zero at the final physical sequence position.
            let index = value.data.len() - width;
            changed.data[index] = 0.0;
            changed
        }))
    }
}

fn trial(
    args: &nemotron_h::ModelArgs,
    depth: usize,
    mode: usize,
    parallel: bool,
    observed: bool,
) -> Vec<(NumericTensor, BTreeMap<String, NumericTensor>)> {
    let context = NumericContext::default();
    let parallel = parallel.then(|| NumericParallelContext::new(0, NumericParallelGroup::new(1)));
    let mut units = (0..2)
        .map(|relative| {
            let mut unit =
                nemotron_h::PredictionUnit::<NumericBackend>::new(args, depth, relative, &context)
                    .unwrap();
            if let nemotron_h::Operator::Sparse(moe) = &mut unit.block.operator {
                let read = &mut moe.shared_experts.up_proj;
                // Keep every shared ReLU² unit active with distinguishable reads.
                for (index, value) in read.weight.data.iter_mut().enumerate() {
                    *value = 0.005 * (1 + index % 4) as f32;
                }
                read.bias.as_mut().unwrap().0.data.fill(1.0);
            }
            unit
        })
        .collect::<Vec<_>>();
    let mut state = DeviceState::<NumericBackend, _>::create(
        nemotron_h::state_layout(args).unwrap(),
        |_, policy| Ok::<_, Error>(NumericHybridLayerState::new(policy)),
    )
    .unwrap();
    let first = format!("model.mtp.layers.{}", depth * 2);
    let last = format!("model.mtp.layers.{}", depth * 2 + 1);
    let zero = match mode {
        0 => None,
        1 => Some(format!("{first}.prediction.embedding.normalized")),
        2 => Some(format!("{first}.prediction.hidden.normalized")),
        3 => Some(format!("{first}.prediction.fusion.output")),
        4 => Some(format!("{first}.attention.channels")),
        5 => Some(format!("{last}.shared.feed_forward.units")),
        6 => Some(format!("{last}.prediction.readout.normalized")),
        _ => unreachable!(),
    };
    let mut outputs = Vec::new();
    for (step, sequence) in [2, 1, 1].into_iter().enumerate() {
        let mut hidden = NumericTensor::new(
            [1, sequence, 8],
            (0..sequence * 8)
                .map(|i| ((i * 7 + step as i32 * 3) % 23) as f32 / 13.0 - 0.5)
                .collect(),
        );
        let embedded = NumericTensor::new(
            [1, sequence, 8],
            (0..sequence * 8)
                .map(|i| ((i * 11 + step as i32 * 5) % 29) as f32 / 17.0 - 0.4)
                .collect(),
        );
        let position = state.layer(1 + depth * 2).unwrap().position();
        let mask = (sequence > 1)
            .then(|| NumericBackend::causal_mask(sequence, position, None, &context).unwrap());
        let mut observer = Observer {
            zero: zero.clone(),
            ..Default::default()
        };
        for (relative, unit) in units.iter_mut().enumerate() {
            let layer = &mut state.as_mut()[1 + depth * 2 + relative];
            let path = format!("model.mtp.layers.{}", depth * 2 + relative);
            let mut provider = eredu_runtime::ResidentExpertProvider;
            hidden = match (parallel.as_ref(), observed) {
                (None, false) => unit.forward(&hidden, &embedded, mask.as_ref(), layer, &context),
                (Some(parallel), false) => unit.forward_parallel_with_provider(
                    &hidden,
                    &embedded,
                    mask.as_ref(),
                    layer,
                    parallel,
                    &context,
                    &mut provider,
                ),
                (None, true) => unit.forward_observed_with_provider(
                    &path,
                    args.n_routed_experts,
                    &hidden,
                    &embedded,
                    mask.as_ref(),
                    layer,
                    &context,
                    &mut provider,
                    &mut observer,
                ),
                (Some(parallel), true) => unit.forward_parallel_observed_with_provider(
                    &path,
                    args.n_routed_experts,
                    &hidden,
                    &embedded,
                    mask.as_ref(),
                    layer,
                    parallel,
                    &context,
                    &mut provider,
                    &mut observer,
                ),
            }
            .unwrap();
        }
        if observed {
            let fusion = units[0].fusion.as_ref().unwrap();
            let input = &observer.values[&format!("{first}.prediction.fusion.input")];
            let expected = linear(input, &fusion.weight, None).unwrap();
            assert_tensor_close(
                &observer.values[&format!("{first}.prediction.fusion.output")],
                &expected,
                "Nemotron prediction fusion reconstruction",
            );
            if let Some(path) = &zero {
                let before = &observer.values[path];
                let after = &observer.values[&format!("{path}.effective")];
                let changed = before.data.len() - *before.shape.last().unwrap() as usize;
                assert_ne!(before.data[changed], 0.0, "nonzero fixture {path}");
                for (index, (&before, &after)) in before.data.iter().zip(&after.data).enumerate() {
                    assert_eq!(after, if index == changed { 0.0 } else { before });
                }
            }
        }
        outputs.push((hidden, observer.values));
    }
    assert_eq!(
        state.layer(0).unwrap().position(),
        0,
        "target state is untouched"
    );
    for other in 0..args.num_nextn_predict_layers as usize {
        assert_eq!(
            state.layer(1 + other * 2).unwrap().position(),
            if other == depth { 4 } else { 0 }
        );
        assert_eq!(state.layer(2 + other * 2).unwrap().position(), 0);
    }
    outputs
}

#[test]
fn nemotron_prediction_fusion_and_components_share_serial_parallel_execution() {
    let args = nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":7, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":1,
        "hybrid_override_pattern":"*", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":4,
        "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":2,
        "mtp_hybrid_override_pattern":"*E", "tie_word_embeddings":false, "mlp_bias":true
    }))
    .unwrap();
    for depth in 0..2 {
        let ordinary = trial(&args, depth, 0, false, false);
        for mode in 0..7 {
            let serial = trial(&args, depth, mode, false, true);
            let parallel = trial(&args, depth, mode, true, true);
            for ((serial, parallel), ordinary) in serial.iter().zip(&parallel).zip(&ordinary) {
                assert_tensor_close(
                    &serial.0,
                    &parallel.0,
                    "Nemotron observed serial/TP1 output",
                );
                assert!(
                    serial.1.keys().eq(parallel.1.keys()),
                    "same observed boundaries"
                );
                for (path, value) in &serial.1 {
                    assert_tensor_close(value, &parallel.1[path], "serial/TP1 component evidence");
                }
                if mode == 0 {
                    assert_tensor_exact(&serial.0, &ordinary.0, "disabled/no-op prediction parity");
                } else {
                    assert_ne!(
                        serial.0.data, ordinary.0.data,
                        "causal prediction mask {mode}"
                    );
                }
            }
        }
        for (parallel, ordinary) in trial(&args, depth, 0, true, false).iter().zip(&ordinary) {
            assert_tensor_close(
                &parallel.0,
                &ordinary.0,
                "disabled serial/TP1 prediction parity",
            );
        }
    }
}
