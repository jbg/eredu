//! Prediction owners, nonzero cached execution, and pre-collapse draft capture.
use super::*;
use eredu_architectures::decoder::ComponentInstrumentation;
use eredu_architectures::qwen4_exp::{
    checkpoint::schema::SafetensorsEncoding,
    config::PredictionGeometry,
    mtp::{PredictionInput, PredictionLimits, PredictionShared, PredictionSpec, PredictionUnit},
    prepared::{PreparedParameters, PreparedPrediction, PreparedTarget},
};
use eredu_runtime::{LayerRuntimeState, RoutedBankId, RuntimeLayerState, RuntimeStateComponents};

fn limits() -> PredictionLimits {
    let target = specification().limits;
    PredictionLimits {
        qsa: target.qsa,
        tile_blocks: target.tile_blocks,
        selection_workspace: target.selection_workspace,
        element: target.element,
    }
}
fn spec(config: &eredu_architectures::qwen4_exp::config::Config) -> PredictionSpec {
    PredictionSpec::from_prepared(
        config,
        limits(),
        RoutedBankId::new(2),
        |depth| {
            let projection = |suffix| {
                eredu_nn::GroupedProjectionSpec::new(
                    ParameterSpec::trainable(format!("mtp.layers.{depth}.mlp.experts.{suffix}"))
                        .unwrap(),
                    None,
                    dense_linear_format(),
                )
            };
            GroupedGatedProductSpec::new(
                config.experts.count,
                config.hidden_size,
                config.experts.intermediate,
                config.hidden_size,
                eredu_nn::GatedProductPolicy::ordinary_silu(),
                eredu_nn::GatedProductGroupLayout::Packed {
                    gate_up: projection("gate_up_proj")?,
                    down: projection("down_proj")?,
                },
            )
        },
        |_| Ok(dense_linear_format()),
    )
    .unwrap()
}
struct Bind<'a> {
    owner: &'a PreparedParameters,
    experts: Vec<PreparedParameters>,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, meta: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = meta.id.as_str();
        let data = if self.owner.recipes().contains_key(name) {
            read_owner_parameter(self.owner, name)
        } else {
            self.experts
                .iter()
                .flat_map(|e| read_owner_parameter(e, name))
                .collect()
        };
        assert_eq!(data.len(), value.data.len(), "{name}");
        value.data = data;
    }
}
fn shared(prepared: &PreparedPrediction, ctx: &NumericContext) -> PredictionShared<NumericBackend> {
    let mut shared = PredictionShared::new(prepared.spec(), ctx).unwrap();
    shared.visit_parameters_mut(&mut Bind {
        owner: prepared.shared(),
        experts: vec![],
    });
    shared
}
fn unit(
    prepared: &PreparedPrediction,
    depth: usize,
    ctx: &NumericContext,
) -> PredictionUnit<NumericBackend> {
    let mut unit = PredictionUnit::new(prepared.spec().units[depth].clone(), ctx).unwrap();
    unit.visit_parameters_mut(&mut Bind {
        owner: prepared.unit(depth).unwrap(),
        experts: (0..3).map(|e| prepared.expert(depth, e).unwrap()).collect(),
    });
    unit
}
fn prediction_state(spec: &PredictionSpec) -> State {
    DeviceState::create(spec.state_layout().unwrap(), |index, policy| {
        let mut layer = NumericHybridLayerState::new(policy);
        if let eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(a) =
            &spec.units[index].mixer
        {
            for stream in a.state.streams() {
                layer.streams.push((
                    stream.slot,
                    0,
                    ResidentAppendStream::new(
                        stream,
                        AppendStreamLimits {
                            entries: 64,
                            page_entries: 2,
                            read_entries: 2,
                        },
                        65536,
                        65536,
                        65536,
                    )
                    .unwrap(),
                ));
            }
        }
        Ok::<_, Error>(layer)
    })
    .unwrap()
}
#[test]
fn qwen4_prepared_prediction_chunk_decode_rollback_and_vocabulary_sharing() {
    std::thread::Builder::new()
        .name("qwen4-prepared-prediction".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(prepared_prediction_case)
        .unwrap()
        .join()
        .unwrap();
}
fn prepared_prediction_case() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut config = configuration();
    config.prediction = Some(PredictionGeometry {
        // The released depth is indexed; a second recurrent fixture exercises
        // independent mixed state without claiming it is a released checkpoint.
        layers: eredu_core::LayerSchedule::new(2, vec![LayerKind::Indexed, LayerKind::Recurrent])
            .unwrap(),
        rope_theta: 7777.,
    });
    let target_spec = specification_for(config.clone());
    let prediction_spec = spec(&config);
    let mut parameters = Parameters::default();
    let mut target =
        TargetModel::<NumericBackend>::new(bind_spec(target_spec.clone()), &ctx).unwrap();
    <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target).visit_parameters_mut(&mut parameters);
    for index in 0..target_spec.units.len() {
        target
            .construct_unit(index, &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    PredictionShared::<NumericBackend>::new(&prediction_spec, &ctx)
        .unwrap()
        .visit_parameters_mut(&mut parameters);
    for unit in &prediction_spec.units {
        PredictionUnit::<NumericBackend>::new(unit.clone(), &ctx)
            .unwrap()
            .visit_parameters_mut(&mut parameters);
    }
    let mut values = table_values();
    // Exercise lexical state numerically in the complete target/MTP lifecycle.
    // The geometry-only table fixture is zero-filled by default.
    for shard in 0..2 {
        let table = values
            .iter_mut()
            .find(|(name, _, _, _)| name.ends_with(&format!("shard_{shard}.weight")))
            .unwrap();
        table.3 = (shard * 12..(shard + 1) * 12)
            .flat_map(|r| [(r as f32 + 0.25) / 17., -(r as f32 + 0.5) / 19.])
            .flat_map(f32::to_le_bytes)
            .collect();
    }
    for (name, value) in &parameters.0 {
        let mut shape: Vec<_> = value.shape.iter().map(|n| *n as usize).collect();
        if name.ends_with(".conv1d.weight") && shape.len() == 2 {
            shape.insert(1, 1);
        }
        values.push((
            name.clone(),
            safetensors::Dtype::F32,
            shape,
            value.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
    }
    let (_directory, source) = super::transforms::fixture(&values);
    let prepared_target = PreparedTarget::safetensors(
        source.clone(),
        config.clone(),
        SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap(),
        target_spec.limits,
    )
    .unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepared = prepared_target.prediction(limits()).unwrap();
    assert_eq!(reads, source.source_diagnostics().unwrap().physical_reads);
    selected::check(&prepared_target, &source, &ctx);
    assert!(prepared.unit(2).is_err());
    assert!(prepared.expert(0, 3).is_err());
    assert!(prepared
        .shared()
        .recipes()
        .keys()
        .all(|name| name.starts_with("mtp.") && !name.starts_with("mtp.layers.")));
    assert!(prepared
        .unit(0)
        .unwrap()
        .source()
        .source_metadata("model.embed_tokens.weight")
        .is_err());
    assert!(prepared
        .vocabulary()
        .source()
        .source_metadata("mtp.fc_hidden.weight")
        .is_err());
    for (name, recipe) in prepared.vocabulary().recipes() {
        assert_eq!(recipe, &prepared_target.static_parameters().recipes()[name]);
    }
    let eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(indexed) =
        &prepared.spec().units[0].mixer
    else {
        panic!()
    };
    assert_eq!(indexed.rotary.base, 7777.);
    assert_eq!(indexed.indexer.rotary.base, 7777.);
    assert_eq!(
        prepared.spec().units[0].feed_forward.feed_forward.bank,
        RoutedBankId::new(2)
    );
    assert_eq!(prepared.spec().state_layout().unwrap().len(), 2);
    assert_ne!(
        prepared.spec().state_layout().unwrap(),
        target_spec.state_layout().unwrap()
    );

    // Shared token lookup and output operator are the same target-owned modules.
    let modules = <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target);
    let tokens: Vec<usize> = (0..35).map(|i| (i % 31) + 1).collect();
    let embedded = modules
        .embeddings
        .forward(&NumericTensor::token_ids(&tokens), &ctx)
        .unwrap();
    let residual = NumericTensor::new(
        [1, 35, 2, 2],
        (0..140)
            .map(|i| ((i * 13 % 41) as f32 - 20.) / 17.)
            .collect(),
    );
    let visible: Vec<bool> = (0..35).map(|i| i != 0 && i != 8).collect();
    let run = |unit: &mut PredictionUnit<NumericBackend>,
               shared: &mut PredictionShared<NumericBackend>,
               state: &mut NumericHybridLayerState,
               range: std::ops::Range<usize>| {
        let embedding = embedded.axis_slice(1, range.start, range.end);
        let residual = residual.axis_slice(1, range.start, range.end);
        unit.forward(
            shared,
            PredictionInput {
                embeddings: &embedding,
                residual: &residual,
                visible: Some(&visible[range.clone()]),
                rotary: None,
            },
            state,
            &mut ResidentExpertProvider,
            &ctx,
            &mut ComponentInstrumentation::disabled(),
        )
        .unwrap()
    };
    for depth in 0..2 {
        let mut whole_state = prediction_state(prepared.spec());
        let mut chunk_state = prediction_state(prepared.spec());
        let mut whole_unit = unit(&prepared, depth, &ctx);
        let mut whole_shared = shared(&prepared, &ctx);
        let expected = run(
            &mut whole_unit,
            &mut whole_shared,
            whole_state.layer(depth).unwrap(),
            0..19,
        );
        let mut captures = Vec::new();
        let mut outputs = Vec::new();
        let mut shared = shared(&prepared, &ctx);
        // Reacquire one unit from its restricted source each time, modeling
        // independent lifetime of shared weights and layer-sized materialization.
        for range in [0..3, 3..8, 8..19] {
            let actual = run(
                &mut unit(&prepared, depth, &ctx),
                &mut shared,
                chunk_state.layer(depth).unwrap(),
                range,
            );
            captures.push(actual.capture);
            outputs.push(actual.hidden);
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&outputs, 1, &ctx).unwrap(),
            &expected.hidden,
            "prediction chunked collapse",
        );
        assert_tensor_close(
            &NumericTensor::concatenate(&captures, 1, &ctx).unwrap(),
            &expected.capture,
            "prediction chunked pre-collapse capture",
        );
        assert_eq!(expected.hidden.shape, [1, 19, 2]);
        assert_eq!(expected.capture.shape, [1, 19, 2, 2]);
        let logits = modules
            .lm_head
            .as_mut()
            .unwrap()
            .forward(&expected.hidden, &ctx)
            .unwrap();
        assert!(logits.data.iter().all(|v| v.is_finite()));
        assert!(logits.data.iter().any(|v| v.abs() > 1e-4));
        for index in 19..35 {
            let expected = run(
                &mut whole_unit,
                &mut whole_shared,
                whole_state.layer(depth).unwrap(),
                index..index + 1,
            );
            let actual = run(
                &mut unit(&prepared, depth, &ctx),
                &mut shared,
                chunk_state.layer(depth).unwrap(),
                index..index + 1,
            );
            assert_tensor_close(
                &actual.hidden,
                &expected.hidden,
                "prediction cached collapse",
            );
            assert_tensor_close(
                &actual.capture,
                &expected.capture,
                "prediction cached capture",
            );
        }
        // Reject malformed visibility and oversized chunks before touching state.
        let before = chunk_state.clone();
        let mut rejected = unit(&prepared, depth, &ctx);
        let mut bounded = prepared.spec().clone();
        bounded.limits.qsa.tokens = 19;
        let mut bounded_shared = PredictionShared::<NumericBackend>::new(&bounded, &ctx).unwrap();
        for (embedding, residual, visibility) in [
            (
                embedded.axis_slice(1, 0, 1),
                residual.axis_slice(1, 0, 1),
                vec![],
            ),
            (embedded.clone(), residual.clone(), visible.clone()),
        ] {
            assert!(rejected
                .forward(
                    &mut bounded_shared,
                    PredictionInput {
                        embeddings: &embedding,
                        residual: &residual,
                        visible: Some(&visibility),
                        rotary: None,
                    },
                    chunk_state.layer(depth).unwrap(),
                    &mut ResidentExpertProvider,
                    &ctx,
                    &mut ComponentInstrumentation::disabled(),
                )
                .is_err());
        }
        let mut before = before;
        let prior = before.layer(depth).unwrap();
        let after = chunk_state.layer(depth).unwrap();
        assert_eq!(prior.position(), after.position());
        let prior: Vec<_> = RuntimeLayerState::retained_values(prior).collect();
        let after: Vec<_> = RuntimeLayerState::retained_values(after).collect();
        assert_eq!(prior.len(), after.len());
        for (a, b) in prior.into_iter().zip(after) {
            assert_tensor_exact(a, b, "prediction admission preserves state");
        }
        // State contents are compared at the established numerical tolerance;
        // exact IDs/positions are asserted separately for chunked arithmetic.
        let a = chunk_state.layer(depth).unwrap();
        let b = whole_state.layer(depth).unwrap();
        assert_eq!(a.position(), b.position());
        let a: Vec<_> = RuntimeLayerState::retained_values(a).collect();
        let b: Vec<_> = RuntimeLayerState::retained_values(b).collect();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.into_iter().zip(b) {
            assert_tensor_close(a, b, "prediction state");
            assert_eq!(a.exact_i32, b.exact_i32);
        }
        let saved = chunk_state.clone();
        let reads_before = source.source_diagnostics().unwrap().physical_reads;
        let proposed = run(
            &mut unit(&prepared, depth, &ctx),
            &mut shared,
            chunk_state.layer(depth).unwrap(),
            34..35,
        );
        let charged = source.source_diagnostics().unwrap().physical_reads;
        assert!(charged > reads_before);
        chunk_state = saved;
        assert_eq!(charged, source.source_diagnostics().unwrap().physical_reads);
        let replay = run(
            &mut unit(&prepared, depth, &ctx),
            &mut shared,
            chunk_state.layer(depth).unwrap(),
            34..35,
        );
        assert_tensor_exact(
            &proposed.capture,
            &replay.capture,
            "prediction rollback capture",
        );
        assert_tensor_exact(
            &proposed.hidden,
            &replay.hidden,
            "prediction rollback collapse",
        );
    }
}

#[test]
fn qwen4_prediction_spec_rejects_invalid_ownership_and_geometry() {
    let mut config = configuration();
    config.prediction = Some(PredictionGeometry {
        layers: eredu_core::LayerSchedule::new(1, vec![LayerKind::Indexed]).unwrap(),
        rope_theta: 7777.,
    });
    let valid = spec(&config);
    for mutate in [
        |s: &mut PredictionSpec| s.units[0].depth = 1,
        |s: &mut PredictionSpec| s.units[0].feed_forward.feed_forward.layer = 1,
        |s: &mut PredictionSpec| s.units.clear(),
        |s: &mut PredictionSpec| s.limits.qsa.tokens = 0,
        |s: &mut PredictionSpec| {
            s.readout.geometry =
                eredu_nn::residual_streams::ResidualStreamGeometry::new(3, 2).unwrap()
        },
    ] {
        let mut invalid = valid.clone();
        mutate(&mut invalid);
        assert!(invalid.state_layout().is_err());
        assert!(
            PredictionShared::<NumericBackend>::new(&invalid, &NumericContext::default()).is_err()
        );
    }
}

#[path = "prediction/executor.rs"]
mod executor;

#[path = "prediction/selected.rs"]
mod selected;

#[path = "prediction/installed.rs"]
mod installed;

#[path = "prediction/driver.rs"]
mod driver;

#[path = "prediction/sampling.rs"]
mod sampling;

#[path = "prediction/snapshots.rs"]
mod snapshots;

#[path = "prediction/resources.rs"]
mod resources;

#[path = "prediction/separate_source.rs"]
mod separate_source;

#[path = "prediction/ordinary.rs"]
mod ordinary;

#[path = "prediction/parallel.rs"]
mod parallel;

#[path = "prediction/ordinary_partition.rs"]
mod ordinary_partition;

#[path = "prediction/ordinary_partition_cold.rs"]
mod ordinary_partition_cold;
