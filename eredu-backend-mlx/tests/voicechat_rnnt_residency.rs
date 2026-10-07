//! RNNT providers preserve recurrent state and greedy feedback after checkpoint conversion.
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntError, RnntExecution, RnntLimits, RnntState},
    rnnt_preparation::PreparedRnntBindings,
};
use eredu_backend_mlx::{
    backend::runtime::checkpoint::binding::binding_bytes,
    native::{materialize_voicechat_rnnt, MlxRnnt},
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorSelection},
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{RetainedTensorRoots, Tensor, TensorElementType};
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions};
use safemlx::{Array, Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::Value;
use std::sync::Arc;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn dimensions(fixture: &Value) -> RnntDimensions {
    let v = &fixture["geometry"];
    let i = |key: &str| v[key].as_i64().unwrap() as i32;
    RnntDimensions {
        vocab_size: i("vocab_size"),
        predictor_width: i("prediction_hidden"),
        predictor_layers: i("prediction_layers") as usize,
        encoder_width: i("encoder_hidden"),
        joint_width: i("joint_hidden"),
    }
}
fn tensor(value: &Value, width: i32) -> MlxTensor {
    MlxTensor::from_array(Array::from_slice(&floats(value), &[1, width]))
}
fn checkpoint(fixture: &Value, f16: bool, path: &std::path::Path) -> SharedCheckpointSource {
    let mut values = fixture["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let shape = value["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let bytes = if f16 {
                floats(&value["values"])
                    .into_iter()
                    .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                floats(&value["values"])
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect()
            };
            (name.clone(), shape, bytes)
        })
        .collect::<Vec<_>>();
    values.push((
        "unrelated.weight".into(),
        vec![1],
        vec![0; if f16 { 2 } else { 4 }],
    ));
    let dtype = if f16 { Dtype::F16 } else { Dtype::F32 };
    serialize_to_file(
        values.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        path,
    )
    .unwrap();
    Arc::new(SafetensorsWeightStore::open(path).unwrap())
}
fn prepared(dimensions: RnntDimensions, source: SharedCheckpointSource) -> PreparedRnntBindings {
    let recipes = source
        .source_keys()
        .into_iter()
        .map(|name| {
            (
                name.clone(),
                DerivedWeightRecipe::Source {
                    key: name,
                    selection: TensorSelection::Full,
                },
            )
        })
        .collect();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let prepared =
        PreparedRnntBindings::from_recipes(dimensions, source.clone(), &recipes).unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "RNNT preparation reads headers only"
    );
    prepared
}
fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, stream: &Stream) {
    let actual = actual.to_f32_vec(stream).unwrap();
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= abs + rel * expected.abs(),
            "{index}: {actual} versus pinned NVIDIA {expected}"
        );
    }
}
fn compare_state(state: &RnntState<MlxTensor>, expected: &Value, fixture: &Value, stream: &Stream) {
    for (index, layer) in state
        .predictor()
        .unwrap()
        .local_state()
        .layers
        .iter()
        .enumerate()
    {
        let width = layer.hidden.shape()[1] as usize;
        for (field, value) in [("h", &layer.hidden), ("c", &layer.cell)] {
            compare(
                value,
                &Value::Array(
                    expected[field].as_array().unwrap()[index * width..(index + 1) * width]
                        .to_vec(),
                ),
                fixture,
                stream,
            );
        }
    }
}
type Retained = Vec<(Vec<i32>, TensorElementType, Vec<f32>)>;
fn retained(state: &RnntState<MlxTensor>, stream: &Stream) -> Retained {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        values.push((
            tensor.shape().to_vec(),
            tensor.element_type().unwrap(),
            tensor.to_f32_vec(stream).unwrap(),
        ))
    });
    values
}
#[derive(Debug, PartialEq)]
struct Trace {
    states: Vec<(Retained, Vec<usize>, u64)>,
    logits: Vec<Vec<f32>>,
    decisions: Vec<(Vec<usize>, bool, bool, bool)>,
}
fn record(trace: &mut Trace, state: &RnntState<MlxTensor>, stream: &Stream) {
    trace.states.push((
        retained(state, stream),
        state.hypothesis().to_vec(),
        state.frames(),
    ));
}
fn run(model: &mut MlxRnnt, fixture: &Value, f16: bool, stream: &Stream) -> Trace {
    let dimensions = model.dimensions();
    let dtype = TensorElementType::F32;
    let initial = RnntState::default();
    let reads = model
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let invalid = MlxTensor::full_f32(0., &[1, dimensions.encoder_width + 1], stream).unwrap();
    let limits = RnntLimits {
        max_symbols_per_frame: 3,
        hypothesis_capacity: 32,
    };
    assert!(matches!(
        model.decode_frame(&invalid, &initial, limits, stream),
        Err(RnntError::InvalidState(_))
    ));
    assert!(matches!(
        model.predict_token(dimensions.vocab_size as usize, &initial, 32, stream),
        Err(RnntError::InvalidState(_))
    ));
    assert!(matches!(
        model.predict_token(0, &initial, 0, stream),
        Err(RnntError::Capacity { capacity: 0 })
    ));
    let frame = tensor(
        &fixture["streaming_frames"][0]["encoder"],
        dimensions.encoder_width,
    );
    assert!(matches!(
        model.decode_frame(
            &frame,
            &initial,
            RnntLimits {
                max_symbols_per_frame: 0,
                ..limits
            },
            stream
        ),
        Err(RnntError::InvalidConfiguration(_))
    ));
    assert_eq!(
        model
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads,
        "invalid ingress acquires no RNNT units"
    );
    let primed = model.prime(&initial, stream).unwrap();
    assert!(retained(&initial, stream).is_empty());
    assert!(retained(&primed, stream)
        .iter()
        .all(|(_, actual, _)| *actual == dtype));
    if !f16 {
        compare_state(&primed, &fixture["predictor_steps"][0], fixture, stream);
    }
    let reads = model
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let unchanged = model.prime(&primed, stream).unwrap();
    assert_eq!(retained(&unchanged, stream), retained(&primed, stream));
    assert_eq!(
        model
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads,
        "already primed state leases no parameters"
    );
    let mut trace = Trace {
        states: Vec::new(),
        logits: Vec::new(),
        decisions: Vec::new(),
    };
    record(&mut trace, &primed, stream);
    let mut state = primed.clone();
    let mut hypothesis = Vec::new();
    for step in fixture["predictor_steps"]
        .as_array()
        .unwrap()
        .iter()
        .skip(1)
    {
        let token = step["token"].as_u64().unwrap() as usize;
        let before = retained(&state, stream);
        let next = model.predict_token(token, &state, 32, stream).unwrap();
        assert_eq!(retained(&state, stream), before);
        hypothesis.push(token);
        assert_eq!(next.hypothesis(), hypothesis);
        assert_eq!(next.frames(), 0);
        if !f16 {
            compare_state(&next, step, fixture, stream);
        }
        record(&mut trace, &next, stream);
        state = next;
    }
    // Exhaustion occurs after a prediction has run; the immutable parent and the
    // provider remain usable for the ordinary retry below.
    let nonblank = tensor(
        &fixture["streaming_frames"][1]["encoder"],
        dimensions.encoder_width,
    );
    let before = retained(&primed, stream);
    assert!(matches!(
        model.decode_frame(
            &nonblank,
            &primed,
            RnntLimits {
                hypothesis_capacity: 1,
                ..limits
            },
            stream
        ),
        Err(RnntError::Capacity { capacity: 1 })
    ));
    assert_eq!(retained(&primed, stream), before);
    for case in ["streaming_frames", "feedback_terminates_frame"] {
        let mut state = primed.clone();
        let mut hypothesis = Vec::new();
        for (index, frame) in fixture[case].as_array().unwrap().iter().enumerate() {
            let input = tensor(&frame["encoder"], dimensions.encoder_width);
            let logits = model.joint_logits(&input, &state, stream).unwrap();
            assert_eq!(logits.element_type(), Some(dtype));
            if !f16 {
                compare(&logits, &frame["joint_calls"][0]["logits"], fixture, stream);
            }
            trace.logits.push(logits.to_f32_vec(stream).unwrap());
            let before = retained(&state, stream);
            let candidate = model.decode_frame(&input, &state, limits, stream).unwrap();
            assert_eq!(retained(&state, stream), before);
            let replay = model
                .decode_frame(&input, &state.clone(), limits, stream)
                .unwrap();
            assert_eq!(candidate.emitted, replay.emitted);
            assert_eq!(
                retained(&candidate.state, stream),
                retained(&replay.state, stream)
            );
            if !f16 {
                let expected = frame["emitted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(candidate.emitted, expected);
                assert_eq!(
                    candidate.first_blank,
                    frame["first_blank"].as_bool().unwrap()
                );
                assert_eq!(
                    candidate.ended_blank,
                    frame["ended_blank"].as_bool().unwrap()
                );
                assert_eq!(
                    candidate.hit_symbol_limit,
                    frame["hit_symbol_limit"].as_bool().unwrap()
                );
                compare_state(&candidate.state, frame, fixture, stream);
            }
            hypothesis.extend_from_slice(&candidate.emitted);
            assert_eq!(candidate.state.hypothesis(), hypothesis);
            assert_eq!(candidate.state.frames(), index as u64 + 1);
            trace.decisions.push((
                candidate.emitted,
                candidate.first_blank,
                candidate.ended_blank,
                candidate.hit_symbol_limit,
            ));
            record(&mut trace, &candidate.state, stream);
            state = candidate.state;
        }
    }
    trace
}

#[test]
fn native_rnnt_residency_preserves_nvidia_states_feedback_and_converted_weights() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::RNNT_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(format!("rnnt_{f16}.safetensors")),
        );
        let (_, _, layout, restricted, units) = prepared(dimensions, source.clone()).into_parts();
        assert_eq!(
            restricted.source_keys().len(),
            fixture["parameters"].as_object().unwrap().len()
        );
        assert!(matches!(
            restricted.source_metadata("unrelated.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let bytes = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect::<Vec<_>>();
        let expected = [32, 192, 192, 48, 36, 64];
        assert_eq!(
            bytes, expected,
            "embedding, per-layer recurrence and three joint projections use prepared F32 arithmetic"
        );
        let largest = *bytes.iter().max().unwrap();
        let depth = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
        )
        .device_depth(units.len());
        let disk_budget = (0..layout.group_count())
            .flat_map(|group| {
                let range = layout.group_range(group).unwrap();
                bytes[range.clone()]
                    .windows(depth.min(range.len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap();
        let mut baseline = None;
        for mode in if f16 { vec![0, 2] } else { vec![0, 1, 2] } {
            let budget = if mode == 2 { disk_budget } else { largest };
            let residency = match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
                ),
            };
            let (mut model, metadata) = materialize_voicechat_rnnt(
                prepared(dimensions, source.clone()),
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(model.static_lease_count(), 0);
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), 564);
            let actual = run(&mut model, &fixture, f16, &stream);
            if let Some(expected) = &baseline {
                assert_eq!(
                    &actual, expected,
                    "RNNT residency parity after checkpoint conversion F16={f16}"
                );
            } else {
                baseline = Some(actual);
            }
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= budget
                );
            }
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert!(report
                    .execution_groups()
                    .iter()
                    .all(|group| group.completed_executions() > 0));
            }
        }
    }
}

#[path = "voicechat_rnnt_residency/parallel.rs"]
mod parallel;

#[path = "voicechat_rnnt_residency/greedy.rs"]
mod greedy;
#[path = "voicechat_rnnt_residency/pipeline.rs"]
mod pipeline;

#[path = "voicechat_rnnt_residency/perception.rs"]
mod perception;

#[path = "voicechat_rnnt_residency/pcm.rs"]
mod pcm;
