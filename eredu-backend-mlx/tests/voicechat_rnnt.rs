//! Native recurrent execution against the pinned NVIDIA/PyTorch component oracle.

use eredu_architectures::voicechat::rnnt::RnntExecution;
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    recurrent::{Lstm, LstmLayerSpec, LstmLayerState, LstmState},
    LinearFormatSpec, LinearSpec, ParameterMetadata, ParameterSpec, ParameterVisitorMut,
    Parameterized, Tensor, TensorElementType,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(eredu_evaluation::voicechat::RNNT_REFERENCE_JSON).unwrap()
}

fn rnnt_dimensions(fixture: &Value) -> eredu_architectures::voicechat::rnnt::RnntDimensions {
    use eredu_architectures::voicechat::rnnt::RnntDimensions;
    let geometry = &fixture["geometry"];
    RnntDimensions {
        vocab_size: geometry["vocab_size"].as_i64().unwrap() as i32,
        predictor_width: geometry["prediction_hidden"].as_i64().unwrap() as i32,
        predictor_layers: geometry["prediction_layers"].as_u64().unwrap() as usize,
        encoder_width: geometry["encoder_hidden"].as_i64().unwrap() as i32,
        joint_width: geometry["joint_hidden"].as_i64().unwrap() as i32,
    }
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|number| number.as_f64().unwrap() as f32)
        .collect()
}

fn tensor(values: &[f32], shape: &[i32]) -> MlxTensor {
    MlxTensor::from_array(Array::from_slice(values, shape))
}

struct Bind<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, output: &'a mut MlxTensor) {
        let parameter = &self.0[metadata.id.as_str()];
        let shape: Vec<i32> = parameter["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect();
        assert_eq!(output.shape(), shape);
        *output = tensor(&floats(&parameter["values"]), &shape);
    }
}

fn assert_values(actual: &MlxTensor, expected: &Value, tolerance: &Value, stream: &Stream) {
    let actual = actual.to_f32_vec(stream).unwrap();
    let expected = floats(expected);
    assert_eq!(actual.len(), expected.len());
    let absolute = tolerance["absolute"].as_f64().unwrap() as f32;
    let relative = tolerance["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "value {index}: {actual} versus NVIDIA reference {expected}"
        );
    }
}

fn assert_state(state: &LstmState<MlxTensor>, expected: &Value, fixture: &Value, stream: &Stream) {
    for (index, layer) in state.layers.iter().enumerate() {
        let width = layer.hidden.shape()[1] as usize;
        for (field, tensor) in [("h", &layer.hidden), ("c", &layer.cell)] {
            let values =
                expected[field].as_array().unwrap()[index * width..(index + 1) * width].to_vec();
            assert_values(tensor, &Value::Array(values), &fixture["tolerance"], stream);
        }
    }
}

fn lstm(fixture: &Value, stream: &Stream) -> Lstm<Backend> {
    let width = fixture["geometry"]["prediction_hidden"].as_i64().unwrap() as i32;
    let count = fixture["geometry"]["prediction_layers"].as_u64().unwrap();
    let prefix = "stt_model.rnnt_decoder.prediction.dec_rnn.lstm";
    let specs = (0..count)
        .map(|index| {
            let projection = |kind: &str| LinearSpec {
                input: width,
                output: width * 4,
                weight: ParameterSpec::trainable(format!("{prefix}.weight_{kind}_l{index}"))
                    .unwrap(),
                bias: Some(
                    ParameterSpec::trainable(format!("{prefix}.bias_{kind}_l{index}")).unwrap(),
                ),
                format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            };
            LstmLayerSpec {
                input: projection("ih"),
                recurrent: projection("hh"),
            }
        })
        .collect();
    let mut lstm = Lstm::new(specs, stream).unwrap();
    lstm.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    lstm
}

#[test]
fn cpu_lstm_matches_nvidia_nonzero_recurrent_sequence() {
    let fixture = fixture();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut lstm = lstm(&fixture, &stream);
    let width = lstm.input_width() as usize;
    let sequence = &fixture["lstm_sequence"];
    let initial_h = floats(&sequence["initial_h"]);
    let initial_c = floats(&sequence["initial_c"]);
    let mut state = LstmState {
        layers: initial_h
            .chunks_exact(width)
            .zip(initial_c.chunks_exact(width))
            .map(|(h, c)| LstmLayerState {
                hidden: tensor(h, &[1, width as i32]),
                cell: tensor(c, &[1, width as i32]),
            })
            .collect(),
    };
    let parent = state.clone();
    let expected_outputs = sequence["output"].as_array().unwrap();
    for (index, input) in floats(&sequence["input"]).chunks_exact(width).enumerate() {
        state = lstm
            .step(&tensor(input, &[1, width as i32]), &state, &stream)
            .unwrap();
        let expected = Value::Array(expected_outputs[index * width..(index + 1) * width].to_vec());
        assert_values(
            state.output().unwrap(),
            &expected,
            &fixture["tolerance"],
            &stream,
        );
    }
    assert_state(&state, sequence, &fixture, &stream);
    assert_state(
        &parent,
        &serde_json::json!({"h": initial_h, "c": initial_c}),
        &fixture,
        &stream,
    );
}

#[test]
fn cpu_lstm_zero_state_preserves_prototype_precision() {
    let fixture = fixture();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let lstm = lstm(&fixture, &stream);
    let prototype = MlxTensor::from_array(
        Array::from_slice(&[0.25, -0.5], &[1, 2])
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap(),
    );
    let state = lstm.zero_state_like(&prototype, &stream).unwrap();
    for layer in state.layers {
        for tensor in [layer.hidden, layer.cell] {
            assert_eq!(tensor.element_type(), Some(TensorElementType::Bf16));
            let actual = tensor
                .as_array()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            assert_eq!(actual.as_slice::<f32>(), &[0.0, 0.0]);
        }
    }

    // A first layer may promote the half-precision prototype. Every layer's
    // initial recurrent zeros still use the original prototype precision.
    use eredu_architectures::voicechat::{
        rnnt::Rnnt,
        rnnt_layered::{LayerwiseRnnt, RnntArchitecture},
    };
    struct MixedPrecision<'s>(&'s Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for MixedPrecision<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            if metadata.id.as_str().ends_with("_l1") {
                *value = value.cast_float(TensorElementType::F16, self.0).unwrap();
            }
        }
    }
    let dimensions = rnnt_dimensions(&fixture);
    assert_eq!(dimensions.predictor_layers, 2);
    let mut ordinary = Rnnt::<Backend>::new(dimensions, &stream).unwrap();
    ordinary.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    ordinary.visit_parameters_mut(&mut MixedPrecision(&stream));
    let architecture = RnntArchitecture::<Backend>::new(dimensions).unwrap();
    let mut units = Vec::new();
    for group in 0..architecture.execution_graph().unwrap().groups().len() {
        for index in 0..architecture.group_unit_count(group).unwrap() {
            let mut unit = architecture.build_unit(group, index, &stream).unwrap();
            unit.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
            unit.visit_parameters_mut(&mut MixedPrecision(&stream));
            units.push(unit);
        }
    }
    let mut layered =
        LayerwiseRnnt::new(architecture, eredu_runtime::ResidentUnitWindow::new(units));
    let mut ordinary_state = None;
    let mut layered_state = None;
    for values in [[0.25, -0.5], [-0.125, 0.375]] {
        let input = tensor(&values, &[1, dimensions.predictor_width])
            .cast_float(TensorElementType::F16, &stream)
            .unwrap();
        let expected = ordinary
            .predictor_step(&input, ordinary_state.as_ref(), &stream)
            .unwrap();
        let actual = layered
            .predictor_step(&input, layered_state.as_ref(), &stream)
            .unwrap();
        assert_eq!(
            expected.local_state().layers[0].hidden.element_type(),
            Some(TensorElementType::F32)
        );
        for (actual, expected) in actual
            .local_state()
            .layers
            .iter()
            .zip(&expected.local_state().layers)
        {
            for (actual, expected) in [
                (&actual.hidden, &expected.hidden),
                (&actual.cell, &expected.cell),
            ] {
                assert_eq!(actual.element_type(), expected.element_type());
                assert_eq!(
                    actual.to_f32_vec(&stream).unwrap(),
                    expected.to_f32_vec(&stream).unwrap()
                );
            }
        }
        ordinary_state = Some(expected);
        layered_state = Some(actual);
    }
}

#[test]
fn cpu_rnnt_matches_nvidia_priming_logits_and_bounded_greedy_frames() {
    use eredu_architectures::voicechat::rnnt::{Rnnt, RnntLimits, RnntState};
    let fixture = fixture();
    let geometry = &fixture["geometry"];
    let dimensions = rnnt_dimensions(&fixture);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut model = Rnnt::<Backend>::new(dimensions, &stream).unwrap();
    model.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let primed = model.prime(&RnntState::default(), &stream).unwrap();
    assert_state(
        primed.predictor().unwrap().local_state(),
        &fixture["predictor_steps"][0],
        &fixture,
        &stream,
    );
    let limits = RnntLimits {
        max_symbols_per_frame: geometry["max_symbols"].as_u64().unwrap() as usize,
        hypothesis_capacity: 32,
    };
    for case in ["streaming_frames", "feedback_terminates_frame"] {
        let mut state = primed.clone();
        for (index, frame) in fixture[case].as_array().unwrap().iter().enumerate() {
            let input = tensor(&floats(&frame["encoder"]), &[1, dimensions.encoder_width]);
            let logits = model.joint_logits(&input, &state, &stream).unwrap();
            assert_values(
                &logits,
                &frame["joint_calls"][0]["logits"],
                &fixture["tolerance"],
                &stream,
            );
            let result = model.decode_frame(&input, &state, limits, &stream).unwrap();
            let emitted: Vec<usize> = frame["emitted"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(result.emitted, emitted);
            assert_eq!(result.first_blank, frame["first_blank"].as_bool().unwrap());
            assert_eq!(
                result.hit_symbol_limit,
                frame["hit_symbol_limit"].as_bool().unwrap()
            );
            assert_eq!(result.ended_blank, frame["ended_blank"].as_bool().unwrap());
            assert_eq!(result.state.frames(), index as u64 + 1);
            assert_state(
                result.state.predictor().unwrap().local_state(),
                frame,
                &fixture,
                &stream,
            );
            state = result.state;
        }
    }
    assert_state(
        primed.predictor().unwrap().local_state(),
        &fixture["predictor_steps"][0],
        &fixture,
        &stream,
    );
}

#[test]
fn cpu_rnnt_encoder_uses_joint_parameter_precision() {
    use eredu_architectures::voicechat::rnnt::{Rnnt, RnntState};
    struct HalfParameters<'s>(&'s Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for HalfParameters<'_> {
        fn visit_mut(&mut self, _: ParameterMetadata, value: &'a mut MlxTensor) {
            *value = value.cast_float(TensorElementType::F16, self.0).unwrap();
        }
    }
    let fixture = fixture();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut model = Rnnt::<Backend>::new(rnnt_dimensions(&fixture), &stream).unwrap();
    model.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    model.visit_parameters_mut(&mut HalfParameters(&stream));
    let state = model.prime(&RnntState::default(), &stream).unwrap();
    // These inputs change when rounded to half precision. The upstream wrapper
    // explicitly rounds perception output before the joint encoder projection.
    let input = tensor(&[-0.812345, 0.213579, 0.312357], &[1, 3]);
    let rounded = input.cast_float(TensorElementType::F16, &stream).unwrap();
    assert_ne!(
        input.to_f32_vec(&stream).unwrap(),
        rounded.to_f32_vec(&stream).unwrap()
    );
    let implicit = model.joint_logits(&input, &state, &stream).unwrap();
    let explicit = model.joint_logits(&rounded, &state, &stream).unwrap();
    assert_eq!(implicit.element_type(), Some(TensorElementType::F16));
    assert_eq!(explicit.element_type(), Some(TensorElementType::F16));
    assert_eq!(
        implicit.to_f32_vec(&stream).unwrap(),
        explicit.to_f32_vec(&stream).unwrap()
    );
}
