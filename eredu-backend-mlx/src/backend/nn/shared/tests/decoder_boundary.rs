use super::*;
use eredu_architectures::{
    decoder::StaticModules, hybrid_decoder::HybridDecoder, qwen4_exp::residual::ResidualBoundary,
};
use eredu_nn::residual_streams::{GatedResidualSpec, ResidualStreamGeometry};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized};

const NORM: [f32; 4] = [0.2, -0.3, 0.5, -0.1];
const DOWN: [f32; 8] = [0.1, 0.2, -0.4, 0.3, -0.2, 0.5, 0.1, 0.4];
const UP: [f32; 8] = [0.3, -0.2, 0.4, 0.1, -0.5, 0.2, 0.6, -0.3];
const HEAD: [f32; 8] = [0.3, 0.6, -0.2, 0.8, 0.7, -0.1, -0.5, -0.4];
struct Load;
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Load {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
        let values: &[f32] = match metadata.id.as_str() {
            "norm" => &NORM,
            "down" => &DOWN,
            "up" => &UP,
            "embedding" | "head" => &HEAD,
            other => panic!("unexpected parameter {other}"),
        };
        *value = MlxTensor::from_array(Array::from_slice(values, value.shape()));
    }
}
fn exercise(device: DeviceType) {
    let execution = ExecutionContext::new(Device::new(device, 0));
    let stream = execution.stream();
    let p = |name| ParameterSpec::trainable(name).unwrap();
    let format = || eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
    let linear = |name, input, output| LinearSpec {
        input,
        output,
        weight: p(name),
        bias: None,
        format: format(),
    };
    let values = [0.3f32, -0.7, 0.8, 0.2, -0.9, 0.4, -0.2, 0.6];
    let input = MlxTensor::from_array(Array::from_slice(&values, &[1, 2, 2, 2]));
    let sigmoid = |v: f64| 1. / (1. + (-v).exp());
    let dot = |w: &[f32], x: &[f64]| w.iter().zip(x).map(|(w, x)| *w as f64 * x).sum::<f64>();
    let mut expected = vec![];
    for x in values.chunks_exact(4) {
        let normalized: Vec<f64> = (0..4)
            .map(|i| {
                let group = i / 2 * 2;
                let variance = (f64::from(x[group]).powi(2) + f64::from(x[group + 1]).powi(2)) / 2.;
                x[i] as f64 / (variance + 1e-4).sqrt() * (1. + NORM[i] as f64)
            })
            .collect();
        let low: Vec<f64> = DOWN
            .chunks_exact(4)
            .map(|w| {
                let v = dot(w, &normalized) / 2.;
                v * sigmoid(v)
            })
            .collect();
        let weighted: Vec<f64> = UP
            .chunks_exact(2)
            .enumerate()
            .map(|(i, w)| sigmoid(dot(w, &low)) * normalized[i])
            .collect();
        let mixed = [
            (weighted[0] + weighted[2]) / 2.,
            (weighted[1] + weighted[3]) / 2.,
        ];
        expected.extend(HEAD.chunks_exact(2).map(|w| dot(w, &mixed) as f32));
    }
    for tied in [true, false] {
        let boundary = ResidualBoundary::<MlxNeuralBackend>::new(
            GatedResidualSpec {
                geometry: ResidualStreamGeometry::new(2, 2).unwrap(),
                normalization: NormalizationConstructionSpec {
                    dimensions: 4,
                    groups: Some(2),
                    epsilon: 1e-4,
                    scale: NormalizationScale::LearnedOffset {
                        weight: p("norm"),
                        offset: 1.,
                    },
                },
                down: linear("down", 4, 2),
                up: linear("up", 2, 4),
                injection: None,
            },
            stream,
        )
        .unwrap();
        let mut modules = StaticModules::from_boundary(
            EmbeddingSpec {
                vocabulary: 4,
                dimensions: 2,
                weight: p("embedding"),
                format: format(),
            },
            (!tied).then(|| linear("head", 2, 4)),
            boundary,
            stream,
        )
        .unwrap();
        modules.visit_parameters_mut(&mut Load);
        let mut decoder = HybridDecoder::new(modules, "model.layers", 2).unwrap();
        let full = decoder.finish_logits(&input, stream).unwrap();
        // Frozen F32 absolute tolerance before evaluating the native boundary.
        close(&full, &expected, 2e-6);
        close(
            &decoder.finish_text_logits(&input, stream).unwrap(),
            &expected[4..],
            2e-6,
        );
        assert_eq!(input.to_f32_vec(stream).unwrap(), values);
        let tokens = MlxTensor::from_array(Array::from_slice(&[0i32, 3], &[1, 2]));
        let expanded = decoder.static_modules_mut().embed(&tokens, stream).unwrap();
        assert_eq!(expanded.shape(), [1, 2, 2, 2]);
        assert_eq!(
            expanded.to_f32_vec(stream).unwrap(),
            [0.3, 0.6, 0.3, 0.6, -0.5, -0.4, -0.5, -0.4]
        );
    }
}
#[test]
fn residual_decoder_boundary_matches_scalar_readout() {
    exercise(DeviceType::Cpu);
}
#[test]
#[ignore = "requires Metal device access"]
fn residual_decoder_boundary_matches_scalar_readout_metal() {
    exercise(DeviceType::Gpu);
}
