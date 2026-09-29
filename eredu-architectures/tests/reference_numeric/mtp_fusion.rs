use super::*;
use eredu_architectures::qwen4_exp::mtp::{PredictionFusion, PredictionFusionSpec};
use eredu_nn::residual_streams::ResidualStreamGeometry;
#[test]
fn prediction_fusion_normalizes_complete_target_before_shared_stream_projection() {
    let context = NumericContext::default();
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let norm = |name, dimensions| NormalizationConstructionSpec {
        dimensions,
        groups: None,
        epsilon: 1e-4,
        scale: NormalizationScale::LearnedOffset {
            weight: parameter(name),
            offset: 1.,
        },
    };
    let linear = |name| LinearSpec {
        input: 2,
        output: 2,
        weight: parameter(name),
        bias: None,
        format: dense_linear_format(),
    };
    let spec = PredictionFusionSpec {
        geometry: ResidualStreamGeometry::new(2, 2).unwrap(),
        embedding_norm: norm("en", 2),
        hidden_norm: norm("hn", 4),
        embedding_projection: linear("ep"),
        hidden_projection: linear("hp"),
    };
    let mut wrong = spec.clone();
    wrong.hidden_norm.groups = Some(2);
    assert!(wrong.validate().is_err());
    let mut fusion = PredictionFusion::<NumericBackend>::new(spec, &context).unwrap();
    struct Load;
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
        fn visit_mut(&mut self, m: ParameterMetadata, v: &'a mut NumericTensor) {
            v.data = match m.id.as_str() {
                "en" => vec![0.2, -0.3],
                "hn" => vec![0.1, 0.3, -0.2, 0.4],
                "ep" => vec![0.1, 0.2, -0.4, 0.3],
                "hp" => vec![0.3, -0.1, 0.2, 0.4],
                _ => panic!(),
            };
        }
    }
    fusion.visit_parameters_mut(&mut Load);
    let embeddings = NumericTensor::new([2, 3, 2], (0..12).map(|i| (i as f32 - 5.) / 4.).collect());
    let target = NumericTensor::new(
        [2, 3, 2, 2],
        (0..24).map(|i| ((i * 11 % 23) as f32 - 10.) / 3.).collect(),
    );
    let mut expected = vec![];
    for row in 0..6 {
        let e = &embeddings.data[row * 2..row * 2 + 2];
        let h = &target.data[row * 4..row * 4 + 4];
        let erms = (e.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 2. + 1e-4).sqrt();
        let hrms = (h.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 4. + 1e-4).sqrt();
        let e = [e[0] as f64 / erms * 1.2, e[1] as f64 / erms * 0.7];
        let h: Vec<f64> = h
            .iter()
            .zip([1.1, 1.3, 0.8, 1.4])
            .map(|(&v, w)| v as f64 / hrms * w)
            .collect();
        for stream in h.chunks_exact(2) {
            expected.extend([
                (0.1 * e[0] + 0.2 * e[1] + 0.3 * stream[0] - 0.1 * stream[1]) as f32,
                (-0.4 * e[0] + 0.3 * e[1] + 0.2 * stream[0] + 0.4 * stream[1]) as f32,
            ]);
        }
    }
    let actual = fusion.forward(&embeddings, &target, &context).unwrap();
    assert_tensor_close(
        &actual,
        &NumericTensor::new([2, 3, 2, 2], expected),
        "independent SGLang residual fusion",
    );
    let chunks: Vec<_> = (0..3)
        .map(|token| {
            fusion
                .forward(
                    &embeddings.axis_slice(1, token, token + 1),
                    &target.axis_slice(1, token, token + 1),
                    &context,
                )
                .unwrap()
        })
        .collect();
    assert_tensor_exact(
        &actual,
        &NumericTensor::concatenate(&chunks, 1, &context).unwrap(),
        "prediction chunks",
    );
    assert!(fusion
        .forward(
            &embeddings,
            &NumericTensor::new([2, 3, 2], vec![0.; 12]),
            &context
        )
        .is_err());
}
