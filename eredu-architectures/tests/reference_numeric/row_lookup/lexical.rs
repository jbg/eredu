use super::*;
use eredu_architectures::qwen4_exp::{
    ngram::{NGramEmbeddingSpec, NGramHashSpec},
    ple::{LexicalInjection, LexicalInjectionSpec},
};
use eredu_nn::{
    residual_streams::ResidualStreamGeometry, CausalDepthwiseConvolutionSpec, ConvolutionActivation,
};
use eredu_runtime::{ParameterBankAccess, ParameterProviders, ResidentExpertProvider};
const KEY: [f32; 16] = [
    0.1, -0.2, 0.3, 0.4, 0.2, 0.1, -0.3, 0.2, -0.1, 0.2, 0.4, -0.3, 0.2, 0.4, 0.1, 0.3,
];
const VALUE: [f32; 8] = [0.05, 0.03, -0.01, 0.02, -0.04, 0.02, 0.01, 0.03];
const KN: [f32; 4] = [0.2, -0.1, 0.3, 0.1];
const QN: [f32; 4] = [0.3, 0.1, -0.2, 0.2];
const CN: [f32; 4] = [-0.1, 0.2, 0.4, -0.3];
const CONV: [f32; 12] = [
    0.2, -0.1, 0.3, 0.1, 0.4, -0.2, -0.3, 0.2, 0.1, 0.4, -0.1, 0.2,
];
fn hash() -> NGramHashSpec {
    NGramHashSpec::new(1007, 7, 3, 1, vec![3, 5, 7], vec![11, 13], vec![0, 11], 24).unwrap()
}
fn construction() -> LexicalInjectionSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let linear = |name, input, output| LinearSpec {
        input,
        output,
        weight: parameter(name),
        bias: None,
        format: dense_linear_format(),
    };
    let norm = |name| NormalizationConstructionSpec {
        groups: Some(2),
        dimensions: 4,
        epsilon: 1e-5,
        scale: NormalizationScale::LearnedOffset {
            weight: parameter(name),
            offset: 1.,
        },
    };
    LexicalInjectionSpec {
        geometry: ResidualStreamGeometry::new(2, 2).unwrap(),
        embedding: NGramEmbeddingSpec::new(
            1007,
            7,
            3,
            1,
            RowLookupSpec { rows: 24, ..spec() },
            5,
            64,
        )
        .unwrap(),
        embedding_width: 4,
        key: linear("key", 4, 4),
        value: linear("value", 4, 2),
        key_norm: norm("kn"),
        query_norm: norm("qn"),
        convolution_norm: norm("cn"),
        convolution: CausalDepthwiseConvolutionSpec {
            channels: 4,
            kernel_size: 3,
            dilation: 3,
            weight: parameter("conv"),
            bias: None,
            activation: ConvolutionActivation::Silu,
        },
        convolution_slot: 1,
    }
}
struct Load;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        value.data = match metadata.id.as_str() {
            "key" => KEY.to_vec(),
            "value" => VALUE.to_vec(),
            "kn" => KN.to_vec(),
            "qn" => QN.to_vec(),
            "cn" => CN.to_vec(),
            "conv" => CONV.to_vec(),
            name => panic!("unexpected {name}"),
        };
    }
}
fn normalized(x: &[f64], weights: &[f32]) -> Vec<f64> {
    x.chunks_exact(2)
        .zip(weights.chunks_exact(2))
        .flat_map(|(x, w)| {
            let rms = ((x[0] * x[0] + x[1] * x[1]) / 2. + 1e-5).sqrt();
            (0..2).map(move |i| x[i] / rms * (1. + w[i] as f64))
        })
        .collect()
}
fn projected(x: &[f64], w: &[f32]) -> Vec<f64> {
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(&w, &x)| w as f64 * x).sum())
        .collect()
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
#[test]
fn lexical_injection_matches_independent_equations_padding_chunks_and_rollback() {
    let context = NumericContext::default();
    let ids = [3u64, 7, 4, 5, 6, 7, 1, 2, 8, 9, 10, 11, 7, 7, 12, 13];
    let residual = NumericTensor::new(
        [2, 8, 2, 2],
        (0..64).map(|i| ((i * 7 % 31) as f32 - 15.) / 7.).collect(),
    );
    let mask = NumericTensor::new(
        [2, 8],
        vec![
            0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.,
        ],
    );
    let lexical_ids: Vec<u64> = ids
        .iter()
        .zip(&mask.data)
        .map(|(&id, &valid)| if valid == 0. { 7 } else { id })
        .collect();
    let mut expected = vec![];
    for batch in 0..2 {
        let mut normalized_values: Vec<Vec<f64>> = vec![];
        for t in 0..8 {
            let at = batch * 8 + t;
            let mut past = [7, 7];
            for lag in 1..=2 {
                if t >= lag {
                    past[lag - 1] = lexical_ids[at - lag];
                }
            }
            if past[0] == 7 {
                past[1] = 7;
            }
            let first = (lexical_ids[at] * 3) ^ (past[0] * 5);
            let rows = [first % 11, ((first ^ (past[1] * 7)) % 13) + 11];
            let embedding: Vec<f64> = rows
                .iter()
                .flat_map(|&row| [row as f64 + 0.25, -(row as f64) - 0.5])
                .collect();
            let key = normalized(&projected(&embedding, &KEY), &KN);
            let query = normalized(
                &residual.data[at * 4..at * 4 + 4]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                &QN,
            );
            let value = projected(&embedding, &VALUE);
            let gated: Vec<f64> = (0..4)
                .map(|channel| {
                    let s = channel / 2 * 2;
                    let score = (key[s] * query[s] + key[s + 1] * query[s + 1]) / 2f64.sqrt();
                    let sign = if score == 0. { 0. } else { score.signum() };
                    sigmoid(score.abs().max(1e-6).sqrt() * sign) * value[channel % 2]
                })
                .collect();
            normalized_values.push(
                normalized(&gated, &CN)
                    .into_iter()
                    .map(|v| v * mask.data[at] as f64)
                    .collect(),
            );
            for channel in 0..4 {
                let conv: f64 = (0..3)
                    .filter_map(|tap| {
                        t.checked_sub((2 - tap) * 3).map(|source| {
                            normalized_values[source][channel] * CONV[channel * 3 + tap] as f64
                        })
                    })
                    .sum();
                expected
                    .push((gated[channel] * mask.data[at] as f64 + conv * sigmoid(conv)) as f32);
            }
        }
    }
    let specification = construction();
    let policy = LayerCachePolicy::fixed_only(specification.state_policies().unwrap()).unwrap();
    let fresh = || ParameterProviders {
        grouped: ResidentExpertProvider,
        rows: BoundedRowLookup::new(
            Rows::default(),
            RowLookupSpec { rows: 24, ..spec() },
            RowLookupLimits {
                requests: 64,
                host_bytes: 8192,
                output_bytes: 1536,
                ..limits()
            },
        )
        .unwrap(),
    };
    let mut layer =
        LexicalInjection::<NumericBackend>::new(specification, hash(), &context).unwrap();
    layer.visit_parameters_mut(&mut Load);
    let mut provider = fresh();
    let mut state = NumericHybridLayerState::new(&policy);
    let whole = layer
        .forward(
            &residual,
            Some(&ids),
            Some(&mask),
            &mut state,
            &mut provider,
            ParameterBankAccess::Bulk,
            &context,
        )
        .unwrap();
    state.advance_fixed(8).unwrap();
    assert_tensor_close(
        &whole,
        &NumericTensor::new([2, 8, 2, 2], expected),
        "independent lexical equation",
    );
    let mut chunked = NumericHybridLayerState::new(&policy);
    let mut pieces = vec![];
    let mut start = 0;
    for length in [1, 2, 1, 4] {
        let chunk_ids: Vec<u64> = (0..2)
            .flat_map(|b| ids[b * 8 + start..b * 8 + start + length].iter().copied())
            .collect();
        pieces.push(
            layer
                .forward(
                    &residual.axis_slice(1, start, start + length),
                    Some(&chunk_ids),
                    Some(&mask.axis_slice(1, start, start + length)),
                    &mut chunked,
                    &mut provider,
                    ParameterBankAccess::Incremental,
                    &context,
                )
                .unwrap(),
        );
        chunked.advance_fixed(length as i32).unwrap();
        start += length;
    }
    assert_tensor_exact(
        &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
        &whole,
        "lexical chunk equivalence",
    );
    for (role, value) in &state.fixed {
        assert_tensor_exact(
            value.as_ref().unwrap(),
            chunked.fixed[role].as_ref().unwrap(),
            "lexical state equivalence",
        );
    }
    let checkpoint = chunked.clone();
    let token = residual.axis_slice(1, 0, 1);
    let first = layer
        .forward(
            &token,
            Some(&[3, 8]),
            None,
            &mut chunked,
            &mut provider,
            ParameterBankAccess::Incremental,
            &context,
        )
        .unwrap();
    chunked = checkpoint;
    let replay = layer
        .forward(
            &token,
            Some(&[3, 8]),
            None,
            &mut chunked,
            &mut provider,
            ParameterBankAccess::Incremental,
            &context,
        )
        .unwrap();
    assert_tensor_exact(&first, &replay, "lexical fork rollback replay");
}
#[test]
fn lexical_gate_signed_zero_and_epsilon_boundary_are_finite() {
    let context = NumericContext::default();
    let gate = NumericTensor::new([6], vec![-0., 0., -1e-12, 1e-12, -4., 9.]);
    let result = gate
        .abs(&context)
        .unwrap()
        .maximum_scalar(1e-6, &context)
        .unwrap()
        .sqrt(&context)
        .unwrap()
        .multiply(&gate.sign(&context).unwrap(), &context)
        .unwrap();
    assert_eq!(result.data, vec![0., 0., -0.001, 0.001, -2., 3.]);
    let mut malformed = construction();
    malformed.convolution.dilation = 1;
    assert!(malformed.validate().is_err());
    malformed = construction();
    malformed.key_norm.groups = Some(1);
    assert!(malformed.validate().is_err());
}

#[test]
fn lexical_binding_preserves_typed_hash_geometry_failure() {
    use eredu_architectures::qwen4_exp::ngram::{NGramEmbeddingError, NGramError};
    let context = NumericContext::default();
    let wrong_reset =
        NGramHashSpec::new(1007, 8, 3, 1, vec![3, 5, 7], vec![11, 13], vec![0, 11], 24).unwrap();
    let error =
        LexicalInjection::<NumericBackend>::new(construction(), wrong_reset, &context).unwrap_err();
    assert!(matches!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<NGramEmbeddingError>(),
        Some(NGramEmbeddingError::Hash(NGramError::Constants))
    ));
}

#[test]
fn lexical_unit_owns_position_original_ids_and_transactional_replay() {
    use eredu_architectures::{
        decoder::ComponentInstrumentation,
        qwen4_exp::{
            ngram::{NGramEmbeddingError, NGramError},
            ple::{LexicalSublayer, LexicalSublayerInput},
        },
    };
    struct Fail;
    impl eredu_runtime::ActivationObserver<NumericTensor, Error> for Fail {
        fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
            if path.ends_with("lexical.write") {
                return Err(Error::backend("injected lexical observation failure"));
            }
            Ok(())
        }
    }
    let context = NumericContext::default();
    let residual = NumericTensor::new(
        [2, 8, 2, 2],
        (0..64).map(|i| ((i * 11 % 41) as f32 - 20.) / 9.).collect(),
    );
    let ids = [3, 7, 4, 5, 6, 7, 1, 2, 8, 9, 10, 11, 7, 7, 12, 13];
    for kernel in [1, 3] {
        let mut spec = construction();
        spec.convolution.kernel_size = kernel;
        let policy = LayerCachePolicy::fixed_only(spec.state_policies().unwrap()).unwrap();
        let mut provider = ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: BoundedRowLookup::new(
                Rows::default(),
                RowLookupSpec {
                    rows: 24,
                    ..super::spec()
                },
                RowLookupLimits {
                    requests: 64,
                    host_bytes: 8192,
                    output_bytes: 1536,
                    ..limits()
                },
            )
            .unwrap(),
        };
        let mut unit =
            LexicalSublayer::<NumericBackend>::new(spec.clone(), hash(), &context).unwrap();
        let mut primitive =
            LexicalInjection::<NumericBackend>::new(spec, hash(), &context).unwrap();
        // This visitor respects both one-tap and dilated parameter geometries.
        unit.visit_parameters_mut(&mut super::super::qwen4_recurrent::Parameters::default());
        primitive.visit_parameters_mut(&mut super::super::qwen4_recurrent::Parameters::default());
        let mut direct_state = NumericHybridLayerState::new(&policy);
        let contribution = primitive
            .forward(
                &residual,
                Some(&ids),
                None,
                &mut direct_state,
                &mut provider,
                ParameterBankAccess::Bulk,
                &context,
            )
            .unwrap();
        let expected = residual.add(&contribution, &context).unwrap();
        let mut state = NumericHybridLayerState::new(&policy);
        let mut pieces = vec![];
        for (start, end) in [(0, 1), (1, 3), (3, 4), (4, 8)] {
            let chunk = residual.axis_slice(1, start, end);
            let chunk_ids: Vec<_> = (0..2)
                .flat_map(|lane| ids[lane * 8 + start..lane * 8 + end].iter().copied())
                .collect();
            let checkpoint = state.clone();
            let before_calls = provider.rows.bank().calls.len();
            let missing = unit.forward(
                LexicalSublayerInput {
                    residual: &chunk,
                    ids: None,
                    padding: None,
                    offset: start as i32,
                },
                &mut state,
                &mut provider,
                ParameterBankAccess::Incremental,
                &context,
                &mut ComponentInstrumentation::disabled(),
            );
            assert!(matches!(
                missing,
                Err(NGramEmbeddingError::Hash(NGramError::MissingTokenIds))
            ));
            assert_eq!(provider.rows.bank().calls.len(), before_calls);
            assert_eq!(state.position(), start as i32);
            let wrong_position = unit.forward(
                LexicalSublayerInput {
                    residual: &chunk,
                    ids: Some(&chunk_ids),
                    padding: None,
                    offset: start as i32 + 1,
                },
                &mut state,
                &mut provider,
                ParameterBankAccess::Incremental,
                &context,
                &mut ComponentInstrumentation::disabled(),
            );
            assert!(matches!(
                wrong_position,
                Err(NGramEmbeddingError::Hash(NGramError::Shape))
            ));
            assert_eq!(provider.rows.bank().calls.len(), before_calls);
            let failed = unit.forward(
                LexicalSublayerInput {
                    residual: &chunk,
                    ids: Some(&chunk_ids),
                    padding: None,
                    offset: start as i32,
                },
                &mut state,
                &mut provider,
                ParameterBankAccess::Incremental,
                &context,
                &mut ComponentInstrumentation::new("model.layers.1", &mut Fail),
            );
            assert!(failed.is_err());
            assert_eq!(state.position(), start as i32);
            let after_failure = provider.rows.bank().completed;
            assert!(provider.rows.bank().calls.len() > before_calls);
            // Only mutable model state rolls back; physical lookup work remains charged.
            state = checkpoint;
            pieces.push(
                unit.forward(
                    LexicalSublayerInput {
                        residual: &chunk,
                        ids: Some(&chunk_ids),
                        padding: None,
                        offset: start as i32,
                    },
                    &mut state,
                    &mut provider,
                    ParameterBankAccess::Incremental,
                    &context,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap(),
            );
            assert_eq!(state.position(), end as i32);
            assert!(provider.rows.bank().completed > after_failure);
        }
        assert_tensor_close(
            &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
            &expected,
            "lexical unit chunked addition",
        );
        for (role, value) in &direct_state.fixed {
            assert_tensor_exact(
                value.as_ref().unwrap(),
                state.fixed[role].as_ref().unwrap(),
                "lexical unit retains complete histories",
            );
        }
        if kernel == 1 {
            assert!(!state
                .fixed
                .contains_key(&StateTensorRole::Convolution { slot: 1 }));
        }
    }
}
