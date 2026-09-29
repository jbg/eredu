use super::*;
use eredu_nn::residual_streams::{GatedResidual, GatedResidualSpec, ResidualStreamGeometry};

fn spec(inject: bool) -> GatedResidualSpec {
    let parameter = |name| ParameterSpec::trainable(name).unwrap();
    let projection = |name, input, output| LinearSpec {
        input,
        output,
        weight: parameter(name),
        bias: None,
        format: dense_linear_format(),
    };
    GatedResidualSpec {
        geometry: ResidualStreamGeometry::new(2, 2).unwrap(),
        normalization: NormalizationConstructionSpec {
            groups: Some(2),
            dimensions: 4,
            epsilon: 1e-4,
            scale: NormalizationScale::LearnedOffset {
                weight: parameter("norm"),
                offset: 1.0,
            },
        },
        down: projection("down", 4, 2),
        up: projection("up", 2, 4),
        injection: inject.then(|| projection("injection", 4, 2)),
    }
}
const NORM: [f32; 4] = [0.2, -0.3, 0.5, -0.1];
const DOWN: [f32; 8] = [0.1, 0.2, -0.4, 0.3, -0.2, 0.5, 0.1, 0.4];
const UP: [f32; 8] = [0.3, -0.2, 0.4, 0.1, -0.5, 0.2, 0.6, -0.3];
const INJECTION: [f32; 8] = [0.2, -0.1, 0.4, 0.3, -0.3, 0.1, 0.2, -0.5];
struct Load;
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Load {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        value.data = match metadata.id.as_str() {
            "norm" => NORM.to_vec(),
            "down" => DOWN.to_vec(),
            "up" => UP.to_vec(),
            "injection" => INJECTION.to_vec(),
            other => panic!("unexpected {other}"),
        };
    }
}
fn dot(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| *a as f64 * b).sum()
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn gated_residual_matches_independent_stream_equations_and_final_collapse() {
    let context = NumericContext::default();
    let input = NumericTensor::new(
        [2, 3, 2, 2],
        (0..24)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 5.0)
            .collect(),
    );
    let mut expected_mixed = Vec::new();
    let mut expected_injected = Vec::new();
    let sublayer = NumericTensor::new([2, 3, 2], (0..12).map(|i| i as f32 * 0.2 - 0.9).collect());
    for (row, x) in input.data.chunks_exact(4).enumerate() {
        let normalized: Vec<f64> = (0..4)
            .map(|i| {
                let g = i / 2 * 2;
                let variance = (x[g] as f64).powi(2) / 2.0 + (x[g + 1] as f64).powi(2) / 2.0;
                x[i] as f64 / (variance + 1e-4).sqrt() * (1.0 + NORM[i] as f64)
            })
            .collect();
        let low: Vec<f64> = DOWN
            .chunks_exact(4)
            .map(|w| {
                let v = dot(w, &normalized) / 2.0;
                v * sigmoid(v)
            })
            .collect();
        let weighted: Vec<f64> = UP
            .chunks_exact(2)
            .enumerate()
            .map(|(i, w)| sigmoid(dot(w, &low)) * normalized[i])
            .collect();
        expected_mixed.extend((0..2).map(|i| ((weighted[i] + weighted[i + 2]) / 2.0) as f32));
        for (stream, w) in INJECTION.chunks_exact(4).enumerate() {
            let gate = 2.0 * sigmoid(dot(w, &normalized) / 2.0);
            expected_injected.extend((0..2).map(|i| {
                (x[stream * 2 + i] as f64 + gate * sublayer.data[row * 2 + i] as f64) as f32
            }));
        }
    }
    for inject in [true, false] {
        let mut mixer = GatedResidual::<NumericBackend>::new(spec(inject), &context).unwrap();
        mixer.visit_parameters_mut(&mut Load);
        let state = mixer.forward(&input, &context).unwrap();
        assert_tensor_close(
            &state.mixed,
            &NumericTensor::new([2, 3, 2], expected_mixed.clone()),
            "gated stream collapse",
        );
        if inject {
            let output = state.inject(&sublayer, &context).unwrap();
            assert_tensor_close(
                &output,
                &NumericTensor::new([2, 3, 2, 2], expected_injected.clone()),
                "gated stream injection",
            );
            // Per-token arithmetic is independent of prefill chunk boundaries.
            let mut pieces = Vec::new();
            for token in 0..3 {
                let state = mixer
                    .forward(&input.axis_slice(1, token, token + 1), &context)
                    .unwrap();
                pieces.push(
                    state
                        .inject(&sublayer.axis_slice(1, token, token + 1), &context)
                        .unwrap(),
                );
            }
            assert_tensor_exact(
                &NumericTensor::concatenate(&pieces, 1, &context).unwrap(),
                &output,
                "chunked gated residual",
            );
        } else {
            assert!(state.inject(&sublayer, &context).is_err());
        }
    }
}

#[test]
fn residual_geometry_checks_complete_stream_transport() {
    let context = NumericContext::default();
    let geometry = ResidualStreamGeometry::new(3, 2).unwrap();
    let hidden = NumericTensor::new([1, 2, 2], vec![1.0, 2.0, -3.0, 4.0]);
    let expanded = geometry.expand(&hidden, &context).unwrap();
    assert_eq!(expanded.shape, [1, 2, 3, 2]);
    assert_eq!(
        expanded.data,
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, -3.0, 4.0, -3.0, 4.0, -3.0, 4.0]
    );
    let flat = geometry.flatten(&expanded, &context).unwrap();
    assert_eq!(flat.shape, [1, 2, 6]);
    assert_tensor_exact(
        &geometry.unflatten(&flat, &context).unwrap(),
        &expanded,
        "stream transport round trip",
    );
    assert!(geometry.flatten(&hidden, &context).is_err());
    assert!(geometry.unflatten(&hidden, &context).is_err());
    assert!(ResidualStreamGeometry::new(0, 2).is_err());
    assert!(ResidualStreamGeometry::new(2, i32::MAX).is_err());
    let mut invalid = spec(true);
    invalid.normalization.groups = None;
    assert!(invalid.validate().is_err());
    invalid = spec(true);
    invalid.up.output = 3;
    assert!(invalid.validate().is_err());
}

const EMBEDDING: [f32; 8] = [0.2, -0.7, 0.5, 0.1, -0.3, 0.8, 0.9, -0.4];
const HEAD: [f32; 8] = [0.3, 0.6, -0.2, 0.8, 0.7, -0.1, -0.5, -0.4];
struct BoundaryLoad {
    start: usize,
}
impl<'a> ParameterVisitorMut<'a, NumericTensor> for BoundaryLoad {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let source = match metadata.id.as_str() {
            "embedding" => &EMBEDDING,
            "head" => &HEAD,
            _ => {
                Load.visit_mut(metadata, value);
                return;
            }
        };
        let count = value.data.len();
        value
            .data
            .copy_from_slice(&source[self.start * 2..self.start * 2 + count]);
    }
}
fn vocabulary_specs(tied: bool) -> (EmbeddingSpec, Option<LinearSpec>) {
    (
        EmbeddingSpec {
            vocabulary: 4,
            dimensions: 2,
            weight: ParameterSpec::trainable("embedding").unwrap(),
            format: dense_linear_format(),
        },
        (!tied).then(|| LinearSpec {
            input: 2,
            output: 4,
            weight: ParameterSpec::trainable("head").unwrap(),
            bias: None,
            format: dense_linear_format(),
        }),
    )
}
fn scalar_readout(input: &NumericTensor, tied: bool, edited: bool) -> NumericTensor {
    let mut output = vec![];
    for x in input.data.chunks_exact(4) {
        let x: Vec<f64> = x
            .iter()
            .map(|v| *v as f64 * if edited { 0.7 } else { 1. })
            .collect();
        let normalized: Vec<f64> = (0..4)
            .map(|i| {
                let g = i / 2 * 2;
                let variance = (x[g].powi(2) + x[g + 1].powi(2)) / 2.;
                x[i] / (variance + 1e-4).sqrt() * (1. + NORM[i] as f64)
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
        let mixed: Vec<f64> = (0..2)
            .map(|i| {
                let v = (weighted[i] + weighted[i + 2]) / 2.;
                if edited {
                    v * 1.4 + 0.13
                } else {
                    v
                }
            })
            .collect();
        output.extend(
            (if tied { &EMBEDDING } else { &HEAD })
                .chunks_exact(2)
                .map(|w| dot(w, &mixed) as f32),
        );
    }
    NumericTensor::new([input.dim(0), input.dim(1), 4], output)
}
#[derive(Default)]
struct BoundaryObservation {
    seen: BTreeMap<String, NumericTensor>,
    edited: bool,
}
impl eredu_runtime::ActivationObserver<NumericTensor, Error> for BoundaryObservation {
    fn observe(&mut self, path: &str, value: &NumericTensor) -> Result<(), Error> {
        self.seen.insert(path.into(), value.clone());
        Ok(())
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &NumericTensor,
    ) -> Result<Option<NumericTensor>, Error> {
        if !self.edited {
            return Ok(None);
        }
        Ok(match path {
            "readout.residual" => Some(value.map(|v| v * 0.7)),
            "readout.normalized" => Some(value.map(|v| v * 1.4 + 0.13)),
            _ => None,
        })
    }
}

#[test]
fn shared_decoder_boundary_preserves_residuals_media_and_actual_readout_edits() {
    use eredu_architectures::{
        decoder::{ComponentInstrumentation, StaticModules},
        hybrid_decoder::HybridDecoder,
        qwen4_exp::residual::ResidualBoundary,
    };
    let context = NumericContext::default();
    let input = NumericTensor::new(
        [2, 3, 2, 2],
        (0..24).map(|i| ((i * 7 % 23) as f32 - 11.) / 5.).collect(),
    );
    for tied in [false, true] {
        let (embedding, head) = vocabulary_specs(tied);
        let boundary = ResidualBoundary::<NumericBackend>::new(spec(false), &context).unwrap();
        let mut modules =
            StaticModules::from_boundary(embedding, head, boundary, &context).unwrap();
        modules.visit_parameters_mut(&mut BoundaryLoad { start: 0 });
        let mut decoder = HybridDecoder::new_with_prediction_groups(
            modules,
            "model.layers",
            2,
            "mtp.layers",
            1,
            1,
        )
        .unwrap();
        let tokens = NumericTensor::token_ids(&[2, 0, 3]);
        let embedded = decoder
            .static_modules_mut()
            .embed(&tokens, &context)
            .unwrap();
        assert_eq!(embedded.shape, [1, 3, 2, 2]);
        assert_eq!(
            embedded.data,
            [-0.3, 0.8, -0.3, 0.8, 0.2, -0.7, 0.2, -0.7, 0.9, -0.4, 0.9, -0.4]
        );
        let media = NumericTensor::new([1, 2, 2], vec![0.45, -0.8, 0.1, 0.3]);
        let expanded = decoder
            .static_modules()
            .expand_embeddings(&media, &context)
            .unwrap();
        assert_eq!(expanded.data, [0.45, -0.8, 0.45, -0.8, 0.1, 0.3, 0.1, 0.3]);
        let before = input.clone();
        let logits = decoder.finish_logits(&input, &context).unwrap();
        assert_tensor_close(
            &logits,
            &scalar_readout(&input, tied, false),
            "shared gated readout",
        );
        assert_tensor_exact(
            &input,
            &before,
            "readout preserves pre-collapse prediction source",
        );
        let prediction = decoder.begin_group(1, &logits, &[&input]).unwrap();
        assert_tensor_exact(
            &prediction,
            &input,
            "prediction receives all residual streams",
        );
        assert_tensor_close(
            &decoder.finish_text_logits(&input, &context).unwrap(),
            &logits.axis_slice(1, 2, 3),
            "last-position readout retains the stream axis",
        );
        let chunks: Vec<_> = (0..3)
            .map(|i| {
                decoder
                    .finish_logits(&input.axis_slice(1, i, i + 1), &context)
                    .unwrap()
            })
            .collect();
        assert_tensor_exact(
            &NumericTensor::concatenate(&chunks, 1, &context).unwrap(),
            &logits,
            "chunked readout",
        );
        let mut observer = BoundaryObservation {
            edited: true,
            ..Default::default()
        };
        let edited = decoder
            .finish_logits_instrumented(
                &input,
                &context,
                &mut ComponentInstrumentation::new("readout", &mut observer),
            )
            .unwrap();
        assert_tensor_close(
            &edited,
            &scalar_readout(&input, tied, true),
            "consumed boundary edits",
        );
        assert_eq!(observer.seen["readout.residual"].shape, [2, 3, 2, 2]);
        assert_eq!(observer.seen["readout.normalized"].shape, [2, 3, 2]);
        assert_tensor_exact(
            &observer.seen["readout.normalized.effective"],
            &observer.seen["readout.projection_input"],
            "projection consumes collapsed edit",
        );
        assert!(decoder
            .finish_text_logits(&NumericTensor::zeros([1, 0, 2, 2]), &context)
            .is_err());
        assert!(decoder
            .finish_text_logits(&NumericTensor::zeros([2]), &context)
            .is_err());
    }
    assert!(ResidualBoundary::<NumericBackend>::new(spec(true), &context).is_err());
    let (mut embedding, head) = vocabulary_specs(false);
    embedding.dimensions = 3;
    assert!(StaticModules::from_boundary(
        embedding,
        head,
        ResidualBoundary::<NumericBackend>::new(spec(false), &context).unwrap(),
        &context
    )
    .is_err());
}

#[test]
fn shared_decoder_boundary_matches_tp2_tp4_vocabulary_and_rebuilt_modules() {
    use eredu_architectures::{
        decoder::{ComponentInstrumentation, StaticModules},
        hybrid_decoder::HybridDecoder,
        qwen4_exp::residual::ResidualBoundary,
    };
    for tied in [true, false] {
        for ranks in [2, 4] {
            let group = NumericParallelGroup::new(ranks);
            let outputs = std::thread::scope(|scope| {
                (0..ranks)
                    .map(|rank| {
                        let group = group.clone();
                        scope.spawn(move || {
                            let context = NumericContext::default();
                            let parallel = NumericParallelContext::new(rank, group);
                            let range = VocabularyParallelRange {
                                global_vocabulary: 4,
                                local: rank * 4 / ranks..(rank + 1) * 4 / ranks,
                            };
                            let mut results = vec![];
                            // Rebuilding unloaded modules and rebinding exact shards must preserve the same readout.
                            for _ in 0..2 {
                                let (embedding, head) = vocabulary_specs(tied);
                                let mut modules = StaticModules::from_parallel_boundary(
                                    embedding,
                                    head,
                                    ResidualBoundary::<NumericBackend>::new(spec(false), &context)
                                        .unwrap(),
                                    range.clone(),
                                    (!tied).then(|| range.clone()),
                                    &context,
                                )
                                .unwrap();
                                modules.visit_parameters_mut(&mut BoundaryLoad {
                                    start: range.local.start,
                                });
                                let expanded = modules
                                    .embed_parallel(
                                        &NumericTensor::token_ids(&[0, 3, 2]),
                                        EmbeddingLookupPolicy::Strict,
                                        &parallel,
                                        &context,
                                    )
                                    .unwrap();
                                let mut decoder =
                                    HybridDecoder::new(modules, "model.layers", 2).unwrap();
                                let logits = decoder
                                    .finish_logits_parallel_instrumented(
                                        &expanded,
                                        &parallel,
                                        &context,
                                        &mut ComponentInstrumentation::disabled(),
                                    )
                                    .unwrap();
                                assert_tensor_close(
                                    &logits,
                                    &scalar_readout(&expanded, tied, false),
                                    "TP gated readout",
                                );
                                results.push(logits);
                            }
                            assert_tensor_exact(
                                &results[0],
                                &results[1],
                                "rebuilt parallel readout",
                            );
                            results.remove(0)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|t| t.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for output in &outputs[1..] {
                assert_tensor_exact(output, &outputs[0], "replicated boundary result");
            }
        }
    }
}
