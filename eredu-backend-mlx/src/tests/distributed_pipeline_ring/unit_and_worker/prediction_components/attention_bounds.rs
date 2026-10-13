// Dynamic FP8 is discontinuous at rounding midpoints: close attention channels
// need not produce equally close writes. Check both executions against scalar
// quantization and F64 projection equations before comparing their writes.
struct PredictionAttentionEvidence {
    group: eredu_core::component::ComponentGroup,
    output: String,
    first: Vec<f32>,
    last: Vec<f32>,
    transforms: [eredu_core::parameters::ProjectionInputTransform; 2],
}

fn prediction_attention_evidence(
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    scope: &eredu_core::component::ComponentExecutionScope,
    source: &safetensors::SafeTensors<'_>,
    facts: &eredu_core::parameters::ParameterDiscovery,
    edits: &[eredu_core::parameters::ParameterEdit],
    limits: eredu_core::capture::CaptureUsage,
) -> Vec<PredictionAttentionEvidence> {
    use eredu_core::parameters::ProjectionInputTransform;
    scope
        .components
        .iter()
        .filter_map(|group| {
            let factor = group.write_input_projection.as_ref()?;
            assert!(factor.bias.is_none() && group.write_bias.is_none());
            let ids = [&factor.weight, &group.write_weight];
            let transforms = ids.map(|id| {
                let parameter = facts.parameters.iter().find(|p| &p.id == id).unwrap();
                if edits.iter().any(|edit| {
                    facts
                        .parameters
                        .iter()
                        .any(|p| p.id == edit.parameter && p.shared_id == parameter.shared_id)
                }) {
                    ProjectionInputTransform::Identity
                } else {
                    parameter.input_transform.clone()
                }
            });
            Some(PredictionAttentionEvidence {
                group: group.clone(),
                output: scope
                    .readout
                    .stream_residual
                    .as_ref()
                    .unwrap()
                    .cycles
                    .iter()
                    .find(|cycle| cycle.node_id == group.node_id)
                    .unwrap()
                    .write
                    .clone(),
                first: independently_edited_fixture_matrix(
                    reference, source, facts, ids[0], edits, limits,
                ),
                last: independently_edited_fixture_matrix(
                    reference, source, facts, ids[1], edits, limits,
                ),
                transforms,
            })
        })
        .collect()
}

fn prediction_projection_input(
    values: &[f32],
    width: usize,
    transform: &eredu_core::parameters::ProjectionInputTransform,
) -> Vec<f32> {
    use eredu_core::parameters::ProjectionInputTransform;
    match transform {
        ProjectionInputTransform::Identity => values.to_vec(),
        ProjectionInputTransform::BlockFp8E4m3 {
            block_width,
            floor,
            magnitude,
        } => {
            let decode = |code: u8| {
                let exponent = i32::from((code >> 3) & 15);
                let mantissa = f32::from(code & 7);
                if exponent == 0 {
                    mantissa * 2_f32.powi(-9)
                } else {
                    (1. + mantissa / 8.) * 2_f32.powi(exponent - 7)
                }
            };
            let levels: [f32; 127] = std::array::from_fn(|code| decode(code as u8));
            values
                .chunks(width)
                .flat_map(|row| {
                    row.chunks(*block_width as usize).flat_map(|block| {
                        let scale = block.iter().fold(floor.value(), |a, b| a.max(b.abs()))
                            / magnitude.value();
                        block.iter().map(move |value| {
                            let scaled = (value / scale).abs();
                            let code = (0_usize..127)
                                .min_by(|a, b| {
                                    (levels[*a] - scaled)
                                        .abs()
                                        .total_cmp(&(levels[*b] - scaled).abs())
                                        .then_with(|| (a & 1).cmp(&(b & 1)))
                                })
                                .unwrap();
                            levels[code] * scale * if value.is_sign_negative() { -1. } else { 1. }
                        })
                    })
                })
                .collect()
        }
        ProjectionInputTransform::Unspecified => {
            panic!("fixture must declare projection arithmetic")
        }
    }
}

impl PredictionAttentionEvidence {
    fn bounds(
        &self,
        actual: &eredu_core::speculative::SpeculativeActivationCapture,
        expected: &eredu_core::speculative::SpeculativeActivationCapture,
    ) -> std::collections::BTreeMap<String, Vec<f64>> {
        let factor = self.group.write_input_projection.as_ref().unwrap();
        let path = self.output.clone();
        let mut result = std::collections::BTreeMap::new();
        let Some(a) = prediction_f32_tensor(actual, &path) else {
            return result;
        };
        let b = prediction_f32_tensor(expected, &path).unwrap();
        let width = self.group.count / factor.groups;
        let latent = factor.groups * factor.rank;
        let hidden = self.last.len() / latent;
        let rows = a.len() / hidden;
        let gamma = |terms: usize| {
            let nu = (2 * terms + 2) as f64 * f64::from(f32::EPSILON) / 2.;
            nu / (1. - nu)
        };
        let check_input = |observed: &[f32], original: &[f32], width, transform| {
            let quantized = prediction_projection_input(original, width, transform);
            assert_eq!(observed.len(), quantized.len());
            for (observed, exact) in observed.iter().zip(quantized) {
                assert!(
                    (observed - exact).abs() <= 2. * f32::EPSILON * exact.abs(),
                    "independent attention input quantization: {observed} != {exact}"
                );
            }
        };
        let reconstruct = |step| {
            let channels = prediction_f32_tensor(step, &self.group.effective_activation).unwrap();
            let first_input = prediction_f32_tensor(step, &factor.input).unwrap();
            let intermediate = prediction_f32_tensor(step, &factor.output).unwrap();
            let last_input = prediction_f32_tensor(step, &factor.final_input).unwrap();
            let mut grouped = Vec::with_capacity(channels.len());
            for group in 0..factor.groups {
                for row in 0..rows {
                    grouped.extend_from_slice(
                        &channels[(row * factor.groups + group) * width
                            ..(row * factor.groups + group + 1) * width],
                    );
                }
            }
            check_input(first_input, &grouped, width, &self.transforms[0]);
            for row in 0..rows {
                for column in 0..latent {
                    let group = column / factor.rank;
                    let terms = (0..width).map(|i| {
                        f64::from(first_input[(group * rows + row) * width + i])
                            * f64::from(self.first[column * width + i])
                    });
                    let (sum, magnitude) = terms.fold((0_f64, 0_f64), |(sum, magnitude), x| {
                        (sum + x, magnitude + x.abs())
                    });
                    assert!(
                        (f64::from(intermediate[row * latent + column]) - sum).abs()
                            <= gamma(width) * magnitude,
                        "independent grouped attention projection {}",
                        self.group.id
                    );
                }
            }
            check_input(last_input, intermediate, latent, &self.transforms[1]);
            (0..a.len())
                .map(|index| {
                    let terms = (0..latent).map(|i| {
                        f64::from(last_input[index / hidden * latent + i])
                            * f64::from(self.last[index % hidden * latent + i])
                    });
                    let (sum, magnitude) = terms.fold((0_f64, 0_f64), |(sum, magnitude), x| {
                        (sum + x, magnitude + x.abs())
                    });
                    (sum, gamma(latent) * magnitude)
                })
                .collect::<Vec<_>>()
        };
        let ax = reconstruct(actual);
        let bx = reconstruct(expected);
        let bounds =
            a.iter()
                .zip(b)
                .zip(ax.iter().zip(&bx))
                .map(|((a, b), ((ax, ab), (bx, bb)))| {
                    for (observed, exact, bound) in [(*a, *ax, *ab), (*b, *bx, *bb)] {
                        assert!((f64::from(observed) - exact).abs() <= bound,
                    "independent attention write {path}: {observed} != {exact}, bound {bound}");
                    }
                    (ax - bx).abs() + ab + bb
                })
                .collect();
        result.insert(path, bounds);
        // These intermediate captures have independent equations above; their
        // cross-execution differences may include an FP8 rounding step.
        for path in [&factor.input, &factor.output, &factor.final_input] {
            let a = prediction_f32_tensor(actual, path).unwrap();
            let b = prediction_f32_tensor(expected, path).unwrap();
            result.insert(
                path.clone(),
                a.iter()
                    .zip(b)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                    .collect(),
            );
        }
        result
    }
}
