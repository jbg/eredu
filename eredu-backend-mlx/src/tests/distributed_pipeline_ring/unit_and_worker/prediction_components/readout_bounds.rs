fn edited_swiglu_equation(
    equation: &eredu_core::component::ComponentActivation,
    input: &[f32],
    gate_weights: &[f32],
    value_weights: &[f32],
) -> (f64, f64) {
    use eredu_core::component::{ComponentActivation, ComponentNonlinearity};
    let ComponentActivation::Gated {
        activation: ComponentNonlinearity::Silu { multiplier },
        gate_upper_bound,
        value_absolute_bound,
        value_offset,
    } = equation
    else {
        panic!("SwiGLU fixture equation")
    };
    let nu = input.len() as f64 * f64::from(f32::EPSILON);
    let gamma = nu / (1. - nu);
    let dot = |weights: &[f32]| {
        weights
            .iter()
            .zip(input)
            .fold((0_f64, 0_f64), |(sum, magnitude), (w, x)| {
                let term = f64::from(*w) * f64::from(*x);
                (sum + term, magnitude + term.abs())
            })
    };
    let (mut gate, gate_magnitude) = dot(gate_weights);
    let (mut value, value_magnitude) = dot(value_weights);
    if let Some(limit) = gate_upper_bound {
        gate = gate.min(f64::from(limit.value()));
    }
    if let Some(limit) = value_absolute_bound {
        value = value.clamp(-f64::from(limit.value()), f64::from(limit.value()));
    }
    value += f64::from(value_offset.value());
    let activated = gate / (1. + (-f64::from(multiplier.value()) * gate).exp());
    // SiLU has derivative magnitude below two; clamping is nonexpansive.
    // Include a finite 16-ulp allowance for elementary-function arithmetic.
    let gate_error = 2. * gamma * gate_magnitude + 16. * f64::from(f32::EPSILON) * activated.abs();
    let value_error = gamma * value_magnitude + f64::from(f32::EPSILON) * value.abs();
    let bound = gate_error * value.abs()
        + (activated.abs() + gate_error) * value_error
        + 2. * f64::from(f32::EPSILON)
            * ((activated.abs() + gate_error) * (value.abs() + value_error));
    (activated * value, bound)
}

// Forward-error evidence for the FP32 head after a quantized overlay promotion.
// Earlier normalized-input comparisons keep their original tolerance. This
// accounts separately for propagation through a cancellation-sensitive readout.
struct EditedReadoutEvidence {
    equation: eredu_core::component::ComponentReadoutEquation,
    width: usize,
    weights: Vec<f32>,
    gated: Vec<EditedGatedEvidence>,
    routed: Vec<EditedRoutedEvidence>,
    ffn_writes: Vec<EditedFfnWriteEvidence>,
}

struct EditedGatedEvidence {
    group: eredu_core::component::ComponentGroup,
    width: usize,
    gate: Vec<f32>,
    value: Vec<f32>,
}

fn prediction_f32_tensor<'a>(
    step: &'a eredu_core::speculative::SpeculativeActivationCapture,
    path: &str,
) -> Option<&'a [f32]> {
    step.captures.records.iter().find_map(|record| {
        if record.path != path {
            return None;
        }
        match &record.payload {
            Some(eredu_core::capture::CapturePayload::Tensor(value)) => match value.data() {
                eredu_core::TensorObservationData::F32(values) => Some(values.as_slice()),
                _ => panic!("FP32 evidence required for {path}"),
            },
            _ => None,
        }
    })
}

// Check each stream equation independently in both executions before allowing
// the measured input/coefficient difference to propagate through cancellation.
// Coefficients and initial inputs retain ordinary comparisons. Writes additionally
// use the independent FFN evidence when that equation is supplied.
fn edited_stream_pair_bounds(
    readout: &eredu_core::component::ComponentReadoutEquation,
    actual: &eredu_core::speculative::SpeculativeActivationCapture,
    expected: &eredu_core::speculative::SpeculativeActivationCapture,
) -> std::collections::BTreeMap<String, Vec<f64>> {
    type Step = eredu_core::speculative::SpeculativeActivationCapture;
    let mixing = readout.stream_residual.as_ref().unwrap();
    let mut result = std::collections::BTreeMap::new();
    let Some(first) = mixing.cycles.first() else {
        return result;
    };
    let Some(input) = prediction_f32_tensor(actual, &first.input) else {
        return result;
    };
    let streams = mixing.streams;
    let rows = prediction_f32_tensor(actual, &first.pre).unwrap().len() / streams;
    let hidden = input.len() / (rows * streams);
    assert_eq!(input.len(), rows * streams * hidden);
    let gamma = |operations: usize| {
        let nu = operations as f64 * f64::from(f32::EPSILON) / 2.;
        assert!(nu < 1.);
        nu / (1. - nu)
    };
    let check = |path: &str, terms: &dyn Fn(&Step, usize) -> Vec<f64>| {
        let a = prediction_f32_tensor(actual, path).unwrap();
        let b = prediction_f32_tensor(expected, path).unwrap();
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).enumerate().map(|(index, (a, b))| {
            let equation = |step: &Step, observed: f32| {
                let terms = terms(step, index);
                let exact = terms.iter().sum::<f64>();
                let bound = gamma(2 * terms.len()) * terms.iter().map(|term| term.abs()).sum::<f64>();
                assert!((f64::from(observed) - exact).abs() <= bound,
                    "independent F64 stream equation {path} index {index}: {observed} != {exact}, bound {bound}");
                (exact, bound)
            };
            let (a_exact, a_bound) = equation(actual, *a);
            let (b_exact, b_bound) = equation(expected, *b);
            (a_exact - b_exact).abs() + a_bound + b_bound
        }).collect::<Vec<_>>()
    };
    for (ordinal, cycle) in mixing.cycles.iter().enumerate() {
        let collapsed = check(&cycle.collapsed, &|step, index| {
            let incoming = prediction_f32_tensor(step, &cycle.input).unwrap();
            let pre = prediction_f32_tensor(step, &cycle.pre).unwrap();
            let row = index / hidden;
            let h = index % hidden;
            (0..streams)
                .map(|s| {
                    f64::from(incoming[(row * streams + s) * hidden + h])
                        * f64::from(pre[row * streams + s])
                })
                .collect()
        });
        result.insert(cycle.collapsed.clone(), collapsed);
        let output = check(&cycle.output, &|step, index| {
            let incoming = prediction_f32_tensor(step, &cycle.input).unwrap();
            let combination = prediction_f32_tensor(step, &cycle.combination).unwrap();
            let post = prediction_f32_tensor(step, &cycle.post).unwrap();
            let write = prediction_f32_tensor(step, &cycle.write).unwrap();
            let row = index / (streams * hidden);
            let to = index / hidden % streams;
            let h = index % hidden;
            std::iter::once(
                f64::from(post[row * streams + to]) * f64::from(write[row * hidden + h]),
            )
            .chain((0..streams).map(|from| {
                f64::from(combination[(row * streams + from) * streams + to])
                    * f64::from(incoming[(row * streams + from) * hidden + h])
            }))
            .collect()
        });
        // These trials have parameter edits but no activation interventions.
        // Verify the declared producer/consumer edge before sharing its bound.
        let consumer = mixing
            .cycles
            .get(ordinal + 1)
            .map_or(&mixing.head.input, |next| &next.input);
        for step in [actual, expected] {
            assert_eq!(
                prediction_f32_tensor(step, &cycle.output).unwrap(),
                prediction_f32_tensor(step, consumer).unwrap(),
                "unmodified stream edge"
            );
        }
        result.insert(consumer.clone(), output.clone());
        result.insert(cycle.output.clone(), output);
    }
    let residual = check(&readout.residual, &|step, index| {
        let incoming = prediction_f32_tensor(step, &mixing.head.input).unwrap();
        let coefficients = prediction_f32_tensor(step, &mixing.head.coefficients).unwrap();
        let row = index / hidden;
        let h = index % hidden;
        (0..streams)
            .map(|s| {
                f64::from(incoming[(row * streams + s) * hidden + h])
                    * f64::from(coefficients[row * streams + s])
            })
            .collect()
    });
    result.insert(readout.residual.clone(), residual);
    result
}

fn independently_edited_fixture_matrix(
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    source: &safetensors::SafeTensors<'_>,
    facts: &eredu_core::parameters::ParameterDiscovery,
    id: &str,
    edits: &[eredu_core::parameters::ParameterEdit],
    limits: eredu_core::capture::CaptureUsage,
) -> Vec<f32> {
    use eredu_core::parameters::*;
    let parameter = facts
        .parameters
        .iter()
        .find(|parameter| parameter.id == id)
        .unwrap();
    assert!((2..=3).contains(&parameter.shape.len()));
    let original = MlxBackend::query_parameter(
        reference,
        &facts.identity,
        id,
        ParameterRegion {
            starts: vec![0; parameter.shape.len()],
            shape: parameter.shape.clone(),
        },
        limits,
    )
    .unwrap();
    verify_v4_fp8_source_row(
        source,
        id,
        &parameter.shape,
        &original.region,
        &original.values,
    );
    let mut weights = original.values;
    for edit in edits {
        let member = facts
            .parameters
            .iter()
            .find(|member| member.id == edit.parameter)
            .unwrap();
        if member.shared_id != parameter.shared_id {
            continue;
        }
        let ParameterUpdate::Add { values } = &edit.update else {
            panic!("additive fixture edit")
        };
        for (offset, value) in values.iter().enumerate() {
            let mut remaining = offset as u64;
            let mut global = 0;
            let mut stride = 1;
            for axis in (0..parameter.shape.len()).rev() {
                global += (edit.region.starts[axis] + remaining % edit.region.shape[axis]) * stride;
                remaining /= edit.region.shape[axis];
                stride *= parameter.shape[axis];
            }
            weights[global as usize] += value;
        }
    }
    weights
}

impl EditedGatedEvidence {
    fn bounds(
        &self,
        actual: &eredu_core::speculative::SpeculativeActivationCapture,
        expected: &eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Option<Vec<f64>> {
        let a_values = prediction_f32_tensor(actual, &self.group.activation)?;
        let b_values = prediction_f32_tensor(expected, &self.group.activation).unwrap();
        assert_eq!(
            a_values,
            prediction_f32_tensor(actual, &self.group.effective_activation).unwrap()
        );
        assert_eq!(
            b_values,
            prediction_f32_tensor(expected, &self.group.effective_activation).unwrap()
        );
        let a_input = prediction_f32_tensor(actual, &self.group.input).unwrap();
        let b_input = prediction_f32_tensor(expected, &self.group.input).unwrap();
        assert_eq!(a_input.len(), b_input.len());
        assert_eq!(
            a_values.len(),
            a_input.len() / self.width * self.group.count
        );
        assert_eq!(b_values.len(), a_values.len());
        let mut bounds = Vec::with_capacity(a_values.len());
        let mut max_effect = 0_f64;
        for (token, (a, b)) in a_input
            .chunks_exact(self.width)
            .zip(b_input.chunks_exact(self.width))
            .enumerate()
        {
            for unit in 0..self.group.count {
                let gate = &self.gate[unit * self.width..(unit + 1) * self.width];
                let value = &self.value[unit * self.width..(unit + 1) * self.width];
                let (a_exact, a_bound) =
                    edited_swiglu_equation(&self.group.activation_equation, a, gate, value);
                let (b_exact, b_bound) =
                    edited_swiglu_equation(&self.group.activation_equation, b, gate, value);
                let index = token * self.group.count + unit;
                for (value, exact, bound) in [
                    (a_values[index], a_exact, a_bound),
                    (b_values[index], b_exact, b_bound),
                ] {
                    assert!((f64::from(value) - exact).abs() <= bound, "SwiGLU independent F64 equation {} token={token} unit={unit}: {value} != {exact}, bound {bound}", self.group.id);
                }
                let effect = (a_exact - b_exact).abs();
                max_effect = max_effect.max(effect);
                bounds.push(effect + a_bound + b_bound);
            }
        }
        eprintln!(
            "edited SwiGLU {:?} {} maximum propagated input effect={max_effect}",
            actual.phase, self.group.id
        );
        Some(bounds)
    }
}

impl EditedReadoutEvidence {
    fn output_path(&self) -> &str {
        if self.equation.score_writes.is_empty() {
            &self.equation.logits
        } else {
            &self.equation.linear_scores
        }
    }

    fn bounds(
        &self,
        actual: &eredu_core::speculative::SpeculativeActivationCapture,
        expected: &eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Option<Vec<f64>> {
        fn tensor<'a>(
            step: &'a eredu_core::speculative::SpeculativeActivationCapture,
            path: &str,
        ) -> Option<&'a [f32]> {
            step.captures.records.iter().find_map(|record| {
                if record.path != path {
                    return None;
                }
                match &record.payload {
                    Some(eredu_core::capture::CapturePayload::Tensor(value)) => {
                        match value.data() {
                            eredu_core::TensorObservationData::F32(values) => {
                                Some(values.as_slice())
                            }
                            _ => panic!("promoted readout must expose FP32 evidence"),
                        }
                    }
                    _ => None,
                }
            })
        }
        let a_output = tensor(actual, self.output_path())?;
        let b_output = tensor(expected, self.output_path()).unwrap();
        let path = self.equation.projection_input.as_ref().unwrap();
        let a_input = tensor(actual, path).unwrap();
        let b_input = tensor(expected, path).unwrap();
        assert_eq!(
            a_input,
            tensor(actual, &self.equation.normalized).unwrap(),
            "promoted head has no input quantization"
        );
        assert_eq!(
            b_input,
            tensor(expected, &self.equation.normalized).unwrap()
        );
        assert_eq!(a_input.len(), b_input.len());
        assert_eq!(a_input.len() % self.width, 0);
        let vocabulary = self.weights.len() / self.width;
        assert_eq!(a_output.len(), a_input.len() / self.width * vocabulary);
        assert_eq!(a_output.len(), b_output.len());
        // At most one rounding for each multiply and add. The F64 oracle
        // retains the signed sum; the bound uses the sum of absolute products.
        let nu = self.width as f64 * f64::from(f32::EPSILON);
        assert!(nu < 1.);
        let gamma = nu / (1. - nu);
        let mut bounds = Vec::with_capacity(a_output.len());
        let mut largest_rounding_ratio = 0_f64;
        let mut largest_input_effect = 0_f64;
        for (token, (a, b)) in a_input
            .chunks_exact(self.width)
            .zip(b_input.chunks_exact(self.width))
            .enumerate()
        {
            for (row, weight) in self.weights.chunks_exact(self.width).enumerate() {
                let dot = |input: &[f32]| {
                    weight
                        .iter()
                        .zip(input)
                        .fold((0_f64, 0_f64), |(signed, magnitude), (w, x)| {
                            let term = f64::from(*w) * f64::from(*x);
                            (signed + term, magnitude + term.abs())
                        })
                };
                let (a_exact, a_magnitude) = dot(a);
                let (b_exact, b_magnitude) = dot(b);
                let index = token * vocabulary + row;
                let a_rounding = gamma * a_magnitude;
                let b_rounding = gamma * b_magnitude;
                for (value, exact, bound) in [
                    (a_output[index], a_exact, a_rounding),
                    (b_output[index], b_exact, b_rounding),
                ] {
                    let error = (f64::from(value) - exact).abs();
                    assert!(error <= bound, "FP32 readout must match its independent signed F64 dot: {value} != {exact}, bound {bound}");
                    if bound > 0. {
                        largest_rounding_ratio = largest_rounding_ratio.max(error / bound);
                    }
                }
                let input_effect = (a_exact - b_exact).abs();
                largest_input_effect = largest_input_effect.max(input_effect);
                bounds.push(input_effect + a_rounding + b_rounding);
            }
        }
        eprintln!("edited readout {:?}: max propagated input difference={largest_input_effect}, max FP32 forward-bound fraction={largest_rounding_ratio}", actual.phase);
        Some(bounds)
    }
}

struct EditedRoutedEvidence {
    group: eredu_core::component::RoutedComponentGroup,
    parameter: String,
    shape: Vec<u64>,
    weights: Vec<f32>,
}

impl EditedRoutedEvidence {
    fn bounds(
        &self,
        actual: &eredu_core::speculative::SpeculativeActivationCapture,
        expected: &eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Option<Vec<Vec<f64>>> {
        use eredu_core::{capture::*, component::ComponentReadRole, TensorObservationData};
        fn sparse<'a>(
            step: &'a eredu_core::speculative::SpeculativeActivationCapture,
            path: &str,
        ) -> Option<&'a RoutedUnitCapture> {
            step.captures.records.iter().find_map(|record| {
                if record.path != path {
                    return None;
                }
                match &record.payload {
                    Some(CapturePayload::RoutedUnits(value)) => Some(value),
                    _ => None,
                }
            })
        }
        let a = sparse(actual, &self.group.activation)?;
        let b = sparse(expected, &self.group.activation).unwrap();
        let ae = sparse(actual, &self.group.effective_activation).unwrap();
        let be = sparse(expected, &self.group.effective_activation).unwrap();
        let input_path = self.group.input.as_ref().unwrap();
        let ai = prediction_f32_tensor(actual, input_path).unwrap();
        let bi = prediction_f32_tensor(expected, input_path).unwrap();
        assert_eq!(ai.len(), bi.len());
        assert_eq!(a.rows.len(), b.rows.len());
        assert_eq!(a.rows.len(), ae.rows.len());
        assert_eq!(b.rows.len(), be.rows.len());
        let read = |role| {
            self.group
                .reads
                .iter()
                .find(|read| read.role == role)
                .unwrap()
        };
        let width = self.group.input_width;
        let rows = self.shape[1] as usize;
        let mut max_effect = 0_f64;
        let bounds = a.rows.iter().zip(&b.rows).zip(ae.rows.iter().zip(&be.rows)).map(|((a,b),(ae,be))| {
            assert_eq!((a.token,a.expert,a.unit_start,a.unit_stride), (b.token,b.expert,b.unit_start,b.unit_stride));
            assert_eq!(a.values, ae.values, "unmasked original/effective routed units");
            assert_eq!(b.values, be.values);
            let (TensorObservationData::F32(av), TensorObservationData::F32(bv)) = (a.values.data(), b.values.data()) else { panic!("FP32 routed units") };
            let start = a.token as usize * width;
            let a_input = &ai[start..start+width];
            let b_input = &bi[start..start+width];
            av.iter().zip(bv).enumerate().map(|(index,(av,bv))| {
                let unit = a.unit_start as usize + index * a.unit_stride as usize;
                let weights = |role| {
                    let range = read(role).rows.row_range(unit).unwrap();
                    assert_eq!(range.len(), 1);
                    let start = (a.expert as usize*rows + range.start)*width;
                    &self.weights[start..start+width]
                };
                let gate = weights(ComponentReadRole::Gate);
                let value = weights(ComponentReadRole::Value);
                let (ax,ab) = edited_swiglu_equation(&self.group.activation_equation, a_input, gate, value);
                let (bx,bb) = edited_swiglu_equation(&self.group.activation_equation, b_input, gate, value);
                for (observed, exact, bound) in [(*av,ax,ab),(*bv,bx,bb)] {
                    assert!((f64::from(observed)-exact).abs() <= bound, "routed independent F64 equation {} token={} expert={} unit={unit}: {observed} != {exact}, bound {bound}", self.group.id, a.token, a.expert);
                }
                let effect = (ax-bx).abs();
                max_effect = max_effect.max(effect);
                effect+ab+bb
            }).collect()
        }).collect();
        eprintln!(
            "edited routed SwiGLU {:?} {} maximum propagated input effect={max_effect}",
            actual.phase, self.group.id
        );
        Some(bounds)
    }
}

// Independent signed reconstruction of a complete routed-plus-shared FFN write.
// Each matrix is checked against decoded source bytes plus the admitted edits.
// All write matrices are promoted by those edits, so the captured effective
// units are their actual F32 multiplication input; no input quantization remains.
struct EditedFfnWriteEvidence {
    path: String,
    shared: eredu_core::component::ComponentGroup,
    routed: eredu_core::component::RoutedComponentGroup,
    shared_weights: Vec<f32>,
    routed_weights: Vec<f32>,
}

impl EditedFfnWriteEvidence {
    fn bounds(
        &self,
        actual: &eredu_core::speculative::SpeculativeActivationCapture,
        expected: &eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Option<Vec<f64>> {
        use eredu_core::{capture::CapturePayload, TensorObservationData};
        let a = prediction_f32_tensor(actual, &self.path)?;
        let b = prediction_f32_tensor(expected, &self.path).unwrap();
        assert_eq!(a.len(), b.len());
        let width = self.routed.output_width;
        let units = self.routed.units_per_expert;
        assert_eq!(self.shared_weights.len(), width * self.shared.count);
        assert_eq!(
            self.routed_weights.len(),
            self.routed.expert_count * width * units
        );
        let reconstruct = |step: &eredu_core::speculative::SpeculativeActivationCapture| {
            let shared = prediction_f32_tensor(step, &self.shared.effective_activation).unwrap();
            let routed = step
                .captures
                .records
                .iter()
                .find_map(|record| {
                    if record.path != self.routed.effective_activation {
                        return None;
                    }
                    match &record.payload {
                        Some(CapturePayload::RoutedUnits(value)) => Some(value),
                        _ => None,
                    }
                })
                .unwrap();
            assert_eq!(shared.len(), a.len() / width * self.shared.count);
            (0..a.len())
                .map(|index| {
                    let token = index / width;
                    let channel = index % width;
                    let mut sum = 0_f64;
                    let mut magnitude = 0_f64;
                    let mut terms = 0_usize;
                    let mut add = |value: f64| {
                        sum += value;
                        magnitude += value.abs();
                        terms += 1;
                    };
                    for unit in 0..self.shared.count {
                        add(f64::from(shared[token * self.shared.count + unit])
                            * f64::from(self.shared_weights[channel * self.shared.count + unit]));
                    }
                    let rows = routed
                        .rows
                        .iter()
                        .filter(|row| row.token as usize == token)
                        .collect::<Vec<_>>();
                    assert!(!rows.is_empty());
                    for row in rows {
                        assert_eq!((row.unit_start, row.unit_stride), (0, 1));
                        let TensorObservationData::F32(values) = row.values.data() else {
                            panic!("FP32 routed write input")
                        };
                        assert_eq!(values.len(), units);
                        for (unit, value) in values.iter().enumerate() {
                            add(f64::from(*value)
                                * f64::from(
                                    self.routed_weights
                                        [(row.expert as usize * width + channel) * units + unit],
                                )
                                * f64::from(row.coefficient));
                        }
                    }
                    // Each term has at most two multiplies plus reductions. Using
                    // all terms also bounds TP/EP reduction regrouping conservatively.
                    let nu = (3 * terms) as f64 * f64::from(f32::EPSILON) / 2.;
                    assert!(nu < 1.);
                    (sum, nu / (1. - nu) * magnitude)
                })
                .collect::<Vec<_>>()
        };
        let ax = reconstruct(actual);
        let bx = reconstruct(expected);
        Some(a.iter().zip(b).zip(ax.iter().zip(&bx)).enumerate().map(|(index, ((a,b), ((ax,ab),(bx,bb))))| {
            for (observed, exact, bound) in [(*a,*ax,*ab),(*b,*bx,*bb)] {
                assert!((f64::from(observed) - exact).abs() <= bound,
                    "independent signed FFN write {} index={index}: {observed} != {exact}, bound {bound}", self.path);
            }
            (ax-bx).abs() + ab + bb
        }).collect())
    }
}
