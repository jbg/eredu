//! Public prediction equations, effective matrices and coordinated reversible edits.
use super::*;
use eredu_core::{
    TensorObservationData, component::*, intervention::InterventionDtype, parameters::*,
};
use std::collections::BTreeMap;

fn values<'a>(records: &'a [CaptureRecord], path: &str) -> &'a [f32] {
    let record = records
        .iter()
        .find(|r| r.path == path)
        .unwrap_or_else(|| panic!("missing {path}"));
    assert_eq!(record.outcome, CaptureOutcome::Captured, "{path}");
    let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
        panic!("tensor {path}")
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("F32 {path}")
    };
    values
}

fn reconstruct(
    records: &[CaptureRecord],
    scope: &ComponentExecutionScope,
    weights: &BTreeMap<String, Vec<f32>>,
) {
    let (inputs, output) = match &scope.residual_base {
        ComponentResidualBase::ProjectedSum { inputs, output, .. } => (inputs.as_slice(), output),
        ComponentResidualBase::Source { output, .. } => (&[][..], output),
        ComponentResidualBase::LinearFusion { .. } => panic!("V4 uses direct or summed sources"),
    };
    let mixing = scope.readout.stream_residual.as_ref().unwrap();
    let readout = &scope.readout;
    let gain = &weights[readout.normalization.gain.as_ref().unwrap()];
    let hidden = gain.len();
    let streams = mixing.streams;
    let fused = values(records, output);
    let rows = fused.len() / (streams * hidden);
    let mut terms = Vec::new();
    if let ComponentResidualBase::Source {
        input, expansion, ..
    } = &scope.residual_base
    {
        assert!(matches!(expansion,
            ComponentFusionExpansion::BroadcastAxis { axis: 2, extent, .. } if *extent == streams));
        let source = values(records, input);
        terms.push(
            (0..fused.len())
                .map(|i| source[i / (streams * hidden) * hidden + i % hidden] as f64)
                .collect(),
        );
    }
    for input in inputs {
        let before = values(records, &input.projection_input);
        let projected = values(records, &input.output);
        let weight = &weights[&input.weight];
        for (row, (before, actual)) in before
            .chunks_exact(hidden)
            .zip(projected.chunks_exact(hidden))
            .enumerate()
        {
            for h in 0..hidden {
                let expected = before
                    .iter()
                    .zip(&weight[h * hidden..(h + 1) * hidden])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>();
                assert!(
                    (expected - actual[h] as f64).abs() < 3e-5,
                    "{} row {row}",
                    input.weight
                );
            }
        }
        let effective = values(records, &input.effective_output);
        terms.push(match input.expansion {
            ComponentFusionExpansion::Identity => {
                effective.iter().map(|v| *v as f64).collect::<Vec<_>>()
            }
            ComponentFusionExpansion::BroadcastAxis {
                axis: 2, extent, ..
            } => {
                assert_eq!(extent, streams);
                (0..fused.len())
                    .map(|i| effective[i / (streams * hidden) * hidden + i % hidden] as f64)
                    .collect()
            }
            _ => panic!("V4 expansion"),
        });
    }
    for (i, actual) in fused.iter().enumerate() {
        assert!((terms.iter().map(|t| t[i]).sum::<f64>() - *actual as f64).abs() < 3e-5);
    }
    for cycle in &mixing.cycles {
        let coefficients = values(records, &cycle.combination);
        for term in &mut terms {
            let mut mixed = vec![0.; term.len()];
            for row in 0..rows {
                for to in 0..streams {
                    for h in 0..hidden {
                        mixed[(row * streams + to) * hidden + h] = (0..streams)
                            .map(|from| {
                                term[(row * streams + from) * hidden + h]
                                    * coefficients[(row * streams + from) * streams + to] as f64
                            })
                            .sum();
                    }
                }
            }
            *term = mixed;
        }
        let post = values(records, &cycle.post);
        let write = values(records, &cycle.write);
        terms.push(
            (0..fused.len())
                .map(|i| {
                    post[i / hidden] as f64
                        * write[i / (streams * hidden) * hidden + i % hidden] as f64
                })
                .collect(),
        );
        for (i, actual) in values(records, &cycle.output).iter().enumerate() {
            assert!(
                (terms.iter().map(|t| t[i]).sum::<f64>() - *actual as f64).abs() < 3e-5,
                "{}",
                cycle.output
            );
        }
    }
    let coefficients = values(records, &mixing.head.coefficients);
    let residual = values(records, &readout.residual);
    let scores = values(records, &readout.logits);
    let head = &weights[&readout.weight];
    let vocabulary = head.len() / hidden;
    for row in 0..rows {
        let rms = (residual[row * hidden..(row + 1) * hidden]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / hidden as f64
            + readout.normalization.epsilon.value() as f64)
            .sqrt();
        let mut reconstructed = [0.; 2];
        for term in &terms {
            for (selected, token) in [2usize, 7].into_iter().enumerate() {
                reconstructed[selected] += (0..hidden)
                    .map(|h| {
                        let collapsed = (0..streams)
                            .map(|stream| {
                                term[(row * streams + stream) * hidden + h]
                                    * coefficients[row * streams + stream] as f64
                            })
                            .sum::<f64>();
                        collapsed / rms * gain[h] as f64 * head[token * hidden + h] as f64
                    })
                    .sum::<f64>();
            }
        }
        for write in &readout.score_writes {
            assert_eq!(write.broadcast_axes, ["sequence"]);
            let input = values(records, &write.projection_input);
            let matrix = &weights[&write.weight];
            let original = values(records, &write.output);
            let effective = values(records, &write.effective_output);
            assert_eq!(matrix.len(), vocabulary * input.len());
            assert_eq!(effective.len(), vocabulary);
            assert!(
                effective.iter().any(|v| v.abs() > 1e-6),
                "nonzero dynamic Markov scores"
            );
            for (selected, token) in [2usize, 7].into_iter().enumerate() {
                let projected = input
                    .iter()
                    .zip(&matrix[token * input.len()..(token + 1) * input.len()])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>();
                assert!((projected - original[token] as f64).abs() < 3e-5);
                reconstructed[selected] += effective[token] as f64;
            }
        }
        let expected = [
            scores[row * vocabulary + 2] as f64,
            scores[row * vocabulary + 7] as f64,
        ];
        for (a, b) in reconstructed.iter().zip(expected) {
            assert!((a - b).abs() < 3e-5, "score {a} vs {b}");
        }
        assert!(((reconstructed[0] - reconstructed[1]) - (expected[0] - expected[1])).abs() < 3e-5);
    }
}

fn verify(device: LocalDevice, fused: bool) {
    for residency in super::super::v3_components::residencies() {
        let root = super::pooling::source(fused);
        let reference = super::pooling::source(fused);
        let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let graph = inspect_architecture(&root.0).unwrap();
        let scope = &graph.component_scopes[0];
        let (inputs, output, effective_output) = match &scope.residual_base {
            ComponentResidualBase::ProjectedSum {
                inputs,
                output,
                effective_output,
            } => (inputs.as_slice(), output, effective_output),
            ComponentResidualBase::Source {
                output,
                effective_output,
                ..
            } => (&[][..], output, effective_output),
            ComponentResidualBase::LinearFusion { .. } => {
                panic!("V4 uses direct or summed sources")
            }
        };
        let readout = &scope.readout;
        let streams = readout.stream_residual.as_ref().unwrap();
        let mut paths = vec![
            output.clone(),
            effective_output.clone(),
            streams.head.input.clone(),
            streams.head.coefficients.clone(),
            readout.residual.clone(),
            readout.normalized.clone(),
            readout.logits.clone(),
        ];
        let mut names = vec![
            readout.normalization.gain.clone().unwrap(),
            readout.weight.clone(),
        ];
        if let ComponentResidualBase::Source { input, .. } = &scope.residual_base {
            paths.push(input.clone());
            paths.extend([
                "dspark.context.normalized".into(),
                "dspark.context.normalized.effective".into(),
            ]);
        }
        for write in &readout.score_writes {
            names.push(write.weight.clone());
            paths.extend([
                write.projection_input.clone(),
                write.output.clone(),
                write.effective_output.clone(),
            ]);
        }
        for input in inputs {
            names.push(input.weight.clone());
            paths.extend([
                input.normalized.clone(),
                input.projection_input.clone(),
                input.output.clone(),
                input.effective_output.clone(),
            ]);
        }
        for cycle in &streams.cycles {
            paths.extend([
                cycle.input.clone(),
                cycle.output.clone(),
                cycle.write.clone(),
                cycle.post.clone(),
                cycle.combination.clone(),
            ]);
        }
        paths.sort();
        paths.dedup();
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency)
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
        let loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (mut model, _) = loaded.into_parts();
        let facts = model.parameter_discovery().unwrap();
        let limits = CaptureUsage {
            captures: 8192,
            retained_bytes: 1 << 30,
            host_bytes: 64 << 20,
            encoded_bytes: 64 << 20,
        };
        let mut weights = BTreeMap::new();
        for name in names {
            let parameter = facts.parameters.iter().find(|p| p.id == name).unwrap();
            assert!(parameter.supported, "{name}: {}", parameter.condition);
            let queried = model
                .query_parameter(
                    &facts.identity,
                    &name,
                    ParameterRegion {
                        starts: vec![0; parameter.shape.len()],
                        shape: parameter.shape.clone(),
                    },
                    limits,
                )
                .unwrap();
            weights.insert(name, queried.values);
        }
        let baseline =
            super::parameters::captures(&mut model, &generation, &paths, &[1, 2, 5], false);
        assert_eq!(
            baseline,
            super::parameters::captures(&mut model, &generation, &paths, &[1, 2, 5], true)
        );
        let mut count = 0;
        for (_, records) in &baseline.1 {
            if records
                .iter()
                .any(|r| r.path == readout.logits && r.outcome == CaptureOutcome::Captured)
            {
                reconstruct(records, scope, &weights);
                count += 1;
            }
        }
        assert!(
            count >= if fused { 2 } else { 3 },
            "repeated prediction evidence"
        );
        if fused {
            // A single accepted target row still seeds every fused context cache.
            assert_eq!(
                super::parameters::captures(&mut model, &generation, &paths, &[1], false),
                super::parameters::captures(&mut model, &generation, &paths, &[1], true)
            );
        }
        let attention = scope
            .components
            .iter()
            .find(|c| c.write_input_projection.is_some())
            .unwrap();
        let ffn = scope
            .components
            .iter()
            .find(|c| matches!(c.activation_equation, ComponentActivation::Gated { .. }))
            .unwrap();
        let mut edits = Vec::new();
        let mut edit_names = vec![
            &attention.reads[0].weight,
            &attention.write_weight,
            &ffn.reads[0].weight,
            &ffn.write_weight,
        ];
        if fused {
            let markov = &readout.score_writes[0];
            edit_names.push(&markov.weight);
            let ComponentFusionSource::TokenEmbedding { weight, .. } = &markov.source else {
                panic!("anchor embedding")
            };
            edit_names.push(weight);
        } else {
            edit_names.push(&inputs[0].weight);
        }
        for name in edit_names {
            let parameter = facts.parameters.iter().find(|p| p.id == *name).unwrap();
            assert!(parameter.supported);
            let width = parameter.shape[1];
            edits.push(ParameterEdit {
                id: format!("coordinate{}", edits.len()),
                parameter: name.clone(),
                parameter_shape: parameter.shape.clone(),
                dtype: InterventionDtype::Float32,
                region: ParameterRegion {
                    starts: vec![0, 0],
                    shape: vec![1, width],
                },
                update: ParameterUpdate::Add {
                    values: (0..width).map(|i| 0.03 + (i % 5) as f32 * 0.004).collect(),
                },
            });
        }
        let admitted = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "V4 prediction fusion and attention-plus-FFN reference edit".into(),
                edits: edits.clone(),
            })
            .unwrap();
        let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
        super::super::parameters::edit_reference(&reference.0, &edits);
        let oracle = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            &execution,
        )
        .unwrap();
        let oracle_generation = oracle.speculative_generation_options().unwrap().unwrap();
        let (mut oracle, _) = oracle.into_parts();
        for prefix in [[1, 2, 5], [5, 2, 1]] {
            let expected = super::parameters::captures(
                &mut oracle,
                &oracle_generation,
                &paths,
                &prefix,
                false,
            );
            let actual =
                super::parameters::captures(&mut model, &generation, &paths, &prefix, true);
            assert_eq!(actual, expected);
            if prefix == [1, 2, 5] {
                assert_ne!(actual, baseline);
            }
        }
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            baseline,
            super::parameters::captures(&mut model, &generation, &paths, &[1, 2, 5], true)
        );
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            original
        );
    }
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_v4_prediction_equations_parameters_and_overlays_cpu() {
    verify(LocalDevice::Cpu, false);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_v4_prediction_equations_parameters_and_overlays_metal() {
    verify(LocalDevice::Accelerator(0), false);
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_dspark_prediction_equations_parameters_and_overlays_cpu() {
    verify(LocalDevice::Cpu, true);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_dspark_prediction_equations_parameters_and_overlays_metal() {
    verify(LocalDevice::Accelerator(0), true);
}
