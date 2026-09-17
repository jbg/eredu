//! Public effective prediction parameters and independent-checkpoint overlay proof.
use super::*;
use eredu_core::{
    component::ComponentResidualBase, intervention::InterventionDtype, parameters::*,
};

pub(super) fn captures<B: eredu_core::SpeculativeGenerationBackend>(
    model: &mut LoadedModel<B>,
    generation: &PreparedChatSpeculativeGenerationOptions,
    paths: &[String],
    prefix: &[u32],
    controlled: bool,
) -> (
    Vec<u32>,
    Vec<(SpeculativeActivationPhase, Vec<CaptureRecord>)>,
) {
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let request = || PreparedChatSpeculativeGenerationRequest {
        input: PreparedChatInput::token_ids(&chat, prefix.to_vec()),
        drafting: eredu_core::SpeculativeDraft::Embedded,
        settings: PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(5),
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        },
        options: generation.clone(),
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_| {},
    };
    let limit = CaptureUsage {
        captures: 128,
        retained_bytes: 8 << 20,
        host_bytes: 8 << 20,
        encoded_bytes: 8 << 20,
    };
    let admitted = model
        .prepare_speculative_activations(SpeculativeActivationPlan {
            schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            captures: CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: paths
                    .iter()
                    .enumerate()
                    .map(|(i, path)| CaptureSelection {
                        id: i.to_string(),
                        path: path.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        transform: CaptureTransform::Preview { max_elements: 4096 },
                    })
                    .collect(),
                limits: CaptureLimits {
                    per_step: limit,
                    cumulative: limit.checked_mul(16).unwrap(),
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            },
            interventions: eredu_core::intervention::InterventionPlan {
                schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
                operations: vec![],
            },
            bounds: CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: None,
                max_predictions: 5,
            },
        })
        .unwrap();
    let identity = admitted.identity().to_owned();
    let options = ControlledSpeculativeOptions {
        activations: Some(admitted),
        snapshots: Some(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
        }),
        ..Default::default()
    };
    let mut records = Vec::new();
    let output = if controlled {
        model
            .with_controlled_text_speculative(request(), options, |session| {
                records.extend(session.step()?.unwrap().activations.iter().cloned());
                let saved = session.snapshot()?;
                let start = records.len();
                while let Some(step) = session.step()? {
                    records.extend(step.activations.iter().cloned());
                }
                let tokens = session.token_ids().to_vec();
                session.restore(&saved)?;
                let mut replay = Vec::new();
                while let Some(step) = session.step()? {
                    replay.extend(step.activations.iter().cloned());
                }
                assert_eq!(session.token_ids(), tokens);
                assert_eq!(replay.len(), records.len() - start);
                for (a, b) in replay.iter().zip(&records[start..]) {
                    assert_eq!(a.captures.as_step().records, b.captures.as_step().records);
                    assert_eq!(a.admission_identity, b.admission_identity);
                }
                Ok(())
            })
            .unwrap()
    } else {
        model
            .generate_observed_text_speculative(request(), options, |step| {
                records.extend(step.activations.iter().cloned());
                ControlFlow::Continue(())
            })
            .unwrap()
    };
    assert!(records.iter().any(|r| matches!(
        r.phase,
        SpeculativeActivationPhase::Proposal { .. } | SpeculativeActivationPhase::FusedProposal
    )));
    assert!(records
        .iter()
        .all(|r| r.admission_identity.as_deref() == Some(identity.as_str()) && r.completed));
    (
        output.token_ids().to_vec(),
        records
            .into_iter()
            .map(|r| (r.phase, r.captures.into_legacy().unwrap().records))
            .collect(),
    )
}

// V3 sources store gate/up separately, with either per-expert or stacked tensors.
// The effective slot concatenates their rows. Expand selected regions using only
// the source header and published layout, independently of production recipes.
fn edit_reference(root: &std::path::Path, edits: &[ParameterEdit]) {
    let bytes = std::fs::read(root.join("model.safetensors")).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    let mut source_edits = Vec::new();
    for edit in edits {
        if header.get(&edit.parameter).is_some() {
            source_edits.push(edit.clone());
            continue;
        }
        assert_eq!(
            edit.parameter_shape.len(),
            3,
            "source alias {}",
            edit.parameter
        );
        let (prefix, projections): (&str, &[&str]) =
            if let Some(prefix) = edit.parameter.strip_suffix("gate_up_proj") {
                (prefix, &["gate_proj", "up_proj"])
            } else {
                (
                    edit.parameter.strip_suffix("down_proj").unwrap(),
                    &["down_proj"],
                )
            };
        let rows = edit.parameter_shape[1] / projections.len() as u64;
        assert_eq!(rows * projections.len() as u64, edit.parameter_shape[1]);
        let selected_rows = edit.region.shape[1];
        let columns = edit.region.shape[2];
        for expert in 0..edit.region.shape[0] {
            for (projection_index, projection) in projections.iter().enumerate() {
                let first = projection_index as u64 * rows;
                let start = first.max(edit.region.starts[1]);
                let end = (first + rows).min(edit.region.starts[1] + selected_rows);
                if start >= end {
                    continue;
                }
                let begin = (expert * selected_rows + start - edit.region.starts[1]) * columns;
                let end_value = begin + (end - start) * columns;
                let values = edit.update.values()[begin as usize..end_value as usize].to_vec();
                let packed_name = format!("{prefix}{projection}");
                let (parameter, parameter_shape, region) = if header.get(&packed_name).is_some() {
                    (
                        packed_name,
                        vec![edit.parameter_shape[0], rows, edit.parameter_shape[2]],
                        ParameterRegion {
                            starts: vec![
                                edit.region.starts[0] + expert,
                                start - first,
                                edit.region.starts[2],
                            ],
                            shape: vec![1, end - start, columns],
                        },
                    )
                } else {
                    (
                        format!(
                            "{prefix}{}.{projection}.weight",
                            edit.region.starts[0] + expert
                        ),
                        vec![rows, edit.parameter_shape[2]],
                        ParameterRegion {
                            starts: vec![start - first, edit.region.starts[2]],
                            shape: vec![end - start, columns],
                        },
                    )
                };
                source_edits.push(ParameterEdit {
                    id: format!("{}-{expert}-{projection_index}", edit.id),
                    parameter,
                    parameter_shape,
                    dtype: edit.dtype,
                    region,
                    update: match edit.update {
                        ParameterUpdate::Replace { .. } => ParameterUpdate::Replace { values },
                        ParameterUpdate::Add { .. } => ParameterUpdate::Add { values },
                    },
                });
            }
        }
    }
    super::super::parameters::edit_reference(root, &source_edits);
}

fn verify(device: LocalDevice) {
    for residency in super::super::v3_components::residencies() {
        let root = super::super::v3_components::source_with_prediction(true, true, 2);
        let reference = super::super::v3_components::source_with_prediction(true, true, 2);
        let original = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let graph = inspect_architecture(&root.0).unwrap();
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
        assert_eq!(graph.component_scopes.len(), 2);
        let mut paths = vec![eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into()];
        for scope in &graph.component_scopes {
            paths.extend([
                scope.readout.normalized.clone(),
                scope.readout.logits.clone(),
            ]);
            let ComponentResidualBase::LinearFusion {
                weight,
                inputs,
                effective_output,
                ..
            } = &scope.residual_base
            else {
                panic!("V3 fixture declares concatenated linear fusion")
            };
            paths.push(effective_output.clone());
            paths.extend(inputs.iter().map(|input| input.output.clone()));
            let mut names = vec![weight.clone(), scope.readout.weight.clone()];
            names.extend(inputs.iter().filter_map(|i| i.normalization.gain.clone()));
            names.extend(scope.readout.normalization.gain.clone());
            for group in &scope.components {
                names.push(group.write_weight.clone());
                for read in &group.reads {
                    names.push(read.weight.clone());
                    names.extend(read.input_projections.iter().map(|p| p.weight.clone()));
                }
            }
            for name in names {
                let parameter = facts
                    .parameters
                    .iter()
                    .find(|p| p.id == name)
                    .unwrap_or_else(|| panic!("missing declared prediction parameter {name}"));
                assert!(
                    parameter.supported,
                    "{}: {}",
                    parameter.id, parameter.condition
                );
            }
        }
        let baseline = captures(&mut model, &generation, &paths, &[1, 2, 5], false);
        let limits = CaptureUsage {
            captures: 8192,
            retained_bytes: 1 << 30,
            host_bytes: 64 << 20,
            encoded_bytes: 64 << 20,
        };
        let mut edits = Vec::new();
        let mut originals = Vec::new();
        // Every effective matrix contributes a bounded first-row edit. This
        // spans target embeddings/readout, MLA, gated and routed FFNs, fusion,
        // both prediction depths and their independent vocabulary heads.
        for parameter in &facts.parameters {
            assert!(
                parameter.supported,
                "{}: {}",
                parameter.id, parameter.condition
            );
            let mut shape = vec![1; parameter.shape.len()];
            *shape.last_mut().unwrap() = *parameter.shape.last().unwrap();
            let region = ParameterRegion {
                starts: vec![0; shape.len()],
                shape,
            };
            let mut region = region;
            if parameter.id == graph.component_readout.as_ref().unwrap().embedding_weight {
                // Token 1 occurs in both fixed prefixes and feeds the shared
                // target embedding used by prediction fusion as well.
                region.starts[0] = 1;
            }
            let queried = model
                .query_parameter(&facts.identity, &parameter.id, region.clone(), limits)
                .unwrap();
            assert!(queried.values.iter().all(|v| v.is_finite()));
            if parameter.shape.len() == 2 {
                let width = region.shape[1] as usize;
                let coefficients: Vec<_> = (0..2 * width)
                    .map(|i| (i % 7) as f32 * 0.125 - 0.25)
                    .collect();
                let projected = model
                    .project_parameter(
                        &facts.identity,
                        &parameter.id,
                        ParameterProjection {
                            region: region.clone(),
                            axis: 1,
                            directions: 2,
                            coefficients: coefficients.clone(),
                        },
                        limits,
                    )
                    .unwrap();
                for direction in 0..2 {
                    let expected = queried
                        .values
                        .iter()
                        .zip(&coefficients[direction * width..(direction + 1) * width])
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>();
                    assert!((projected.values[direction] as f64 - expected).abs() < 2e-6);
                }
                let column = model
                    .query_parameter(
                        &facts.identity,
                        &parameter.id,
                        ParameterRegion {
                            starts: vec![0, 0],
                            shape: vec![parameter.shape[0], 1],
                        },
                        limits,
                    )
                    .unwrap();
                assert_eq!(column.values.len(), parameter.shape[0] as usize);
            }
            if parameter.shape.len() < 2 {
                continue;
            }
            let values = (0..queried.values.len())
                .map(|i| 0.01 + (i % 5) as f32 * 0.003)
                .collect();
            originals.push(queried.values);
            edits.push(ParameterEdit {
                id: format!("edit{}", edits.len()),
                parameter: parameter.id.clone(),
                parameter_shape: parameter.shape.clone(),
                dtype: InterventionDtype::Float32,
                region,
                update: ParameterUpdate::Add { values },
            });
        }
        let admitted = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "coordinated target and embedded prediction component edit".into(),
                edits: edits.clone(),
            })
            .unwrap();
        let used = model.parameter_discovery().unwrap().usage;
        assert!(model.activate_parameter_overlay(&admitted, used).is_err());
        assert_eq!(model.parameter_discovery().unwrap().usage, used);
        let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
        assert_eq!(
            active.overlay_identity.as_deref(),
            Some(admitted.identity())
        );
        for (edit, before) in edits.iter().zip(&originals) {
            let actual = model
                .query_parameter(
                    &active.identity,
                    &edit.parameter,
                    edit.region.clone(),
                    limits,
                )
                .unwrap();
            for ((a, b), c) in before.iter().zip(edit.update.values()).zip(actual.values) {
                assert_eq!(*a + *b, c);
            }
        }
        edit_reference(&reference.0, &edits);
        let loaded_oracle = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            &execution,
        )
        .unwrap();
        let oracle_generation = loaded_oracle
            .speculative_generation_options()
            .unwrap()
            .unwrap();
        let (mut oracle, _) = loaded_oracle.into_parts();
        for prefix in [[1, 2, 5], [5, 2, 1]] {
            let expected = captures(&mut oracle, &oracle_generation, &paths, &prefix, false);
            let actual = captures(&mut model, &generation, &paths, &prefix, true);
            assert_eq!(actual, expected);
            if prefix == [1, 2, 5] {
                assert_ne!(actual, baseline);
            }
        }
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            captures(&mut model, &generation, &paths, &[1, 2, 5], true),
            baseline
        );
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            original
        );
    }
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_v3_prediction_parameters_and_overlays_cpu() {
    verify(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_v3_prediction_parameters_and_overlays_metal() {
    verify(LocalDevice::Accelerator(0));
}
