// Exercises the public backend admission/factory through real partition sessions.
pub(super) fn verify_partitioned(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    checkpoint: &Path,
    stream: &Stream,
    options: &eredu_runtime::NormalizedLoadRequest,
    rank: usize,
) {
    use eredu_core::{capture::*, intervention::*, speculative::*};
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let config =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let graph = resolved.architecture_plan().architecture_descriptor();
    let media = config.get("vision_config").is_some();
    let scope = &graph.component_scopes[0];
    let fused = scope.kind == eredu_core::component::ComponentExecutionScopeKind::FusedPrediction;
    let channel = scope
        .components
        .iter()
        .find(|group| {
            matches!(
                group.activation_equation,
                eredu_core::component::ComponentActivation::Attention { .. }
            )
        })
        .unwrap();
    let shared = scope.components.iter().find(|group| {
        matches!(
            group.activation_equation,
            eredu_core::component::ComponentActivation::Gated { .. }
                | eredu_core::component::ComponentActivation::Unary { .. }
        )
    });
    let mut paths = vec![
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH.to_owned(),
        scope.readout.normalized.clone(),
        scope.readout.logits.clone(),
    ];
    if let Some(path) = &scope.readout.projection_input {
        paths.push(path.clone());
    }
    // Include both sides of the fusion normalization and the actual shared
    // embedding input, whose logical node belongs to this prediction scope.
    if let eredu_core::component::ComponentResidualBase::LinearFusion {
        inputs,
        projection_input,
        output,
        effective_output,
        ..
    } = &scope.residual_base
    {
        paths.extend([
            projection_input.clone(),
            output.clone(),
            effective_output.clone(),
        ]);
        for input in inputs {
            paths.push(input.output.clone());
            paths.push(input.output.strip_suffix(".effective").unwrap().into());
            if let eredu_core::component::ComponentFusionSource::Observation { path } =
                &input.source
            {
                paths.push(path.clone());
            }
        }
        for point in &graph.observations.points {
            if point.position == eredu_core::ObservationPosition::ReadOnly
                && graph.nodes.iter().any(|node| {
                    node.id == point.node_id
                        && matches!(
                            node.kind,
                            eredu_core::ArchitectureNodeKind::Embedding
                                | eredu_core::ArchitectureNodeKind::Normalization
                        )
                        && node.parent.as_deref() == Some(scope.node_id.as_str())
                })
            {
                paths.push(point.path.clone());
            }
        }
    }
    for component in &scope.components {
        paths.extend([
            component.activation.clone(),
            component.effective_activation.clone(),
        ]);
        if matches!(
            component.activation_equation,
            eredu_core::component::ComponentActivation::Gated { .. }
        ) {
            paths.push(component.input.clone());
        }
    }
    for component in &scope.routed_components {
        if let Some(input) = &component.input {
            paths.push(input.clone());
        }
        paths.extend([
            component.activation.clone(),
            component.effective_activation.clone(),
        ]);
    }
    if fused {
        paths.extend([
            "dspark.context.normalized".into(),
            "dspark.context.normalized.effective".into(),
            scope.readout.score_writes[0]
                .input
                .trim_end_matches(".effective")
                .into(),
            scope.readout.score_writes[0].input.clone(),
            scope.readout.score_writes[0].output.clone(),
            scope.readout.score_writes[0].effective_output.clone(),
        ]);
        let streams = scope.readout.stream_residual.as_ref().unwrap();
        let eredu_core::component::ComponentResidualBase::Source {
            input,
            output,
            effective_output,
            ..
        } = &scope.residual_base
        else {
            panic!("fused prediction source");
        };
        paths.extend([
            input.clone(),
            output.clone(),
            effective_output.clone(),
            streams.head.input.clone(),
            streams.head.coefficients.clone(),
            scope.readout.residual.clone(),
            scope.readout.linear_scores.clone(),
        ]);
        for write in &scope.readout.score_writes {
            paths.push(write.projection_input.clone());
        }
        for cycle in &streams.cycles {
            paths.extend([
                cycle.input.clone(),
                cycle.output.clone(),
                cycle.collapsed.clone(),
                cycle.write.clone(),
                cycle.pre.clone(),
                cycle.post.clone(),
                cycle.combination.clone(),
            ]);
        }
        paths.sort();
        paths.dedup();
    }
    if std::env::var_os(OPAQUE_INKLING_COMPONENTS).is_some() {
        let mut target_paths = vec!["readout.embedding".to_owned()];
        for component in &graph.components {
            target_paths.extend([
                component.input.clone(),
                component.activation.clone(),
                component.effective_activation.clone(),
            ]);
        }
        for transform in &graph.component_transforms {
            if transform.id.starts_with("decoder.") {
                target_paths.push(transform.input.clone());
                target_paths.push(transform.output.clone());
                if let Some(path) = &transform.effective_output {
                    target_paths.push(path.clone());
                }
            }
        }
        target_paths.extend(paths);
        let mut seen = std::collections::BTreeSet::new();
        paths = target_paths
            .into_iter()
            .filter(|path| seen.insert(path.clone()))
            .collect();
    }
    let routed = scope.routed_components.first();
    let weights_stream = fixture_weights_stream(stream);
    let backend = MlxBackend::new(stream, &weights_stream);
    let model = load_model(
        &backend,
        checkpoint,
        MlxLoadRequest::from_normalized(options.clone()),
    )
    .unwrap();
    let mut reference = ModelRuntime::from_prepared(backend, model).unwrap();
    let run =
        |runtime: &mut ModelRuntime<MlxBackend<'_>>, mode: u8, tokens: [u32; 3], replay: bool| {
            let input = prediction_component_prompt(&tokens, media);
            let count = <MlxBackend as eredu_core::ModelCapabilityBackend>::count_prepared_input(
                runtime, &input,
            )
            .unwrap();
            assert_eq!(count.text_tokens, 3);
            assert_eq!(count.media_positions, if media { 2 } else { 0 });
            assert_eq!(count.model_positions, if media { 5 } else { 3 });
            assert_eq!(count.kind, eredu_core::ObservationKind::Exact);
            let discovery = MlxBackend::speculative_activation_discovery(runtime).unwrap();
            for path in &paths {
                let point = discovery
                    .captures
                    .support
                    .points
                    .iter()
                    .find(|point| &point.path == path)
                    .unwrap();
                assert_eq!(
                    point.prefill,
                    eredu_core::ObservationSupportStatus::Supported,
                    "{path}"
                );
                assert_eq!(
                    point.decode,
                    eredu_core::ObservationSupportStatus::Supported,
                    "{path}"
                );
            }
            let allowance = CaptureUsage {
                captures: 4096,
                retained_bytes: 512 << 20,
                host_bytes: 4 << 30,
                encoded_bytes: 256 << 20,
            };
            let plan = SpeculativeActivationPlan {
                schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                bounds: CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: count.model_positions,
                    max_context: None,
                    max_predictions: 8,
                },
                captures: CapturePlan {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selections: paths
                        .iter()
                        .map(|path| CaptureSelection {
                            id: path.clone(),
                            path: path.clone(),
                            schedule: Default::default(),
                            slices: vec![],
                            transform: if matches!(
                                discovery.captures.catalog.get(path).unwrap().value_type,
                                eredu_core::ObservationValueType::RoutedUnits { .. }
                            ) {
                                CaptureTransform::RoutedUnits
                            } else {
                                CaptureTransform::FullTensor
                            },
                        })
                        .collect(),
                    limits: CaptureLimits {
                        per_step: allowance,
                        cumulative: allowance.checked_mul(32).unwrap(),
                        physical_native_bytes: None,
                        on_limit: CaptureLimitPolicy::Fail,
                    },
                },
                interventions: InterventionPlan {
                    schema_version: INTERVENTION_SCHEMA_VERSION,
                    operations: if matches!(mode, 9 | 10) {
                        assert!(fused);
                        vec![InterventionOperation {
                            id: "fused-context-or-markov".into(),
                            target: if mode == 9 {
                                "dspark.context.normalized".into()
                            } else {
                                scope.readout.score_writes[0]
                                    .input
                                    .trim_end_matches(".effective")
                                    .into()
                            },
                            schedule: Default::default(),
                            slices: vec![],
                            action: InterventionAction::Zero {
                                dtype: InterventionDtype::Float32,
                            },
                            evidence: InterventionEvidence::Preview { max_elements: 2048 },
                        }]
                    } else if mode != 0 {
                        let masks: &[u8] = if mode == 8 { &[3, 6] } else { &[mode] };
                        masks
                            .iter()
                            .map(|mode| {
                                let channel_mask = matches!(mode, 1 | 3 | 7);
                                let shared_mask = matches!(mode, 5 | 6)
                                    || (routed.is_none() && matches!(mode, 2 | 4));
                                let selected_position = matches!(mode, 3..=6);
                                InterventionOperation {
                                    id: "prediction-components".into(),
                                    target: if channel_mask {
                                        channel.activation.clone()
                                    } else if shared_mask {
                                        shared.unwrap().activation.clone()
                                    } else {
                                        routed.unwrap().activation.clone()
                                    },
                                    schedule: CaptureSchedule {
                                        prefill: !(fused && selected_position),
                                        decode: fused || !selected_position,
                                        // A later fused proposal may have only one remaining
                                        // row. Select row one of the first proposal exactly.
                                        end_prediction: (fused && selected_position).then_some(2),
                                        ..Default::default()
                                    },
                                    slices: if selected_position {
                                        vec![CaptureSlice {
                                            axis: if channel_mask || shared_mask {
                                                "sequence"
                                            } else {
                                                "token"
                                            }
                                            .into(),
                                            start: 1,
                                            end: 2,
                                            stride: 1,
                                        }]
                                    } else {
                                        vec![]
                                    },
                                    action: InterventionAction::MaskComponents {
                                        dtype: InterventionDtype::Float32,
                                        indices: if *mode == 7 {
                                            (0..channel.count as u32).collect()
                                        } else if channel_mask || shared_mask {
                                            vec![0]
                                        } else {
                                            (0..routed.unwrap().expert_count)
                                                .map(|expert| {
                                                    (expert * routed.unwrap().units_per_expert)
                                                        as u32
                                                })
                                                .collect()
                                        },
                                        keep_selected: matches!(mode, 3 | 4 | 6 | 7),
                                    },
                                    evidence: if channel_mask || shared_mask {
                                        InterventionEvidence::Preview { max_elements: 2048 }
                                    } else {
                                        InterventionEvidence::None
                                    },
                                }
                            })
                            .enumerate()
                            .map(|(i, mut operation)| {
                                operation.id = format!("prediction-components-{i}");
                                operation
                            })
                            .collect()
                    } else {
                        vec![]
                    },
                },
            }
            .admit(&discovery)
            .unwrap();
            let mut records = Vec::new();
            let mut failure = None;
            let visitor = DriveControlledSpeculation::new(
                Default::default(),
                ControlledSpeculativeOptions {
                    activations: Some(plan),
                    snapshots: replay.then_some(eredu_core::execution_control::SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 64 << 20,
                        cumulative_copy_bytes: 512 << 20,
                    }),
                    ..Default::default()
                },
                |session: &mut dyn ControlledSpeculativeSession| {
                    if replay {
                        records.extend(session.step()?.unwrap().activations.iter().cloned());
                        let saved = session.snapshot()?;
                        let sibling = session.fork(&saved)?;
                        let start = records.len();
                        while let Some(step) = session.step()? {
                            records.extend(step.activations.iter().cloned());
                        }
                        let tokens = session.token_ids().to_vec();
                        let used = session.snapshot_usage().cumulative_copy_bytes;
                        session.restore(&saved)?;
                        let mut repeated = Vec::new();
                        while let Some(step) = session.step()? {
                            repeated.extend(step.activations.iter().cloned());
                        }
                        assert_eq!(session.token_ids(), tokens);
                        assert_prediction_replay(&repeated, &records[start..]);
                        session.exchange(&sibling)?;
                        let mut branched = Vec::new();
                        while let Some(step) = session.step()? {
                            branched.extend(step.activations.iter().cloned());
                        }
                        assert_eq!(session.token_ids(), tokens);
                        assert_prediction_replay(&branched, &records[start..]);
                        session.exchange(&sibling)?;
                        assert_eq!(session.token_ids(), tokens);
                        assert!(session.snapshot_usage().cumulative_copy_bytes > used);
                        session.release_branch(&sibling)?;
                        session.release_snapshot(&saved)?;
                    } else {
                        while let Some(step) = session.step()? {
                            records.extend(step.activations.iter().cloned());
                        }
                    }
                    Ok(())
                },
                &mut failure,
            );
            let (output, _) = execute_neutral_embedded_mtp_with(
                runtime,
                input,
                SpeculativeConfig {
                    max_tokens: 4,
                    max_draft_tokens: if fused { 2 } else { 1 },
                    temperature: 0.0,
                    eos_token_ids: vec![],
                },
                visitor,
            );
            assert!(failure.is_none(), "{failure:?}");
            let output = output.unwrap();
            assert!(records.iter().all(|record| record.completed));
            if fused {
                let context = records
                    .iter()
                    .find(|r| r.phase == Phase::PredictionPrefill)
                    .unwrap();
                assert_eq!(
                    context.captures.as_step().invocation.unwrap().sequence,
                    tokens.len() as u64
                );
                let proposal = records
                    .iter()
                    .find(|r| r.phase == Phase::FusedProposal)
                    .unwrap();
                assert_eq!(proposal.captures.as_step().invocation.unwrap().sequence, 2);
                for record in &records {
                    for value in &record.captures.as_step().records {
                        if value.path == "dspark.context.normalized" {
                            assert_eq!(
                                value.outcome == CaptureOutcome::Captured,
                                matches!(
                                    record.phase,
                                    Phase::PredictionPrefill | Phase::PredictionReplay
                                )
                            );
                        } else if value.path == channel.activation {
                            assert_eq!(
                                value.outcome == CaptureOutcome::Captured,
                                record.phase == Phase::FusedProposal
                            );
                        }
                    }
                }
            }
            (output.token_ids().to_vec(), records)
        };
    let mut references = Vec::new();
    for mode in 0..3 {
        eprintln!("prediction component comparison rank={rank} mode={mode}");
        let expected = run(&mut reference, mode, [1, 2, 3], false);
        let actual = run(runtime, mode, [1, 2, 3], false);
        assert_partitioned_prediction_result(&actual, &expected);
        if fused {
            verify_fused_readout(runtime, scope, &actual.1);
        }
        references.push(expected.1);
    }
    let logits = |records: &[SpeculativeActivationCapture]| {
        records
            .iter()
            .flat_map(|record| &record.captures.as_step().records)
            .filter(|record| record.path == scope.readout.logits)
            .filter_map(|record| record.payload.clone())
            .collect::<Vec<_>>()
    };
    assert_ne!(
        logits(&references[0]),
        logits(&references[1]),
        "nonzero channel deletion must change actual prediction scores"
    );
    assert_ne!(
        logits(&references[0]),
        logits(&references[2]),
        "nonzero feed-forward-unit deletion must change actual prediction scores"
    );
    if shared.is_some() {
        // Exercise global component coordinates and physical token positions
        // independently of the model's residual-stream geometry.
        for mode in 3..=8 {
            eprintln!("prediction selected-component trial rank={rank} mode={mode}");
            let expected = run(&mut reference, mode, [1, 2, 3], false);
            let actual = run(runtime, mode, [1, 2, 3], false);
            assert_partitioned_prediction_result(&actual, &expected);
            verify_prediction_mask_values(&actual.1, scope, mode, if media { 4 } else { 2 });
            if fused {
                verify_fused_readout(runtime, scope, &actual.1);
            }
            if mode == 7 {
                assert_eq!(
                    logits(&references[0]),
                    logits(&expected.1),
                    "all-keep is a no-op"
                );
            } else {
                assert_ne!(
                    logits(&references[0]),
                    logits(&expected.1),
                    "causal mask {mode}"
                );
            }
            references.push(expected.1);
        }
        let shared = shared.unwrap();
        let kept = prediction_trial_tensor(&references[6], &shared.effective_activation);
        let recomputed = prediction_trial_tensor(&references[8], &shared.effective_activation);
        assert_ne!(
            kept[shared.count], recomputed[shared.count],
            "a surviving shared unit must recompute from the channel-masked residual"
        );
    }
    if fused {
        for mode in [9, 10] {
            let expected = run(&mut reference, mode, [1, 2, 3], false);
            let actual = run(runtime, mode, [1, 2, 3], false);
            assert_partitioned_prediction_result(&actual, &expected);
            verify_fused_readout(runtime, scope, &actual.1);
            assert_ne!(
                logits(&references[0]),
                logits(&expected.1),
                "context/Markov edit {mode} must affect actual fused scores"
            );
        }
    }
    let configured = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let checkpoint_plan = configured.architecture.checkpoint();
    let integer_buffers = checkpoint_plan
        .common_tensors
        .iter()
        .chain(
            checkpoint_plan
                .layout_groups
                .iter()
                .flat_map(|g| &g.variants)
                .flat_map(|v| &v.tensors),
        )
        .filter(|tensor| {
            matches!(
                tensor.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::I32
                )
            )
        })
        .map(|tensor| tensor.key.clone())
        .collect();
    verify_partitioned_prediction_parameters(
        runtime,
        &mut reference,
        rank,
        &graph,
        &integer_buffers,
        run,
    );
}

fn prediction_component_prompt(tokens: &[u32; 3], media: bool) -> MlxModelInput {
    let text = Array::from_slice(tokens, &[1, 3]);
    let mut parts = vec![text_input_part(&text)];
    let mut identity = tokens.to_vec();
    if media {
        let pixels = Array::from_slice(
            &(0..96)
                .map(|index| (index as f32 - 41.0) / 97.0)
                .collect::<Vec<_>>(),
            &[8, 12],
        );
        let grid = Array::from_slice(&[1i32, 2, 4], &[1, 3]);
        parts.push(input_part(
            InputModality::Image,
            InputPayload::Tensor(pixels),
            [(InputMetadataKey::PatchGrid, grid)],
            [],
        ));
        identity.extend([42, 42]);
    }
    synthetic_prediction_input(&parts, &identity)
}

type PredictionResult = (
    Vec<u32>,
    Vec<eredu_core::speculative::SpeculativeActivationCapture>,
);

fn verify_fused_readout(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    scope: &eredu_core::component::ComponentExecutionScope,
    records: &[eredu_core::speculative::SpeculativeActivationCapture],
) {
    use eredu_core::component::{ComponentFusionExpansion, ComponentResidualBase};
    let ComponentResidualBase::Source {
        input,
        output,
        expansion,
        ..
    } = &scope.residual_base
    else {
        panic!("fused source");
    };
    let ComponentFusionExpansion::BroadcastAxis {
        axis: 2,
        extent: streams,
        ..
    } = expansion
    else {
        panic!("fused stream expansion");
    };
    let steps = records
        .iter()
        .filter(|r| r.phase == Phase::FusedProposal)
        .map(|r| {
            let source = prediction_trial_tensor(std::slice::from_ref(r), input);
            let expanded = prediction_trial_tensor(std::slice::from_ref(r), output);
            let hidden = source.len() / r.captures.as_step().invocation.unwrap().sequence as usize;
            assert_eq!(expanded.len(), source.len() * streams);
            for (i, actual) in expanded.iter().enumerate() {
                assert_eq!(
                    *actual,
                    source[i / (streams * hidden) * hidden + i % hidden]
                );
            }
            r.captures.as_step().clone()
        })
        .collect::<Vec<_>>();
    assert!(!steps.is_empty());
    v4_components::verify_partitioned_readout(runtime, &scope.readout, &steps);
}

fn prediction_trial_payload<'a>(
    records: &'a [eredu_core::speculative::SpeculativeActivationCapture],
    path: &str,
) -> &'a eredu_core::capture::CapturePayload {
    records
        .iter()
        .filter(|record| {
            matches!(
                record.phase,
                Phase::PredictionPrefill | Phase::FusedProposal
            )
        })
        .flat_map(|record| &record.captures.as_step().records)
        .filter(|record| record.path == path)
        .find_map(|record| record.payload.as_ref())
        .unwrap_or_else(|| panic!("first invoked prediction payload {path}"))
}

fn prediction_trial_tensor<'a>(
    records: &'a [eredu_core::speculative::SpeculativeActivationCapture],
    path: &str,
) -> &'a [f32] {
    let eredu_core::capture::CapturePayload::Tensor(value) =
        prediction_trial_payload(records, path)
    else {
        panic!("tensor {path}")
    };
    let eredu_core::TensorObservationData::F32(values) = value.data() else {
        panic!("F32 {path}")
    };
    values
}

fn verify_prediction_mask_values(
    records: &[eredu_core::speculative::SpeculativeActivationCapture],
    scope: &eredu_core::component::ComponentExecutionScope,
    mode: u8,
    expected_sequence: usize,
) {
    use eredu_core::{capture::CapturePayload, component::ComponentActivation};
    let masks: &[u8] = if mode == 8 { &[3, 6] } else { &[mode] };
    for mode in masks {
        if *mode == 4 && !scope.routed_components.is_empty() {
            let routed = &scope.routed_components[0];
            let CapturePayload::RoutedUnits(original) =
                prediction_trial_payload(records, &routed.activation)
            else {
                panic!("original routed values")
            };
            let CapturePayload::RoutedUnits(effective) =
                prediction_trial_payload(records, &routed.effective_activation)
            else {
                panic!("effective routed values")
            };
            assert_eq!(original.rows.len(), effective.rows.len());
            let mut removed = 0;
            for (before, after) in original.rows.iter().zip(&effective.rows) {
                assert_eq!(
                    (before.token, before.slot, before.expert),
                    (after.token, after.slot, after.expert)
                );
                let (
                    eredu_core::TensorObservationData::F32(before_values),
                    eredu_core::TensorObservationData::F32(after_values),
                ) = (before.values.data(), after.values.data())
                else {
                    panic!("F32 routed values")
                };
                for (i, (&before_value, &after_value)) in
                    before_values.iter().zip(after_values).enumerate()
                {
                    let unit = before.unit_start + i as u64 * before.unit_stride;
                    let delete = before.token == 1 && unit != 0;
                    assert_eq!(after_value, if delete { 0. } else { before_value });
                    removed += usize::from(delete && before_value != 0.);
                }
            }
            assert!(
                removed > 0,
                "selected-token routed keep-only must remove nonzero units"
            );
        } else {
            let attention = matches!(mode, 3 | 7);
            let group = scope
                .components
                .iter()
                .find(|group| {
                    matches!(
                        group.activation_equation,
                        ComponentActivation::Attention { .. }
                    ) == attention
                })
                .unwrap();
            let before = prediction_trial_tensor(records, &group.activation);
            let after = prediction_trial_tensor(records, &group.effective_activation);
            // Physical sequence positions include media placeholders and are
            // distinct from the logical prediction index.
            let sequence = records
                .iter()
                .find(|record| {
                    matches!(
                        record.phase,
                        Phase::PredictionPrefill | Phase::FusedProposal
                    ) && record.captures.as_step().records.iter().any(|capture| {
                        capture.path == group.activation && capture.payload.is_some()
                    })
                })
                .unwrap()
                .captures.as_step().invocation
                .unwrap()
                .sequence as usize;
            assert_eq!(sequence, expected_sequence);
            assert_eq!(before.len(), sequence * group.count);
            assert_eq!(before.len(), after.len());
            let mut removed = 0;
            for (i, (&before, &after)) in before.iter().zip(after).enumerate() {
                let selected_token = i / group.count == 1;
                let selected_component = i % group.count == 0;
                let delete = *mode != 7
                    && selected_token
                    && if matches!(*mode, 2 | 5) {
                        selected_component
                    } else {
                        !selected_component
                    };
                assert_eq!(
                    after,
                    if delete { 0. } else { before },
                    "mask {mode}, scalar {i}"
                );
                removed += usize::from(delete && before != 0.);
            }
            assert!(
                *mode == 7 || removed > 0,
                "selected-token mask {mode} must remove nonzero values"
            );
        }
    }
}

fn assert_partitioned_prediction_result(actual: &PredictionResult, expected: &PredictionResult) {
    assert_partitioned_prediction_result_with_readout(actual, expected, None, None);
}

fn assert_partitioned_prediction_result_with_readout(
    actual: &PredictionResult,
    expected: &PredictionResult,
    readout: Option<&EditedReadoutEvidence>,
    stream_readout: Option<&eredu_core::component::ComponentReadoutEquation>,
) {
    use eredu_core::capture::*;
    // The block-FP8 fixture uses 256-wide projections (complete 128-wide TP
    // tiles). Repeated reductions and coordinated overlays amplify FP32 sum
    // order differences. Keep its numerical envelope separate from small F32
    // fixtures; identities, effective parameter queries and replay remain exact.
    let tolerance = if std::env::var_os("EREDU_RING_PREDICTION_FP8").is_some()
        || std::env::var_os("EREDU_RING_DEEPSEEK_FP8").is_some()
    {
        2e-4
    } else {
        2e-5
    };
    assert_eq!(actual.0, expected.0);
    assert_eq!(actual.1.len(), expected.1.len());
    if std::env::var_os("EREDU_RING_DEEPSEEK_FP8").is_some() {
        for (left, right) in actual.1.iter().zip(&expected.1) {
            if !left
                .captures.as_step().partitions
                .iter()
                .any(|part| part.context.overlay_identity.is_some())
            {
                continue;
            }
            for (a, b) in left.captures.as_step().records.iter().zip(&right.captures.as_step().records) {
                if !a.path.contains("readout") {
                    continue;
                }
                if let (
                    Some(CapturePayload::Tensor(a_values)),
                    Some(CapturePayload::Tensor(b_values)),
                ) = (&a.payload, &b.payload)
                {
                    if let (
                        eredu_core::TensorObservationData::F32(a_values),
                        eredu_core::TensorObservationData::F32(b_values),
                    ) = (a_values.data(), b_values.data())
                    {
                        let (absolute, scaled) = a_values.iter().zip(b_values).fold(
                            (0_f32, 0_f32),
                            |(absolute, scaled), (a, b)| {
                                let error = (a - b).abs();
                                (absolute.max(error), scaled.max(error / b.abs().max(1.)))
                            },
                        );
                        eprintln!("edited DeepSeek readout {:?} {} max_abs={absolute} max_scaled={scaled}", left.phase, a.path);
                    }
                }
            }
        }
    }
    for (actual, expected) in actual.1.iter().zip(&expected.1) {
        let mut conditioned_bounds = stream_readout
            .map_or_else(std::collections::BTreeMap::new, |equation| {
                edited_stream_pair_bounds(equation, actual, expected)
            });
        let mut routed_bounds = std::collections::BTreeMap::new();
        if let Some(readout) = readout {
            if let Some(bounds) = readout.bounds(actual, expected) {
                conditioned_bounds.insert(readout.output_path().to_owned(), bounds);
            }
            for write in &readout.ffn_writes {
                if let Some(bounds) = write.bounds(actual, expected) {
                    conditioned_bounds.insert(write.path.clone(), bounds);
                }
            }
            for routed in &readout.routed {
                if let Some(bounds) = routed.bounds(actual, expected) {
                    routed_bounds.insert(routed.group.activation.clone(), bounds.clone());
                    routed_bounds.insert(routed.group.effective_activation.clone(), bounds);
                }
            }
            for gated in &readout.gated {
                if let Some(bounds) = gated.bounds(actual, expected) {
                    conditioned_bounds.insert(gated.group.activation.clone(), bounds.clone());
                    conditioned_bounds.insert(gated.group.effective_activation.clone(), bounds);
                }
            }
        }
        assert_eq!(
            (actual.phase, actual.origin, actual.captures.as_step().invocation),
            (
                expected.phase,
                expected.origin,
                expected.captures.as_step().invocation
            )
        );
        assert_eq!(
            actual.captures.as_step().records.len(),
            expected.captures.as_step().records.len()
        );
        let phase = actual.phase;
        for (actual, expected) in actual
            .captures.as_step().records
            .iter()
            .zip(&expected.captures.as_step().records)
        {
            assert_eq!(actual.outcome, expected.outcome, "{}", actual.path);
            let path = &actual.path;
            match (&actual.payload, &expected.payload) {
                (Some(CapturePayload::Tensor(actual)), Some(CapturePayload::Tensor(expected))) => {
                    assert_eq!(actual.shape(), expected.shape());
                    let (
                        eredu_core::TensorObservationData::F32(actual),
                        eredu_core::TensorObservationData::F32(expected),
                    ) = (actual.data(), expected.data())
                    else {
                        panic!("floating component fixture")
                    };
                    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                        let bound = conditioned_bounds
                            .get(path)
                            .map_or(f64::from(tolerance * expected.abs().max(1.)), |bounds| {
                                bounds[index]
                            });
                        assert!(
                            (f64::from(*actual) - f64::from(*expected)).abs() <= bound,
                            "component {path} phase {phase:?} index {index}: {actual} != {expected}"
                        );
                    }
                }
                (None, None) => {}
                // Sparse records preserve coordinates/counts; native reduction
                // order may change unit values within the same tolerance.
                (
                    Some(CapturePayload::RoutedUnits(actual)),
                    Some(CapturePayload::RoutedUnits(expected)),
                ) => {
                    assert_eq!(actual.geometry, expected.geometry);
                    assert!(actual.source_token_ranges.is_empty());
                    assert_eq!(actual.rows.len(), expected.rows.len());
                    for (row_index, (actual, expected)) in
                        actual.rows.iter().zip(&expected.rows).enumerate()
                    {
                        assert_eq!(
                            (
                                actual.source_peer,
                                actual.token,
                                actual.slot,
                                actual.expert,
                                actual.unit_start,
                                actual.unit_stride
                            ),
                            (
                                expected.source_peer,
                                expected.token,
                                expected.slot,
                                expected.expert,
                                expected.unit_start,
                                expected.unit_stride
                            )
                        );
                        assert!((actual.coefficient - expected.coefficient).abs() <= 2e-5);
                        assert_eq!(actual.values.shape(), expected.values.shape());
                        let (
                            eredu_core::TensorObservationData::F32(actual),
                            eredu_core::TensorObservationData::F32(expected),
                        ) = (actual.values.data(), expected.values.data())
                        else {
                            panic!("floating routed units")
                        };
                        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                            let bound = routed_bounds
                                .get(path)
                                .map_or(f64::from(tolerance * expected.abs().max(1.)), |bounds| {
                                    bounds[row_index][index]
                                });
                            assert!(
                                (f64::from(*actual) - f64::from(*expected)).abs() <= bound,
                                "routed unit {path} phase {phase:?} row={row_index} index={index}: {actual} != {expected}, bound {bound}"
                            );
                        }
                    }
                }
                _ => panic!("capture payload differs for {}", actual.path),
            }
        }
        assert!(
            !actual.captures.as_step().partitions.is_empty()
                || actual
                    .captures.as_step().records
                    .iter()
                    .all(|record| record.payload.is_none())
        );
    }
}

fn assert_prediction_replay(
    actual: &[eredu_core::speculative::SpeculativeActivationCapture],
    expected: &[eredu_core::speculative::SpeculativeActivationCapture],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.phase, expected.phase);
        assert_eq!(actual.captures.as_step().records, expected.captures.as_step().records);
        assert_eq!(actual.admission_identity, expected.admission_identity);
        assert!(actual.completed);
    }
}
