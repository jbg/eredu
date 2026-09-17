const MODES: [ExecutionResidency; 3] = [
    ExecutionResidency::FullyResident,
    ExecutionResidency::LayerwiseHost,
    ExecutionResidency::DenseDiskStream,
];
fn persona_case() -> Case {
    Case {
        convention: eredu_core::RealtimeFrameConvention::AbsoluteDelayedSlots,
        chunks: [3, 5, 3],
        ..Default::default()
    }
}
fn fixture(change: Option<&str>) -> (tempfile::TempDir, moshi::MoshiConfig) {
    let config = moshi::config::personaplex_numeric_fixture();
    let plan = moshi::safetensors_plan(&config).unwrap();
    assert!(plan.layout_groups.is_empty());
    let tensors = plan
        .common_tensors
        .iter()
        .map(|c| {
            let shape = c
                .shape
                .iter()
                .map(|d| i32::try_from(*d).unwrap())
                .collect::<Vec<_>>();
            let seed = c
                .key
                .bytes()
                .fold(17u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
            let mut data = (0..elements(&shape))
                .map(|i| {
                    let d = ((i * 11 + (seed % 83) as usize) % 37) as f32 - 18.;
                    if c.key.ends_with(".alpha") {
                        0.9 + d * 0.003
                    } else {
                        d * 0.021 + 0.004
                    }
                })
                .collect::<Vec<_>>();
            if change == Some("norm") && c.key == "depformer.layers.0.norm1.alpha" {
                data[0] += 0.7;
                data[3] -= 0.4;
            }
            if change == Some("last-qkv") && c.key == "depformer.layers.0.self_attn.in_proj_weight"
            {
                assert_eq!(shape, [16 * 3 * 8, 8]);
                for (i, value) in data[15 * 3 * 8 * 8..].iter_mut().enumerate() {
                    *value = *value * 1.3 + ((i % 7) as f32 - 3.) * 0.04;
                }
            }
            (c.key.clone(), NumericTensor::new(shape, data))
        })
        .collect();
    let artifact = tempfile::tempdir().unwrap();
    // This explicit private marker is deliberately rejected by public parsing.
    std::fs::write(
        artifact.path().join("config.json"),
        r#"{"model_type":"personaplex","version":"private-scalar-proof"}"#,
    )
    .unwrap();
    write_payload_tensors(
        &artifact.path().join("model.safetensors"),
        tensors,
        &BTreeSet::new(),
    );
    (artifact, config)
}
fn selected_with_residency(
    path: &std::path::Path,
    config: &moshi::MoshiConfig,
    residency: ExecutionResidency,
) -> moshi::PreparedMoshiRealtime {
    select_preparation(
        moshi::artifact::prepare_personaplex_numeric_fixture(path, config.clone()).unwrap(),
        config,
        residency,
    )
}
fn changed(a: &NumericTensor, b: &NumericTensor) -> bool {
    assert_eq!(a.shape, b.shape);
    a.data
        .iter()
        .zip(&b.data)
        .any(|(a, b)| (a - b).abs() > 1e-6)
}
fn callback<'a>(frame: &'a Frame, path: &str) -> &'a NumericTensor {
    &frame
        .values
        .iter()
        .find(|(p, _)| p == path)
        .unwrap_or_else(|| panic!("missing real callback {path}"))
        .1
}
fn complete_nonzero(report: &Report, batch: usize) {
    assert_eq!(report.status, RequestStatus::Active);
    assert_eq!(report.frames.len(), 11);
    assert_eq!(report.committed_work, 11);
    assert_eq!(report.completion_calls, 11);
    assert_eq!(report.executions, 10);
    let init = &report.frames[0];
    assert!(init.values.is_empty() && init.diagnostics.is_empty());
    assert_eq!(init.outputs[1].shape, [batch as i32, 0]);
    assert_eq!(init.outputs[2].shape, [batch as i32, 8]);
    assert!(init.snapshot.calls.iter().all(|n| *n == 0));
    assert!(init
        .snapshot
        .state
        .as_ref()
        .iter()
        .all(|s| s.position() == 0 && s.attention.as_ref().unwrap().keys.is_none()));
    for (i, frame) in report.frames.iter().enumerate().skip(1) {
        assert_eq!(frame.snapshot.schedule.frontier(), i + 1);
        assert_eq!(frame.snapshot.calls.len(), 17);
        assert_eq!(frame.outputs[0].shape, [batch as i32, 1]);
        assert_eq!(frame.outputs[1].shape, [batch as i32, 16]);
        assert_eq!(frame.outputs[2].shape, [batch as i32, 8]);
        assert_eq!(frame.diagnostics.len(), 17);
        assert!(!frame.snapshot.history.is_empty());
        for (layer, s) in frame.snapshot.state.as_ref().iter().enumerate() {
            let a = s.attention.as_ref().unwrap();
            assert_eq!(a.window, Some(if layer < 2 { 3001 } else { 8 }));
            assert_eq!(a.offset, if layer < 2 { i as i32 } else { 16 });
            let keys = a.keys.as_ref().unwrap();
            let values = a.values.as_ref().unwrap();
            assert!(keys.data.iter().any(|v| v.abs() > 1e-9));
            assert!(values.data.iter().any(|v| v.abs() > 1e-9));
            assert_eq!(keys.shape[2], if layer < 2 { i as i32 } else { 8 });
            assert_eq!(keys.shape, values.shape);
        }
    }
    assert!(report.final_state.calls[0] > 0);
    assert!(report.final_state.calls[1..9].iter().all(|n| *n > 0));
    assert!(report.final_state.calls[9..].iter().all(|n| *n == 0));
}

#[test]
fn personaplex_private_fixture_keeps_released_gate_and_actual_selected_shared_sources() {
    let (artifact, config) = fixture(None);
    let public =
        moshi::MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
    assert_eq!(public.temporal().hidden_size(), 4096);
    assert_eq!(public.depth_template().hidden_size(), 1024);
    assert!(moshi::MoshiConfig::from_json(
        r#"{"model_type":"personaplex","version":"7b-v1","dim":8}"#
    )
    .is_err());
    assert!(moshi::prepare_realtime_model(artifact.path()).is_err());
    assert_ne!(public.identity(), config.identity());
    assert_eq!(
        config.frame_schedule().delays(),
        public.frame_schedule().delays()
    );
    assert_eq!(config.frame_schedule().total_audio_codebooks(), 16);
    assert_eq!(config.frame_schedule().depth_audio_codebooks(), 16);
    assert_eq!(config.frame_schedule().generated_audio_codebooks(), 8);
    assert_eq!(config.frame_schedule().input_audio_codebooks(), 8);
    assert_eq!(
        config.parameter_sharing(),
        moshi::ParameterSharing::SharedDepthNorms
    );
    assert_eq!(
        config.checkpoint_layout(),
        moshi::CheckpointLayout::PersonaPlexPytorch
    );
    for mode in MODES {
        let selected = selected_with_residency(artifact.path(), &config, mode);
        let tasks = selected.materialization_tasks().to_vec();
        let source = moshi::prepare_selected_moshi_realtime_source(selected).unwrap();
        source.artifact_identity().unwrap();
        let before = source.source().source_diagnostics().unwrap();
        eredu_runtime::preflight_realtime_materialization_tasks::<NumericBackend>(
            &tasks,
            source.source().as_ref(),
        )
        .unwrap();
        let (pinned, units) =
            eredu_runtime::realtime_task_binding_plan(&tasks, source.source().as_ref())
                .unwrap()
                .into_parts();
        let after = source.source().source_diagnostics().unwrap();
        assert_eq!(before.physical_reads, after.physical_reads);
        assert_eq!(before.physical_read_bytes, after.physical_read_bytes);
        assert!(!pinned.is_empty());
        assert_eq!(units.len(), 18);
        let norms = units
            .values()
            .flat_map(|bindings| bindings.iter())
            .filter(|b| {
                b.name().starts_with("depformer.slices.") && b.name().ends_with(".norm1.weight")
            })
            .collect::<Vec<_>>();
        assert_eq!(norms.len(), 32);
        // Each independent slice has its own real recipe-backed normalization.
        assert!(norms.iter().all(|b| b.alias_of().is_none()));
        let family = norms
            .iter()
            .filter(|b| b.name().ends_with(".layers.0.norm1.weight"))
            .collect::<Vec<_>>();
        assert_eq!(family.len(), 16);
        let resolved = family
            .iter()
            .map(|b| {
                payload::recipe_value(
                    &b.source_recipe(),
                    source.source().as_ref(),
                    &NumericContext::default(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        for value in &resolved[1..] {
            assert_tensor_exact(
                value,
                &resolved[0],
                "one physical norm recipe, sixteen unit bindings",
            );
        }
        assert!(tasks
            .iter()
            .flat_map(|t| t.components())
            .any(|c| c.requirement().recipe_owner().is_some()));
    }
}

#[test]
fn personaplex_selected_residencies_match_all_nonzero_state_across_prefix_continuations() {
    let (artifact, config) = fixture(None);
    for batch in [1, 2] {
        let reference = run(
            artifact.path(),
            &config,
            Case {
                batch,
                ..persona_case()
            },
        );
        complete_nonzero(&reference, batch);
        for mode in MODES {
            let actual = run(
                artifact.path(),
                &config,
                Case {
                    batch,
                    residency: mode,
                    bounded: true,
                    ..persona_case()
                },
            );
            complete_nonzero(&actual, batch);
            residency::same_report(&reference, &actual);
            if mode != ExecutionResidency::FullyResident {
                residency::storage_proof(&actual, mode);
            }
        }
    }
}

#[test]
fn personaplex_real_delays_mixed_targets_and_required_demand_control_actual_projections() {
    let (artifact, config) = fixture(None);
    for mode in MODES {
        let base = Case {
            residency: mode,
            ..persona_case()
        };
        let diagnostic = run(artifact.path(), &config, base);
        assert_eq!(vocabulary(&diagnostic.projections).len(), 10 * 17);
        let lean = run(
            artifact.path(),
            &config,
            Case {
                diagnostics: false,
                readout: eredu_core::OutputDemand::StateOnly,
                bounded: true,
                ..base
            },
        );
        let equivalent = run(
            artifact.path(),
            &config,
            Case {
                diagnostics: false,
                ..base
            },
        );
        // Identical execution permission: all mutable state compares in full.
        for (a, b) in equivalent.frames.iter().zip(&lean.frames) {
            same_snapshot(&a.snapshot, &b.snapshot);
            same_outputs(&a.outputs, &b.outputs);
            assert!(b.diagnostics.is_empty());
        }
        // Diagnostics execute forced user-audio bodies; lean execution skips
        // exactly the remaining forced suffix after its last sampled decision.
        for (i, (a, b)) in diagnostic.frames.iter().zip(&lean.frames).enumerate() {
            same_outputs(&a.outputs, &b.outputs);
            assert_eq!(a.snapshot.calls, b.snapshot.calls);
            assert_eq!(a.snapshot.random, b.snapshot.random);
            assert_eq!(a.snapshot.schedule, b.snapshot.schedule);
            assert_eq!(a.snapshot.history.len(), b.snapshot.history.len());
            for ((ac, av), (bc, bv)) in a.snapshot.history.iter().zip(&b.snapshot.history) {
                assert_eq!(ac, bc);
                assert_tensor_exact(
                    av,
                    bv,
                    "identical delayed payloads despite omitted forced bodies",
                );
            }
            for layer in 0..2 {
                let aa = a.snapshot.state.as_ref()[layer].attention.as_ref().unwrap();
                let bb = b.snapshot.state.as_ref()[layer].attention.as_ref().unwrap();
                assert_eq!(aa.offset, bb.offset);
                assert_eq!(aa.window, bb.window);
                same_tensor_option(&aa.keys, &bb.keys, "temporal keys");
                same_tensor_option(&aa.values, &bb.values, "temporal values");
            }
            for layer in 2..4 {
                assert_eq!(
                    a.snapshot.state.as_ref()[layer].position(),
                    if i == 0 { 0 } else { 16 }
                );
                assert_eq!(
                    b.snapshot.state.as_ref()[layer].position(),
                    if i == 0 {
                        0
                    } else if i == 1 {
                        1
                    } else {
                        8
                    }
                );
            }
        }
        let projections = vocabulary(&lean.projections);
        assert_eq!(
            projections.len(),
            lean.final_state.calls.iter().sum::<usize>()
        );
        assert!(projections.iter().all(|(_, shape)| shape == &[1, 1, 8]));
        assert!(projections.iter().all(|(p, _)| p == "text_linear.weight"
            || p.strip_prefix("depformer.slices.")
                .unwrap()
                .split('.')
                .next()
                .unwrap()
                .parse::<usize>()
                .unwrap()
                < 8));
        assert!(
            lean.frames[1].snapshot.calls[2..9].iter().all(|n| *n == 0),
            "released delay-one warmup padding"
        );
        assert!(lean.frames[2].snapshot.calls[2..9].iter().all(|n| *n > 0));
        let forced = run(
            artifact.path(),
            &config,
            Case {
                forced: true,
                diagnostics: false,
                readout: eredu_core::OutputDemand::StateOnly,
                ..base
            },
        );
        assert!(vocabulary(&forced.projections).is_empty());
        assert!(forced.final_state.calls.iter().all(|n| *n == 0));
        assert!(forced.final_state.state.as_ref()[2..]
            .iter()
            .all(|s| s.position() == 0));
        let observed = run(
            artifact.path(),
            &config,
            Case {
                forced: true,
                diagnostics: false,
                required_observer: true,
                observe_after_initialization: true,
                readout: eredu_core::OutputDemand::StateOnly,
                ..base
            },
        );
        let full_forced = run(
            artifact.path(),
            &config,
            Case {
                forced: true,
                ..base
            },
        );
        for (a, b) in full_forced.frames.iter().zip(&observed.frames) {
            same_snapshot(&a.snapshot, &b.snapshot);
            same_outputs(&a.outputs, &b.outputs);
            same_values(&a.values, &b.values);
        }
        assert_eq!(vocabulary(&observed.projections).len(), 170);
        assert!(observed.final_state.state.as_ref()[2..]
            .iter()
            .all(|s| s.position() == 16));
        for (a, b) in forced.frames.iter().zip(&observed.frames) {
            same_outputs(&a.outputs, &b.outputs);
        }
        let intervention = run(
            artifact.path(),
            &config,
            Case {
                diagnostics: false,
                required_observer: true,
                observe_after_initialization: true,
                intervene_text: true,
                readout: eredu_core::OutputDemand::StateOnly,
                ..base
            },
        );
        for frame in &intervention.frames[1..] {
            assert_eq!(frame.outputs[0].data, [3.]);
            assert_eq!(callback(frame, "model.logits").data[3], 100.);
        }
    }
}

#[test]
fn personaplex_initialization_observation_rejection_and_later_terminal_prefix_are_transactional() {
    let (artifact, config) = fixture(None);
    for mode in MODES {
        let base = Case {
            residency: mode,
            ..persona_case()
        };
        let rejected = run(
            artifact.path(),
            &config,
            Case {
                required_observer: true,
                reject_initialization: true,
                ..base
            },
        );
        assert!(rejected.frames.is_empty());
        assert_eq!(rejected.status, RequestStatus::Failed);
        assert_eq!(rejected.final_state.schedule.frontier(), 0);
        assert!(rejected.final_state.history.is_empty());
        assert!(rejected.final_state.calls.iter().all(|n| *n == 0));
        assert_eq!(rejected.final_state.random, Some(11));
        assert!(rejected.projections.is_empty());
        assert_eq!(rejected.completion_calls, 0);
        assert_eq!(rejected.executions, 0);
        assert!(rejected.retired_state.is_some());
        let reference = run(artifact.path(), &config, base);
        for (fail, cancel) in [(Some("depformer.slices.15.logits"), false), (None, true)] {
            let terminal = run(
                artifact.path(),
                &config,
                Case {
                    fail,
                    cancel,
                    bounded: true,
                    ..base
                },
            );
            assert_eq!(terminal.frames.len(), 3);
            assert_eq!(terminal.committed_work, 3);
            assert!(terminal.retired_state.is_some());
            for (a, b) in reference.frames.iter().zip(&terminal.frames) {
                same_snapshot(&a.snapshot, &b.snapshot);
                same_values(&a.values, &b.values);
                same_outputs(&a.outputs, &b.outputs);
            }
            same_snapshot(&reference.frames[2].snapshot, &terminal.final_state);
            if fail.is_some() {
                assert_eq!(terminal.status, RequestStatus::Failed);
                assert!(!terminal.failed_values.is_empty());
            }
        }
    }
}

#[test]
fn personaplex_physical_shared_norm_and_last_packed_slice_affect_their_actual_consumers() {
    let (original, config) = fixture(None);
    let (norm, _) = fixture(Some("norm"));
    let (last, _) = fixture(Some("last-qkv"));
    for mode in MODES {
        let case = Case {
            residency: mode,
            forced: true,
            ..persona_case()
        };
        let reference = run(original.path(), &config, case);
        let normalized = run(norm.path(), &config, case);
        let sliced = run(last.path(), &config, case);
        for ((a, b), c) in reference
            .frames
            .iter()
            .zip(&normalized.frames)
            .zip(&sliced.frames)
            .skip(1)
        {
            same_outputs(&a.outputs, &b.outputs);
            same_outputs(&a.outputs, &c.outputs);
            assert_tensor_exact(
                callback(a, "text_linear.logits"),
                callback(b, "text_linear.logits"),
                "depth norm does not alter temporal model",
            );
            assert!(changed(
                callback(a, "depformer.slices.0.logits"),
                callback(b, "depformer.slices.0.logits")
            ));
            assert!(changed(
                callback(a, "depformer.slices.15.logits"),
                callback(b, "depformer.slices.15.logits")
            ));
            for slice in 0..15 {
                let path = format!("depformer.slices.{slice}.logits");
                assert_tensor_exact(
                    callback(a, &path),
                    callback(c, &path),
                    "earlier packed slices and forced feedback remain exact",
                );
            }
            assert!(changed(
                callback(a, "depformer.slices.15.logits"),
                callback(c, "depformer.slices.15.logits")
            ));
        }
    }
}
