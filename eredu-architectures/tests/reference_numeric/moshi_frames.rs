//! Selected nonzero frames, using the production realtime coordinator.

#[path = "moshi_frames/demand.rs"]
mod demand;
#[path = "moshi_frames/residency.rs"]
mod residency;
include!("../support/numeric/frame_imports.rs");

fn config(depth: usize, delayed: bool) -> serde_json::Value {
    serde_json::json!({
        "model_type":"moshi", "dim":4,"text_card":7,"n_q":4,"dep_q":depth,
        "generated_audio_codebooks":2,"card":6,"num_heads":1,"num_layers":2,
        "dim_feedforward":12,"causal":true,"context":3,"max_period":10000.0,
        "positional_embedding":"rope","depformer_dim":4,"depformer_dim_feedforward":12,
        "depformer_num_heads":1,"depformer_num_layers":2,"depformer_context":4,
        "depformer_max_period":10000.0,"depformer_pos_emb":"none",
        "delays":if delayed {vec![0,0,1,2,1]} else {vec![0,0,0,0,0]}
    })
}
fn fixture(depth: usize, delayed: bool, scale: f32) -> (tempfile::TempDir, moshi::MoshiConfig) {
    let value = config(depth, delayed);
    let (artifact, _) =
        prepared_adapter::payload_fixture_config_with(&value, 1.0, |name, shape| {
            let seed = name
                .bytes()
                .fold(17_u32, |a, b| a.wrapping_mul(31).wrapping_add(b.into()));
            Some(NumericTensor::new(
                shape.to_vec(),
                (0..elements(shape))
                    .map(|i| {
                        let d = ((i * 11 + (seed % 83) as usize) % 37) as f32 - 18.;
                        if name.contains("norm") && name.ends_with("weight") {
                            0.9 + d * 0.003
                        } else {
                            scale * (d * 0.021 + 0.004)
                        }
                    })
                    .collect(),
            ))
        });
    (
        artifact,
        moshi::MoshiConfig::from_json(&value.to_string()).unwrap(),
    )
}

include!("../support/numeric/frame_support.rs");
include!("../support/numeric/frame_selection.rs");

fn selected(path: &std::path::Path, config: &moshi::MoshiConfig) -> moshi::PreparedMoshiRealtime {
    selected_with_residency(path, config, ExecutionResidency::FullyResident)
}
fn selected_with_residency(
    path: &std::path::Path,
    config: &moshi::MoshiConfig,
    residency: ExecutionResidency,
) -> moshi::PreparedMoshiRealtime {
    select_preparation(
        moshi::prepare_realtime_model(path).unwrap(),
        config,
        residency,
    )
}

include!("../support/numeric/frames.rs");

#[test]
fn moshi_selected_multidepth_nonzero_frames_preserve_complete_state_and_history() {
    for depth in [2, 3] {
        let (artifact, config) = fixture(depth, true, 1.0);
        // Published temporal context counts previous positions, while the
        // depth context is already the complete visible-key window.
        assert_eq!(config.temporal().context(), 3);
        assert_eq!(config.depth_template().context(), 4);
        let temporal_window = config.temporal().attention_window();
        let depth_window = config.depth_template().attention_window();
        assert_eq!(temporal_window, config.temporal().context() + 1);
        assert_eq!(depth_window, config.depth_template().context());
        for batch in [1, 2] {
            let ordinary = run(
                artifact.path(),
                &config,
                Case {
                    batch,
                    ..Default::default()
                },
            );
            let bounded = run(
                artifact.path(),
                &config,
                Case {
                    batch,
                    bounded: true,
                    ..Default::default()
                },
            );
            assert_eq!(ordinary.frames.len(), 10);
            assert_eq!(ordinary.status, RequestStatus::Active);
            assert_eq!(ordinary.bound_parameters, bounded.bound_parameters);
            assert_eq!(ordinary.completion_calls, 10);
            assert_eq!(ordinary.committed_work, 10);
            assert!(ordinary.retired_state.is_none());
            assert_eq!(ordinary.executions, bounded.executions);
            assert_eq!(ordinary.executions, 10);
            let expected = moshi::observation_points(&config)
                .into_iter()
                .map(|p| p.path())
                .chain(std::iter::once("model.logits".into()))
                .collect::<Vec<_>>();
            for (position, (a, b)) in ordinary.frames.iter().zip(&bounded.frames).enumerate() {
                for (index, layer) in a.snapshot.state.as_ref().iter().enumerate() {
                    let cache = layer.attention.as_ref().unwrap();
                    if index < 2 {
                        assert_eq!(cache.offset, (position + 1) as i32);
                        assert_eq!(cache.window, Some(temporal_window));
                        assert_eq!(
                            cache.retained(),
                            ((position + 1) as i32).min(temporal_window)
                        );
                        assert_eq!(layer.resets, 0);
                    } else {
                        assert_eq!(cache.offset, depth as i32);
                        assert_eq!(cache.window, Some(depth_window));
                        assert_eq!(cache.retained(), (depth as i32).min(depth_window));
                        assert_eq!(layer.resets, position + 1);
                    }
                }
                same_snapshot(&a.snapshot, &b.snapshot);
                same_values(&a.values, &b.values);
                same_outputs(&a.outputs, &b.outputs);
                same_outputs(&a.diagnostics, &b.diagnostics);
                assert_eq!(
                    a.values.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>(),
                    expected
                );
            }
            let state = &bounded.final_state.state;
            for (index, layer) in state.as_ref().iter().enumerate() {
                assert!(layer.position() > 0);
                let cache = layer.attention.as_ref().unwrap();
                assert!(cache
                    .keys
                    .as_ref()
                    .unwrap()
                    .data
                    .iter()
                    .any(|v| v.abs() > 1e-9));
                if index < 2 {
                    assert_eq!(layer.resets, 0);
                    assert!(cache.offset > cache.retained());
                } else {
                    assert_eq!(cache.offset, depth as i32);
                    assert_eq!(layer.resets, bounded.executions);
                }
            }
            assert!(!bounded.final_state.history.is_empty());
            assert!(bounded.final_state.calls.iter().any(|n| *n > 0));
            assert!(bounded.final_state.random.unwrap() > 11);
            if depth == 2 && batch == 1 {
                let (changed, changed_config) = fixture(depth, true, 0.63);
                let changed = run(changed.path(), &changed_config, Case::default());
                let text = |report: &Report| {
                    report
                        .frames
                        .last()
                        .unwrap()
                        .values
                        .iter()
                        .find(|(path, _)| path == "text_linear.logits")
                        .unwrap()
                        .1
                        .data
                        .clone()
                };
                let original_logits = text(&ordinary);
                let changed_logits = text(&changed);
                assert!(
                    original_logits
                        .iter()
                        .zip(changed_logits)
                        .any(|(a, b)| (*a - b).abs() > 1e-5),
                    "real source payload must influence complete text logits"
                );
            }
        }
    }
}

#[test]
fn moshi_selected_forced_tail_preserves_real_diagnostics_and_skip_boundary() {
    let (artifact, config) = fixture(2, false, 1.0);
    let full = run(
        artifact.path(),
        &config,
        Case {
            forced: true,
            ..Default::default()
        },
    );
    let skipped = run(
        artifact.path(),
        &config,
        Case {
            forced: true,
            diagnostics: false,
            bounded: true,
            ..Default::default()
        },
    );
    assert_eq!(full.executions, 10);
    assert_eq!(skipped.executions, 10);
    for (a, b) in full.frames.iter().zip(&skipped.frames) {
        assert_eq!(a.snapshot.schedule, b.snapshot.schedule);
        assert_eq!(a.snapshot.calls, b.snapshot.calls);
        assert_eq!(a.snapshot.random, b.snapshot.random);
        assert_eq!(a.snapshot.history.len(), b.snapshot.history.len());
        for ((ac, av), (bc, bv)) in a.snapshot.history.iter().zip(&b.snapshot.history) {
            assert_eq!(ac, bc);
            assert_tensor_exact(av, bv, "forced coordinate payload");
        }
        same_outputs(&a.outputs, &b.outputs);
        assert!(!a.diagnostics.is_empty());
        assert!(b.diagnostics.is_empty());
        assert_eq!(
            a.values
                .iter()
                .filter(|(p, _)| p.starts_with("depformer.slices."))
                .count(),
            2
        );
        assert!(b
            .values
            .iter()
            .all(|(p, _)| !p.starts_with("depformer.slices.")));
        for (a, b) in a.snapshot.state.as_ref()[..2]
            .iter()
            .zip(&b.snapshot.state.as_ref()[..2])
        {
            assert_eq!(a.position(), b.position());
            same_tensor_option(
                &a.attention.as_ref().unwrap().keys,
                &b.attention.as_ref().unwrap().keys,
                "forced temporal keys",
            );
            same_tensor_option(
                &a.attention.as_ref().unwrap().values,
                &b.attention.as_ref().unwrap().values,
                "forced temporal values",
            );
        }
    }
    assert!(skipped.final_state.calls.iter().all(|n| *n == 0));
    assert_eq!(skipped.final_state.random, Some(11));
    for layer in &skipped.final_state.state.as_ref()[2..] {
        let cache = layer.attention.as_ref().unwrap();
        assert_eq!(cache.offset, 0);
        assert!(cache.keys.is_none() && cache.values.is_none());
        assert_eq!(layer.resets, 10);
    }
}

#[test]
fn moshi_selected_callback_failure_and_cancellation_preserve_committed_prefix() {
    let (artifact, config) = fixture(3, false, 1.0);
    let ordinary = run(artifact.path(), &config, Case::default());
    for (fail, cancel) in [
        (Some("text_linear.logits"), false),
        (Some("depformer.slices.1.logits"), false),
        (None, true),
    ] {
        let failed = run(
            artifact.path(),
            &config,
            Case {
                fail,
                cancel,
                bounded: true,
                ..Default::default()
            },
        );
        assert_eq!(failed.frames.len(), 2);
        assert_eq!(
            failed.status,
            if cancel {
                RequestStatus::Cancelled
            } else {
                RequestStatus::Failed
            }
        );
        same_snapshot(&failed.final_state, &ordinary.frames[1].snapshot);
        same_state(
            failed.retired_state.as_ref().unwrap(),
            &ordinary.frames[1].snapshot.state,
        );
        assert_eq!(failed.committed_work, 2);
        assert_eq!(failed.completion_calls, 2);
        if let Some(path) = fail {
            assert_eq!(failed.failed_values.last().unwrap().0, path);
        } else {
            assert!(failed.failed_values.is_empty());
        }
    }
}

#[test]
fn moshi_selected_source_handoff_rejects_changed_physical_payload_before_reads() {
    let (first, config) = fixture(2, true, 1.0);
    let (second, other) = fixture(3, true, 1.0);
    let admitted = selected(first.path(), &config);
    let catalog = admitted.source_metadata().clone();
    let selected_key = admitted
        .materialization_tasks()
        .iter()
        .flat_map(|task| task.components())
        .flat_map(|component| component.source_provenance())
        .next()
        .expect("actual selected physical source")
        .catalog_key
        .clone();
    let original =
        moshi::prepare_selected_moshi_realtime_source(selected(first.path(), &config)).unwrap();
    let foreign =
        moshi::prepare_selected_moshi_realtime_source(selected(second.path(), &other)).unwrap();
    assert_ne!(
        original.artifact_identity().unwrap(),
        foreign.artifact_identity().unwrap()
    );
    drop(original);
    drop(foreign);
    let original_path = std::fs::canonicalize(first.path().join("model.safetensors")).unwrap();
    let changed_path = second.path().join("model.safetensors");
    assert_ne!(
        std::fs::metadata(&original_path).unwrap().len(),
        std::fs::metadata(&changed_path).unwrap().len(),
        "different depth catalog must change the admitted file length"
    );
    std::fs::copy(changed_path, &original_path).unwrap();
    // Handoff validates the already-admitted immutable header/resolution. It
    // deliberately does not reopen payloads or repeat filesystem discovery.
    let source = moshi::prepare_selected_moshi_realtime_source(admitted).unwrap();
    let store = source.source().clone();
    assert_eq!(
        store.source_keys(),
        catalog.keys().cloned().collect::<Vec<_>>()
    );
    for (key, metadata) in &catalog {
        assert_eq!(&store.source_metadata(key).unwrap(), metadata);
    }
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
    for _ in 0..2 {
        let acquired = store.acquire_lease(eredu_checkpoint::store::TensorReadRequest {
            key: selected_key.clone(),
            selection: eredu_checkpoint::store::TensorSelection::Full,
            policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
        });
        assert!(matches!(
            acquired,
            Err(eredu_checkpoint::store::StoreError::AdmittedFileChanged { path })
                if path == original_path
        ));
    }
    // Actual selected construction may create unloaded modules, but its first
    // required source materialization must fail without publishing an executor.
    let started = Rc::new(Cell::new(false));
    let result = moshi::visit_selected_moshi_realtime_architecture::<NumericBackend, State, _>(
        source,
        &NumericContext::default(),
        Visitor {
            case: Case::default(),
            config,
            context: NumericContext::default(),
            started: started.clone(),
        },
    );
    assert!(matches!(
        result,
        Err(moshi::MoshiRealtimeDispatchError::Mechanism(_))
    ));
    assert!(started.get());
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
    assert!(diagnostics.payload_shard_paths.is_empty());
}
