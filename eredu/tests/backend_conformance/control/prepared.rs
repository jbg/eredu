use super::*;
use eredu::api::*;
use eredu_core::PreparedPromptAttribution;

// Inspect the preserved typed cause, including transparent outer error variants.
fn has_cause<T: std::error::Error + 'static>(
    mut error: &(dyn std::error::Error + 'static),
    predicate: impl Fn(&T) -> bool,
) -> bool {
    loop {
        if error.downcast_ref::<T>().is_some_and(&predicate) {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}
fn token_ranges(records: &[ControlledGenerationRecord]) -> Vec<(u32, [u64; 2])> {
    records
        .iter()
        .filter_map(|r| match &r.event {
            ControlledGenerationEvent::Progress {
                event:
                    ObservedGenerationEvent::Token {
                        token_id,
                        input_range,
                        captures,
                        ..
                    },
            } => {
                assert!(captures.is_none());
                Some((*token_id, *input_range))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn prepared_prepared_run_and_steps_share_unicode_commit_ranges_and_no_startup_prediction() {
    let mut outputs = Vec::new();
    for stepped in [false, true] {
        let (mut model, chat, settings, first) = setup();
        let pool = model.original_pool().clone();
        let guard = crate::host_authority::Guard::new(&pool);
        let prepared = PreparedChatRequest::new(&chat, original_sources::settings(settings));
        let mut records = Vec::new();
        let mut session = model
            .start_controlled_chat(prepared, limits(), Default::default(), |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap()
            .unwrap();
        assert_eq!(
            session.prompt_attribution().decoder_positions,
            u64::from(first)
        );
        assert_eq!(session.status(), GenerationStatus::Prepared);
        assert!(guard.update(|p| p.steps.is_empty()));
        assert!(session.token_ids().is_empty());
        if stepped {
            while matches!(
                session.status(),
                GenerationStatus::Prepared | GenerationStatus::Paused
            ) {
                session
                    .step(|r| {
                        records.push(r);
                        ControlFlow::Continue(())
                    })
                    .unwrap();
            }
        } else {
            session
                .run(|r| {
                    records.push(r);
                    ControlFlow::Continue(())
                })
                .unwrap();
        }
        assert_eq!(session.token_ids(), [first, first + 1, first + 2]);
        assert_eq!(
            token_ranges(&records),
            vec![
                (first, [0, first as u64]),
                (first + 1, [first as u64, first as u64 + 1]),
                (first + 2, [first as u64 + 1, first as u64 + 2])
            ]
        );
        let semantics: Vec<_> = records
            .iter()
            .filter_map(|r| match &r.event {
                ControlledGenerationEvent::Progress {
                    event: ObservedGenerationEvent::Semantic { event, .. },
                } => Some(event.clone()),
                _ => None,
            })
            .collect();
        assert!(semantics.contains(&SemanticEvent::TextDelta("é".into())));
        assert!(records
            .iter()
            .all(|r| r.instrumentation == PreparedInstrumentationRecord::Unobserved));
        outputs.push((
            session.token_ids().to_vec(),
            serde_json::to_value(semantics).unwrap(),
        ));
        drop(session);
        drop(chat);
        drop(model);
        assert!(
            pool.used_bytes().unwrap() > 0,
            "escaped control records retain actual host custody"
        );
        // Eight ordinary aliases can retire concurrently after the run/model.
        // Every alias must take the same consuming finalization path.
        let aliases: Vec<_> = (0..8).map(|_| records.clone()).collect();
        drop(records);
        std::thread::scope(|scope| {
            let joins: Vec<_> = aliases
                .into_iter()
                .map(|records| scope.spawn(move || drop(records)))
                .collect();
            for join in joins {
                join.join().unwrap();
            }
        });
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn prepared_cancelled_start_and_consumer_break_do_not_predict() {
    for case in ["cancel", "break", "zero", "panic"] {
        let (mut model, chat, mut settings, _) = setup();
        let pool = model.original_pool().clone();
        let guard = crate::host_authority::Guard::new(&pool);
        if case == "zero" {
            settings.overrides.max_new_tokens = Some(0);
        }
        let request = PreparedChatRequest::new(&chat, original_sources::settings(settings));
        let control = GenerationControlHandle::default();
        if case == "cancel" {
            control.cancel();
        }
        if case == "panic" {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = model.start_controlled_chat(request, limits(), control, |_| {
                    panic!("startup disconnect")
                });
            }))
            .is_err());
        } else if case == "zero" {
            let error = model
                .start_controlled_chat(request, limits(), control, |_| {
                    panic!("zero budget delivery")
                })
                .err()
                .unwrap();
            assert!(has_cause::<eredu_core::generation::GenerationError>(
                &error,
                |e| matches!(e, eredu_core::generation::GenerationError::ZeroTokenBudget)
            ));
        } else {
            let session = model
                .start_controlled_chat(request, limits(), control, |_| {
                    if case == "break" {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .unwrap();
            if case == "cancel" {
                assert!(session.is_none());
            } else {
                let mut session = session.unwrap();
                session.run(|_| ControlFlow::Continue(())).unwrap();
                assert_eq!(session.status(), GenerationStatus::Cancelled);
                assert!(session.token_ids().is_empty());
            }
        }
        assert!(guard.update(|p| p.steps.is_empty()));
        drop(chat);
        drop(model);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn prepared_capacity_mismatch_refuses_before_prediction_and_empty_capture_is_supported() {
    let (mut model, chat, settings, _) = setup();
    let pool = model.original_pool().clone();
    let guard = crate::host_authority::Guard::new(&pool);
    let mut settings = original_sources::settings(settings);
    settings.inference.managed_memory_capacity_bytes = Some(1);
    let error = model
        .start_controlled_chat(
            PreparedChatRequest::new(&chat, settings),
            limits(),
            Default::default(),
            |_| panic!("unadmitted delivery"),
        )
        .err()
        .unwrap();
    assert!(error.session_failure().unwrap().input_rejection().is_some());
    assert!(guard.update(|p| p.steps.is_empty()));
    drop(error);
    settings.inference.managed_memory_capacity_bytes = Some(original_sources::CAPACITY);
    let capture = CapturePlan::none();
    let mut request = PreparedChatRequest::new(&chat, settings);
    request.capture = Some(&capture);
    let mut records = Vec::new();
    let mut session = model
        .start_controlled_chat(request, limits(), Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    assert!(session.token_ids().is_empty());
    session
        .run(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(!session.token_ids().is_empty());
    assert!(records
        .iter()
        .all(|r| !matches!(r.instrumentation, PreparedInstrumentationRecord::Unobserved)));
    assert!(records
        .iter()
        .filter_map(|r| r.event.progress())
        .filter_map(ObservedGenerationEvent::captures)
        .all(|frame| frame.records.is_empty()));
    drop(session);
    drop(records);
    drop(chat);
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn prepared_foreign_session_and_trace_failure_preserve_exact_source_custody() {
    let (model, chat, settings, _) = setup();
    let (mut other, other_chat, _, _) = setup();
    let error = other
        .start_controlled_chat(
            PreparedChatRequest::new(&chat, original_sources::settings(settings)),
            limits(),
            Default::default(),
            |_| panic!("foreign delivery"),
        )
        .err()
        .unwrap();
    assert!(error.session_failure().unwrap().input_rejection().is_some());
    drop(error);
    drop(other_chat);
    drop(other);
    drop(chat);
    drop(model);
    let (mut model, chat, settings, _) = setup();
    let pool = model.original_pool().clone();
    let guard = crate::host_authority::Guard::new(&pool);
    let error = model
        .start_controlled_chat(
            PreparedChatRequest::new(&chat, original_sources::settings(settings)),
            TraceLimits {
                per_record_bytes: 1,
                total_bytes: 1,
            },
            Default::default(),
            |_| panic!("oversized delivery"),
        )
        .err()
        .unwrap();
    assert!(guard.update(|p| p.steps.is_empty()));
    drop(chat);
    drop(model);
    assert!(
        matches!(
            &error,
            ControlledGenerationError::Capture(eredu_core::capture::CaptureError::Limit { .. })
        ),
        "{error:?}"
    );
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "fixed quota refusal owns no destination payload"
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn wire_version_and_exact_record_transport_keep_source_attribution() {
    let (mut model, chat, settings, _) = setup();
    let prepared = PreparedChatRequest::new(&chat, original_sources::settings(settings));
    let mut record = None;
    let session = model
        .start_controlled_chat(prepared, limits(), Default::default(), |r| {
            record = Some(r);
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    let record = record.unwrap();
    let json = serde_json::to_string(&record).unwrap();
    let wire = ControlledWireRecord::from_json(&json).unwrap();
    match wire.event {
        ControlledGenerationEvent::Started {
            prompt_attribution, ..
        } => assert_eq!(&prompt_attribution, session.prompt_attribution()),
        _ => panic!("started"),
    }
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value["schema_version"] = 999.into();
    assert!(ControlledWireRecord::from_json(&value.to_string()).is_err());
    for short in [false, true] {
        let bytes = json.len() as u64 - u64::from(short);
        let mut budget = eredu_runtime::execution_control::TraceBudget::new(TraceLimits {
            per_record_bytes: bytes,
            total_bytes: bytes,
        });
        assert_eq!(budget.charge(&record).is_ok(), !short);
    }
}

#[test]
fn prepared_snapshot_restore_and_serial_branch_keep_attribution_and_monotone_budgets() {
    let (mut model, chat, settings, first) = setup();
    let pool = model.original_pool().clone();
    let guard = crate::host_authority::Guard::new(&pool);
    let prepared = PreparedChatRequest::new(&chat, original_sources::settings(settings));
    let mut session = model
        .start_controlled_chat(prepared, limits(), Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap()
        .unwrap();
    let snapshot_limits = SnapshotLimits {
        max_snapshots: 3,
        max_branches: 1,
        retained_bytes: 16_000_000,
        cumulative_copy_bytes: 64_000_000,
    };
    session
        .enable_snapshots(
            snapshot_limits,
            original_sources::CAPACITY,
            eredu_runtime::working_memory::WorkspaceCopyLimits::new(original_sources::CAPACITY),
        )
        .unwrap();
    let saved = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(
        saved.complete_token_ids(),
        Some(session.prompt_attribution().canonical_token_ids.as_slice())
    );
    assert_eq!(
        saved.metadata().instrumentation,
        PreparedInstrumentationRecord::Unobserved
    );
    session.step(|_| ControlFlow::Continue(())).unwrap();
    let before = session.snapshot_usage().unwrap();
    session
        .restore(&saved, |_| ControlFlow::Continue(()))
        .unwrap();
    assert!(session.token_ids().is_empty());
    assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > before.cumulative_copy_bytes);
    let before_fork = session.snapshot_usage().unwrap();
    let child_trace = limits();
    let mut lineage = None;
    let mut branch = session
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: child_trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |r| {
                lineage = Some(r);
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    let after_fork = session.snapshot_usage().unwrap();
    assert_eq!(after_fork.branches, before_fork.branches + 1);
    assert!(after_fork.retained_bytes > before_fork.retained_bytes);
    assert!(after_fork.cumulative_copy_bytes > before_fork.cumulative_copy_bytes);
    let error = session
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: child_trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |_| panic!("over-limit branch delivery"),
        )
        .err()
        .unwrap();
    assert!(
        matches!(
            error.session_failure().and_then(|e| e.snapshot_failure()),
            Some(
                eredu_runtime::execution_control::TextSnapshotError::Control(
                    ExecutionControlError::Limit(_)
                )
            )
        ),
        "{error:?}"
    );
    assert_eq!(session.snapshot_usage().unwrap(), after_fork);
    drop(error);
    let lineage = lineage.unwrap();
    let branch_json = serde_json::to_string(&lineage).unwrap();
    assert!(ControlledWireRecord::from_json(&branch_json).is_ok());
    let mut malformed: serde_json::Value = serde_json::from_str(&branch_json).unwrap();
    malformed["event"]["lineage"]["parent"]["schema_version"] = 999.into();
    assert!(ControlledWireRecord::from_json(&malformed.to_string()).is_err());
    match &lineage.event {
        ControlledGenerationEvent::BranchStarted {
            prompt_attribution,
            inherited_token_ids,
            ..
        } => {
            assert_eq!(prompt_attribution.attribution(), saved.prompt_attribution());
            assert!(inherited_token_ids.is_empty());
        }
        _ => panic!("branch attribution"),
    }
    session
        .exchange(&mut branch, |_| ControlFlow::Continue(()))
        .unwrap();
    session.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), [first, first + 1, first + 2]);
    assert_eq!(session.prompt_attribution(), saved.prompt_attribution());
    assert!(session.emitted_bytes() > 0);
    assert!(session.emitted_bytes() <= child_trace.total_bytes);
    assert!(
        session.snapshot_usage().unwrap().cumulative_copy_bytes >= after_fork.cumulative_copy_bytes
    );
    let metadata = saved.metadata().clone();
    let wire = serde_json::to_string(&metadata).unwrap();
    let dto: GenerationSnapshotData<PreparedPromptAttribution> =
        serde_json::from_str(&wire).unwrap();
    assert_eq!(dto.prompt_attribution, *session.prompt_attribution());
    drop(dto);
    drop(lineage);
    drop(branch);
    drop(saved);
    assert!(
        session.snapshot_usage().unwrap().retained_bytes >= metadata.retained_bytes,
        "escaped snapshot metadata retains its actual copy reservation"
    );
    drop(session);
    drop(chat);
    drop(model);
    assert!(pool.used_bytes().unwrap() > 0);
    let aliases: Vec<_> = (0..8).map(|_| metadata.clone()).collect();
    drop(metadata);
    std::thread::scope(|scope| {
        let joins: Vec<_> = aliases
            .into_iter()
            .map(|metadata| scope.spawn(move || drop(metadata)))
            .collect();
        for join in joins {
            join.join().unwrap();
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn prepared_terminal_sampling_facts_survive_restore_and_exchange_without_new_work() {
    use eredu_runtime::{execution_control::SnapshotBudget, working_memory::WorkspaceCopyLimits};

    let (mut model, chat, settings, _) = snapshots::snapshot_setup();
    let pool = model.original_pool().clone();
    let guard = crate::host_authority::Guard::new(&pool);
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let prepared = PreparedChatRequest::new(&chat, original_sources::settings(settings));
    let mut session = model.start_prepared_chat(prepared, &cancellation).unwrap().unwrap();
    assert_eq!(session.preparation_report().unwrap().geometry.max_output_tokens, 8);
    let sampling = session.sampling_state().unwrap();
    while session.finish_reason().is_none() {
        session = session.advance(&cancellation, &mut |_| {}).unwrap();
    }
    assert_eq!(session.status(), GenerationStatus::Completed);
    let history = session.token_ids().to_vec();
    let prediction = session.next_prediction();
    let steps = guard.update(|probe| probe.steps.len());
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: original_sources::CAPACITY,
        cumulative_copy_bytes: original_sources::CAPACITY * 8,
    });
    let saved = session.snapshot(&budget, original_sources::CAPACITY,
        WorkspaceCopyLimits::new(original_sources::CAPACITY)).unwrap();

    for stage in 0..3 {
        let mut branch = match stage {
            1 => {
                assert!(session.restore_snapshot(&saved, PreparedChatResumeSettings::default(),
                    original_sources::CAPACITY, &cancellation).unwrap());
                None
            }
            2 => {
                let mut branch = session.fork_snapshot(&saved, PreparedChatResumeSettings::default(),
                    original_sources::CAPACITY, &cancellation).unwrap().unwrap();
                session.exchange(&mut branch).unwrap();
                Some(branch)
            }
            _ => None,
        };
        assert_eq!(session.sampling_state().unwrap(), sampling);
        if stage != 0 {
            let geometry = session.preparation_report().unwrap().geometry;
            assert_eq!((geometry.input_positions, geometry.max_output_tokens,
                geometry.prefill_chunk_positions), (0, 0, 0));
            assert_eq!(geometry.output, eredu_core::OutputDemand::StateOnly);
        }
        assert!(session.override_sampling(SamplingOverride {
            temperature: Some(0.75), reseed: Some(99),
        }).is_err(), "terminal sampler cannot be replaced");
        assert_eq!(session.sampling_state().unwrap(), sampling);
        session = session.advance(&cancellation, &mut |_| panic!("terminal event")).unwrap();
        if let Some(branch) = &mut branch {
            session.exchange(branch).unwrap();
        }
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(session.token_ids(), history);
        assert_eq!(session.next_prediction(), prediction);
        assert_eq!(guard.update(|probe| probe.steps.len()), steps);
    }
    drop((saved, session));
    drop((chat, model));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn prepared_direct_choice_and_snapshot_configuration_refuse_without_prediction() {
    use eredu_core::capture::CaptureError;
    for operation in 0..4 {
        let (mut model, chat, settings, _) = setup();
        let pool = model.original_pool().clone();
        let guard = crate::host_authority::Guard::new(&pool);
        let prepared = PreparedChatRequest::new(&chat, original_sources::settings(settings));
        let control = GenerationControlHandle::default();
        let mut session = model
            .start_controlled_chat(prepared, limits(), control.clone(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap()
            .unwrap();
        let error = if operation == 3 {
            let limits = SnapshotLimits {
                max_snapshots: 3,
                max_branches: 1,
                retained_bytes: 16_000_000,
                cumulative_copy_bytes: 64_000_000,
            };
            session
                .enable_snapshots(
                    limits,
                    original_sources::CAPACITY,
                    eredu_runtime::working_memory::WorkspaceCopyLimits::new(
                        original_sources::CAPACITY,
                    ),
                )
                .unwrap();
            session
                .enable_snapshots(
                    limits,
                    original_sources::CAPACITY,
                    eredu_runtime::working_memory::WorkspaceCopyLimits::new(
                        original_sources::CAPACITY,
                    ),
                )
                .unwrap_err()
        } else {
            control.cancel();
            match operation {
                0 => session.force_next_token(0).unwrap_err(),
                1 => session.clear_forced_token().unwrap_err(),
                _ => session.sampling_state().unwrap_err(),
            }
        };
        let expected = if operation == 3 {
            "snapshot limits are already configured"
        } else {
            "generation cancellation was requested"
        };
        assert!(matches!(&error,ControlledGenerationError::Rejected(reason) if *reason==expected));
        assert!(guard.update(|p| p.steps.is_empty()));
        drop(session);
        drop(chat);
        drop(model);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
