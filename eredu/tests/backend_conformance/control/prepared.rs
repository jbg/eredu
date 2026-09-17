use super::*;
use eredu::api::*;
use eredu_core::{
    InputModality, InputPartDescriptor, InputPayloadKind, InputTensorIdentity,
    PreparedControlInput, PreparedControlInputBackend, PreparedControlInputError,
    PreparedPromptAttribution, PreparedPromptSegment, PreparedPromptSegmentPlan,
    PromptTokenAttribution, SharedPromptAttribution,
};
use std::cell::Cell;

thread_local! { static FACTORIES: Cell<usize> = const { Cell::new(0) }; }
pub(crate) struct Input {
    prompt: Vec<u32>,
    attribution: SharedPromptAttribution,
    session: String,
}
impl PreparedControlInput for Input {
    type Prompt = Vec<u32>;
    fn attribution(&self) -> &PreparedPromptAttribution {
        self.attribution.attribution()
    }
    fn shared_attribution(&self) -> &SharedPromptAttribution {
        &self.attribution
    }
}
impl PreparedControlInputBackend for MockBackend {
    type ControlInput = Input;
    fn prepare_control_input(
        runtime: &ModelRuntime<Self>,
        prompt: Vec<u32>,
    ) -> Result<Input, eredu_core::BackendFailure> {
        let host = Self::acquire_host_preparation(runtime)?;
        FACTORIES.with(|count| count.set(count.get() + 1));
        let positions = prompt.len() as u64;
        // This existing neutral backend has no persistent model state. Its actual
        // prefill is exactly this Vec, and its real opening frontier is zero.
        let prepared = eredu_core::PreparedInputIdentity::new(vec![InputPartDescriptor::new(
            InputModality::Text,
            InputPayloadKind::TokenIds,
            InputTensorIdentity::new(TensorDtype::U32, vec![1, prompt.len()]).unwrap(),
            [],
        )
        .unwrap()])
        .unwrap();
        let value = PreparedPromptAttribution {
            schema_version: eredu_core::PREPARED_PROMPT_ATTRIBUTION_VERSION,
            prepared,
            semantic_content_identity: eredu_core::cache::prompt_cache_token_fingerprint(&prompt),
            opening_position: 0,
            decoder_positions: positions,
            batch: 1,
            segments: vec![PreparedPromptSegment {
                plan: PreparedPromptSegmentPlan {
                    source_part: 0,
                    modality: InputModality::Text,
                    payload: InputPayloadKind::TokenIds,
                    decoder_range: [0, positions],
                },
                tokens: PromptTokenAttribution::Canonical {
                    range: [0, positions],
                },
            }],
            canonical_token_ids: prompt.clone(),
        };
        let attribution = SharedPromptAttribution::from_prepared(value, host).unwrap();
        Ok(Input {
            prompt,
            attribution,
            session: runtime.session().intervention_identity.clone(),
        })
    }
    fn consume_control_input(
        runtime: &ModelRuntime<Self>,
        input: Input,
    ) -> Result<(Vec<u32>, SharedPromptAttribution), eredu_core::BackendFailure> {
        if input.session != runtime.session().intervention_identity {
            return Err(PreparedControlInputError::SourceMismatch.into_backend_failure());
        }
        runtime
            .session()
            .authority
            .require_idle()
            .map_err(eredu_core::BackendFailure::from_error)?;
        Ok((input.prompt, input.attribution))
    }
}
fn prepare(
    model: &LoadedModel<MockBackend>,
    chat: &PreparedChat,
    settings: PreparedChatGenerationSettings,
) -> PreparedControlledInput<MockBackend> {
    let ids = model.encode(chat.rendered_prompt(), false).unwrap();
    model
        .prepare_controlled_input(
            chat,
            ids,
            settings,
            PreparedInputInstrumentation::Unobserved,
            limits(),
        )
        .unwrap()
}
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
fn token_ranges(records: &[PreparedControlledGenerationRecord]) -> Vec<(u32, [u64; 2])> {
    records
        .iter()
        .filter_map(|r| match &r.event {
            PreparedControlledGenerationEvent::Progress {
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
fn v2_prepared_run_and_steps_share_unicode_commit_ranges_and_no_startup_prediction() {
    let mut outputs = Vec::new();
    for stepped in [false, true] {
        let (mut model, chat, settings, first) = setup();
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let guard = crate::host_authority::Guard::new(&pool);
        let prepared = prepare(&model, &chat, settings);
        assert_eq!(
            prepared.prompt_attribution().decoder_positions,
            u64::from(first)
        );
        let mut records = Vec::new();
        let mut session = model
            .start_controlled_prepared_text(prepared, &[], Default::default(), |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
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
                PreparedControlledGenerationEvent::Progress {
                    event: ObservedGenerationEvent::Semantic { event, .. },
                } => Some(event.clone()),
                _ => None,
            })
            .collect();
        assert!(semantics.contains(&SemanticEvent::TextDelta("é".into())));
        assert!(records
            .iter()
            .all(|r| r.instrumentation == PreparedInstrumentationRecord::Unobserved));
        outputs.push((session.token_ids().to_vec(), semantics));
        drop(session);
        drop(model);
        assert!(
            pool.unquoted_owner_count().unwrap() > 0,
            "escaped V2 records retain actual host custody"
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
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn v2_cancelled_start_and_consumer_break_do_not_predict() {
    for case in ["cancel", "break", "zero", "panic"] {
        let (mut model, chat, mut settings, _) = setup();
        if case == "zero" {
            settings.overrides.max_new_tokens = Some(0);
        }
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let guard = crate::host_authority::Guard::new(&pool);
        if case == "zero" {
            // The existing resolver rejects zero before startup, after the real
            // V2 input factory. The returned error must retain preparation custody.
            let ids = model.encode(chat.rendered_prompt(), false).unwrap();
            let factories = FACTORIES.with(Cell::get);
            let error = model
                .prepare_controlled_input(
                    &chat,
                    ids,
                    settings,
                    PreparedInputInstrumentation::Unobserved,
                    limits(),
                )
                .err()
                .expect("zero token budget must reject during preparation");
            assert!(has_cause::<PreparedChatError>(&error, |cause| matches!(
                cause,
                PreparedChatError::Generation(
                    eredu_core::generation::GenerationError::ZeroTokenBudget
                )
            )));
            assert_eq!(FACTORIES.with(Cell::get), factories + 1);
            assert!(guard.update(|p| p.steps.is_empty()));
            drop(model);
            assert!(pool.unquoted_owner_count().unwrap() > 0);
            drop(error);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            continue;
        }
        let prepared = prepare(&model, &chat, settings);
        let control = GenerationControlHandle::default();
        if case == "cancel" {
            control.cancel();
        }
        if case == "panic" {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = model.start_controlled_prepared_text(prepared, &[], control, |_| {
                    panic!("startup disconnect")
                });
            }));
            assert!(result.is_err());
            assert!(guard.update(|p| p.steps.is_empty()));
            drop(model);
            assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
            continue;
        }
        let mut session = model
            .start_controlled_prepared_text(prepared, &[], control, |_| {
                if case == "break" {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        session.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(session.status(), GenerationStatus::Cancelled);
        assert!(session.token_ids().is_empty());
        assert!(guard.update(|p| p.steps.is_empty()));
    }
}

#[test]
fn v2_managed_rejects_before_factory_and_empty_capture_rejects_at_unimplemented_bind() {
    fn rejection<'a>(
        error: &'a (dyn std::error::Error + 'static),
    ) -> Option<&'a PreparedControlInputError> {
        let mut cause = Some(error);
        while let Some(error) = cause {
            if let Some(value) = error.downcast_ref::<PreparedControlInputError>() {
                return Some(value);
            }
            cause = error.source();
        }
        None
    }
    let (model, chat, settings, _) = setup();
    let ids = model.encode(chat.rendered_prompt(), false).unwrap();
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let guard = crate::host_authority::Guard::new(&pool);
    let before = FACTORIES.with(Cell::get);
    let mut managed = settings;
    managed.inference.managed_memory_capacity_bytes = Some(1 << 20);
    let error = model
        .prepare_controlled_input(
            &chat,
            ids.clone(),
            managed,
            PreparedInputInstrumentation::Unobserved,
            limits(),
        )
        .err()
        .expect("managed gate");
    assert!(matches!(
        rejection(&error),
        Some(PreparedControlInputError::UnknownBound)
    ));
    assert_eq!(FACTORIES.with(Cell::get), before);
    assert!(guard.update(|p| p.attempts == 0 && p.steps.is_empty()));
    drop(error);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);

    // Ordinary empty Capture is eligible for V2 preparation. This mock exposes
    // valid discovery, but intentionally retains the default unsupported binder.
    let error = model
        .prepare_controlled_input(
            &chat,
            ids.clone(),
            settings,
            PreparedInputInstrumentation::Capture {
                plan: CapturePlan::none(),
            },
            limits(),
        )
        .err()
        .expect("unimplemented prepared capture binding");
    assert!(matches!(
        rejection(&error),
        Some(PreparedControlInputError::InstrumentationUnavailable)
    ));
    assert_eq!(FACTORIES.with(Cell::get), before + 1);
    assert!(guard.update(|p| p.attempts == 2 && p.steps.is_empty()));
    // The established installed empty plan remains supported by the legacy route.
    assert!(model
        .prepare_observed_token_ids(&chat, ids, settings, CapturePlan::none(), limits())
        .is_ok());
    assert!(guard.update(|p| p.steps.is_empty()));
    drop(model);
    assert_eq!(
        pool.unquoted_owner_count().unwrap(),
        1,
        "escaped bind error retains its facade host authority"
    );
    drop(error);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn v2_foreign_session_and_trace_failure_retain_safe_ordinary_custody() {
    {
        let (model, chat, settings, _) = setup();
        let ids = model.encode(chat.rendered_prompt(), false).unwrap();
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let guard = crate::host_authority::Guard::new(&pool);
        // Provider source custody succeeds, then the actual facade host request fails.
        guard.update(|p| p.reject_at = Some(2));
        let error = model
            .prepare_controlled_input(
                &chat,
                ids,
                settings,
                PreparedInputInstrumentation::Unobserved,
                limits(),
            )
            .err()
            .unwrap();
        assert!(guard.update(|p| p.attempts == 2 && p.steps.is_empty()));
        drop(model);
        assert!(pool.unquoted_owner_count().unwrap() > 0);
        drop(error);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
    let (model, chat, settings, _) = setup();
    let prepared = prepare(&model, &chat, settings);
    let (mut other, _, _, _) = setup();
    assert!(other
        .start_controlled_prepared_text(prepared, &[], Default::default(), |_| panic!(
            "foreign delivery"
        ))
        .err()
        .unwrap()
        .to_string()
        .contains("another loaded session"));
    let (mut model, chat, settings, _) = setup();
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let guard = crate::host_authority::Guard::new(&pool);
    let ids = model.encode(chat.rendered_prompt(), false).unwrap();
    let prepared = model
        .prepare_controlled_input(
            &chat,
            ids,
            settings,
            PreparedInputInstrumentation::Unobserved,
            TraceLimits {
                per_record_bytes: 1,
                total_bytes: 1,
            },
        )
        .unwrap();
    let error = model
        .start_controlled_prepared_text(prepared, &[], Default::default(), |_| {
            panic!("oversized delivery")
        })
        .err()
        .unwrap();
    assert!(guard.update(|p| p.steps.is_empty()));
    drop(model);
    assert!(
        pool.unquoted_owner_count().unwrap() > 0,
        "owning preparation failure outlives session"
    );
    drop(error);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn v2_wire_version_and_exact_record_transport_keep_source_attribution() {
    let legacy = ControlledGenerationRecord {
        schema_version: 1,
        sequence: 4,
        epoch: 2,
        timing: Default::default(),
        generation: ObservedGenerationRecord {
            schema_version: 1,
            run_id: "r".into(),
            artifact_identity: None,
            session_id: "s".into(),
            capture_plan_id: "p".into(),
            intervention_plan_id: None,
            parameter_overlay_id: None,
            event: ObservedGenerationEvent::Lifecycle {
                status: GenerationStatus::Paused,
                next_prediction: 3,
            },
        },
    };
    let legacy_wire = serde_json::to_string(&legacy).unwrap();
    assert_eq!(
        legacy_wire,
        r#"{"schema_version":1,"sequence":4,"epoch":2,"timing":{"time_to_first_token":null},"generation":{"schema_version":1,"run_id":"r","artifact_identity":null,"session_id":"s","capture_plan_id":"p","event":{"kind":"lifecycle","status":"paused","next_prediction":3}}}"#
    );
    let (mut model, chat, settings, _) = setup();
    let prepared = prepare(&model, &chat, settings);
    let mut record = None;
    let session = model
        .start_controlled_prepared_text(prepared, &[], Default::default(), |r| {
            record = Some(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    let record = record.unwrap();
    let json = serde_json::to_string(&record).unwrap();
    let wire = PreparedControlledWireRecord::from_json(&json).unwrap();
    match wire.event {
        PreparedControlledGenerationEvent::Started {
            prompt_attribution, ..
        } => assert_eq!(&prompt_attribution, session.prompt_attribution()),
        _ => panic!("started"),
    }
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value["schema_version"] = 999.into();
    assert!(PreparedControlledWireRecord::from_json(&value.to_string()).is_err());
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
fn v2_snapshot_restore_and_serial_branch_keep_attribution_and_monotone_budgets() {
    let (mut model, chat, settings, first) = setup();
    let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let guard = crate::host_authority::Guard::new(&pool);
    let prepared = prepare(&model, &chat, settings);
    let mut session = model
        .start_controlled_prepared_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    let snapshot_limits = SnapshotLimits {
        max_snapshots: 3,
        max_branches: 1,
        retained_bytes: 16_000_000,
        cumulative_copy_bytes: 64_000_000,
    };
    session.enable_snapshots(snapshot_limits).unwrap();
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
    // The original 1 MiB child trace's possible semantic prefix alone exceeds
    // this retained limit. Reject it before copying, without spending the budget.
    let oversized_trace = limits();
    assert!(
        oversized_trace
            .total_bytes
            .checked_mul(std::mem::size_of::<SemanticEvent>() as u64 + 1)
            .unwrap()
            > snapshot_limits.retained_bytes
    );
    let before_fork = session.snapshot_usage().unwrap();
    let copies = guard.update(|p| p.copies.len());
    let mut rejected_emission = false;
    let error = session
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: oversized_trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |_| {
                rejected_emission = true;
                ControlFlow::Continue(())
            },
        )
        .err()
        .expect("oversized child trace must reject before native copy");
    assert!(has_cause::<ControlledGenerationError>(&error, |cause| {
        matches!(
            cause,
            ControlledGenerationError::Snapshot(
                eredu_runtime::execution_control::TextSnapshotError::Control(
                    ExecutionControlError::Limit("retained bytes")
                )
            )
        )
    }));
    assert!(!rejected_emission);
    assert_eq!(session.snapshot_usage().unwrap(), before_fork);
    assert_eq!(guard.update(|p| p.copies.len()), copies);
    drop(error);
    // This three-token branch uses one existing record's allowance as its whole
    // trace budget. The actual run below must fit it; snapshot limits stay fixed.
    let child_trace = TraceLimits {
        per_record_bytes: oversized_trace.per_record_bytes,
        total_bytes: oversized_trace.per_record_bytes,
    };
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
    let lineage = lineage.unwrap();
    let branch_json = serde_json::to_string(&lineage).unwrap();
    assert!(PreparedControlledWireRecord::from_json(&branch_json).is_ok());
    let mut malformed: serde_json::Value = serde_json::from_str(&branch_json).unwrap();
    malformed["event"]["lineage"]["parent"]["schema_version"] = 999.into();
    assert!(PreparedControlledWireRecord::from_json(&malformed.to_string()).is_err());
    match &lineage.event {
        PreparedControlledGenerationEvent::BranchStarted {
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
    let dto: PreparedGenerationSnapshotData<PreparedPromptAttribution> =
        serde_json::from_str(&wire).unwrap();
    assert_eq!(dto.prompt_attribution, *session.prompt_attribution());
    drop(dto);
    drop(lineage);
    drop(branch);
    drop(saved);
    drop(session);
    drop(model);
    assert!(pool.unquoted_owner_count().unwrap() > 0);
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
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn v2_direct_choice_and_snapshot_configuration_errors_retain_actual_host_after_session() {
    use eredu_core::capture::CaptureError;
    for operation in 0..4 {
        let (mut model, chat, settings, _) = setup();
        let pool = eredu_runtime::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let guard = crate::host_authority::Guard::new(&pool);
        let prepared = prepare(&model, &chat, settings);
        let control = GenerationControlHandle::default();
        let mut session = model
            .start_controlled_prepared_text(prepared, &[], control.clone(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        let error = if operation == 3 {
            let limits = SnapshotLimits {
                max_snapshots: 3,
                max_branches: 1,
                retained_bytes: 16_000_000,
                cumulative_copy_bytes: 64_000_000,
            };
            session.enable_snapshots(limits).unwrap();
            session.enable_snapshots(limits).unwrap_err()
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
        assert!(has_cause::<ControlledGenerationError>(&error, |e| {
            matches!(e, ControlledGenerationError::Capture(capture)
                if matches!(capture.cause(), CaptureError::Invalid(message) if message == expected))
        }));
        assert!(guard.update(|p| p.steps.is_empty()));
        drop(session);
        drop(model);
        assert!(pool.unquoted_owner_count().unwrap() > 0);
        drop(error);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}
