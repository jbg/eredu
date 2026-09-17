use super::*;

fn setup(
    total: u64,
    shifted: bool,
    layout: usize,
    kind: usize,
    maximum: u64,
    slice: usize,
    limits: Option<CaptureLimits>,
) -> (Observer, Rc<RefCell<Backend>>, CaptureDiscovery) {
    let (mut plan, mut catalog, mut support, caps) = fixture(CaptureTransform::Preview {
        max_elements: maximum,
    });
    catalog.points[0].dtype = match kind {
        0 => ObservationDtype::Floating,
        1 | 2 => ObservationDtype::Integer,
        3 => ObservationDtype::Boolean,
        _ => ObservationDtype::Unknown,
    };
    let row = TensorAxis {
        name: "sequence".into(),
        dimension: SymbolicDimension::Sequence,
    };
    let fixed = |name: &str, n| TensorAxis {
        name: name.into(),
        dimension: SymbolicDimension::Known(n),
    };
    catalog.points[0].axes = Some(match layout {
        0 => vec![row, fixed("hidden", 4)],
        1 => vec![fixed("left", 2), row, fixed("right", 2)],
        _ => vec![fixed("hidden", 4), row],
    });
    support.capture = caps.clone();
    let original = plan.selections.pop().unwrap();
    for lane in 0..2 {
        let mut selected = original.clone();
        selected.id = format!("preview-{lane}");
        let rows = total - u64::from(lane == 1 && shifted);
        if rows != 0 && slice != 0 {
            selected.slices = vec![CaptureSlice {
                axis: "sequence".into(),
                start: match slice {
                    1 | 2 => 1,
                    _ => rows - 1,
                },
                end: if slice == 2 { 1 } else { rows },
                stride: if slice == 1 { 2 } else { 1 },
            }];
        }
        plan.selections.push(selected);
    }
    if let Some(limits) = limits {
        plan.limits = limits;
    } else {
        plan.limits.per_step.captures = 100;
        plan.limits.cumulative.captures = 100;
    }
    let admitted = plan
        .admit_invocations(
            &catalog,
            &support,
            &caps,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: total,
                max_context: None,
                max_predictions: 1,
            },
        )
        .unwrap();
    let backend = Rc::new(RefCell::new(Backend::default()));
    let observer = SpeculativeCaptureObserver::new(
        CaptureSession::new(admitted),
        Provider(backend.clone()),
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        vec![
            SpeculativeCaptureScope::Target,
            SpeculativeCaptureScope::Prediction { depth: 0 },
        ],
        vec![],
    )
    .unwrap();
    (
        observer,
        backend,
        CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog,
            support,
        },
    )
}
fn whole(rows: u64, layout: usize, kind: usize) -> Value {
    let shape = match layout {
        0 => vec![rows, 4],
        1 => vec![2, rows, 2],
        _ => vec![4, rows],
    };
    let count = (rows * 4) as usize;
    let data = match kind {
        0 => TensorObservationData::F32(
            (0..count)
                .map(|n| match n {
                    0 => -0.,
                    1 => f32::NAN,
                    2 => f32::INFINITY,
                    3 => f32::NEG_INFINITY,
                    _ => n as f32 + 0.25,
                })
                .collect(),
        ),
        1 => TensorObservationData::I64((0..count).map(|n| -(1i64 << 54) + n as i64).collect()),
        2 => TensorObservationData::U64((0..count).map(|n| (1u64 << 63) + n as u64).collect()),
        _ => TensorObservationData::Bool((0..count).map(|n| n % 3 == 1).collect()),
    };
    Value { shape, data }
}
fn local(global: &Value, axis: usize, start: u64, end: u64) -> Value {
    let mut shape = global.shape.clone();
    shape[axis] = end - start;
    let mut slice = ResolvedCaptureSlice {
        starts: vec![0; shape.len()],
        ends: global.shape.clone(),
        strides: vec![1; shape.len()],
        shape: shape.clone(),
    };
    slice.starts[axis] = start;
    slice.ends[axis] = end;
    let selected = selected_indices(&global.shape, &slice);
    let data = match &global.data {
        TensorObservationData::F32(v) => {
            TensorObservationData::F32(selected.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::I64(v) => {
            TensorObservationData::I64(selected.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::U64(v) => {
            TensorObservationData::U64(selected.iter().map(|i| v[*i]).collect())
        }
        TensorObservationData::Bool(v) => {
            TensorObservationData::Bool(selected.iter().map(|i| v[*i]).collect())
        }
    };
    Value { shape, data }
}
fn emit(
    observer: &mut Observer,
    total: u64,
    input_start: u64,
    input_end: u64,
    hidden: u64,
    axis: usize,
    seed: bool,
    value: &Value,
    hook: bool,
) -> Result<(), String> {
    let width = value.shape[axis];
    observer.set_prefill_span(Some(SpeculativePrefillSpan {
        prompt_tokens: total,
        input_start,
        input_end,
        position: input_start,
        hidden_start: hidden,
        token_start: input_end - width,
        sequence: width,
        seed_start: hidden,
    }));
    let result = with_speculative_activation(
        Some(observer),
        if seed {
            SpeculativeActivationPhase::PredictionPrefill
        } else {
            SpeculativeActivationPhase::TargetPrefill
        },
        width as usize,
        |o| {
            if hook {
                o.unwrap().observe("block.output", value)
            } else {
                Ok(())
            }
        },
    );
    observer.set_prefill_span(None);
    result
}
fn execute(
    observer: &mut Observer,
    total: u64,
    chunk: u64,
    shifted: bool,
    layout: usize,
    kind: usize,
) -> Result<(), String> {
    begin(observer, total, shifted);
    let axis = usize::from(layout != 0);
    for start in (0..total).step_by(chunk as usize) {
        let end = (start + chunk).min(total);
        for seed in [false, true] {
            let skip = u64::from(seed && shifted && start == 0);
            let width = end - start - skip;
            if width == 0 {
                continue;
            }
            let hidden = if seed && shifted {
                start.saturating_sub(1)
            } else {
                start
            };
            let value = local(
                &whole(total - u64::from(seed && shifted), layout, kind),
                axis,
                hidden,
                hidden + width,
            );
            emit(
                observer, total, start, end, hidden, axis, seed, &value, true,
            )?;
        }
    }
    observer.complete_prefill_reductions()
}
fn equal(a: &CapturePayload, b: &CapturePayload) {
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
    if let (Some(a), Some(b)) = (a.as_tensor(), b.as_tensor()) {
        if let (TensorObservationData::F32(a), TensorObservationData::F32(b)) = (a.data(), b.data())
        {
            assert_eq!(
                a.iter().map(|n| n.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|n| n.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
fn assert_report(
    observer: &Observer,
    report: &SpeculativePrefillReductions,
    discovery: &CaptureDiscovery,
    layout: usize,
    kind: usize,
) {
    for entry in &report.records {
        if entry.logical_sequence == 0 {
            assert_eq!(entry.status, SpeculativePrefillReductionStatus::Skipped);
            assert!(entry.record.payload.is_none());
            continue;
        }
        assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
        assert_eq!(entry.covered_sequence, entry.logical_sequence);
        let selected = &observer.session().plan().plan().selections[entry.selection_index];
        let global = whole(entry.logical_sequence, layout, kind);
        let slice = resolve_slice(&discovery.catalog.points[0], selected, &global.shape).unwrap();
        let expected = Backend::default()
            .transform(&global, selected, &slice)
            .unwrap();
        equal(entry.record.payload.as_ref().unwrap(), &expected);
        assert_eq!(entry.record.source_shape, Some(global.shape));
        assert_eq!(entry.record.selected_shape, Some(slice.shape.clone()));
        assert_eq!(
            entry.record.outcome,
            completed_capture_outcome(&selected.transform, elements(&slice.shape).unwrap())
        );
    }
}

#[test]
fn global_preview_matches_full_order_for_all_scalar_kinds_windows_and_strides() {
    for layout in 0..3 {
        for kind in 0..4 {
            for slice in 0..4 {
                for maximum in [0, 1, 7, 8, 9, 20, 25] {
                    let (mut observer, backend, discovery) =
                        setup(5, false, layout, kind, maximum, slice, None);
                    execute(&mut observer, 5, 2, false, layout, kind).unwrap();
                    let mut physical = drain(&mut observer);
                    assert!(physical.iter().all(|p| p.prefill_reductions.is_none()));
                    assert!(observer.session().checkpoint(&discovery).is_err());
                    let usage = observer.session().cumulative_usage();
                    observer.finish_prefill_reductions(true);
                    assert_eq!(observer.session().cumulative_usage(), usage);
                    physical.extend(drain(&mut observer));
                    let report = physical
                        .last()
                        .unwrap()
                        .prefill_reductions
                        .as_ref()
                        .unwrap();
                    assert_report(&observer, report, &discovery, layout, kind);
                    assert_eq!(physical.len(), 6);
                    for envelope in &physical {
                        assert!(envelope.completed);
                        let seed = envelope.phase == SpeculativeActivationPhase::PredictionPrefill;
                        let record = &envelope.captures.as_step().records[usize::from(seed)];
                        let span = envelope.prefill_span.unwrap();
                        assert_eq!(
                            envelope.captures.as_step().invocation.unwrap().sequence,
                            span.sequence
                        );
                        let value = local(
                            &whole(5, layout, kind),
                            usize::from(layout != 0),
                            span.hidden_start,
                            span.hidden_start + span.sequence,
                        );
                        let selected =
                            &observer.session().plan().plan().selections[usize::from(seed)];
                        let projection = CaptureInvocationWindow {
                            logical_sequence: 5,
                            start: span.hidden_start,
                        }
                        .project(
                            envelope.captures.as_step().invocation.unwrap(),
                            &discovery.catalog.points[0],
                            selected,
                            &value.shape,
                        )
                        .unwrap();
                        if let Some(fragment) = projection.fragments().first() {
                            let expected = Backend::default()
                                .transform(&value, selected, fragment.local())
                                .unwrap();
                            equal(record.payload.as_ref().unwrap(), &expected);
                            assert_eq!(
                                record.outcome,
                                completed_capture_outcome(
                                    &selected.transform,
                                    elements(&fragment.local().shape).unwrap()
                                )
                            );
                        } else {
                            assert!(record.payload.is_none());
                            assert!(matches!(
                                record.outcome,
                                CaptureOutcome::Skipped {
                                    reason: CaptureSkipReason::NotInvoked
                                }
                            ));
                        }
                    }
                    if slice == 2 {
                        assert_eq!(backend.borrow().transforms, 0);
                    }
                    if maximum == 0 {
                        assert_eq!(backend.borrow().exported, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn preview_preserves_shifted_lanes_zero_seed_and_nonrefunding_restore() {
    for (total, chunk, shifted) in [(5, 2, true), (6, 2, false), (5, 1, true), (1, 1, true)] {
        let (mut observer, _, discovery) = setup(total, shifted, 0, 1, 7, 0, None);
        let saved = observer.checkpoint(&discovery).unwrap();
        execute(&mut observer, total, chunk, shifted, 0, 1).unwrap();
        observer.finish_prefill_reductions(true);
        let physical = drain(&mut observer);
        let report = physical
            .last()
            .unwrap()
            .prefill_reductions
            .as_ref()
            .unwrap();
        assert_report(&observer, report, &discovery, 0, 1);
        for entry in &report.records {
            let windows: Vec<_> = physical.iter().filter(|e| e.phase == entry.phase).collect();
            assert_eq!(entry.windows, windows.len() as u64);
            assert_eq!(
                entry.first_invocation,
                windows.first().map(|e| e.invocation)
            );
            assert_eq!(entry.last_invocation, windows.last().map(|e| e.invocation));
        }
        let spent = observer.session().cumulative_usage();
        observer.set_activation_origin(None);
        observer.restore(&saved).unwrap();
        assert_eq!(observer.session().cumulative_usage(), spent);
    }
}

#[test]
fn preview_cancellation_and_failures_never_publish_a_filled_prefix_as_complete() {
    for sealed in [false, true] {
        let (mut observer, _, _) = setup(5, false, 0, 0, 1, 0, None);
        if sealed {
            execute(&mut observer, 5, 2, false, 0, 0).unwrap();
        } else {
            begin(&mut observer, 5, false);
            emit(
                &mut observer,
                5,
                0,
                2,
                0,
                0,
                false,
                &local(&whole(5, 0, 0), 0, 0, 2),
                true,
            )
            .unwrap();
        }
        let used = observer.session().cumulative_usage();
        observer.finish_prefill_reductions(false);
        assert_eq!(observer.session().cumulative_usage(), used);
        let physical = drain(&mut observer);
        assert!(physical.iter().any(|e| e.completed));
        for e in physical
            .last()
            .unwrap()
            .prefill_reductions
            .as_ref()
            .unwrap()
            .records
            .iter()
        {
            assert_eq!(e.status, SpeculativePrefillReductionStatus::Aborted);
            assert!(e.record.payload.is_none());
        }
    }
    for failure in 0..4 {
        let (mut observer, backend, _) = setup(5, false, 0, 0, 0, 0, None);
        begin(&mut observer, 5, false);
        emit(
            &mut observer,
            5,
            0,
            2,
            0,
            0,
            false,
            &local(&whole(5, 0, 0), 0, 0, 2),
            true,
        )
        .unwrap();
        let before = backend.borrow().transforms;
        let mut value = local(&whole(5, 0, if failure == 0 { 2 } else { 0 }), 0, 2, 4);
        if failure == 1 {
            value.shape[1] = 5;
        }
        if failure == 3 {
            backend.borrow_mut().fail = true;
        }
        assert!(emit(&mut observer, 5, 2, 4, 2, 0, false, &value, failure != 2).is_err());
        observer.finish_prefill_reductions(false);
        assert!(drain(&mut observer)
            .iter()
            .filter_map(|e| e.prefill_reductions.as_ref())
            .flat_map(|r| &r.records)
            .all(|r| r.status != SpeculativePrefillReductionStatus::Complete
                && r.record.payload.is_none()));
        assert_eq!(
            backend.borrow().transforms,
            before + usize::from(failure == 0 || failure == 3)
        );
    }
}

#[test]
fn preview_exact_and_one_short_limits_and_terminal_truncated_wire() {
    let (mut measured, _, _) = setup(5, true, 1, 2, 7, 0, None);
    let limits = measured.session().plan().plan().limits.clone();
    execute(&mut measured, 5, 2, true, 1, 2).unwrap();
    let exact = measured.session().cumulative_usage();
    measured.finish_prefill_reductions(true);
    let envelopes = drain(&mut measured);
    let report = envelopes
        .last()
        .unwrap()
        .prefill_reductions
        .as_ref()
        .unwrap();
    for encoded in [false, true] {
        let required = if encoded {
            exact.encoded_bytes
        } else {
            exact.host_bytes
        };
        for (limit, accepted) in [(required, true), (required - 1, false)] {
            let mut limits = limits.clone();
            if encoded {
                limits.cumulative.encoded_bytes = limit;
            } else {
                limits.cumulative.host_bytes = limit;
            }
            let (mut observer, _, _) = setup(5, true, 1, 2, 7, 0, Some(limits));
            assert_eq!(execute(&mut observer, 5, 2, true, 1, 2).is_ok(), accepted);
            let spent = observer.session().cumulative_usage();
            observer.finish_prefill_reductions(accepted);
            assert_eq!(observer.session().cumulative_usage(), spent);
            assert!(
                if encoded {
                    spent.encoded_bytes
                } else {
                    spent.host_bytes
                } <= limit
            );
            if accepted {
                assert_eq!(spent, exact);
            } else {
                assert!(drain(&mut observer)
                    .iter()
                    .filter_map(|e| e.prefill_reductions.as_ref())
                    .flat_map(|r| &r.records)
                    .all(|r| r.status != SpeculativePrefillReductionStatus::Complete));
            }
        }
    }
    // Pin the real terminal wire change, not a guessed number of enum bytes.
    let mut terminal = report.as_reductions().clone();
    for row in &mut terminal.records {
        assert!(matches!(
            row.record.outcome,
            CaptureOutcome::Truncated {
                emitted_elements: 7,
                ..
            }
        ));
        for _ in 0..4 {
            row.record.charged.encoded_bytes =
                serde_json::to_vec(&row.record).unwrap().len() as u64;
        }
    }
    for _ in 0..4 {
        terminal.charged.encoded_bytes = serde_json::to_vec(&terminal).unwrap().len() as u64;
    }
    let mut pending = terminal.clone();
    for row in &mut pending.records {
        row.status = SpeculativePrefillReductionStatus::Pending;
        row.record.outcome = CaptureOutcome::Missing;
    }
    let delta =
        serde_json::to_vec(&terminal).unwrap().len() - serde_json::to_vec(&pending).unwrap().len();
    assert!(delta > 2 * pending.records.len());
    assert!(crate::capture::encoded::prefill_reductions_fit_encoding(
        &pending
    ));
    pending.charged.encoded_bytes -= 1;
    assert!(!crate::capture::encoded::prefill_reductions_fit_encoding(
        &pending
    ));
    pending.charged.encoded_bytes += 1;
    pending.records[0].record.charged.encoded_bytes -= 1;
    assert!(!crate::capture::encoded::prefill_reductions_fit_encoding(
        &pending
    ));
}

#[test]
fn preview_initial_and_later_skip_or_fail_preserve_cumulative_work() {
    let (mut measured, backend, _) = setup(128, false, 0, 0, 0, 0, None);
    let mut initial_limits = measured.session().plan().plan().limits.clone();
    begin(&mut measured, 128, false);
    measured.set_prefill_span(Some(SpeculativePrefillSpan {
        prompt_tokens: 128,
        input_start: 0,
        input_end: 64,
        position: 0,
        hidden_start: 0,
        token_start: 0,
        sequence: 64,
        seed_start: 0,
    }));
    measured
        .begin_activation_invocation(SpeculativeActivationPhase::TargetPrefill, 64)
        .unwrap();
    initial_limits.per_step.host_bytes = measured.session().cumulative_usage().host_bytes;
    assert_eq!(backend.borrow().transforms, 0);
    measured.finish_activation_invocation(false);
    measured.finish_prefill_reductions(false);
    for policy in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let mut limits = initial_limits.clone();
        limits.on_limit = policy;
        let (mut observer, backend, _) = setup(128, false, 0, 0, 512, 0, Some(limits));
        let result = execute(&mut observer, 128, 64, false, 0, 0);
        assert_eq!(result.is_ok(), policy == CaptureLimitPolicy::Skip);
        let spent = observer.session().cumulative_usage();
        observer.finish_prefill_reductions(result.is_ok());
        assert_eq!(observer.session().cumulative_usage(), spent);
        assert_eq!(backend.borrow().transforms, 0);
        if result.is_ok() {
            let records = drain(&mut observer);
            for row in &records
                .last()
                .unwrap()
                .prefill_reductions
                .as_ref()
                .unwrap()
                .records
            {
                assert_eq!(row.status, SpeculativePrefillReductionStatus::Skipped);
                assert_eq!(row.covered_sequence, 128);
                assert!(row.record.payload.is_none());
                assert!(matches!(
                    row.record.outcome,
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Limit { .. }
                    }
                ));
            }
        }
    }
    for policy in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let (default, _, _) = setup(5, false, 0, 0, 1, 0, None);
        let mut limits = default.session().plan().plan().limits.clone();
        limits.on_limit = policy;
        limits.cumulative.captures = 2;
        let (mut observer, backend, _) = setup(5, false, 0, 0, 1, 0, Some(limits));
        let result = execute(&mut observer, 5, 2, false, 0, 0);
        assert_eq!(result.is_ok(), policy == CaptureLimitPolicy::Skip);
        observer.finish_prefill_reductions(result.is_ok());
        assert_eq!(backend.borrow().transforms, 2);
        assert_eq!(observer.session().cumulative_usage().captures, 2);
        for row in drain(&mut observer)
            .iter()
            .filter_map(|e| e.prefill_reductions.as_ref())
            .flat_map(|r| &r.records)
        {
            assert_ne!(row.status, SpeculativePrefillReductionStatus::Complete);
            assert!(row.record.payload.is_none());
        }
    }
}

#[test]
fn preview_checks_no_overlap_precision_coverage_and_generated_sources() {
    for failure in 0..5 {
        let (mut observer, backend, _) = setup(5, false, 0, 1, 0, 3, None);
        begin(&mut observer, 5, false);
        emit(
            &mut observer,
            5,
            0,
            2,
            0,
            0,
            false,
            &local(&whole(5, 0, 1), 0, 0, 2),
            true,
        )
        .unwrap();
        assert_eq!(backend.borrow().transforms, 0); // the selection lies in the last span
        let start = if failure == 1 {
            0
        } else if failure == 2 {
            3
        } else {
            2
        };
        if failure == 3 {
            let mut foreign = origin();
            foreign.prediction += 1;
            observer.set_activation_origin(Some(foreign));
        }
        let value = local(
            &whole(5, 0, if failure == 0 { 2 } else { 1 }),
            0,
            start,
            start + 2,
        );
        assert!(emit(
            &mut observer,
            5,
            start,
            start + 2,
            start,
            0,
            false,
            &value,
            failure != 4
        )
        .is_err());
        observer.finish_prefill_reductions(false);
        assert!(drain(&mut observer)
            .iter()
            .filter_map(|e| e.prefill_reductions.as_ref())
            .flat_map(|r| &r.records)
            .all(|r| r.status != SpeculativePrefillReductionStatus::Complete
                && r.record.payload.is_none()));
        assert_eq!(backend.borrow().transforms, 0);
    }
    for unwind in [false, true] {
        let (mut observer, backend, _) = setup(5, false, 0, 0, 1, 0, None);
        begin(&mut observer, 5, false);
        observer.set_prefill_span(Some(SpeculativePrefillSpan {
            prompt_tokens: 5,
            input_start: 0,
            input_end: 2,
            position: 0,
            hidden_start: 0,
            token_start: 0,
            sequence: 2,
            seed_start: 0,
        }));
        let value = local(&whole(5, 0, 0), 0, 0, 2);
        let mut factories = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_speculative_activation(
                Some(&mut observer),
                SpeculativeActivationPhase::TargetPrefill,
                2,
                |o| {
                    if unwind {
                        std::panic::panic_any(54321u32);
                    }
                    o.unwrap().observe_generated(
                        "block.output",
                        &value,
                        &GeneratedCaptureSource {
                            creation_bytes: 0,
                            source_dtype: Some(TensorDtype::F32),
                        },
                        &mut || {
                            factories += 1;
                            Ok(value.clone())
                        },
                    )
                },
            )
        }));
        if unwind {
            assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 54321);
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(factories, 0);
        assert_eq!(backend.borrow().transforms, 0);
        observer.set_prefill_span(None);
        observer.finish_prefill_reductions(false);
        assert!(drain(&mut observer)
            .iter()
            .filter_map(|e| e.prefill_reductions.as_ref())
            .flat_map(|r| &r.records)
            .all(|r| r.status != SpeculativePrefillReductionStatus::Complete
                && r.record.payload.is_none()));
    }
}

#[test]
fn unfinished_preview_retires_actual_source_before_host_preparation_custody() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Weak,
    };
    struct Retired {
        source: Weak<AdmittedCapturePlan>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Retired {
        fn drop(&mut self) {
            assert!(self.source.upgrade().is_none());
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    for kind in [0, 1] {
        let (mut observer, _, _) = setup(5, false, 1, kind, 7, 0, None);
        let source = Arc::downgrade(observer.session().plan.legacy_arc().unwrap());
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = eredu_core::HostPreparationAuthority::retain(Retired {
            source: source.clone(),
            drops: drops.clone(),
        });
        observer.session().retain_host_preparation(&owner).unwrap();
        drop(owner);
        begin(&mut observer, 5, false);
        emit(
            &mut observer,
            5,
            0,
            2,
            0,
            1,
            false,
            &local(&whole(5, 1, kind), 1, 0, 2),
            true,
        )
        .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(source.upgrade().is_some());
        drop(observer);
        assert!(source.upgrade().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
