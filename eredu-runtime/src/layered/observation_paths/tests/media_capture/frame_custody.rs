use super::*;
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn owned(binding: OrdinaryPrefillCapture) -> (CaptureSession, Arc<AtomicUsize>) {
    let retired = Arc::new(AtomicUsize::new(0));
    let host = HostPreparationAuthority::retain(Retired(retired.clone()));
    let run = CaptureSession::with_ordinary_prefill(binding, &host).unwrap();
    drop(host);
    (run, retired)
}
#[test]
fn ordinary_frame_retains_tensor_metadata_empty_skip_and_failed_diagnostics_after_session() {
    for case in 0..5 {
        let mut selected = binding(false, case == 1, limits());
        if case == 2 {
            // A legitimate schedule-only skip, preserving the installed media mode.
            let mut plan = selected.source().admission().plan().clone();
            plan.selections[0].schedule.first_prediction = 1;
            let d = discovery(&selected);
            let source = SharedCapturePlan::new(
                plan.admit_with_text_origin(
                    &d.catalog,
                    &d.support,
                    &d.support.capture,
                    selected.source().admission().request(),
                    CaptureTextOrigin {
                        cached_positions: 0,
                    },
                )
                .unwrap(),
            );
            selected = OrdinaryPrefillCapture::new(
                paths().prepare_media_capture_selection(&source).unwrap(),
                selected.geometry(),
            )
            .unwrap();
        }
        let (mut run, retired) = owned(selected);
        let mut backend = Backend {
            fail: if case == 3 { Some(2) } else { None },
            ..Default::default()
        };
        let success = advance(&mut run, &mut backend, None, case != 4);
        assert_eq!(success, case != 3);
        assert!(run.has_pending_step());
        assert!(run.has_pending_step());
        let frame = run.take_shared_step().unwrap();
        assert!(!run.has_pending_step());
        assert_eq!(
            frame.outcome(),
            if case >= 3 {
                CaptureStepOutcome::Aborted
            } else {
                CaptureStepOutcome::Committed
            }
        );
        if case == 1 {
            assert!(frame.records().is_empty())
        }
        if case == 2 {
            assert!(matches!(
                frame.records()[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ))
        }
        if case == 3 {
            assert!(
                matches!(&frame.records()[0].outcome,CaptureOutcome::Failed{message,..}if !message.is_empty())
            )
        }
        // Drop an independently escaped tensor first: bare metadata must still
        // retain the host through the frame, even after every session/source dies.
        let tensor = frame
            .records()
            .first()
            .and_then(|r| r.payload.as_ref())
            .and_then(|p| match p {
                CapturePayload::SharedTensor(t) => Some(t.clone()),
                _ => None,
            });
        drop(tensor);
        let alias = frame.clone();
        drop(frame);
        drop(run);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(!alias.records().first().is_some_and(|r| r.path.is_empty()));
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn ordinary_empty_frame_control_exact_short_is_required_even_under_skip() {
    let control =
        PreparedCapturedStep::retained_control_bytes::<HostPreparationAuthority>().unwrap();
    for skip in [false, true] {
        for short in [false, true] {
            let mut quota = limits();
            quota.per_step.host_bytes = control - u64::from(short);
            quota.cumulative.host_bytes = quota.per_step.host_bytes;
            if skip {
                quota.on_limit = CaptureLimitPolicy::Skip
            }
            let (mut run, retired) = owned(binding(false, true, quota));
            let mut backend = Backend::default();
            assert_eq!(advance(&mut run, &mut backend, None, true), !short);
            assert_eq!(backend.validates.get(), 0);
            assert_eq!(backend.copies.get(), 0);
            let frame = run.take_shared_step();
            assert_eq!(frame.is_some(), !short);
            assert_eq!(
                run.cumulative_usage().host_bytes,
                if short { 0 } else { control }
            );
            if let Some(frame) = &frame {
                assert!(frame.records().is_empty());
                assert_eq!(frame.step_usage().host_bytes, control)
            }
            drop(run);
            assert_eq!(retired.load(Ordering::SeqCst), usize::from(short));
            drop(frame);
            assert_eq!(retired.load(Ordering::SeqCst), 1);
        }
    }
    // Control succeeds but the first record's mandatory metadata cannot fit.
    // No partial record/value escapes and the consumed control is not refunded.
    let selected = binding(false, false, limits());
    let admission = selected.source().admission();
    let metadata = crate::capture::metadata_reservation(
        &admission.plan().selections[0],
        &admission.points()[0],
    )
    .unwrap();
    let mut quota = limits();
    quota.per_step.host_bytes = control + metadata.host_bytes - 1;
    quota.cumulative.host_bytes = quota.per_step.host_bytes;
    let (mut run, retired) = owned(binding(false, false, quota));
    let mut backend = Backend::default();
    assert!(!advance(&mut run, &mut backend, None, true));
    assert_eq!(backend.validates.get(), 0);
    assert_eq!(backend.copies.get(), 0);
    assert!(run.take_shared_step().is_none());
    assert_eq!(run.cumulative_usage().host_bytes, control);
    drop(run);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
#[test]
fn ordinary_installed_mode_decode_guard_and_restore_preserve_frame_controls() {
    let selected = binding(false, true, limits());
    let d = discovery(&selected);
    let (mut run, retired) = owned(selected);
    let saved = run.checkpoint(&d).unwrap();
    assert!(!run.has_pending_step());
    assert!(advance(&mut run, &mut Backend::default(), None, true));
    let prefill = run.take_shared_step().unwrap();
    let first = run.cumulative_usage();
    assert!(run.ordinary_prefill_capture().is_none());
    let epoch = DistributedCommitEpoch::FIRST
        .next()
        .unwrap()
        .next()
        .unwrap()
        .next()
        .unwrap();
    run.prepare_step_transaction(epoch, crate::ExpertPass::Decode, 1)
        .unwrap();
    assert!(run.has_pending_step());
    assert!(run.take_shared_step().is_none());
    assert!(run
        .prepare_step_transaction(epoch.next().unwrap(), crate::ExpertPass::Decode, 2)
        .is_err());
    run.complete_transaction(epoch).unwrap();
    assert!(run.take_shared_step().is_none());
    run.finish_transaction(epoch, true);
    let decoded = run.take_shared_step().unwrap();
    assert_eq!(decoded.phase(), CapturePhase::Decode);
    assert_eq!(decoded.prediction_index(), 1);
    assert_eq!(decoded.step_usage(), prefill.step_usage());
    assert_eq!(decoded.cumulative_usage().host_bytes, first.host_bytes * 2);
    run.restore(&saved).unwrap();
    assert_eq!(run.cumulative_usage(), decoded.cumulative_usage());
    drop((run, saved, prefill));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(decoded);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
#[test]
fn ordinary_frame_concurrent_final_aliases_keep_actual_host_until_last() {
    let (mut run, retired) = owned(binding(false, true, limits()));
    assert!(advance(&mut run, &mut Backend::default(), None, true));
    let frame = run.take_shared_step().unwrap();
    drop(run);
    let gate = Arc::new(std::sync::Barrier::new(5));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let frame = frame.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                gate.wait();
                assert!(frame.records().is_empty());
                drop(frame)
            })
        })
        .collect();
    drop(frame);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    gate.wait();
    for thread in threads {
        thread.join().unwrap()
    }
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_pre_frame_error_and_poisoned_checkpoint_outlive_actual_source() {
    for poison in [false, true] {
        let selected = binding(false, true, limits());
        let discovery = discovery(&selected);
        let (mut run, retired) = owned(selected);
        let saved = run.checkpoint(&discovery).unwrap();
        let error = if poison {
            run.poison_ordinary_host_for_test();
            match run.checkpoint(&discovery) {
                Err(error) => error,
                Ok(_) => panic!("poison rejects"),
            }
        } else {
            run.begin_ordinary_prefill(&PrefillChunk {
                input: 1..3,
                position: 1,
                output: OutputDemand::StateOnly,
            })
            .unwrap_err()
        };
        assert!(matches!(error.cause(), CaptureError::Invalid(_)));
        assert!(!run.has_pending_step());
        let alias = error.clone();
        drop((run, saved, error));
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(!alias.to_string().is_empty());
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn ordinary_refused_frame_control_returns_retained_typed_limit_without_refund_or_frame() {
    let control =
        PreparedCapturedStep::retained_control_bytes::<HostPreparationAuthority>().unwrap();
    for policy in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let mut quota = limits();
        quota.on_limit = policy;
        quota.per_step.host_bytes = control - 1;
        quota.cumulative.host_bytes = control - 1;
        let (mut run, retired) = owned(binding(false, true, quota));
        let error = run.begin_step(CapturePhase::Prefill, 0).unwrap_err();
        assert!(matches!(
            error.cause(),
            CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: false
            }
        ));
        assert!(run.take_shared_step().is_none());
        assert_eq!(run.cumulative_usage().host_bytes, 0);
        drop(run);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn ordinary_after_frame_error_and_frame_aliases_have_independent_final_custody() {
    for keep_error in [false, true] {
        let (mut run, retired) = owned(binding(false, true, limits()));
        assert!(advance(&mut run, &mut Backend::default(), None, true));
        let frame = run.take_shared_step().unwrap();
        assert!(run.ordinary_prefill_capture().is_none());
        let error = run.begin_step(CapturePhase::Decode, u64::MAX).unwrap_err();
        drop(run);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        if keep_error {
            drop(frame);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            let alias = error.clone();
            drop(error);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(alias);
        } else {
            drop(error);
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            drop(frame);
        }
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
