use super::*;

fn charged(step: &ScheduledCaptureStep<'_>) -> CaptureUsage {
    step.records()
        .iter()
        .fold(CaptureUsage::default(), |sum, record| {
            sum.checked_add(record.charged).unwrap()
        })
}
fn fill(step: &mut ScheduledCaptureStep<'_>, encoded_bytes: u64) -> usize {
    let mut tensor = step.take_tensor(0).unwrap().prepare().unwrap();
    while tensor.initialized_count() < tensor.len() {
        tensor
            .push_f32(tensor.initialized_count() as f32 * 0.25 - 3.5)
            .unwrap();
    }
    let tensor = tensor.finish().unwrap();
    let TensorObservationData::F32(values) = tensor.observation().data() else {
        unreachable!()
    };
    let pointer = values.as_ptr() as usize;
    step.record_tensor(
        tensor,
        checkpoint::TensorDtype::F32,
        CaptureUsage {
            encoded_bytes,
            ..Default::default()
        },
    )
    .unwrap();
    pointer
}

#[test]
fn detached_aborted_payload_is_owned_and_retains_custody_until_retirement() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    fill(&mut step, 20_000);
    let usage = charged(&step);
    let pending = step.into_aborted_pending(usage, usage, 1.25);
    // No reference to the source or mutable claim row remains in this owner.
    drop((bank, source));
    let erased: Box<dyn Send + Sync> = Box::new(pending);
    drop(erased);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop((r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn recovered_aborted_delivery_preserves_nonzero_payload_identity_and_schedule_records() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let data = fill(&mut step, 20_000);
    let records = step.records().as_ptr();
    let id = step.records()[0].selection_id.as_ptr();
    let usage = charged(&step);
    let pending = step.into_aborted_pending(usage, usage, 1.25);
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    drop((bank, source));
    let delivered = pending.finish().unwrap();
    let alias = delivered.clone();
    let frame = delivered.as_ref();
    assert_eq!(frame.outcome, CaptureStepOutcome::Aborted);
    assert_eq!(frame.records.as_ptr(), records);
    assert_eq!(frame.records[0].selection_id.as_ptr(), id);
    let Some(CapturePayload::SharedTensor(tensor)) = &frame.records[0].payload else {
        panic!("shared payload")
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        unreachable!()
    };
    assert_eq!(values.as_ptr() as usize, data);
    assert_eq!(values[0], -3.5);
    assert_eq!(values[1], -3.25);
    assert!(matches!(
        frame.records[1].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Schedule
        }
    ));
    assert!(matches!(frame.records[2].outcome, CaptureOutcome::Missing));
    assert_eq!(frame.step_usage, usage);
    assert_eq!(frame.cumulative_usage, usage);
    assert_eq!(frame.capture_seconds, 1.25);
    drop((delivered, r, run));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn detach_never_validates_a_closed_or_quarantined_parent_and_error_keeps_custody() {
    for quarantine in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let usage = charged(&step);
        if quarantine {
            drop(run.scope().unwrap());
        } else {
            drop(run);
        }
        let before = TRANSFER_ALLOCATIONS.get();
        let pending = step.into_aborted_pending(usage, usage, 0.0);
        assert_eq!(TRANSFER_ALLOCATIONS.get(), before);
        let error = pending.finish().unwrap_err();
        assert!(matches!(
            error.error(),
            CaptureStepError::Memory(WorkingMemoryError::ExecutionFenced)
        ));
        let error = error.into_pending().finish().unwrap_err();
        drop((bank, r, source));
        let erased: Box<dyn std::error::Error + Send + Sync> = Box::new(error);
        assert_eq!(pool.used_bytes().unwrap(), h);
        drop(erased);
        assert_eq!(pool.used_bytes().unwrap(), if quarantine { h } else { 0 });
    }
}

#[test]
fn invalid_completion_metadata_stays_rejected_without_losing_payload_or_refunding_claim() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let _ = fill(&mut step, 20_000);
    let usage = charged(&step);
    let pending = step.into_aborted_pending(CaptureUsage::default(), usage, 0.0);
    let error = pending.finish().unwrap_err();
    assert!(matches!(error.error(), CaptureStepError::InvalidCompletion));
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    let error = error.into_pending().finish().unwrap_err();
    assert!(matches!(error.error(), CaptureStepError::InvalidCompletion));
    drop((bank, r, run, source));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn encoded_failure_can_only_recover_same_aborted_frame_with_original_usage() {
    let mut declaration = point();
    declaration.axes = Some(vec![TensorAxis {
        name: "width".into(),
        dimension: SymbolicDimension::Known(1024),
    }]);
    let source = admit(raw(), declaration, 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    fill(&mut step, 0);
    let records = step.records().as_ptr();
    let usage = charged(&step);
    let pending = step.into_aborted_pending(usage, usage, 0.5);
    let error = pending.finish().unwrap_err();
    assert!(matches!(
        error.error(),
        CaptureStepError::EncodedSize { index: 0 }
    ));
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    let delivered = error.into_pending().finish().unwrap();
    assert_eq!(delivered.outcome(), CaptureStepOutcome::Aborted);
    assert_eq!(delivered.records().as_ptr(), records);
    assert!(matches!(
        delivered.records()[0].outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Invalid,
            ..
        }
    ));
    assert!(delivered.records()[0].payload.is_none());
    assert_eq!(delivered.step_usage(), usage);
    assert_eq!(delivered.cumulative_usage(), usage);
    drop((bank, r, run, source));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(delivered);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn observer_unwind_can_detach_frame_in_drop_and_preserves_spent_row_and_nonzero_values() {
    struct AbortOnDrop<'a, 'b> {
        frame: Option<ScheduledCaptureStep<'a>>,
        output: &'b mut Option<PendingCaptureDelivery>,
        usage: CaptureUsage,
    }
    impl Drop for AbortOnDrop<'_, '_> {
        fn drop(&mut self) {
            if let Some(frame) = self.frame.take() {
                *self.output = Some(frame.into_aborted_pending(self.usage, self.usage, 0.0));
            }
        }
    }
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let pointer = fill(&mut step, 20_000);
    let usage = charged(&step);
    let mut pending = None;
    let before = (
        ledger(&pool),
        CLAIM_ALLOCATIONS.get(),
        TRANSFER_ALLOCATIONS.get(),
    );
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _guard = AbortOnDrop {
            frame: Some(step),
            output: &mut pending,
            usage,
        };
        panic!("observer unwind after successful capture");
    }))
    .is_err());
    assert_eq!(
        (
            ledger(&pool),
            CLAIM_ALLOCATIONS.get(),
            TRANSFER_ALLOCATIONS.get()
        ),
        before
    );
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    let frame = pending.take().unwrap().finish().unwrap();
    assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
    let Some(CapturePayload::SharedTensor(tensor)) = &frame.records()[0].payload else {
        panic!("shared tensor")
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        unreachable!()
    };
    assert_eq!(values.as_ptr() as usize, pointer);
    drop((bank, r, run, source));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(frame);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn one_byte_short_rejects_entire_abort_owner_program_before_claim_or_frame_allocation() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h - 1, 0).unwrap();
    let (r, run) = fresh(&pool, h - 1);
    let before = (
        ledger(&pool),
        CLAIM_ALLOCATIONS.get(),
        TRANSFER_ALLOCATIONS.get(),
    );
    assert!(run.prepare_capture_run(&r, plan(&source)).is_err());
    assert_eq!(
        (
            ledger(&pool),
            CLAIM_ALLOCATIONS.get(),
            TRANSFER_ALLOCATIONS.get()
        ),
        before
    );
    drop((r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
