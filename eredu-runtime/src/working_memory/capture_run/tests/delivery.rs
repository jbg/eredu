use super::*;

fn populated(frame: &mut ScheduledCaptureStep<'_>, index: usize) {
    let mut tensor = frame.take_tensor(index).unwrap().prepare().unwrap();
    let values = [1.25, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    while tensor.initialized_count() < tensor.len() {
        tensor
            .push_f32(values[tensor.initialized_count() % values.len()])
            .unwrap();
    }
    let tensor = tensor.finish().unwrap();
    frame
        .record_tensor(tensor, TensorDtype::Bf16, CaptureUsage::default())
        .unwrap();
}
fn required(frame: &ScheduledCaptureStep<'_>) -> CaptureUsage {
    frame
        .records()
        .iter()
        .try_fold(CaptureUsage::default(), |n, r| n.checked_add(r.charged))
        .unwrap()
}
#[test]
fn exact_precommit_hold_and_one_short_keep_original_account_and_all_payload_pointers() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (short_r, short_run) = fresh(&pool, h - 1);
    let before = ledger(&pool);
    assert!(matches!(
        short_run.prepare_capture_run(&short_r, plan(&source)),
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(ledger(&pool), before);
    drop(short_r);
    drop(short_run);
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    populated(&mut frame, 0);
    let records = frame.records().as_ptr();
    let shape = frame.records()[0].source_shape.as_ref().unwrap().as_ptr();
    let payload = frame.records()[0]
        .payload
        .as_ref()
        .unwrap()
        .as_tensor()
        .unwrap()
        .data();
    let TensorObservationData::F32(values) = payload else {
        panic!("f32")
    };
    let data = values.as_ptr();
    let spent = required(&frame);
    let before = ledger(&pool);
    let delivery = frame.prepare_delivery(spent, spent, 0.125).unwrap();
    assert_eq!(ledger(&pool), before);
    assert_eq!(bank.spent_steps(), 1); // the row/source borrow has ended
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    let published = delivery.finish(CaptureStepOutcome::Committed);
    assert_eq!(ledger(&pool), before);
    assert_eq!(published.records().as_ptr(), records);
    assert_eq!(
        published.records()[0]
            .source_shape
            .as_ref()
            .unwrap()
            .as_ptr(),
        shape
    );
    let TensorObservationData::F32(values) = published.records()[0]
        .payload
        .as_ref()
        .unwrap()
        .as_tensor()
        .unwrap()
        .data()
    else {
        panic!("f32")
    };
    assert_eq!(values.as_ptr(), data);
    assert_eq!(values[1].to_bits(), (-0.0f32).to_bits());
    assert!(values[4].is_nan());
    let alias = published.clone();
    drop(published);
    drop(bank);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn invalid_usage_and_timing_return_same_builder_without_refunding_claim() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    populated(&mut frame, 0);
    let records = frame.records().as_ptr();
    let spent = required(&frame);
    let before = ledger(&pool);
    let error = frame
        .prepare_delivery(CaptureUsage::default(), spent, 0.)
        .unwrap_err();
    assert!(matches!(error.error(), CaptureStepError::InvalidCompletion));
    let mut frame = error.into_builder();
    assert_eq!(frame.records().as_ptr(), records);
    assert!(frame.take_tensor(0).is_err());
    let error = frame.prepare_delivery(spent, spent, f64::NAN).unwrap_err();
    let frame = error.into_builder();
    assert_eq!(frame.records().as_ptr(), records);
    assert_eq!(ledger(&pool), before);
    let delivery = frame.prepare_delivery(spent, spent, 0.).unwrap();
    let published = delivery.finish(CaptureStepOutcome::Aborted);
    assert_eq!(published.step_usage(), spent);
    drop(published);
    drop(bank);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn completed_delivery_survives_parent_close_or_quarantine_without_certifying_native_work() {
    for quarantine in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let spent = required(&frame);
        let delivery = frame.prepare_delivery(spent, spent, 0.).unwrap();
        let native = quarantine.then(|| run.scope().unwrap());
        drop(native); // deliberately unresolved native work, if present
        drop(bank);
        drop(r);
        drop(run);
        let before = ledger(&pool);
        let published = delivery.finish(CaptureStepOutcome::Aborted);
        assert_eq!(ledger(&pool), before); // no validation/refund/certification
        assert_eq!(published.outcome(), CaptureStepOutcome::Aborted);
        drop(published);
        assert_eq!(pool.used_bytes().unwrap(), if quarantine { h } else { 0 });
    }
}
#[test]
fn parent_close_before_sealing_rejects_and_keeps_same_partial_owner() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    populated(&mut frame, 0);
    let records = frame.records().as_ptr();
    let spent = required(&frame);
    drop(run);
    let error = frame.prepare_delivery(spent, spent, 0.).unwrap_err();
    assert!(matches!(
        error.error(),
        CaptureStepError::Memory(WorkingMemoryError::ExecutionFenced)
    ));
    let frame = error.into_builder();
    assert_eq!(frame.records().as_ptr(), records);
    assert!(frame.records()[0].payload.is_some());
    drop(frame);
    drop(bank);
    drop(r);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn encoded_counter_matches_nonfinite_wire_at_exact_limit_and_one_short() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    populated(&mut frame, 0);
    // Test-owned reference copy, never a construction path or inherited grant.
    let mut record = frame.records()[0].clone();
    for _ in 0..8 {
        let encoded = serde_json::to_vec(&record).unwrap().len() as u64;
        if record.charged.encoded_bytes == encoded {
            break;
        }
        record.charged.encoded_bytes = encoded;
    }
    assert_eq!(
        record.charged.encoded_bytes,
        serde_json::to_vec(&record).unwrap().len() as u64
    );
    assert!(crate::capture::record_fits_encoding(&record));
    record.charged.encoded_bytes -= 1;
    assert!(!crate::capture::record_fits_encoding(&record));
    record.charged.encoded_bytes = u64::MAX;
    assert!(crate::capture::record_fits_encoding(&record));
    drop(record);
    drop(frame);
    drop(bank);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn encoded_failure_retains_metadata_charges_and_escaped_tensor_then_seals_aborted() {
    let mut raw = raw();
    raw.selections.truncate(1);
    let mut point = point();
    point.axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(512);
    let source = admit(raw, point, 4, false);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let receipt = complete(frame.take_tensor(0).unwrap().prepare().unwrap(), 123.456);
    let alias = receipt.observation().clone();
    frame
        .record_tensor(receipt, TensorDtype::F16, CaptureUsage::default())
        .unwrap();
    let source_shape = frame.records()[0].source_shape.as_ref().unwrap().as_ptr();
    let selected_shape = frame.records()[0].selected_shape.as_ref().unwrap().as_ptr();
    let charged = frame.records()[0].charged;
    let before = ledger(&pool);
    let error = frame.prepare_delivery(charged, charged, 0.).unwrap_err();
    assert!(matches!(
        error.error(),
        CaptureStepError::EncodedSize { index: 0 }
    ));
    let mut frame = error.into_builder();
    let record = &frame.records()[0];
    assert!(matches!(
        record.outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Invalid,
            ..
        }
    ));
    assert!(record.payload.is_none());
    assert_eq!(record.charged, charged);
    assert_eq!(record.source_shape.as_ref().unwrap().as_ptr(), source_shape);
    assert_eq!(
        record.selected_shape.as_ref().unwrap().as_ptr(),
        selected_shape
    );
    assert_eq!(record.source_dtype, Some(TensorDtype::F16));
    assert!(frame.validate_record_encoding(0).is_err());
    assert!(frame.take_tensor(0).is_err());
    assert_eq!(ledger(&pool), before);
    let published = frame
        .prepare_delivery(charged, charged, 0.)
        .unwrap()
        .finish(CaptureStepOutcome::Aborted);
    assert_eq!(published.step_usage(), charged);
    drop(published);
    drop(bank);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), h);
    assert_eq!(alias.shape(), &[5, 512]);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn unwinding_with_ready_delivery_retires_payload_and_does_not_refund_frame_claim() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let frame = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let spent = required(&frame);
        let _delivery = frame.prepare_delivery(spent, spent, 0.).unwrap();
        panic!("after completed frame before outcome");
    }))
    .is_err());
    assert_eq!(bank.spent_steps(), 1);
    assert!(bank.begin_step(CapturePhase::Prefill, 0).is_err());
    drop(bank);
    drop(r);
    drop(run);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
