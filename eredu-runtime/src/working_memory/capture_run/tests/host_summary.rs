use super::*;

#[test]
fn original_host_summary_uses_exact_source_spent_claim_and_retained_account() {
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.selections[0].transform = CaptureTransform::Summary;
    let source = admit(raw, point(), 1, false);
    let bytes = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(bytes * 2, 0).unwrap();
    let (short, short_run) = fresh(&pool, bytes - 1);
    assert!(short_run.prepare_capture_run(&short, plan(&source)).is_err());
    drop((short_run, short));
    let (reservation, run) = fresh(&pool, bytes);
    let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
    let mut frame = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
    let values = [1.0, -2.0, 3.0, 0.0, f32::NAN, f32::INFINITY,
        f32::NEG_INFINITY, 4.0, 5.0, 6.0];
    let claim = frame.take_summary(0).unwrap();
    assert_eq!(claim.geometry().source_shape(), [5, 2]);
    let receipt = claim.summarize_host_f32(&[5, 2], CaptureHostF32::Dense(&values)).unwrap();
    let value = receipt.observation();
    assert_eq!((value.elements, value.finite, value.non_finite), (10, 7, 3));
    assert_eq!((value.nan, value.positive_infinity, value.negative_infinity), (1, 1, 1));
    assert_eq!((value.min, value.max, value.mean), (Some(-2.0), Some(6.0), Some(17.0 / 7.0)));
    assert!(frame.take_summary(0).is_err());
    drop(frame);
    drop((bank, run, reservation));
    assert!(pool.used_bytes().unwrap() >= bytes);
    drop(receipt);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_host_summary_refuses_wrong_backing_and_uniform_uses_same_reduction() {
    for wrong in [true, false] {
        let mut raw = raw(); raw.selections.truncate(1);
        raw.selections[0].transform = CaptureTransform::Summary;
        let source = admit(raw, point(), 1, false);
        let bytes = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let (reservation, run) = fresh(&pool, bytes);
        let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
        let mut frame = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
        let claim = frame.take_summary(0).unwrap();
        let result = claim.summarize_host_f32(&[5, 2], if wrong {
            CaptureHostF32::Dense(&[1.0])
        } else { CaptureHostF32::Uniform(-3.0) });
        assert_eq!(result.is_err(), wrong);
        if let Ok(value) = &result {
            assert_eq!((value.observation().mean, value.observation().rms), (Some(-3.0), Some(3.0)));
        }
        assert!(frame.take_summary(0).is_err());
        drop(frame); drop((bank, run, reservation));
        assert!(pool.used_bytes().unwrap() >= bytes);
        drop(result);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_host_summary_selects_declared_strides_before_existing_reduction() {
    let mut raw = raw(); raw.selections.truncate(1);
    raw.selections[0].transform = CaptureTransform::Summary;
    raw.selections[0].slices = vec![CaptureSlice {
        axis: "context".into(), start: 1, end: 5, stride: 2,
    }, CaptureSlice { axis: "width".into(), start: 1, end: 2, stride: 1 }];
    let source = admit(raw, point(), 1, false);
    let bytes = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (reservation, run) = fresh(&pool, bytes);
    let mut bank = run.prepare_capture_run(&reservation, plan(&source)).unwrap();
    let mut frame = bank.begin_step(CapturePhase::Prefill, 0).unwrap().prepare().unwrap();
    let claim = frame.take_summary(0).unwrap();
    let values = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let receipt = claim.summarize_host_f32(&[5, 2], CaptureHostF32::Dense(&values)).unwrap();
    assert_eq!((receipt.observation().elements, receipt.observation().mean,
        receipt.observation().rms), (2, Some(5.0), Some(29f64.sqrt())));
}
