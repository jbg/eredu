use super::super::super::bounded_capture;
use super::*;
use eredu_core::{BackendProvider as _, TensorObservationData};

fn find_source<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(found) = error.downcast_ref::<T>() {
            return Some(found);
        }
        error = error.source()?;
    }
}

fn submit(
    session: &mut MlxModelSession,
    backend: &MlxBackend<'_>,
    decode: bool,
    token: u32,
    observer: &mut impl RuntimeActivationObserver<MlxTensor, Error>,
) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
    if decode {
        session.submit_decode_with_observer(
            backend,
            || Ok(Array::from_slice(&[token], &[1, 1])),
            observer,
        )
    } else {
        session.submit_prefill_with_observer(
            backend,
            MlxBackend::prepare_text_prompt(backend, vec![1, 2])?,
            observer,
        )
    }
}

fn settle(session: &MlxModelSession) {
    super::super::super::recovery::wait_for_retirement(|| {
        session.ensure_no_submission_in_flight().is_ok()
    });
}

fn verify_capture_failures(device: safemlx::DeviceType) {
    let context = crate::backend::ExecutionContext::new(safemlx::Device::new(device, 0));
    let stream = context.stream();
    let weights =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxBackend::new(stream, weights.stream());
    let root = crate::composition::mlx::replicated_text::tests::tiny_artifact("llama", true);
    let host = eredu_runtime::LayerwiseLoadOptions::new(
        eredu_core::residency::OffloadConfig::new(Some(u64::MAX), Some(u64::MAX), 3).unwrap(),
    );
    let disk = eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 30, 2 << 30, 3, 2).unwrap();
    for residency in [
        eredu_runtime::WeightResidency::fully_resident(),
        eredu_runtime::WeightResidency::layerwise_host(host),
        eredu_runtime::WeightResidency::dense_disk_stream(disk),
    ] {
        for decode in [false, true] {
            let mut reference = None;
            for fail in [false, true] {
                let options = crate::MlxLoadRequest::from_normalized(
                    eredu_runtime::NormalizedLoadRequest::default()
                        .with_weight_residency(residency),
                );
                let prepared = eredu_core::load_model(&backend, root.path(), options).unwrap();
                let mut session = backend.create_session(prepared).unwrap();
                if decode {
                    submit(
                        &mut session,
                        &backend,
                        false,
                        0,
                        &mut eredu_runtime::NoopObserver,
                    )
                    .unwrap()
                    .wait()
                    .unwrap();
                }
                let discovery = session
                    .capture_discovery
                    .as_ref()
                    .unwrap()
                    .capture()
                    .unwrap();
                let point = discovery
                    .catalog
                    .points
                    .iter()
                    .find(|point| point.path.ends_with(".feed_forward.units"))
                    .unwrap();
                let budget = CaptureUsage {
                    captures: 100,
                    retained_bytes: 10_000_000,
                    host_bytes: 10_000_000,
                    encoded_bytes: 10_000_000,
                };
                let plan = CapturePlan {
                    schema_version: 1,
                    selections: vec![CaptureSelection {
                        id: "ffn".into(),
                        path: point.path.clone(),
                        schedule: CaptureSchedule::default(),
                        slices: vec![],
                        transform: CaptureTransform::FullTensor,
                    }],
                    limits: CaptureLimits {
                        per_step: budget,
                        cumulative: budget,
                        on_limit: CaptureLimitPolicy::Fail,
                    },
                }
                .admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    CaptureRequestShape {
                        batch: 1,
                        prompt_tokens: 2,
                        max_predictions: 3,
                    },
                )
                .unwrap();
                let mut capture =
                    CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
                let checkpoint = capture.checkpoint(&discovery).unwrap();
                let prediction = u64::from(decode);
                if fail {
                    let before = session.payload.model.erased().state_snapshot();
                    // A cold collector rejection crosses the same native/neutral adapters
                    // before source work and does not establish native settlement itself.
                    let cold = match submit(
                        &mut session,
                        &backend,
                        decode,
                        3,
                        &mut RejectedCapture(
                            CaptureError::Unsupported("cold collector fixture".into()),
                            None,
                        ),
                    ) {
                        Err(error) => error,
                        Ok(_) => panic!("cold rejection entered forward"),
                    };
                    assert!(cold.model_state_preserved());
                    let cold = eredu_core::BackendFailure::from_error(cold);
                    assert!(matches!(find_source::<CaptureError>(&cold),
                        Some(CaptureError::Unsupported(message)) if message == "cold collector fixture"));
                    assert_eq!(session.payload.model.erased().state_snapshot(), before);
                    assert_eq!(capture.cumulative_usage(), CaptureUsage::default());
                    settle(&session);

                    // Deliver an actual MLX exception after the selected native source
                    // has been evaluated. The injected delivery is deterministic; the
                    // ordinary collector, reservation, rollback and completion are real.
                    let exception = Array::from_slice(&[1.0f32], &[1])
                        .reshape(&[2], stream)
                        .unwrap_err();
                    let expected = (exception.what().to_owned(), exception.location());
                    let fault = bounded_capture::fail_next_transform(exception);
                    let failed = match with_observer(
                        &mut session,
                        &mut capture,
                        stream,
                        None,
                        prediction,
                        |session, observer| {
                            submit(
                                session,
                                &backend,
                                decode,
                                3,
                                &mut eredu_runtime::BorrowedActivationObserver(observer),
                            )
                        },
                    ) {
                        Err(error) => error,
                        Ok(_) => panic!("injected capture failure succeeded"),
                    };
                    drop(fault);
                    assert!(failed.model_state_preserved(), "{failed}");
                    let public = eredu_core::BackendFailure::from_error(failed);
                    let original =
                        find_source::<Exception>(&public).expect("original native cause");
                    assert_eq!(original.what(), expected.0);
                    assert_eq!(original.location(), expected.1);
                    let failed = capture
                        .take_shared_step()
                        .map(|frame| frame.as_step().clone())
                        .unwrap();
                    assert!(matches!(
                        failed.records[0].outcome,
                        CaptureOutcome::Failed {
                            reason: CaptureFailureReason::Native,
                            ..
                        }
                    ));
                    assert!(failed.records[0].payload.is_none());
                    assert!(failed.step_usage.retained_bytes > 0);
                    assert!(capture.checkpoint(&discovery).is_err());
                    settle(&session);
                    assert_eq!(session.payload.model.erased().state_snapshot(), before);
                    capture.restore(&checkpoint).unwrap();
                    assert_eq!(capture.cumulative_usage(), failed.cumulative_usage);
                }
                let charged_before = capture.cumulative_usage();
                let output = with_observer(
                    &mut session,
                    &mut capture,
                    stream,
                    None,
                    prediction,
                    |session, observer| {
                        submit(
                            session,
                            &backend,
                            decode,
                            3,
                            &mut eredu_runtime::BorrowedActivationObserver(observer),
                        )
                    },
                )
                .unwrap()
                .wait()
                .unwrap();
                let logits = output.evaluated().unwrap().as_slice::<f32>().to_vec();
                let captured = capture
                    .take_shared_step()
                    .map(|frame| frame.as_step().clone())
                    .unwrap();
                let CapturePayload::Tensor(units) = captured.records[0].payload.as_ref().unwrap()
                else {
                    panic!("component tensor");
                };
                let TensorObservationData::F32(values) = units.data() else {
                    panic!("component precision");
                };
                assert!(values.iter().any(|value| value.abs() > 1e-6));
                assert!(captured.cumulative_usage.retained_bytes > charged_before.retained_bytes);
                let next = submit(
                    &mut session,
                    &backend,
                    true,
                    if decode { 4 } else { 3 },
                    &mut eredu_runtime::NoopObserver,
                )
                .unwrap()
                .wait()
                .unwrap();
                let next = next.evaluated().unwrap().as_slice::<f32>().to_vec();
                let result = (logits, next, units.clone());
                if let Some(reference) = &reference {
                    assert_eq!(&result, reference);
                } else {
                    reference = Some(result);
                }
            }
        }
    }
}

#[test]
fn native_capture_failure_causes_and_retry_cpu() {
    verify_capture_failures(safemlx::DeviceType::Cpu);
}

#[test]
#[ignore = "requires Metal device"]
#[cfg(feature = "metal")]
fn native_capture_failure_causes_and_retry_metal() {
    assert!(safemlx::metal::is_available().unwrap());
    verify_capture_failures(safemlx::DeviceType::Gpu);
}

#[test]
fn native_duplicate_observation_retains_portable_error() {
    let stream = crate::test_stream();
    let value = MlxTensor::from_array(Array::from_slice(&[1.25f32, -2.5], &[1, 2]));
    let request = eredu_core::ObservationRequest::all();
    let mut collector = super::super::super::observation::InspectionCollector::new(&request);
    collector.capture("duplicate", &value);
    collector.capture("duplicate", &value);
    let failure =
        eredu_core::BackendFailure::from_error(collector.materialize(stream).unwrap_err());
    assert!(
        matches!(find_source::<eredu_core::ObservationError>(&failure),
        Some(eredu_core::ObservationError::DuplicatePath(path)) if path == "duplicate")
    );
}
