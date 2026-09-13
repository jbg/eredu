fn check_speculative_control_delivery<'world>(
    runtime: &mut ModelRuntime<MlxBackend<'world>>,
    input: MlxModelInput,
    rank: usize,
    case: &str,
) {
    use eredu_core::{
        generation::SpeculativeRequestStatus as Status,
        speculative::SpeculativeControlError as Error,
    };
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let before = runtime.text_preparation_usage().unwrap();
    let mut options = ControlledSpeculativeOptions::default();
    if rank == 0 && case == "record" {
        options.trace_limits.per_record_bytes = 1;
        options.trace_limits.total_bytes = 1;
    }
    if case == "sampling" {
        use eredu_core::capture::*;
        let discovery =
            <MlxBackend<'_> as eredu_core::TextGenerationBackend>::capture_discovery(runtime)
                .unwrap();
        let usage = CaptureUsage {
            captures: 1,
            retained_bytes: 1 << 20,
            host_bytes: 1 << 20,
            encoded_bytes: 1 << 20,
        };
        options.capture = Some(
            CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: vec![CaptureSelection {
                    id: "sampling-budget".into(),
                    path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::Preview { max_elements: 4 },
                }],
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage.checked_mul(if rank == 0 { 1 } else { 16 }).unwrap(),
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            }
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                CaptureRequestShape {
                    batch: 1,
                    prompt_tokens: 1,
                    max_predictions: 3,
                },
            )
            .unwrap(),
        );
    }
    let mut failure = None;
    let mut steps = 0;
    let (result, publications) = execute_neutral_embedded_mtp_with(
        runtime,
        input,
        SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: 1,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        },
        DriveControlledSpeculation::new(
            Default::default(),
            options,
            |session: &mut dyn ControlledSpeculativeSession| -> Result<(), Error> {
                while let Some(step) = session.step()? {
                    steps += 1;
                    if rank == 0 {
                        match case {
                            "cancel_prefill" if steps == 1 => {
                                session.cancel()?;
                                break;
                            }
                            "cancel_pending"
                                if step.status == Status::TargetVerificationInFlight =>
                            {
                                session.cancel()?;
                                break;
                            }
                            "caller" => {
                                return Err(Error::Invalid("injected caller delivery failure"));
                            }
                            _ => {}
                        }
                    }
                }
                if rank == 0 && case == "terminal" {
                    return Err(Error::Invalid("injected terminal delivery failure"));
                }
                Ok(())
            },
            &mut failure,
        ),
    );
    if case.starts_with("cancel_") {
        let output = result.unwrap();
        assert!(failure.is_none());
        assert_eq!(output.token_ids().len(), 1, "rank={rank} case={case}");
        assert!(runtime.synchronize().is_ok());
    } else {
        let error = result
            .err()
            .expect("one failing controller must stop all peers");
        let original: &(dyn std::error::Error + 'static) = match &failure {
            Some(error) => error,
            None => &error,
        };
        if rank == 0 && case == "sampling" {
            let mut source = Some(original);
            let mut found = false;
            while let Some(error) = source {
                found |= matches!(
                    error.downcast_ref::<eredu_core::capture::CaptureError>(),
                    Some(eredu_core::capture::CaptureError::Limit {
                        budget: eredu_core::capture::CaptureBudget::Captures,
                        cumulative: true
                    })
                );
                source = error.source();
            }
            assert!(found, "rank={rank} case={case} original={original}");
        } else if rank == 0 {
            assert!(
                matches!(
                    (&failure, case),
                    (
                        Some(Error::Capture(
                            eredu_core::capture::CaptureError::Limit { .. }
                        )),
                        "record"
                    ) | (
                        Some(Error::Invalid("injected caller delivery failure")),
                        "caller"
                    ) | (
                        Some(Error::Invalid("injected terminal delivery failure")),
                        "terminal"
                    )
                ),
                "rank={rank} case={case} original={original}"
            );
        } else {
            let mut source = Some(original);
            let mut found = false;
            while let Some(error) = source {
                found |= error
                    .downcast_ref::<eredu_core::run_preparation::TextPreparationRejected>()
                    .is_some_and(|error| {
                        error.rank == 0
                            && error.stage
                                == eredu_core::run_preparation::TextPreparationStage::Delivery
                    });
                source = error.source();
            }
            assert!(found, "rank={rank} case={case} original={original}");
        }
        if case == "terminal" {
            assert!(steps > 1);
        } else {
            assert_eq!(steps, if case == "record" { 0 } else { 1 });
        }
        assert!(runtime.synchronize().is_err());
    }
    assert!(publications > 0);
    let after = runtime.text_preparation_usage().unwrap();
    assert!(after.attempts > before.attempts);
    assert!(after.host_bytes > before.host_bytes);
    eprintln!(
        "speculative control delivery rank={rank} case={case} steps={steps} publications={publications}"
    );
}
