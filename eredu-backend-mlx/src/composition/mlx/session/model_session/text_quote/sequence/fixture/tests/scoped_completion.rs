//! Actual native carrier contribution participates in the unchanged original Q.
use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;

#[test]
fn c2_native_carrier_exact_and_one_short_precede_chunk_source_for_each_driver() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let carrier = safemlx::PreparedPrefillFailure::<
        eredu_runtime::working_memory::OriginalPrefillRootCustody,
    >::layout()
    .unwrap();
    assert!(carrier.native_owner_bytes > 0 && carrier.rust_node_bytes > 0);
    assert!(carrier.total_bytes().unwrap() > carrier.native_owner_bytes);
    for driver in 0..3 {
        let mut required = None;
        for pass in 0..3 {
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let (mut runtime, _artifact) = load(&stream, &pool, 0);
            let probe = Probe::new(&runtime, None, false);
            let baseline = pool.fixture_host_charge().unwrap();
            let input_attempts = paths::session_input_creation_attempts();
            let native = paths::snapshot();
            let capacity = required.map_or(u64::MAX, |n| baseline + n - u64::from(pass == 1));
            let configuration = chunked_original(config(4, capacity));
            let mut callback_called = false;
            let result = start_chunked(&mut runtime, configuration, driver, |sequence| {
                callback_called = true;
                assert_ne!(
                    pass, 1,
                    "one-short construction must not call the inspector"
                );
                let facts = probe.facts();
                if pass == 0 {
                    required = Some(facts.required);
                } else {
                    assert_eq!(Some(facts.required), required);
                }
                let preparation = probe.take();
                let quote = preparation.quote.as_ref().unwrap();
                let step = preparation
                    .request
                    .as_ref()
                    .unwrap()
                    .claim_step(&probe.context(), eredu_core::PendingTextInput::Prefill(()))
                    .unwrap();
                let mut bank = quote.prefill_scopes.as_ref().unwrap().borrow_mut();
                let scopes = bank.claim(&step).unwrap();
                drop(bank);
                let facts = scopes.facts();
                assert_eq!(facts.plan().span_count(), 3);
                assert_eq!(facts.plan().scope_count(), 17);
                assert!(
                    facts.total_bytes().unwrap().unwrap()
                        >= u64::try_from(carrier.total_bytes().unwrap()).unwrap()
                            * facts.plan().scope_count()
                );
                drop((scopes, step));
                drop((sequence, preparation));
            });
            if pass == 1 {
                assert!(!callback_called);
                let error = result.unwrap_err();
                assert!(matches!(
                    cause::<WorkingMemoryError>(&error),
                    WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    )
                ));
                assert!(probe.0.admitted.borrow().is_none());
                assert_eq!(probe.0.calls.get(), 0);
                assert_eq!(paths::session_input_creation_attempts(), input_attempts);
                assert_eq!(paths::snapshot(), native);
                assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
            } else {
                result.unwrap();
                assert!(callback_called);
            }
            drop(probe);
            finish(runtime, &stream);
            settle(&pool, 0);
        }
    }
}

fn start_chunked(
    runtime: &mut Runtime,
    config: TextGenerationConfig,
    driver: usize,
    inspect: impl FnOnce(RetainedGenerationSequence),
) -> Result<(), BackendFailure> {
    let input = TextGenerationInput::TokenIds(vec![2, 5, 7, 3, 11]);
    let claim = GenerationSequenceRequest::new(4, &[]);
    match driver {
        0 => {
            let mut run = TextGeneration::from_input_with_sequence(
                runtime,
                input,
                config,
                TokenFilter::All,
                None,
                claim,
            )?;
            inspect(run.take_prepared_sequence().unwrap());
            Ok(())
        }
        1 => {
            let mut run = ControlledTextGeneration::from_input_with_sequence(
                runtime,
                input,
                config,
                disk::Controller::default(),
                None,
                claim,
            )
            .map_err(preparation_error)?;
            inspect(run.take_prepared_sequence().unwrap());
            Ok(())
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut run = driver
                .start_input_with_sequence(input, config, disk::Controller::default(), None, claim)
                .map_err(preparation_error)?;
            inspect(driver.take_prepared_sequence(&mut run).unwrap().unwrap());
            Ok(())
        }
    }
}
fn preparation_error<B: std::error::Error + 'static, C: std::error::Error + 'static>(
    error: ControlledTextGenerationError<B, C>,
) -> BackendFailure {
    match error {
        ControlledTextGenerationError::Preparation(error) => error,
        other => panic!("unexpected controlled error: {other}"),
    }
}
