use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::working_memory::{LoadedDecodeSourceBackend, LoadedGenerationDecoderInput};
use eredu_text::decoder_storage::DecodeCompilePlan;

#[test]
fn two_actual_native_requests_share_one_cold_source_across_residency_and_cached_continuation() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let mut expected_by_request = [None, None];
    for residency in 0..3 {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, residency);
        let tokenizer = tokenizer();
        let snapshot = tokenizer.snapshot();
        let source = MlxBackend::compile_loaded_decode_source(
            &runtime,
            DecodeCompilePlan::prepare(&snapshot).unwrap(),
        )
        .unwrap();
        let cold = source.original_bytes();
        let mut final_ids = None;
        let mut final_held = 0;
        for request in 0..2 {
            let input = LoadedGenerationDecoderInput::new(&source, 4, false).unwrap();
            assert!(input.source().same_source(&source));
            let probe = Probe::new(&runtime, None, false);
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut run = driver
                .start_token_ids_with_sequence(
                    eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
                    config(4, u64::MAX).with_inference_policy(eredu_core::TextInferencePolicy {
                        prefill_chunk_positions: std::num::NonZeroU64::new(2),
                        memory_limits: eredu_core::MemoryLimitDeclarations::new([(
                            "host".into(),
                            eredu_core::MemoryLimit::Finite(u64::MAX),
                        )]),
                        submission_tracking_capacity_bytes: None,
                        graph_metadata_capacity_bytes: None,
                    }),
                    disk::Controller::default(),
                    None,
                    GenerationSequenceRequest::new(4, &[]).with_decoder(&input),
                )
                .unwrap();
            let preparation = probe.take();
            let facts = probe.facts();
            let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
            assert!(driver.take_prepared_sequence(&mut run).unwrap().is_none());
            assert!(sequence.matches_decoder_input(Some(&input)));
            let mut sequence = sequence.prepare_storage().unwrap();
            let ptr = sequence.tokens().as_ptr();
            let mut actual = Vec::new();
            let mut text = String::new();
            for _ in 0..4 {
                let token = driver.advance(&mut run).unwrap().unwrap();
                let id = token.token_id();
                actual.push(id);
                if let Some(suffix) = sequence.decode_token(id).unwrap() {
                    text.push_str(suffix);
                }
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                assert!(driver.take_completed_delivery(&mut run).unwrap().is_none());
            }
            sequence.finish_decoder().unwrap();
            assert_eq!(text, snapshot.decode(&actual, false).unwrap());
            assert!(!text.is_empty());
            assert_eq!(probe.0.decoder_takes.get(), 1);
            let ids = sequence.into_token_ids();
            assert_eq!(ids.as_ptr(), ptr);
            if let Some(expected) = &expected_by_request[request] {
                assert_eq!(&actual, expected);
            } else {
                expected_by_request[request] = Some(actual);
            }
            drop((run, driver, preparation, probe, input));
            runtime.synchronize().unwrap();
            assert_eq!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .retained_inference_authority()
                    .unwrap()
                    .admission()
                    .unwrap()
                    .position(),
                6 * (request + 1) as u64,
            );
            if request == 0 {
                drop(ids);
                // Completion preserves the cached request's original reservation.
                // Reset still needs a separately admitted memory path; rejection
                // must preserve the frontier used by the second request.
                let error = runtime.reset().unwrap_err();
                assert!(matches!(
                    cause::<WorkingMemoryError>(&error),
                    WorkingMemoryError::ReservedWorkActive
                ));
                drop(error);
                assert_eq!(
                    runtime
                        .session()
                        .payload
                        .model
                        .erased()
                        .retained_inference_authority()
                        .unwrap()
                        .admission()
                        .unwrap()
                        .position(),
                    6,
                );
                assert!(pool.fixture_host_charge().unwrap() >= cold);
            } else {
                final_ids = Some(ids);
                final_held = facts.held;
            }
        }
        finish(runtime, &stream);
        settle(&pool, cold + final_held);
        // Only the cold owner now retains the compiled source: freeze already
        // retired decoder destinations/source lease before returning raw IDs.
        drop(source);
        settle(&pool, final_held);
        assert_eq!(final_ids.as_ref().unwrap().len(), 4);
        drop(final_ids);
        settle(&pool, 0);
    }
}
#[test]
fn native_shared_source_fenced_provider_error_outlives_model_and_input() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let snapshot = tokenizer().snapshot();
    let source = MlxBackend::compile_loaded_decode_source(
        &runtime,
        DecodeCompilePlan::prepare(&snapshot).unwrap(),
    )
    .unwrap();
    let cold = source.original_bytes();
    let input = LoadedGenerationDecoderInput::new(&source, 4, false).unwrap();
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::FenceProvider);
    let error =
        start_with_decoder(&mut runtime, 4, &[], 0, u64::MAX, None, Some(&input)).unwrap_err();
    let _ = cause::<eredu_core::RetainedSequencePreparationError>(&error);
    let held = probe.facts().held;
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::ExecutionFenced
    ));
    drop((probe.take(), probe, input, source));
    finish(runtime, &stream);
    settle(&pool, cold + held);
    drop(error);
    settle(&pool, 0);
}
