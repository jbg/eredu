use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::working_memory::{
    LoadedDecodeSourceBackend, LoadedGenerationDecoderInput, OriginalStopSourceBackend,
};
use eredu_text::decoder_storage::DecodeCompilePlan;

#[test]
fn native_plain_requests_reuse_and_override_original_stops_across_residency_and_cached_continuation()
 {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
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
        let stops = MlxBackend::compile_original_stop_source(
            &runtime,
            eredu_text::stop_storage::StopCompilePlan::prepare_refs(&["\0halt"]).unwrap(),
        )
        .unwrap();
        let mut cold = source.original_bytes() + stops.original_bytes();
        let mut override_stops = None;
        let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
            eredu_core::RetainedGenerationSequence,
            WorkingMemoryError,
            WorkingMemoryError,
        >()
        .unwrap()
        .with_plain_text_output()
        .unwrap();
        let mut final_ids = None;
        let mut final_held = 0;
        for request in 0..3 {
            if request == 2 {
                let changed = MlxBackend::compile_original_stop_source(
                    &runtime,
                    eredu_text::stop_storage::StopCompilePlan::prepare_refs(&["\0changed"])
                        .unwrap(),
                )
                .unwrap();
                assert!(!changed.same_source(&stops));
                cold += changed.original_bytes();
                override_stops = Some(changed);
            }
            let request_stops = override_stops.as_ref().unwrap_or(&stops);
            let input =
                LoadedGenerationDecoderInput::new_plain_text(&source, request_stops, 4, false)
                    .unwrap();
            assert!(input.source().same_source(&source));
            assert!(input.stop_source().unwrap().same_source(request_stops));
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
                    GenerationSequenceRequest::new(4, &[])
                        .with_decoder(&input)
                        .with_consumer(&consumer),
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
                let output = sequence.project_plain_text(id).unwrap();
                assert!(!output.stop_matched);
                text.push_str(output.visible);
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                assert!(driver.take_completed_delivery(&mut run).unwrap().is_none());
            }
            text.push_str(sequence.finish_plain_text().unwrap());
            assert_eq!(text, snapshot.decode(&actual, false).unwrap());
            assert!(!text.is_empty());
            assert_eq!(probe.0.decoder_takes.get(), 1);
            let ids = sequence.into_token_ids();
            assert_eq!(ids.as_ptr(), ptr);
            drop((run, driver, preparation, probe, input));
            if request < 2 {
                drop(ids);
                // Cached native owners can keep the prior original request
                // charge alive. This is continuation, not an unfunded reset.
                runtime.synchronize().unwrap();
                assert!(pool.fixture_host_charge().unwrap() >= cold);
            } else {
                final_ids = Some(ids);
                final_held = facts.held;
            }
        }
        finish(runtime, &stream);
        settle(&pool, cold + final_held);
        // Only the cold owners now retain the compiled programs: freeze already
        // retired both destinations/source leases before returning raw IDs.
        drop((source, stops, override_stops));
        settle(&pool, final_held);
        assert_eq!(final_ids.as_ref().unwrap().len(), 4);
        drop(final_ids);
        settle(&pool, 0);
    }
}
#[test]
fn native_plain_shared_sources_fenced_error_outlives_model_and_header() {
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
    let stops = MlxBackend::compile_original_stop_source(
        &runtime,
        eredu_text::stop_storage::StopCompilePlan::prepare_refs(&["halt"]).unwrap(),
    )
    .unwrap();
    let cold = source.original_bytes() + stops.original_bytes();
    let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
        eredu_core::RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
    .with_plain_text_output()
    .unwrap();
    let input = LoadedGenerationDecoderInput::new_plain_text(&source, &stops, 4, false).unwrap();
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::FenceProvider);
    let error = {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let result = driver.start_token_ids_with_sequence(
            eredu_core::TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
            config(4, u64::MAX),
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(4, &[])
                .with_decoder(&input)
                .with_consumer(&consumer),
        );
        match result {
            Err(eredu_core::ControlledTextGenerationError::Preparation(error)) => error,
            Err(_) => panic!("unexpected setup error"),
            Ok(_) => panic!("fenced provider succeeded"),
        }
    };
    let _ = cause::<eredu_core::RetainedSequencePreparationError>(&error);
    let held = probe.facts().held;
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        WorkingMemoryError::ExecutionFenced
    ));
    drop((probe.take(), probe, input, source, stops));
    finish(runtime, &stream);
    settle(&pool, cold + held);
    drop(error);
    settle(&pool, 0);
}
