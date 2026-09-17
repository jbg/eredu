use super::*;
use crate::working_memory::{LoadedDecodeSource, LoadedGenerationDecoderInput, OriginalStopSource};
use eredu_core::{
    GenerationDecoderInput, GenerationDecoderOutput, GenerationSequenceBankRejection,
};
use eredu_text::stop_storage::StopCompilePlan;
fn decoder(pool: &WorkingMemoryPool) -> LoadedDecodeSource {
    super::loaded::loaded(pool)
}
fn stops(pool: &WorkingMemoryPool, strings: &[&str]) -> OriginalStopSource {
    pool.compile_stop_source(StopCompilePlan::prepare_refs(strings).unwrap())
        .unwrap()
}
fn plain() -> eredu_core::GenerationSequenceConsumerLayout {
    consumer().with_plain_text_output().unwrap()
}
fn attach(state: &Rc<RefCell<State>>, input: &LoadedGenerationDecoderInput) {
    state.borrow_mut().loaded_bytes =
        input.source().original_bytes() + input.stop_source().unwrap().original_bytes();
}
#[test]
fn two_plain_requests_share_exact_sources_across_three_real_routes_and_capture() {
    for route in 0..3 {
        for capture in [false, true] {
            let (mut runtime, state) = runtime(Mode {
                capture,
                audit_decoder: true,
                ..Mode::default()
            });
            let pool = state.borrow().pool.clone();
            let decoder = decoder(&pool);
            let stops = stops(&pool, &[" é", " é", ""]);
            let cold = decoder.original_bytes() + stops.original_bytes();
            let mut previous_r = None;
            for run in 0..2 {
                let input = LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 5, true)
                    .unwrap();
                attach(&state, &input);
                let foreign =
                    LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 5, true)
                        .unwrap();
                let changed_skip =
                    LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 5, false)
                        .unwrap();
                let options = capture.then(|| eredu_core::TextPreparationOptions {
                    interventions: None, capture: Some(capture_source_for_geometry(InferenceGeometry {
                        max_output_tokens: 5,
                        ..geometry()
                    })),
                });
                let sequence =
                    extract_decoder(&mut runtime, route, 5, &input, &plain(), options).unwrap();
                assert_eq!(
                    sequence.decoder_output(),
                    GenerationDecoderOutput::PlainText
                );
                assert!(sequence.matches_decoder_input(Some(&input)));
                assert!(!sequence.matches_decoder_input(Some(&foreign)));
                assert!(!sequence.matches_decoder_input(Some(&changed_skip)));
                let mut sequence = sequence.prepare_storage().unwrap();
                let address = sequence.tokens().as_ptr();
                assert_eq!(
                    sequence.decode_token(0),
                    Err(GenerationDecoderError::Unavailable)
                );
                let first = sequence.project_plain_text(0).unwrap();
                assert_eq!((first.visible, first.stop_matched), ("hello", false));
                sequence.commit(0, TokenTerminalSignals::default()).unwrap();
                let second = sequence.project_plain_text(1).unwrap();
                assert_eq!((second.visible, second.stop_matched), ("", true));
                sequence
                    .commit(
                        1,
                        TokenTerminalSignals {
                            stop_sequence: true,
                            ..TokenTerminalSignals::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    sequence.finish_reason(),
                    Some(eredu_core::FinishReason::StopSequence)
                );
                assert_eq!(sequence.finish_plain_text().unwrap(), "");
                let r = state.borrow().r;
                if let Some(previous) = previous_r {
                    assert_eq!(r, previous);
                }
                previous_r = Some(r);
                let (_, held) = retire_request(&state);
                drop((input, foreign, changed_skip));
                assert_eq!(pool.used_bytes().unwrap(), cold + held);
                let tokens = sequence.into_token_ids();
                assert_eq!(tokens.as_ptr(), address);
                assert_eq!(tokens.as_ref(), [0, 1]);
                drop(tokens);
                assert_eq!(pool.used_bytes().unwrap(), cold);
                if run == 0 {
                    state.borrow_mut().root = Some(pool.register_storage([(1u32, 64)]).unwrap());
                }
            }
            assert_eq!(state.borrow().decoder_takes, 2);
            drop(runtime);
            assert_eq!(pool.used_bytes().unwrap(), cold);
            drop((decoder, stops));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn shared_stop_request_prices_destinations_without_copying_original_source_bytes() {
    let mut results = Vec::new();
    for literals in [
        vec!["halt"],
        vec!["halt", "also"],
        vec!["longhalt"],
        vec!["halt", "halt", ""],
    ] {
        let (mut runtime, state) = runtime(Mode::default());
        let pool = state.borrow().pool.clone();
        let decoder = decoder(&pool);
        let stops = stops(&pool, &literals);
        let input =
            LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 5, false).unwrap();
        attach(&state, &input);
        let sequence = extract_decoder(&mut runtime, 0, 5, &input, &plain(), None).unwrap();
        results.push((state.borrow().r, stops.original_bytes()));
        drop(sequence);
        retire_request(&state);
        drop((runtime, input, decoder, stops));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    assert_eq!(results[0].0, results[1].0);
    assert!(results[1].1 > results[0].1);
    assert_eq!(results[2].0 - results[0].0, 4); // only four more proper-prefix destination bytes
    assert_eq!(results[3], results[0]);
}
#[test]
fn both_original_pools_preflight_and_one_claim_reject_foreign_replay_and_short_admission() {
    let (mut runtime, state) = runtime(Mode::default());
    let pool = state.borrow().pool.clone();
    let decoder = decoder(&pool);
    let stops = stops(&pool, &[" é"]);
    let cold = decoder.original_bytes() + stops.original_bytes();
    let foreign_pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let foreign_stops = self::stops(&foreign_pool, &[" é"]);
    for _ in 0..32 {
        assert!(matches!(
            LoadedGenerationDecoderInput::new_plain_text(&decoder, &foreign_stops, 3, true),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
    }
    assert_eq!(pool.used_bytes().unwrap(), cold + 64);
    let input = LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 3, true).unwrap();
    attach(&state, &input);
    assert!(extract_decoder(&mut runtime, 0, 3, &input, &consumer(), None).is_err());
    assert!(extract_decoder(&mut runtime, 0, 4, &input, &plain(), None).is_err());
    let (mut foreign, other) = harness::runtime(Mode::default());
    let error = extract_decoder(&mut foreign, 0, 3, &input, &plain(), None).unwrap_err();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceBankRejection>(),
        Some(&GenerationSequenceBankRejection::IdentityMismatch)
    );
    assert_eq!(state.borrow().decoder_takes, 0);
    assert_eq!(other.borrow().decoder_takes, 0);
    state.borrow_mut().mode.short = true;
    assert!(extract_decoder(&mut runtime, 0, 3, &input, &plain(), None).is_err());
    assert_eq!(state.borrow().order, ["admit"]);
    assert_eq!(pool.used_bytes().unwrap(), cold + 64);
    state.borrow_mut().mode.short = false;
    let replays: Vec<_> = (0..32)
        .map(|_| extract_decoder(&mut runtime, 0, 3, &input, &plain(), None).unwrap_err())
        .collect();
    assert!(replays.iter().all(|error| error
        .source()
        .unwrap()
        .downcast_ref::<GenerationSequenceBankRejection>()
        == Some(&GenerationSequenceBankRejection::Unavailable)));
    assert_eq!(state.borrow().decoder_takes, 1);
    let fresh = LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 3, true).unwrap();
    let sequence = extract_decoder(&mut runtime, 1, 3, &fresh, &plain(), None).unwrap();
    drop(sequence);
    retire_request(&state);
    retire_request(&other);
    drop((
        runtime,
        foreign,
        input,
        fresh,
        decoder,
        stops,
        foreign_stops,
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
    drop(replays);
}
#[test]
fn explicit_empty_plain_owner_and_dormant_cancel_preserve_sources_until_freeze() {
    for maximum in [0, 3] {
        for empty in [false, true] {
            let (mut runtime, state) = runtime(Mode::default());
            let pool = state.borrow().pool.clone();
            let decoder = decoder(&pool);
            let stops = stops(&pool, if empty { &[] } else { &[" é"] });
            let input =
                LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, maximum, true)
                    .unwrap();
            attach(&state, &input);
            assert!(input.stop_source().is_some());
            assert_eq!(input.output_kind(), GenerationDecoderOutput::PlainText);
            let mut sequence =
                extract_decoder(&mut runtime, 2, maximum, &input, &plain(), None).unwrap();
            if maximum == 0 {
                assert_eq!(sequence.finish_plain_text().unwrap(), "");
            } else if empty {
                sequence = sequence.prepare_storage().unwrap();
                let first = sequence.project_plain_text(0).unwrap();
                assert_eq!((first.visible, first.stop_matched), ("hello", false));
                sequence.commit(0, TokenTerminalSignals::default()).unwrap();
                assert!(sequence.cancel());
            } else {
                assert!(sequence.cancel());
            }
            let cold = decoder.original_bytes() + stops.original_bytes();
            let (_, held) = retire_request(&state);
            drop((runtime, input, decoder, stops));
            assert_eq!(pool.used_bytes().unwrap(), cold + held);
            let tokens = sequence.into_token_ids();
            assert_eq!(pool.used_bytes().unwrap(), held);
            drop(tokens);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn consuming_plain_provider_failure_keeps_both_sources_after_model_and_header_drop() {
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let pool = state.borrow().pool.clone();
    let decoder = decoder(&pool);
    let stops = stops(&pool, &[" é"]);
    let cold = decoder.original_bytes() + stops.original_bytes();
    let input = LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 3, true).unwrap();
    attach(&state, &input);
    let sequence = extract_decoder(&mut runtime, 1, 3, &input, &plain(), None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let (_, held) = retire_request(&state);
    drop((runtime, input, decoder, stops));
    let failure = sequence.prepare_storage().unwrap_err();
    assert!(failure.cause().source().unwrap().is::<WorkingMemoryError>());
    assert_eq!(pool.used_bytes().unwrap(), cold + held + 64);
    let failure = failure.into_sequence().prepare_storage().unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), cold + held + 64);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn concurrent_genuine_claims_take_the_entire_plain_pair_once() {
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let decoder = decoder(&pool);
    let stops = stops(&pool, &[" é"]);
    let input = LoadedGenerationDecoderInput::new_plain_text(&decoder, &stops, 3, true).unwrap();
    let cold = decoder.original_bytes() + stops.original_bytes();
    let barrier = std::sync::Barrier::new(2);
    let outcomes = std::thread::scope(|threads| {
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (pool, input, barrier) = (pool.clone(), &input, &barrier);
            workers.push(threads.spawn(move || {
                let (mut runtime, state) = harness::runtime_with_pool(Mode::default(), pool);
                attach(&state, input);
                barrier.wait();
                let result = extract_decoder(&mut runtime, 0, 3, input, &plain(), None);
                let success = result.is_ok();
                if let Err(error) = &result {
                    assert_eq!(
                        error
                            .source()
                            .unwrap()
                            .downcast_ref::<GenerationSequenceBankRejection>(),
                        Some(&GenerationSequenceBankRejection::Unavailable)
                    );
                }
                drop(result);
                retire_request(&state);
                drop(runtime);
                success
            }));
        }
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(outcomes.iter().filter(|x| **x).count(), 1);
    assert_eq!(pool.used_bytes().unwrap(), cold);
    drop((input, decoder, stops));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn foreign_combined_bank_preserves_its_original_pair_and_rejects_equal_new_sources() {
    let (mut first, old) = runtime(Mode {
        defer_bank: true,
        explicit_source: true,
        ..Mode::default()
    });
    let old_pool = old.borrow().pool.clone();
    let a_decoder = decoder(&old_pool);
    let a_stops = stops(&old_pool, &["é"]);
    let old_cold = a_decoder.original_bytes() + a_stops.original_bytes();
    let a = LoadedGenerationDecoderInput::new_plain_text(&a_decoder, &a_stops, 3, true).unwrap();
    attach(&old, &a);
    let initial = extract_decoder(&mut first, 0, 3, &a, &plain(), None).unwrap_err();
    let original = old.borrow().active.as_ref().unwrap().clone();
    let bank = old.borrow_mut().pending_bank.take().unwrap();
    let (mut second, new) = runtime(Mode {
        reject_decoder_attachment: true,
        ..Mode::default()
    });
    let new_pool = new.borrow().pool.clone();
    let b_decoder = decoder(&new_pool);
    let b_stops = stops(&new_pool, &["é"]);
    let b = LoadedGenerationDecoderInput::new_plain_text(&b_decoder, &b_stops, 3, true).unwrap();
    attach(&new, &b);
    assert!(a_stops.source().stops().eq(b_stops.source().stops()));
    assert!(!a_stops.same_source(&b_stops));
    new.borrow_mut().foreign = Some(original.clone());
    new.borrow_mut().pending_bank = Some(bank);
    let error = extract_decoder(&mut second, 2, 3, &b, &plain(), None).unwrap_err();
    assert_eq!(new.borrow().order, ["admit"]);
    assert_eq!(new.borrow().decoder_rejections_untaken, 1);
    drop(original);
    let (_, held) = retire_request(&old);
    retire_request(&new);
    drop((first, second, a, b, a_decoder, a_stops, b_decoder, b_stops));
    assert_eq!(new_pool.used_bytes().unwrap(), 0);
    assert_eq!(old_pool.used_bytes().unwrap(), old_cold + held + 64);
    drop(error);
    assert_eq!(old_pool.used_bytes().unwrap(), 0);
    drop(initial);
}
