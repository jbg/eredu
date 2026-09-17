use super::*;
use crate::working_memory::{LoadedDecodeSource, LoadedGenerationDecoderInput};
use eredu_text::decoder_storage::DecodeCompilePlan;

pub(super) fn loaded(pool: &WorkingMemoryPool) -> LoadedDecodeSource {
    let tokenizer = Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":7,"content":"<stop>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"é":1,"":2,"<stop>":7,"[UNK]":8},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap();
    pool.compile_decode_source(DecodeCompilePlan::prepare(&tokenizer.snapshot()).unwrap())
        .unwrap()
}
fn attach(state: &Rc<RefCell<State>>, source: &LoadedDecodeSource) {
    state.borrow_mut().loaded_bytes = source.original_bytes();
}
#[test]
fn two_successive_original_requests_share_one_source_and_freeze_only_request_storage() {
    for route in 0..3 {
        let (mut runtime, state) = runtime(Mode {
            audit_decoder: true,
            ..Mode::default()
        });
        let pool = state.borrow().pool.clone();
        let source = loaded(&pool);
        let cold = source.original_bytes();
        attach(&state, &source);
        let mut previous_r = None;
        for run in 0..2 {
            let input = LoadedGenerationDecoderInput::new(&source, 5, true).unwrap();
            assert!(input.source().same_source(&source));
            let foreign_header = LoadedGenerationDecoderInput::new(&source, 5, true).unwrap();
            let opposite = LoadedGenerationDecoderInput::new(&source, 5, false).unwrap();
            let mut sequence =
                extract_decoder(&mut runtime, route, 5, &input, &consumer(), None).unwrap();
            assert!(sequence.matches_decoder_input(Some(&input)));
            assert!(!sequence.matches_decoder_input(Some(&foreign_header)));
            assert!(!sequence.matches_decoder_input(Some(&opposite)));
            assert_eq!(
                sequence.decode_token(0),
                Err(GenerationDecoderError::Unprepared)
            );
            sequence = sequence.prepare_storage().unwrap();
            let ptr = sequence.tokens().as_ptr();
            let mut output = String::new();
            for id in [0, 1, 99, 2, 7] {
                if let Some(text) = sequence.decode_token(id).unwrap() {
                    output.push_str(text);
                }
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
            }
            assert_eq!(output, "hello é ");
            sequence.finish_decoder().unwrap();
            let r = state.borrow().r;
            if let Some(previous) = previous_r {
                assert_eq!(r, previous);
            }
            previous_r = Some(r);
            let (_, held) = retire_request(&state);
            drop((input, foreign_header, opposite));
            assert_eq!(pool.used_bytes().unwrap(), cold + held);
            let tokens = sequence.into_token_ids();
            assert_eq!(tokens.as_ptr(), ptr);
            assert_eq!(tokens.as_ref(), [0, 1, 99, 2, 7]);
            assert_eq!(pool.used_bytes().unwrap(), cold + held);
            drop(tokens);
            assert_eq!(pool.used_bytes().unwrap(), cold);
            if run == 0 {
                state.borrow_mut().root = Some(pool.register_storage([(1u32, 64)]).unwrap());
            }
        }
        assert_eq!(state.borrow().decoder_takes, 2);
        drop(runtime);
        assert_eq!(pool.used_bytes().unwrap(), cold);
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn shared_preflight_foreign_pool_wrong_n_replay_and_one_short_preserve_cold_owner() {
    let (mut runtime, state) = runtime(Mode::default());
    let pool = state.borrow().pool.clone();
    let source = loaded(&pool);
    attach(&state, &source);
    let cold = source.original_bytes();
    assert!(matches!(
        LoadedGenerationDecoderInput::new(&source, usize::MAX, true),
        Err(eredu_text::decoder_storage::DecodeStorageError::Overflow)
    ));
    let input = LoadedGenerationDecoderInput::new(&source, 3, true).unwrap();
    let plain = consumer().with_plain_text_output().unwrap();
    assert!(extract_decoder(&mut runtime, 0, 3, &input, &plain, None).is_err());
    assert_eq!(state.borrow().decoder_takes, 0);
    let (mut foreign, foreign_state) = harness::runtime(Mode::default());
    let error = extract_decoder(&mut foreign, 0, 3, &input, &consumer(), None).unwrap_err();
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
        Some(&eredu_core::GenerationSequenceBankRejection::IdentityMismatch)
    );
    assert_eq!(foreign_state.borrow().decoder_takes, 0);
    assert!(extract_decoder(&mut runtime, 0, 4, &input, &consumer(), None).is_err());
    assert_eq!(state.borrow().decoder_takes, 0);
    state.borrow_mut().mode.short = true;
    assert!(extract_decoder(&mut runtime, 0, 3, &input, &consumer(), None).is_err());
    assert_eq!(pool.used_bytes().unwrap(), cold + 64);
    state.borrow_mut().mode.short = false;
    let failures: Vec<_> = (0..32)
        .map(|_| extract_decoder(&mut runtime, 0, 3, &input, &consumer(), None).unwrap_err())
        .collect();
    assert!(failures.iter().all(|error| error
        .source()
        .unwrap()
        .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(
    ) == Some(
        &eredu_core::GenerationSequenceBankRejection::Unavailable
    )));
    assert_eq!(state.borrow().decoder_takes, 1);
    let next = LoadedGenerationDecoderInput::new(&source, 3, true).unwrap();
    let sequence = extract_decoder(&mut runtime, 1, 3, &next, &consumer(), None).unwrap();
    drop(sequence);
    retire_request(&state);
    retire_request(&foreign_state);
    drop((runtime, foreign, input, next, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(failures);
}
#[test]
fn shared_source_survives_model_and_consuming_provider_failure_without_repricing() {
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let pool = state.borrow().pool.clone();
    let source = loaded(&pool);
    let cold = source.original_bytes();
    attach(&state, &source);
    let input = LoadedGenerationDecoderInput::new(&source, 3, true).unwrap();
    let sequence = extract_decoder(&mut runtime, 1, 3, &input, &consumer(), None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let (_, held) = retire_request(&state);
    drop((runtime, input, source));
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), cold + held + 64);
    assert!(failure.cause().source().unwrap().is::<WorkingMemoryError>());
    let sequence = failure.into_sequence();
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), cold + held + 64);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn two_concurrent_genuine_claims_take_one_atomic_header_lease() {
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let source = loaded(&pool);
    let input = LoadedGenerationDecoderInput::new(&source, 3, true).unwrap();
    let cold = source.original_bytes();
    let barrier = std::sync::Barrier::new(2);
    let outcomes = std::thread::scope(|threads| {
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (pool, input, barrier) = (pool.clone(), &input, &barrier);
            workers.push(threads.spawn(move || {
                let (mut runtime, state) = harness::runtime_with_pool(Mode::default(), pool);
                attach(&state, input.source());
                barrier.wait();
                let result = extract_decoder(&mut runtime, 0, 3, input, &consumer(), None);
                let success = result.is_ok();
                if let Err(error) = &result {
                    assert_eq!(
                        error
                            .source()
                            .unwrap()
                            .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
                        Some(&eredu_core::GenerationSequenceBankRejection::Unavailable)
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
    drop((input, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
