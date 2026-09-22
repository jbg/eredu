use super::decoder::extract_decoder;
use super::*;
use crate::working_memory::OriginalGenerationDecoderInput;
use eredu_core::{
    GenerationDecoderError, GenerationDecoderOutput, GenerationSequenceConsumerLayout,
};
use eredu_text::{
    decoder_storage::PreparedDecodeSource, stop_storage::PreparedStopSource, tokenizer::Tokenizer,
};
fn source(maximum: usize, stops: &[&str]) -> OriginalGenerationDecoderInput {
    let tokenizer = Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"é":1,"":2,"[UNK]":7},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap();
    OriginalGenerationDecoderInput::new_plain_text(
        PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap(),
        maximum,
        false,
        PreparedStopSource::prepare(stops.iter().copied()).unwrap(),
    )
    .unwrap()
}
fn consumer() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
    .with_plain_text_output()
    .unwrap()
}
#[test]
fn actual_plain_source_admission_and_borrowed_stop_output_share_all_three_routes() {
    for route in 0..3 {
        for capture in [false, true] {
            let (mut runtime, state) = runtime(Mode {
                capture,
                audit_decoder: true,
                ..Mode::default()
            });
            let source = source(5, &[" é"]);
            let equal_foreign = self::source(5, &[" é"]);
            let changed_stops = self::source(5, &["hello"]);
            let options = capture.then(|| TextPreparationOptions {
                interventions: None,
                capture: Some(capture_source_for_geometry(InferenceGeometry {
                    max_output_tokens: 5,
                    ..geometry()
                })),
            });
            let sequence =
                extract_decoder(&mut runtime, route, 5, &source, &consumer(), options).unwrap();
            assert_eq!(state.borrow().decoder_takes, 1);
            assert_eq!(
                state.borrow().order,
                ["admit", "bind", "extract", "prompt", "sampling"]
            );
            assert_eq!(
                sequence.decoder_output(),
                GenerationDecoderOutput::PlainText
            );
            assert!(sequence.matches_decoder_input(Some(&source)));
            assert!(!sequence.matches_decoder_input(Some(&equal_foreign)));
            assert!(!sequence.matches_decoder_input(Some(&changed_stops)));
            assert!(!sequence.matches_decoder_input(None));
            let mut sequence = sequence.prepare_storage().unwrap();
            let address = sequence.tokens().as_ptr();
            assert_eq!(
                sequence.decode_token(0),
                Err(GenerationDecoderError::Unavailable)
            );
            assert_eq!(
                sequence.finish_decoder(),
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
            assert_eq!(sequence.tokens(), [0, 1]);
            let (pool, held) = retire_request(&state);
            drop(runtime);
            drop((source, equal_foreign, changed_stops));
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            let tokens = sequence.into_token_ids();
            assert_eq!(tokens.as_ptr(), address);
            let mut iterator = tokens.clone().into_iter();
            drop(tokens);
            assert_eq!(iterator.next(), Some(0));
            assert_eq!(pool.payload_used_bytes().unwrap(), held);
            drop(iterator);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn original_stop_extents_have_exact_admission_delta_and_minus_one_rejects_before_work() {
    let mut rs = vec![];
    for stops in [vec!["halt"], vec!["longhalt"], vec!["halt", "halt", ""]] {
        let (mut runtime, state) = runtime(Mode::default());
        let input = source(5, &stops);
        let sequence = extract_decoder(&mut runtime, 0, 5, &input, &consumer(), None).unwrap();
        rs.push(state.borrow().r);
        drop(sequence);
        let (pool, _) = retire_request(&state);
        drop(runtime);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    // Four more immutable UTF-8 bytes and four more proper-prefix work bytes;
    // one same-shape index entry and every named control are unchanged.
    assert_eq!(rs[1] - rs[0], 8);
    assert_eq!(rs[2], rs[0]);
    let (mut runtime, state) = runtime(Mode {
        short: true,
        ..Mode::default()
    });
    let input = source(5, &["halt"]);
    let error = extract_decoder(&mut runtime, 0, 5, &input, &consumer(), None).unwrap_err();
    assert_eq!(state.borrow().order, ["admit"]);
    assert_eq!(state.borrow().votes, [(Stage::Admission, Status::Failed)]);
    assert_eq!(state.borrow().pool.payload_used_bytes().unwrap(), 64);
    drop(error);
    state.borrow_mut().mode.short = false;
    let replay = extract_decoder(&mut runtime, 0, 5, &input, &consumer(), None).unwrap_err();
    assert_eq!(
        replay
            .source()
            .unwrap()
            .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
        Some(&eredu_core::GenerationSequenceBankRejection::Unavailable)
    );
    assert_eq!(state.borrow().decoder_takes, 1);
    let fresh = source(5, &["halt"]);
    let wrong = GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap();
    assert!(extract_decoder(&mut runtime, 0, 5, &fresh, &wrong, None).is_err());
    assert!(extract_decoder(&mut runtime, 0, 4, &fresh, &consumer(), None).is_err());
    // Both mismatches preceded the unique cold take.
    let sequence = extract_decoder(&mut runtime, 0, 5, &fresh, &consumer(), None).unwrap();
    drop(sequence);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn foreign_combined_source_stays_cold_and_original_error_retains_only_its_own_bank() {
    let (mut first, old) = runtime(Mode {
        defer_bank: true,
        explicit_source: true,
        ..Mode::default()
    });
    let a = source(3, &["é"]);
    let initial = extract_decoder(&mut first, 0, 3, &a, &consumer(), None).unwrap_err();
    let original = old.borrow().active.as_ref().unwrap().clone();
    let bank = old.borrow_mut().pending_bank.take().unwrap();
    let (mut second, new) = runtime(Mode {
        reject_decoder_attachment: true,
        ..Mode::default()
    });
    new.borrow_mut().foreign = Some(original.clone());
    new.borrow_mut().pending_bank = Some(bank);
    let b = source(3, &["é"]);
    let error = extract_decoder(&mut second, 2, 3, &b, &consumer(), None).unwrap_err();
    assert_eq!(new.borrow().order, ["admit"]);
    assert_eq!(new.borrow().decoder_rejections_untaken, 1);
    drop(original);
    let (old_pool, held) = retire_request(&old);
    drop(first);
    let (new_pool, _) = retire_request(&new);
    drop(second);
    drop((a, b));
    assert_eq!(new_pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(old_pool.payload_used_bytes().unwrap(), held + 64);
    drop(error);
    assert_eq!(old_pool.payload_used_bytes().unwrap(), 0);
    drop(initial);
}
#[test]
fn original_plain_dormant_cancel_and_failed_preparation_keep_consuming_custody() {
    for maximum in [0, 3] {
        let (mut runtime, state) = runtime(Mode::default());
        let input = source(maximum, &[" é"]);
        let mut sequence =
            extract_decoder(&mut runtime, 2, maximum, &input, &consumer(), None).unwrap();
        if maximum == 0 {
            assert_eq!(sequence.finish_plain_text().unwrap(), "");
        } else {
            assert!(sequence.cancel());
        }
        let (pool, held) = retire_request(&state);
        drop(runtime);
        drop(input);
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        let tokens = sequence.into_token_ids();
        assert!(tokens.is_empty());
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        drop(tokens);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let input = source(3, &["é"]);
    let sequence = extract_decoder(&mut runtime, 1, 3, &input, &consumer(), None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let (pool, held) = retire_request(&state);
    drop(runtime);
    drop(input);
    let error = sequence.prepare_storage().unwrap_err();
    assert!(error.cause().source().unwrap().is::<WorkingMemoryError>());
    assert_eq!(pool.payload_used_bytes().unwrap(), held + 64);
    let error = error.into_sequence().prepare_storage().unwrap_err();
    assert_eq!(pool.payload_used_bytes().unwrap(), held + 64);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
