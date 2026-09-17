use super::*;
use crate::working_memory::OriginalGenerationDecoderInput;
use eredu_core::{GenerationDecoderError, GenerationSequenceConsumerLayout, TextGenerationDriver};
use eredu_text::{decoder_storage::PreparedDecodeSource, tokenizer::Tokenizer};

fn decoder(maximum: usize, skip: bool) -> OriginalGenerationDecoderInput {
    let tokenizer = Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":7,"content":"<stop>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"é":1,"":2,"<stop>":7,"[UNK]":8},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap();
    let source = PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap();
    OriginalGenerationDecoderInput::new(source, maximum, skip).unwrap()
}
fn consumer() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_types::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
}
pub(super) fn extract_decoder(
    runtime: &mut ModelRuntime<Backend>,
    route: usize,
    maximum: usize,
    source: &dyn eredu_core::GenerationDecoderInput,
    consumer: &GenerationSequenceConsumerLayout,
    options: Option<TextPreparationOptions>,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let request = GenerationSequenceRequest::new(maximum, &[7])
        .with_consumer(consumer)
        .with_decoder(source);
    let input = TextGenerationInput::TokenIds(vec![2]);
    let map = |e| match e {
        eredu_core::ControlledTextGenerationError::Preparation(e) => e,
        _ => panic!("unexpected original setup failure"),
    };
    match route {
        0 => {
            let mut run = TextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(maximum),
                TokenFilter::All,
                options,
                request,
            )?;
            Ok(run.take_prepared_sequence().unwrap())
        }
        1 => {
            let mut run = ControlledTextGeneration::from_input_with_sequence(
                runtime,
                input,
                config(maximum),
                Controller,
                options,
                request,
            )
            .map_err(map)?;
            Ok(run.take_prepared_sequence().unwrap())
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut run = driver
                .start_input_with_sequence(input, config(maximum), Controller, options, request)
                .map_err(map)?;
            Ok(driver.take_prepared_sequence(&mut run).unwrap().unwrap())
        }
    }
}
#[test]
fn actual_original_decoder_source_joins_all_core_routes_and_optional_capture_once() {
    for route in 0..3 {
        for capture in [false, true] {
            for skip in [false, true] {
                let (mut runtime, state) = runtime(Mode {
                    capture,
                    audit_decoder: true,
                    ..Mode::default()
                });
                let source = decoder(5, skip);
                let foreign = decoder(5, skip); // equal vocabulary/N/policy, distinct physical source
                let opposite_policy = decoder(5, !skip);
                let layout = consumer();
                let options = capture.then(|| TextPreparationOptions {
                    interventions: None, capture: Some(capture_source_for_geometry(InferenceGeometry {
                        max_output_tokens: 5,
                        ..geometry()
                    })),
                });
                let mut sequence =
                    extract_decoder(&mut runtime, route, 5, &source, &layout, options).unwrap();
                assert_eq!(state.borrow().decoder_takes, 1);
                assert!(sequence.matches_decoder_input(Some(&source)));
                assert!(!sequence.matches_decoder_input(Some(&foreign)));
                assert!(!sequence.matches_decoder_input(Some(&opposite_policy)));
                assert!(!sequence.matches_decoder_input(None));
                assert_eq!(sequence.consumer_layout(), Some(&layout));
                assert_eq!(
                    sequence.decode_token(0),
                    Err(GenerationDecoderError::Unprepared)
                );
                assert_eq!(
                    state.borrow().order,
                    ["admit", "bind", "extract", "prompt", "sampling"]
                );
                sequence = sequence.prepare_storage().unwrap();
                let address = sequence.tokens().as_ptr();
                let mut output = String::new();
                for id in [0, 1, 99, 2, 7] {
                    if let Some(text) = sequence.decode_token(id).unwrap() {
                        output.push_str(text);
                    }
                    sequence
                        .commit(id, TokenTerminalSignals::default())
                        .unwrap();
                }
                assert_eq!(
                    output,
                    if skip {
                        "hello é "
                    } else {
                        "hello é  <stop>"
                    }
                );
                sequence.finish_decoder().unwrap();
                assert_eq!(sequence.tokens(), [0, 1, 99, 2, 7]);
                let (pool, held) = retire_request(&state);
                drop(runtime);
                assert_eq!(pool.used_bytes().unwrap(), held);
                // The taken input is now a nonowning immutable header; the two
                // rejected comparison inputs still own their separate cold
                // sources. None owns the provider's transferred source.
                drop((source, foreign, opposite_policy));
                assert_eq!(pool.used_bytes().unwrap(), held);
                let tokens = sequence.into_token_ids();
                assert_eq!(tokens.as_ptr(), address);
                assert_eq!(pool.used_bytes().unwrap(), held);
                let mut iter = tokens.clone().into_iter();
                drop(tokens);
                assert_eq!(iter.next(), Some(0));
                assert_eq!(pool.used_bytes().unwrap(), held);
                drop(iter);
                assert_eq!(pool.used_bytes().unwrap(), 0);
            }
        }
    }
}
#[test]
fn exact_minus_one_source_replay_and_mismatched_n_do_not_prepare_destinations() {
    let source = decoder(5, false);
    let (mut runtime, state) = runtime(Mode {
        short: true,
        ..Mode::default()
    });
    let layout = consumer();
    let failure = extract_decoder(&mut runtime, 0, 5, &source, &layout, None).unwrap_err();
    assert_eq!(state.borrow().decoder_takes, 1);
    assert_eq!(state.borrow().order, ["admit"]);
    assert_eq!(state.borrow().votes, [(Stage::Admission, Status::Failed)]);
    assert_eq!(state.borrow().pool.used_bytes().unwrap(), 64);
    drop(failure);
    state.borrow_mut().mode.short = false;
    for _ in 0..16 {
        let failure = extract_decoder(&mut runtime, 0, 5, &source, &layout, None).unwrap_err();
        assert_eq!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
            Some(&eredu_core::GenerationSequenceBankRejection::Unavailable)
        );
    }
    assert_eq!(state.borrow().decoder_takes, 1);
    assert_eq!(state.borrow().pool.used_bytes().unwrap(), 64);
    let wrong_n = decoder(4, false);
    assert!(extract_decoder(&mut runtime, 0, 5, &wrong_n, &layout, None).is_err());
    // Wrong N rejects before the take; its actual matching request still works.
    let sequence = extract_decoder(&mut runtime, 0, 4, &wrong_n, &layout, None).unwrap();
    drop(sequence);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn fenced_decoder_preparation_error_retains_actual_source_and_original_registered_roots() {
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let source = decoder(3, true);
    let sequence = extract_decoder(&mut runtime, 1, 3, &source, &consumer(), None).unwrap();
    // Genuine startup completed; closing the actual source run invalidates local
    // provider preparation without synthesizing a new original bank or guard.
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
    drop(source);
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    assert!(failure.cause().source().unwrap().is::<WorkingMemoryError>());
    let sequence = failure.into_sequence();
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), held + 64);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn dormant_zero_and_cancelled_decoder_sources_freeze_without_destinations() {
    for maximum in [0, 3] {
        let (mut runtime, state) = runtime(Mode::default());
        let source = decoder(maximum, true);
        let mut sequence =
            extract_decoder(&mut runtime, 2, maximum, &source, &consumer(), None).unwrap();
        if maximum == 0 {
            sequence.finish_decoder().unwrap();
        } else {
            assert!(sequence.cancel());
        }
        let (pool, held) = retire_request(&state);
        drop(runtime);
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), held);
        let tokens = sequence.into_token_ids();
        assert!(tokens.is_empty());
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop(tokens);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn foreign_genuine_decoder_claim_keeps_the_original_source_and_bank_in_one_error() {
    for rejected_attachment in [false, true] {
        let (mut first, old) = runtime(Mode {
            defer_bank: true,
            explicit_source: true,
            ..Mode::default()
        });
        let a = decoder(3, false);
        let initial = extract_decoder(&mut first, 0, 3, &a, &consumer(), None).unwrap_err();
        let original = old.borrow().active.as_ref().unwrap().clone();
        let bank = old.borrow_mut().pending_bank.take().unwrap();
        let (mut second, new) = runtime(Mode {
            reject_decoder_attachment: rejected_attachment,
            ..Mode::default()
        });
        new.borrow_mut().foreign = Some(original.clone());
        new.borrow_mut().pending_bank = Some(bank);
        let b = decoder(3, false); // equal actual payload/policy; different source/claim
        let error = extract_decoder(&mut second, 2, 3, &b, &consumer(), None).unwrap_err();
        assert_eq!(error.kind(), eredu_core::BackendFailureKind::InvalidSession);
        assert_eq!(
            new.borrow().order,
            if rejected_attachment {
                vec!["admit"]
            } else {
                vec!["admit", "bind", "extract"]
            }
        );
        assert_eq!(
            new.borrow().decoder_rejections_untaken,
            usize::from(rejected_attachment)
        );
        assert!(old.borrow().pending_bank.is_none());
        drop(original);
        let (old_pool, held) = retire_request(&old);
        drop(first);
        let (new_pool, _) = retire_request(&new);
        drop(second);
        drop((a, b));
        assert_eq!(new_pool.used_bytes().unwrap(), 0);
        assert_eq!(old_pool.used_bytes().unwrap(), held + 64);
        let replays = std::array::from_fn::<_, 64, _>(|_| {
            eredu_core::GenerationSequenceBankRejection::Unavailable.into_backend_failure()
        });
        drop(error);
        assert_eq!(old_pool.used_bytes().unwrap(), 0);
        assert!(replays.iter().all(|e| e
            .source()
            .unwrap()
            .is::<eredu_core::GenerationSequenceBankRejection>()));
        drop((replays, initial));
    }
}

mod loaded;

mod shared_plain;

mod aggregate;
