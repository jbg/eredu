use super::*;
use crate::working_memory::{AggregateGenerationDecoderInput, OriginalTokenizer};
use eredu_text::tokenizer_storage::TokenizerPlan;
const AGGREGATE_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[{"id":4,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2,"Ġ":3},"merges":[["h","i"]]}}"#;
fn aggregate(pool: &WorkingMemoryPool) -> OriginalTokenizer {
    pool.compile_tokenizer(TokenizerPlan::prepare_json(AGGREGATE_JSON.as_bytes()).unwrap())
        .unwrap()
}
fn aggregate_file(pool: &WorkingMemoryPool) -> OriginalTokenizer {
    use std::io::Write as _;
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(AGGREGATE_JSON.as_bytes()).unwrap();
    let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file).unwrap();
    pool.compile_tokenizer_file(read).unwrap()
}

#[test]
fn two_successive_original_requests_share_one_source_and_freeze_only_request_storage() {
    two_successive_requests(false, 0);
}
#[cfg(unix)]
#[test]
fn file_ingress_source_uses_same_ordinary_controlled_detached_provider_and_no_i_tail() {
    two_successive_requests(true, 0);
}
#[cfg(unix)]
#[test]
fn file_regex_source_uses_same_original_provider_across_two_requests_and_all_drivers() {
    two_successive_requests(true, 1);
}
#[cfg(unix)]
#[test]
fn file_implicit_source_uses_same_original_provider_across_two_requests_and_all_drivers() {
    two_successive_requests(true, 2);
}
#[cfg(unix)]
#[test]
fn file_template_source_uses_the_same_provider_for_two_requests_and_all_drivers() {
    two_successive_requests(true, 3);
}
#[cfg(unix)]
#[test]
fn file_two_phase_source_preserves_original_c_e_and_two_request_provider_lifetimes() {
    two_successive_requests(true, 4);
}
#[cfg(unix)]
#[test]
fn file_nfc_source_keeps_actual_e_and_two_request_decoder_lifetimes_on_all_drivers() {
    two_successive_requests(true, 5);
}
fn two_successive_requests(from_file: bool, profile: u8) {
    let regex = profile != 0;
    for route in 0..3 {
        let (mut runtime, state) = harness::runtime_with_pool(
            Mode {
                audit_decoder: true,
                ..Mode::default()
            },
            WorkingMemoryPool::new(if regex { u64::MAX } else { 1_000_000 }, 0).unwrap(),
        );
        let pool = state.borrow().pool.clone();
        let source = if regex {
            use std::io::Write as _;
            let json=AGGREGATE_JSON.replace("\"pre_tokenizer\":null",concat!("\"pre_tokenizer\":",r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###));
            let json = if profile == 2 {
                AGGREGATE_JSON.replace("\"pre_tokenizer\":null",concat!("\"pre_tokenizer\":",r###"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"###))
            } else {
                json
            };
            let json = if profile >= 3 {
                json.replace("\"post_processor\":null", r#""post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"start","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"start":{"id":"start","ids":[2],"tokens":["hi"]}}}"#)
            } else {
                json
            };
            let json = if profile == 4 {
                let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
                value["pre_tokenizer"]=serde_json::from_str(r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###).unwrap();
                value["added_tokens"].as_array_mut().unwrap().push(serde_json::from_str(r#"{"id":2,"content":"hi","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}"#).unwrap());
                value.to_string()
            } else {
                json
            };
            let json = if profile == 5 {
                let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
                value["normalizer"] = serde_json::json!({"type":"NFC"});
                value["model"]["continuing_subword_prefix"] = serde_json::json!("");
                value["model"]["end_of_word_suffix"] = serde_json::json!("");
                // The decoder fixture uses ID4 as its raw special boundary.
                value["model"]["vocab"]["<S>"] = serde_json::json!(4);
                value["model"]["vocab"]["Ã"] = serde_json::json!(5);
                value["model"]["vocab"]["©"] = serde_json::json!(6);
                value.to_string()
            } else {
                json
            };
            let mut file = tempfile::tempfile().unwrap();
            file.write_all(json.as_bytes()).unwrap();
            let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file).unwrap();
            pool.compile_tokenizer_file(read).unwrap()
        } else if from_file {
            aggregate_file(&pool)
        } else {
            aggregate(&pool)
        };
        if profile == 5 {
            assert_eq!(source.token_id("<S>"), Some(4));
            assert_eq!(source.spelling(4), Some("<S>"));
            assert!(source.is_special("<S>"));
            assert_eq!(source.token_id("Ã"), Some(5));
            assert_eq!(source.token_id("©"), Some(6));
        }
        let cold = source.original_bytes();
        if profile >= 4 {
            // The harness already owns its 64-byte model root before E.
            let baseline = pool.used_bytes().unwrap();
            assert_eq!(baseline, cold + 64);
            let ids = pool
                .encode_tokenizer_ids(
                    &source,
                    if profile == 5 {
                        "hi<S> e\u{301}"
                    } else {
                        "hi<S> hi"
                    },
                    true,
                )
                .unwrap();
            assert_eq!(
                ids.ids(),
                if profile == 5 {
                    &[2, 2, 4, 3, 5, 6][..]
                } else {
                    &[2, 2, 4, 3, 2][..]
                }
            );
            if profile == 5 {
                assert_eq!(ids.normalization_capacities(), [8, 8, 9]);
            }
            assert!(ids.matches_source(&source));
            assert_eq!(pool.used_bytes().unwrap(), baseline + ids.original_bytes());
            drop(ids);
            assert_eq!(pool.used_bytes().unwrap(), baseline);
        }

        state.borrow_mut().loaded_bytes = source.original_bytes();
        let mut previous_r = None;
        for run in 0..2 {
            let input = AggregateGenerationDecoderInput::new(&source, 5, true).unwrap();
            assert!(input.source().same_source(&source));
            let foreign_header = AggregateGenerationDecoderInput::new(&source, 5, true).unwrap();
            let opposite = AggregateGenerationDecoderInput::new(&source, 5, false).unwrap();
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
            for id in [2, 3, 4, 0, 1] {
                if let Some(text) = sequence.decode_token(id).unwrap() {
                    output.push_str(text);
                }
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
            }
            assert_eq!(output, "hi hi");
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
            assert_eq!(tokens.as_ref(), [2, 3, 4, 0, 1]);
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
    let source = aggregate(&pool);
    state.borrow_mut().loaded_bytes = source.original_bytes();
    let cold = source.original_bytes();
    assert!(matches!(
        AggregateGenerationDecoderInput::new(&source, usize::MAX, true),
        Err(eredu_text::decoder_storage::DecodeStorageError::Overflow)
    ));
    let input = AggregateGenerationDecoderInput::new(&source, 3, true).unwrap();
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
    let next = AggregateGenerationDecoderInput::new(&source, 3, true).unwrap();
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
    let source = aggregate(&pool);
    let cold = source.original_bytes();
    state.borrow_mut().loaded_bytes = source.original_bytes();
    let input = AggregateGenerationDecoderInput::new(&source, 3, true).unwrap();
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
    let source = aggregate(&pool);
    let input = AggregateGenerationDecoderInput::new(&source, 3, true).unwrap();
    let cold = source.original_bytes();
    let barrier = std::sync::Barrier::new(2);
    let outcomes = std::thread::scope(|threads| {
        let mut workers = Vec::new();
        for _ in 0..2 {
            let (pool, input, barrier) = (pool.clone(), &input, &barrier);
            workers.push(threads.spawn(move || {
                // No fallible fixture construction precedes this rendezvous.
                barrier.wait();
                let (mut runtime, state) = harness::runtime_with_pool(Mode::default(), pool);
                state.borrow_mut().loaded_bytes = input.source().original_bytes();
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

#[test]
fn aggregate_plain_requests_bind_actual_stop_overrides_and_empty_cancel_without_refunding_source() {
    use eredu_core::{GenerationDecoderInput, GenerationDecoderOutput};
    use eredu_text::stop_storage::StopCompilePlan;
    for maximum in [0, 3] {
        let (mut runtime, state) = runtime(Mode::default());
        let pool = state.borrow().pool.clone();
        let source = aggregate(&pool);
        let foreign_pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let foreign = foreign_pool
            .compile_stop_source(StopCompilePlan::prepare_refs(&["hi"]).unwrap())
            .unwrap();
        assert!(matches!(
            AggregateGenerationDecoderInput::new_plain_text(&source, &foreign, maximum, true),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        for run in 0..2 {
            let policy: &[&str] = if run == 0 { &["hi"] } else { &[] };
            let stops = pool
                .compile_stop_source(StopCompilePlan::prepare_refs(policy).unwrap())
                .unwrap();
            let cold = source.original_bytes() + stops.original_bytes();
            let input =
                AggregateGenerationDecoderInput::new_plain_text(&source, &stops, maximum, true)
                    .unwrap();
            state.borrow_mut().loaded_bytes = cold;
            assert_eq!(input.output_kind(), GenerationDecoderOutput::PlainText);
            assert!(input.stop_source().is_some());
            let layout = consumer().with_plain_text_output().unwrap();
            let mut sequence =
                extract_decoder(&mut runtime, run, maximum, &input, &layout, None).unwrap();
            if maximum == 0 {
                assert_eq!(sequence.finish_plain_text().unwrap(), "");
            } else {
                sequence = sequence.prepare_storage().unwrap();
                let output = sequence.project_plain_text(2).unwrap();
                assert_eq!(
                    (output.visible, output.stop_matched),
                    if run == 0 { ("", true) } else { ("hi", false) }
                );
                sequence
                    .commit(
                        2,
                        TokenTerminalSignals {
                            stop_sequence: run == 0,
                            ..TokenTerminalSignals::default()
                        },
                    )
                    .unwrap();
                if run == 1 {
                    assert!(sequence.cancel());
                }
            }
            let (_, held) = retire_request(&state);
            drop((input, stops));
            assert_eq!(pool.used_bytes().unwrap(), cold + held);
            let tokens = sequence.into_token_ids();
            assert_eq!(pool.used_bytes().unwrap(), source.original_bytes() + held);
            drop(tokens);
            assert_eq!(pool.used_bytes().unwrap(), source.original_bytes());
            if run == 0 {
                state.borrow_mut().root = Some(pool.register_storage([(1u32, 64)]).unwrap());
            }
        }
        drop((runtime, source, foreign));
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
    }
}

mod terminal_text;
