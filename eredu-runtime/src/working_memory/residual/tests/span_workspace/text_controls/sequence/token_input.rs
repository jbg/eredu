use super::*;
use crate::working_memory::{OriginalTokenInputFailure, OriginalTokenInputLayout};
use eredu_core::{TextGenerationDriver, TokenIdsInputPlan, TokenInputRejection};

fn extract_input(
    runtime: &mut ModelRuntime<Backend>,
    route: usize,
    ids: &[u32],
    maximum: usize,
    options: Option<TextPreparationOptions>,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let request = GenerationSequenceRequest::new(maximum, &[99]);
    extract_input_request(runtime, route, ids, maximum, options, request)
}
fn extract_input_request(
    runtime: &mut ModelRuntime<Backend>,
    route: usize,
    ids: &[u32],
    maximum: usize,
    options: Option<TextPreparationOptions>,
    request: GenerationSequenceRequest<'_>,
) -> Result<RetainedGenerationSequence, eredu_core::BackendFailure> {
    let input = TokenIdsInputPlan::new(ids).unwrap();
    let map = |e| match e {
        eredu_core::ControlledTextGenerationError::Preparation(e) => e,
        _ => panic!("unexpected input setup failure"),
    };
    match route {
        0 => {
            let mut run = TextGeneration::from_token_ids_with_sequence(
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
            let mut run = ControlledTextGeneration::from_token_ids_with_sequence(
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
                .start_token_ids_with_sequence(input, config(maximum), Controller, options, request)
                .map_err(map)?;
            Ok(driver.take_prepared_sequence(&mut run).unwrap().unwrap())
        }
    }
}
#[test]
fn original_input_real_claim_all_routes_capture_zero_and_source_borrow_end() {
    for route in 0..3 {
        for capture in [false, true] {
            // Zero exercises the low-level terminal configuration only. Public
            // capture admission requires a positive prediction allowance.
            let maximums: &[usize] = if capture { &[3] } else { &[0, 3] };
            for &maximum in maximums {
                let (mut runtime, state) = runtime(Mode {
                    capture,
                    ..Mode::default()
                });
                let mut ids = vec![3, 11, 2, 17, 5];
                let g = InferenceGeometry {
                    input_positions: 5,
                    max_output_tokens: maximum as u64,
                    ..geometry()
                };
                let options = capture.then(|| TextPreparationOptions {
                    interventions: None,
                    capture: Some(capture_source_for_geometry(g)),
                });
                let sequence = extract_input(&mut runtime, route, &ids, maximum, options).unwrap();
                ids.fill(99);
                assert_eq!(state.borrow().input_observed, [3, 11, 2, 17, 5]);
                assert_eq!(state.borrow().input_builds, 1);
                assert_eq!(
                    state.borrow().order,
                    ["admit", "bind", "extract", "original prompt", "sampling"]
                );
                assert_eq!(
                    state.borrow().held,
                    state.borrow().p + 51 + state.borrow().r + state.borrow().input_bytes
                );
                let original = state.borrow().active.as_ref().unwrap().clone();
                let rejected: Vec<_> = (0..32).map(|_| {
            assert!(original.owner.borrow_mut().take_token_input_bank().is_none());
            <Backend as eredu_core::TextGenerationBackend>::prepare_original_text_prompt_admitted(runtime.backend(), &original).unwrap_err()
        }).collect();
                assert!(rejected.iter().all(|e| {
                    e.source().unwrap().downcast_ref::<TokenInputRejection>()
                        == Some(&TokenInputRejection::Unavailable)
                }));
                drop(original);
                let (pool, held) = retire_request(&state);
                drop(runtime);
                assert_eq!(pool.used_bytes().unwrap(), held);
                drop(sequence);
                assert_eq!(
                    pool.used_bytes().unwrap(),
                    0,
                    "replay failures carry no cloned custody"
                );
                drop(rejected);
            }
        }
    }
}
#[test]
fn original_input_exact_short_and_spare_source_capacity_do_not_change_destination_charge() {
    let mut charges = Vec::new();
    for spare in [0, 4096] {
        let mut ids = Vec::with_capacity(5 + spare);
        ids.extend([3, 11, 2, 17, 5]);
        let (mut runtime, state) = runtime(Mode {
            short: true,
            ..Mode::default()
        });
        assert!(extract_input(&mut runtime, 0, &ids, 3, None).is_err());
        assert_eq!(state.borrow().input_builds, 0);
        assert_eq!(state.borrow().order, ["admit"]);
        assert_eq!(state.borrow().pool.used_bytes().unwrap(), 64);
        state.borrow_mut().mode.short = false;
        let sequence = extract_input(&mut runtime, 0, &ids, 3, None).unwrap();
        charges.push((state.borrow().held, state.borrow().input_bytes));
        assert_eq!(
            state.borrow().input_bytes,
            OriginalTokenInputLayout::prepare(&TokenIdsInputPlan::new(&ids).unwrap())
                .unwrap()
                .protected_bytes()
        );
        drop(sequence);
        let (pool, _) = retire_request(&state);
        drop(runtime);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    assert_eq!(charges[0], charges[1]);
    let small = OriginalTokenInputLayout::prepare(&TokenIdsInputPlan::new(&[1]).unwrap()).unwrap();
    let big =
        OriginalTokenInputLayout::prepare(&TokenIdsInputPlan::new(&[1, 2, 3]).unwrap()).unwrap();
    assert_eq!(big.protected_bytes() - small.protected_bytes(), 8);
}
#[test]
fn consumed_foreign_input_bank_error_retains_only_original_account_not_new_source() {
    let (mut old, a) = runtime(Mode {
        defer_input: true,
        ..Mode::default()
    });
    let ids = [3, 11, 2];
    assert!(extract_input(&mut old, 0, &ids, 3, None).is_err());
    let bank = a.borrow_mut().pending_input.take().unwrap();
    let (mut new, b) = runtime(Mode::default());
    b.borrow_mut().pending_input = Some(bank);
    let error = extract_input(&mut new, 1, &ids, 3, None).unwrap_err();
    assert!(
        error
            .source()
            .unwrap()
            .downcast_ref::<OriginalTokenInputFailure>()
            .is_some()
    );
    assert_eq!(b.borrow().input_builds, 0);
    let untouched = b
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .owner
        .borrow_mut()
        .take_token_input_bank()
        .expect("foreign-bank failure leaves the new request's bank unspent");
    drop(untouched);
    let (pool_a, held) = retire_request(&a);
    let (pool_b, _) = retire_request(&b);
    drop((old, new));
    assert_eq!(pool_b.used_bytes().unwrap(), 0);
    assert_eq!(pool_a.used_bytes().unwrap(), held);
    drop(error);
    assert_eq!(pool_a.used_bytes().unwrap(), 0);
}
#[test]
fn real_input_reserve_failure_and_fenced_failure_escape_with_original_custody() {
    for fault in [0, 1] {
        let (mut runtime, state) = runtime(Mode {
            fence_input: fault == 0,
            explicit_source: true,
            ..Mode::default()
        });
        let error =
            crate::working_memory::residual::span_workspace::token_input_fault(fault, || {
                extract_input(&mut runtime, 2, &[3, 11, 2], 3, None)
            })
            .unwrap_err();
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<OriginalTokenInputFailure>()
            .unwrap();
        if fault == 1 {
            assert_eq!(source.partial_tokens(), Some([].as_slice()));
            assert!(
                source
                    .source()
                    .unwrap()
                    .source()
                    .unwrap()
                    .is::<std::collections::TryReserveError>()
            );
        } else {
            assert!(source.partial_tokens().is_none());
        }
        assert_eq!(state.borrow().input_builds, 0);
        let (pool, held) = retire_request(&state);
        drop(runtime);
        assert!(pool.pin_registered_storage([(1u32, 64)]).is_ok());
        assert_eq!(pool.used_bytes().unwrap(), held + 64);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn input_postfill_unwind_spends_bank_and_retires_destination_before_original_hold() {
    let (mut runtime, state) = runtime(Mode::default());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::working_memory::residual::span_workspace::token_input_fault(2, || {
            extract_input(&mut runtime, 1, &[3, 11, 2], 3, None)
        })
    }));
    assert!(panic.is_err());
    assert!(
        state
            .borrow()
            .active
            .as_ref()
            .unwrap()
            .owner
            .borrow_mut()
            .take_token_input_bank()
            .is_none()
    );
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_input_is_borrowed_across_actual_adaptive_candidates_and_copied_once_after_acceptance() {
    let (mut runtime, state) = runtime(Mode {
        input_retry: true,
        ..Mode::default()
    });
    let ids = [3, 11, 2, 17, 5];
    let sequence = extract_input(&mut runtime, 0, &ids, 3, None).unwrap();
    assert!(state.borrow().input_candidates > 1);
    assert_eq!(state.borrow().input_builds, 1);
    assert_eq!(state.borrow().input_observed, ids);
    assert!(
        state
            .borrow()
            .active
            .as_ref()
            .unwrap()
            .request
            .request()
            .geometry()
            .prefill_chunk_positions
            < 5
    );
    drop(sequence);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_encoded_ids_lend_once_to_real_input_all_routes_without_adopting_e() {
    original_encoded_input(0);
}
#[test]
fn original_regex_encoded_ids_use_same_real_input_all_routes_and_destination_once() {
    original_encoded_input(1);
}
#[test]
fn original_implicit_encoded_ids_use_same_real_input_all_routes_and_destination_once() {
    original_encoded_input(2);
}
#[test]
fn original_template_ids_and_decoder_share_one_real_admission_on_all_drivers() {
    original_encoded_input(3);
}
#[test]
fn original_two_phase_ids_and_decoder_keep_same_input_authority_on_all_drivers() {
    original_encoded_input(4);
}
#[test]
fn original_nfc_empty_affix_ids_lend_to_actual_input_and_decoder_on_all_drivers() {
    original_encoded_input(5);
}
fn original_encoded_input(profile: u8) {
    let regex = profile != 0;
    use eredu_text::tokenizer_storage::TokenizerPlan;
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":3,"b":11,"c":2,"d":17,"e":5},"merges":[]}}"#;
    let json = if profile == 2 {
        json.replace("\"pre_tokenizer\":null",concat!("\"pre_tokenizer\":",r###"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"###))
    } else if regex {
        json.replace("\"pre_tokenizer\":null", concat!("\"pre_tokenizer\":", r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###))
    } else {
        json.to_owned()
    };
    let json = if profile >= 3 {
        json.replace("\"post_processor\":null", r#""post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"start","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"start":{"id":"start","ids":[3],"tokens":["a"]}}}"#)
    } else {
        json
    };
    let json = if profile == 4 {
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["pre_tokenizer"]=serde_json::from_str(r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###).unwrap();
        value["added_tokens"]=serde_json::from_str(r#"[{"id":11,"content":"b","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},{"id":2,"content":"c","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}]"#).unwrap();
        value.to_string()
    } else {
        json
    };
    let json = if profile == 5 {
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["normalizer"] = serde_json::json!({"type":"NFC"});
        value["model"]["continuing_subword_prefix"] = serde_json::json!("");
        value["model"]["end_of_word_suffix"] = serde_json::json!("");
        value["model"]["vocab"].as_object_mut().unwrap().remove("e");
        value["model"]["vocab"]["Ã"] = serde_json::json!(19);
        value["model"]["vocab"]["©"] = serde_json::json!(20);
        value["model"]["vocab"]["Ã©"] = serde_json::json!(5);
        value["model"]["merges"] = serde_json::json!([["Ã", "©"]]);
        value.to_string()
    } else {
        json
    };
    for route in 0..3 {
        let (mut runtime, state) = harness::runtime_with_pool(
            Mode {
                input_retry: true,
                audit_decoder: profile >= 3,
                ..Mode::default()
            },
            WorkingMemoryPool::new(if regex { u64::MAX } else { 1_000_000 }, 0).unwrap(),
        );
        let pool = state.borrow().pool.clone();
        let source = pool
            .compile_tokenizer(TokenizerPlan::prepare_json(json.as_bytes()).unwrap())
            .unwrap();
        let c = source.original_bytes();
        let encoded = pool
            .encode_tokenizer_ids(
                &source,
                if profile == 5 {
                    "bcde\u{301}"
                } else if profile >= 3 {
                    "bcde"
                } else {
                    "abcde"
                },
                true,
            )
            .unwrap();
        assert_eq!(encoded.ids(), [3, 11, 2, 17, 5]);
        let e = encoded.original_bytes();
        let ptr = encoded.ids().as_ptr();
        state.borrow_mut().loaded_bytes = c + e;
        let header = (profile >= 3).then(|| {
            crate::working_memory::AggregateGenerationDecoderInput::new(&source, 3, false).unwrap()
        });
        let sequence = if let Some(header) = &header {
            let layout = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
                RetainedGenerationSequence,
                WorkingMemoryError,
                WorkingMemoryError,
            >()
            .unwrap();
            let request = GenerationSequenceRequest::new(3, &[99])
                .with_consumer(&layout)
                .with_decoder(header);
            let mut sequence =
                extract_input_request(&mut runtime, route, encoded.ids(), 3, None, request)
                    .unwrap();
            assert!(sequence.matches_decoder_input(Some(header)));
            sequence = sequence.prepare_storage().unwrap();
            let mut text = String::new();
            for id in [3, 11, 2] {
                if let Some(piece) = sequence.decode_token(id).unwrap() {
                    text.push_str(piece);
                }
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
            }
            sequence.finish_decoder().unwrap();
            assert_eq!(text, "abc");
            assert_eq!(state.borrow().decoder_takes, 1);
            sequence
        } else {
            extract_input(&mut runtime, route, encoded.ids(), 3, None).unwrap()
        };
        assert_eq!(state.borrow().input_observed, [3, 11, 2, 17, 5]);
        assert_eq!(state.borrow().input_builds, 1);
        assert!(state.borrow().input_candidates > 1);
        assert_ne!(state.borrow().input_address, ptr as usize);
        assert_eq!(encoded.ids().as_ptr(), ptr);
        assert!(encoded.matches_source(&source));
        assert_eq!(
            state.borrow().order,
            ["admit", "bind", "extract", "original prompt", "sampling"]
        );
        // The existing genuine I owns its separate copy; E is neither adopted nor requoted.
        drop(encoded);
        drop(header);
        let (_, held) = retire_request(&state);
        drop(runtime);
        assert_eq!(pool.used_bytes().unwrap(), c + held);
        drop(source);
        assert_eq!(
            pool.used_bytes().unwrap(),
            held + if profile >= 3 { c } else { 0 }
        );
        drop(sequence);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
