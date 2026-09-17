use super::*;
use eredu_core::{GenerationPlainTextOutput, GenerationSequenceConsumerLayout, GenerationTiming};
use eredu_text::stop_storage::StopCompilePlan;
fn terminal() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap()
}
#[test]
fn terminal_text_uses_same_r_payload_and_stop_byte_conservation_on_all_core_routes() {
    for route in 0..3 {
        for policy in [&[][..], &["hiz"][..], &["hi h"][..], &[" x"][..]] {
            let (mut runtime, state) = runtime(Mode::default());
            let pool = state.borrow().pool.clone();
            let source = aggregate(&pool);
            let stops = pool
                .compile_stop_source(StopCompilePlan::prepare_refs(policy).unwrap())
                .unwrap();
            let cold = source.original_bytes() + stops.original_bytes();
            state.borrow_mut().loaded_bytes = cold;
            let input =
                AggregateGenerationDecoderInput::new_plain_text(&source, &stops, 5, true).unwrap();
            let mut sequence = extract_decoder(&mut runtime, route, 5, &input, &terminal(), None)
                .unwrap()
                .prepare_storage()
                .unwrap();
            let ids_address = sequence.tokens().as_ptr();
            let mut visible = String::new();
            for id in [2, 3, 4, 0, 1] {
                let projected = sequence.project_plain_text(id).unwrap();
                visible.push_str(projected.visible);
                let matched = projected.stop_matched;
                sequence
                    .commit(
                        id,
                        TokenTerminalSignals {
                            stop_sequence: matched,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                if matched {
                    break;
                }
            }
            visible.push_str(sequence.finish_plain_text().unwrap());
            assert_eq!(
                sequence.finish_plain_text().unwrap(),
                "",
                "finish cannot duplicate lookbehind"
            );
            assert_eq!(visible, if policy == ["hi h"] { "" } else { "hi hi" });
            let (_, held) = retire_request(&state);
            let tokens = sequence.into_token_ids();
            assert_eq!(tokens.as_ptr(), ids_address);
            let text = tokens.terminal_text().unwrap();
            let text_address = text.as_str().as_ptr();
            let extra = text.clone();
            drop((runtime, input, source, stops));
            assert_eq!(pool.used_bytes().unwrap(), held);
            assert_eq!(text.as_str(), visible);
            let mut iterator = tokens.into_iter();
            assert_eq!(iterator.next(), Some(2));
            drop(text);
            assert_eq!(extra.as_str().as_ptr(), text_address);
            assert_eq!(pool.used_bytes().unwrap(), held);
            drop(extra);
            assert_eq!(pool.used_bytes().unwrap(), held);
            drop(iterator);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn terminal_target_reserve_is_real_partial_failure_and_repeated_prepare_cannot_retry() {
    let (mut runtime, state) = runtime(Mode {
        fail_terminal_reserve: true,
        ..Mode::default()
    });
    let pool = state.borrow().pool.clone();
    let source = aggregate(&pool);
    let stops = pool
        .compile_stop_source(StopCompilePlan::prepare_refs(&["hiz"]).unwrap())
        .unwrap();
    let cold = source.original_bytes() + stops.original_bytes();
    state.borrow_mut().loaded_bytes = cold;
    let input = AggregateGenerationDecoderInput::new_plain_text(&source, &stops, 5, true).unwrap();
    let sequence = extract_decoder(&mut runtime, 1, 5, &input, &terminal(), None).unwrap();
    let failed = sequence.prepare_storage().unwrap_err();
    assert!(failed
        .cause()
        .source()
        .unwrap()
        .is::<std::collections::TryReserveError>());
    // Injection reaches the terminal Vec's actual reserve after token slots,
    // decoder buffers and stop destination have all returned successful prepare.
    let (_, held) = retire_request(&state);
    drop((runtime, input, source, stops));
    assert_eq!(pool.used_bytes().unwrap(), cold + held);
    let failed = failed.into_sequence().prepare_storage().unwrap_err();
    assert!(
        !failed
            .cause()
            .source()
            .unwrap()
            .is::<std::collections::TryReserveError>(),
        "core terminal-attempt rejection, no second allocator call"
    );
    assert_eq!(pool.used_bytes().unwrap(), cold + held);
    drop(failed);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn terminal_empty_cancel_retains_zero_text_without_preparation_and_plain_mode_cannot_upgrade() {
    for terminal_mode in [false, true] {
        let (mut runtime, state) = runtime(Mode::default());
        let pool = state.borrow().pool.clone();
        let source = aggregate(&pool);
        let stops = pool
            .compile_stop_source(StopCompilePlan::prepare_refs(&[]).unwrap())
            .unwrap();
        state.borrow_mut().loaded_bytes = source.original_bytes() + stops.original_bytes();
        let input =
            AggregateGenerationDecoderInput::new_plain_text(&source, &stops, 3, true).unwrap();
        let layout = if terminal_mode {
            terminal()
        } else {
            consumer().with_plain_text_output().unwrap()
        };
        let mut sequence = extract_decoder(&mut runtime, 0, 3, &input, &layout, None).unwrap();
        assert!(sequence.cancel());
        let (_, held) = retire_request(&state);
        let ids = sequence.into_token_ids();
        drop((runtime, input, source, stops));
        assert_eq!(pool.used_bytes().unwrap(), held);
        assert_eq!(ids.terminal_text().is_some(), terminal_mode);
        if terminal_mode {
            let output = GenerationPlainTextOutput::from_retained(
                ids,
                eredu_core::FinishReason::Cancelled,
                GenerationTiming::default(),
            )
            .unwrap();
            assert!(output.text.as_str().is_empty());
            assert!(output.token_ids.is_empty());
            drop(output);
        } else {
            drop(ids);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn utf8_pending_flush_and_mismatch_exceed_one_call_capacity_but_fit_selected_terminal_bound() {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"Ã©":1,"z":2,"x":3},"merges":[]}}"#;
    for flush in [false, true] {
        let (mut runtime, state) = runtime(Mode::default());
        let pool = state.borrow().pool.clone();
        let source = pool
            .compile_tokenizer(TokenizerPlan::prepare_json(json.as_bytes()).unwrap())
            .unwrap();
        let maximum = 21;
        let l = eredu_text::decoder_storage::DecodeStreamLayout::for_source(
            source.decode_source(),
            maximum,
            false,
        )
        .unwrap()
        .text_capacity();
        let prefix = "a".repeat(20);
        let policy = format!("{}{}", prefix, if flush { "éz" } else { "x" });
        let stops = pool
            .compile_stop_source(StopCompilePlan::prepare_refs(&[&policy]).unwrap())
            .unwrap();
        state.borrow_mut().loaded_bytes = source.original_bytes() + stops.original_bytes();
        let input =
            AggregateGenerationDecoderInput::new_plain_text(&source, &stops, maximum, false)
                .unwrap();
        let mut sequence = extract_decoder(&mut runtime, 1, maximum, &input, &terminal(), None)
            .unwrap()
            .prepare_storage()
            .unwrap();
        for _ in 0..20 {
            assert_eq!(sequence.project_plain_text(0).unwrap().visible, "");
            sequence.commit(0, TokenTerminalSignals::default()).unwrap();
        }
        let visible = sequence.project_plain_text(1).unwrap().visible.to_owned();
        sequence.commit(1, TokenTerminalSignals::default()).unwrap();
        let pending = sequence.finish_plain_text().unwrap().to_owned();
        let expected = format!("{prefix}é");
        let one_call = eredu_text::decoder_storage::DecodeStreamLayout::for_source(
            source.decode_source(),
            1,
            false,
        )
        .unwrap()
        .text_capacity();
        assert!(expected.len() > one_call);
        // L is the full retained-history candidate capacity. M*L deliberately
        // keeps the general per-emission envelope, not a linear total proof.
        assert_eq!(if flush { &pending } else { &visible }, &expected);
        assert_eq!(if flush { &visible } else { &pending }, "");
        assert!(visible.len() + pending.len() <= maximum * l);
        let (_, held) = retire_request(&state);
        let ids = sequence.into_token_ids();
        assert_eq!(ids.terminal_text().unwrap().as_str(), expected);
        drop((runtime, input, stops, source));
        assert_eq!(pool.used_bytes().unwrap(), held);
        drop(ids);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
