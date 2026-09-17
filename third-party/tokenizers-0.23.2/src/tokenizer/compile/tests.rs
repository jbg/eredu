use super::*;
use crate::Model;
use serde_json::{Value, json};

fn added(content: &str, normalized: bool, special: bool) -> Value {
    json!({"id":9999,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":normalized,"special":special})
}
fn source() -> Value {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, c) in alphabet.into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(id));
    }
    vocab.insert("hi".into(), json!(256));
    let byte =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false});
    json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[added("<S>",false,true),added("HI",true,false)],
        "normalizer":null,
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},byte.clone()]},
        "post_processor":{"type":"Sequence","processors":[byte.clone()]},
        "decoder":{"type":"Sequence","decoders":[byte]},
        "model":{"type":"BPE","vocab":vocab,"merges":[["h","i"]]}})
}
fn compare(actual: &Tokenizer, legacy: &Tokenizer, text: &str) {
    for special in [false, true] {
        let a = actual.encode(text, special).unwrap();
        let b = legacy.encode(text, special).unwrap();
        assert_eq!(a.get_ids(), b.get_ids(), "{text:?}");
        assert_eq!(a.get_tokens(), b.get_tokens(), "{text:?}");
        assert_eq!(a.get_offsets(), b.get_offsets(), "{text:?}");
        assert_eq!(a.get_attention_mask(), b.get_attention_mask());
        assert_eq!(a.get_special_tokens_mask(), b.get_special_tokens_mask());
        for skip in [false, true] {
            assert_eq!(
                actual.decode(a.get_ids(), skip).unwrap(),
                legacy.decode(b.get_ids(), skip).unwrap()
            );
        }
    }
}
#[test]
fn fresh_inline_aggregate_matches_legacy_nonzero_encoding_offsets_and_decode() {
    for nfc in [false, true] {
        let mut value = source();
        if nfc {
            value["normalizer"] = json!({"type":"NFC"});
            value["added_tokens"][1]["normalized"] = json!(false);
        }
        let input = value.to_string();
        let actual = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .compile()
            .unwrap();
        let legacy = Tokenizer::from_bytes(input.as_bytes()).unwrap();
        drop(input);
        for text in [
            "hi HI <S> hi",
            "é e\u{301} 🦀",
            " 12\t345\nhi",
            "<S><S>HIhi",
            "",
            "\0\u{7f}",
        ] {
            compare(&actual, &legacy, text);
        }
        let bytes: String = (0u8..=255).map(char::from).collect();
        compare(&actual, &legacy, &bytes);
        let a = actual.encode(("hi 12", "é <S>"), true).unwrap();
        let b = legacy.encode(("hi 12", "é <S>"), true).unwrap();
        assert_eq!(a.get_ids(), b.get_ids());
        assert_eq!(a.get_offsets(), b.get_offsets());
        assert_eq!(a.get_type_ids(), b.get_type_ids());
        assert!(actual.token_to_id("hi").is_some());
        // Compatibility copies remain ordinary HF work; no aggregate owner exposes these APIs.
        let cloned = actual.clone();
        compare(&cloned, &legacy, "HIhi<S>");
        let restored =
            Tokenizer::from_bytes(serde_json::to_string(&actual).unwrap().as_bytes()).unwrap();
        compare(&restored, &legacy, "hi é HI");
    }
}
#[test]
fn selected_regex_and_actual_normalizer_authority_reject_before_construction() {
    for pre in [
        json!({"type":"ByteLevel","use_regex":true}),
        json!({"type":"Split","pattern":{"Regex":"\\w+"},"behavior":"Isolated","invert":false}),
    ] {
        let mut value = source();
        value["pre_tokenizer"] = pre;
        // Invalid model demonstrates selected regex is rejected before even model planning.
        value["model"] = json!(false);
        let input = value.to_string();
        assert_eq!(
            TokenizerCompilePlan::prepare_json(input.as_bytes())
                .unwrap_err()
                .kind(),
            K::RegexProfile
        );
    }
    let mut value = source();
    value["normalizer"] = json!({"type":"NFC"});
    let input = value.to_string();
    assert!(matches!(
        TokenizerCompilePlan::prepare_json(input.as_bytes()),
        Err(Error::Added(_))
    ));
    for field in ["single_word", "lstrip", "rstrip"] {
        let mut value = source();
        value["added_tokens"][0][field] = json!(true);
        assert!(matches!(
            TokenizerCompilePlan::prepare_json(value.to_string().as_bytes()),
            Err(Error::Added(_))
        ));
    }
    for input in [
        b"{}".as_slice(),
        b"{\"model\":{},\"model\":{}}",
        b"{\"unknown\":null}",
        b"[]",
        b"\xff",
    ] {
        assert!(TokenizerCompilePlan::prepare_json(input).is_err());
    }
    assert_eq!(add(usize::MAX, 1).unwrap_err().kind(), K::Overflow);
}
#[test]
fn all_sequence_reserve_failures_retain_the_actual_completed_model_and_prior_components() {
    for stage in 0..3 {
        let input = source().to_string();
        let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        assert!(error.allocation_error().is_some());
        assert!(error.completed_model());
        assert_eq!(
            error.partial.model.as_ref().unwrap().token_to_id("hi"),
            Some(256)
        );
        assert_eq!(error.partial.pre.is_some(), stage > 0);
        assert_eq!(error.partial.post.is_some(), stage > 1);
        assert!(error.partial.decoder.is_none());
        assert!(error.partial.added.is_none());
        assert_eq!(error.partial.pre_items.capacity(), 0);
        assert_eq!(error.partial.post_items.capacity(), 0);
        assert_eq!(error.partial.decode_items.capacity(), 0);
    }
}
#[test]
fn every_model_and_added_target_reserve_preserves_its_real_prefix_in_the_root_failure() {
    let mut value = source();
    value["model"] = json!({"type":"BPE","vocab":{"a":0,"##b</w>":1,"ab</w>":2,"[UNK]":3},"merges":[["a","##b</w>"]],"unk_token":"[UNK]","continuing_subword_prefix":"##","end_of_word_suffix":"</w>"});
    for stage in 0..7 {
        let input = value.to_string();
        let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_model_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        let Cause::Model(failure) = &error.cause else {
            panic!("actual BPE failure");
        };
        assert!(failure.allocation_error().is_some());
        assert!(!error.completed_model());
        let caps = failure.buffer_capacities();
        assert!(caps[..stage].iter().all(|&n| n > 0));
        assert!(caps[stage..].iter().all(|&n| n == 0));
    }
    for stage in 0..4 {
        let input = source().to_string();
        let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_added_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        let Cause::Added(failure) = &error.cause else {
            panic!("actual added failure");
        };
        assert!(failure.allocation_error().is_some());
        assert!(error.completed_model());
        assert!(
            error.partial.pre.is_some()
                && error.partial.post.is_some()
                && error.partial.decoder.is_some()
        );
        let caps = failure.buffer_capacities();
        assert!(caps[..stage].iter().all(|&n| n > 0));
        assert!(caps[stage..].iter().all(|&n| n == 0));
    }
}
#[test]
fn late_added_collision_retains_complete_components_without_rebinding_a_foreign_model() {
    let mut value = source();
    value["model"] = json!({"type":"BPE","vocab":{"a":0,"b":2},"merges":[]});
    value["added_tokens"] = json!([added("new", false, true), added("b", false, false)]);
    let input = value.to_string();
    let plan = TokenizerCompilePlan::prepare_json(input.as_bytes()).unwrap();
    let expected = plan.added.requirements().buffer_bytes();
    let error = plan.compile().unwrap_err();
    drop(input);
    let Cause::Added(failure) = &error.cause else {
        panic!("actual late collision");
    };
    assert!(failure.source_error().is_some());
    assert_eq!(failure.allocated_bytes(), Some(expected));
    assert!(error.completed_model());
    assert_eq!(
        error.partial.model.as_ref().unwrap().token_to_id("b"),
        Some(2)
    );
    assert!(
        error.partial.pre.is_some()
            && error.partial.post.is_some()
            && error.partial.decoder.is_some()
    );
}

#[test]
fn fallback_literal_reservation_failure_retains_actual_partial_constructor_and_model() {
    let mut value = source();
    value["decoder"] = json!({"type":"Sequence","decoders":[
        {"type":"ByteFallback"}, {"type":"Fuse"},
        {"type":"Replace","pattern":{"String":"▁"},"content":" "}]});
    for stage in 3..5 {
        let input = value.to_string();
        let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        assert!(error.allocation_error().is_some());
        assert!(error.completed_model());
        assert!(error.partial.pre.is_some() && error.partial.post.is_some());
        assert!(error.partial.decoder.is_none());
        assert_eq!(error.partial.decode_items.len(), 2);
        assert!(error.partial.decode_items.capacity() >= 3);
        assert_eq!(
            error.partial.decode_strings[0].capacity(),
            if stage == 4 { 3 } else { 0 }
        );
        assert_eq!(error.partial.decode_strings[1].capacity(), 0);
        if stage == 4 {
            assert_eq!(error.partial.decode_strings[0], "▁".as_bytes());
        }
    }
    for sequence in [
        json!([{ "type":"Fuse" },{ "type":"ByteFallback" },
            {"type":"Replace","pattern":{"String":"▁"},"content":" "}]),
        json!([{ "type":"ByteFallback" },{ "type":"Fuse" },
            {"type":"Replace","pattern":{"String":"▁"},"content":"xx"}]),
        json!([{ "type":"ByteFallback" },{ "type":"Fuse" },
            {"type":"Replace","pattern":{"Regex":"▁"},"content":" "}]),
    ] {
        value["decoder"]["decoders"] = sequence;
        // Unsupported component settings are refused before model construction.
        value["model"] = json!(false);
        let bytes = value.to_string();
        assert!(TokenizerCompilePlan::prepare_json(bytes.as_bytes()).is_err());
    }
}
