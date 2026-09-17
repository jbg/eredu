use super::*;
use crate::{EncodeIdsPlan, Encoding};
use serde_json::{json, Value};

pub(crate) fn source() -> Value {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, c) in alphabet.into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(id));
    }
    vocab.insert("hi".into(), json!(256));
    vocab.insert("Hello".into(), json!(257));
    let added = |id, word| json!({"id":id,"content":word,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true});
    let mut value = json!({"version":"1.0","normalizer":null,"padding":null,"truncation":null,
        "pre_tokenizer":null,
        "post_processor":{"type":"TemplateProcessing",
            "single":[{"SpecialToken":{"id":"start","type_id":3}},{"Sequence":{"id":"A","type_id":7}},{"SpecialToken":{"id":"end","type_id":5}}],
            "pair":[{"SpecialToken":{"id":"start","type_id":3}},{"Sequence":{"id":"A","type_id":7}},{"SpecialToken":{"id":"end","type_id":5}},{"Sequence":{"id":"B","type_id":9}}],
            "special_tokens":{"start":{"id":"stored start","ids":[259,258],"tokens":["<B>","different spelling"]},"end":{"id":"end","ids":[260],"tokens":["<E>"]}}},
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "added_tokens":[added(258,"<S>"),added(259,"<B>"),added(260,"<E>")],
        "model":{"type":"BPE","vocab":vocab,"merges":[["h","i"]],"ignore_merges":true}});
    #[cfg(feature = "fancy-regex")]
    {
        value["pre_tokenizer"] = json!({"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":fancy_regex::workspace::construction::patterns().nth(2).unwrap()},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]});
    }
    value
}
fn compare(a: &Encoding, b: &Encoding) {
    assert_eq!(a.get_ids(), b.get_ids());
    assert_eq!(a.get_tokens(), b.get_tokens());
    assert_eq!(a.get_type_ids(), b.get_type_ids());
    assert_eq!(a.get_offsets(), b.get_offsets());
    assert_eq!(a.get_word_ids(), b.get_word_ids());
    assert_eq!(a.get_attention_mask(), b.get_attention_mask());
    assert_eq!(a.get_special_tokens_mask(), b.get_special_tokens_mask());
    assert_eq!(a.get_sequence_ids(), b.get_sequence_ids());
    assert_eq!(a.get_overflowing().len(), b.get_overflowing().len());
    for (a, b) in a.get_overflowing().iter().zip(b.get_overflowing()) {
        compare(a, b);
    }
}
fn compiled(value: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
#[test]
fn packed_template_preserves_single_pair_encoding_and_ordinary_clone_serde() {
    for sequence in [false, true] {
        let mut value = source();
        if sequence {
            value["post_processor"] = json!({"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},value["post_processor"].clone()]});
        }
        let input = value.to_string();
        let mut actual = compiled(&value);
        let mut legacy = Tokenizer::from_bytes(input.as_bytes()).unwrap();
        drop(input);
        for encode_special in [false, true] {
            actual.set_encode_special_tokens(encode_special);
            legacy.set_encode_special_tokens(encode_special);
            for special in [false, true] {
                for text in [
                    "Hello hi<S>",
                    "é e\u{301}🦀",
                    " can't 1234\r\n ",
                    "",
                    "\0\u{7f}",
                ] {
                    compare(
                        &actual.encode(text, special).unwrap(),
                        &legacy.encode(text, special).unwrap(),
                    );
                    compare(
                        &actual.encode((text, "hi<B>"), special).unwrap(),
                        &legacy.encode((text, "hi<B>"), special).unwrap(),
                    );
                    let ids = EncodeIdsPlan::prepare(&actual, text, special)
                        .unwrap()
                        .encode()
                        .unwrap();
                    assert_eq!(
                        ids.ids(),
                        legacy.encode(text, special).unwrap().get_ids()
                    );
                    assert_eq!(
                        ids.capacities()[2],
                        text.len() + if special { 3 } else { 0 }
                    );
                }
            }
        }
        let cloned = actual.clone();
        drop(actual);
        let restored =
            Tokenizer::from_bytes(serde_json::to_vec(&cloned).unwrap())
                .unwrap();
        compare(
            &cloned.encode(("hi", "Hello"), true).unwrap(),
            &restored.encode(("hi", "Hello"), true).unwrap(),
        );
        let value = serde_json::to_value(cloned.get_post_processor().unwrap())
            .unwrap();
        let expected =
            serde_json::to_value(legacy.get_post_processor().unwrap())
                .unwrap();
        assert_eq!(value, expected);
    }
}
#[test]
fn template_special_policy_empty_and_all_target_failure_prefixes_use_real_destinations(
) {
    let value = source();
    let actual = compiled(&value);
    assert_eq!(
        EncodeIdsPlan::prepare(&actual, "", true)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [259, 258, 260]
    );
    assert!(EncodeIdsPlan::prepare(&actual, "", false)
        .unwrap()
        .encode()
        .unwrap()
        .ids()
        .is_empty());
    for stage in 0..5 {
        let input = value.to_string();
        let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_template_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        assert!(error.completed_model());
        let failure = error.template_failure().unwrap();
        assert!(failure.allocation_error().is_some());
        let capacities = failure.capacities();
        assert!(capacities[..stage].iter().all(|n| *n > 0));
        assert!(capacities[stage..].iter().all(|n| *n == 0));
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    for stage in 0..4 {
        let error = EncodeIdsPlan::prepare(&actual, "hi<S>Hello", true)
            .unwrap()
            .fail_reservation(stage)
            .encode();
        assert!(error.is_err());
    }
    let mut late = source();
    late["model"]["vocab"] = json!({"h":0,"i":2});
    late["model"]["merges"] = json!([]);
    late["added_tokens"] = json!([{"id":2,"content":"new","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true},{"id":2,"content":"i","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":false}]);
    let input = late.to_string();
    let error = TokenizerCompilePlan::prepare_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap_err();
    assert!(matches!(&error.cause, Cause::Added(_)));
    assert!(matches!(
        &error.partial.post,
        Some(PostProcessorWrapper::CompiledTemplate(_))
    ));
}
#[test]
fn template_source_validation_preserves_decoded_keys_sparse_ids_and_pair_metadata(
) {
    let mut value = source();
    value["post_processor"]["special_tokens"]["start"]["ids"][0] =
        json!(u32::MAX);
    let input = value.to_string().replace("\"start\"", "\"st\\u0061rt\"");
    let actual = TokenizerCompilePlan::prepare_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    let legacy = Tokenizer::from_bytes(input.as_bytes()).unwrap();
    assert_eq!(
        EncodeIdsPlan::prepare(&actual, "hi", true)
            .unwrap()
            .encode()
            .unwrap()
            .ids()[0],
        u32::MAX
    );
    compare(
        &actual.encode(("hi", "hi"), true).unwrap(),
        &legacy.encode(("hi", "hi"), true).unwrap(),
    );
    // Distinct keys may intentionally carry the same numeric ID and a different
    // token spelling. Object order and escaped names cannot change that meaning.
    let mut shared = source();
    shared["post_processor"]["special_tokens"]["end"]["ids"] = json!([259]);
    let encoded = shared.to_string();
    let ordinary = Tokenizer::from_bytes(encoded.as_bytes()).unwrap();
    let packed = compiled(&shared);
    compare(
        &packed.encode(("hi", "hi"), true).unwrap(),
        &ordinary.encode(("hi", "hi"), true).unwrap(),
    );
    let post = &shared["post_processor"];
    let reordered = format!(
        r#"{{"special_tokens":{},"pair":{},"single":{},"type":"TemplateProcessing"}}"#,
        post["special_tokens"], post["pair"], post["single"]
    );
    let encoded = encoded.replace(&post.to_string(), &reordered);
    let reordered = TokenizerCompilePlan::prepare_json(encoded.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    compare(
        &reordered.encode("hi", true).unwrap(),
        &packed.encode("hi", true).unwrap(),
    );
    for change in 0..7 {
        let mut bad = source();
        let p = &mut bad["post_processor"];
        match change {
            0 => p["single"][0]["SpecialToken"]["id"] = json!("missing"),
            1 => p["pair"][3]["Sequence"]["id"] = json!("C"),
            2 => p["special_tokens"]["start"]["ids"] = json!([1]),
            3 => p["single"][1]["Sequence"]["type_id"] = json!(-1),
            4 => p["unknown"] = json!(0),
            5 => p["single"][0]["Sequence"] = json!({"id":"A","type_id":0}),
            _ => p["special_tokens"]["end"]["ids"] = json!([4294967296u64]),
        }
        bad["model"] = json!(false);
        assert!(matches!(
            TokenizerCompilePlan::prepare_json(bad.to_string().as_bytes()),
            Err(Error::Root { .. })
        ));
    }
    let duplicate=source().to_string().replace("\"special_tokens\":{","\"special_tokens\":{\"end\":{\"id\":\"end\",\"ids\":[1],\"tokens\":[\"x\"]},");
    assert_eq!(
        TokenizerCompilePlan::prepare_json(duplicate.as_bytes())
            .unwrap_err()
            .kind(),
        K::DuplicateField
    );
}
#[test]
fn unsupported_single_repetition_rejects_e_and_legacy_missing_special_false_stays_valid(
) {
    let mut value = source();
    value["post_processor"]["single"]
        .as_array_mut()
        .unwrap()
        .push(json!({"Sequence":{"id":"A","type_id":1}}));
    let actual = compiled(&value);
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    compare(
        &actual.encode("hi", true).unwrap(),
        &legacy.encode("hi", true).unwrap(),
    );
    assert!(matches!(
        EncodeIdsPlan::prepare(&actual, "hi", true),
        Err(crate::EncodeIdsError::PipelineProfile)
    ));
    let mut no_a = source();
    no_a["post_processor"]["single"] =
        json!([{"SpecialToken":{"id":"end","type_id":0}}]);
    let no_a = compiled(&no_a);
    for special in [false, true] {
        assert!(matches!(
            EncodeIdsPlan::prepare(&no_a, "hi", special),
            Err(crate::EncodeIdsError::PipelineProfile)
        ));
    }
    let mut zero = source();
    zero["post_processor"]["single"] =
        json!([{"Sequence":{"id":"A","type_id":0}}]);
    zero["post_processor"]["pair"] = json!([{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}]);
    zero["post_processor"]["special_tokens"] = json!({});
    let zero = compiled(&zero);
    assert!(EncodeIdsPlan::prepare(&zero, "", true)
        .unwrap()
        .encode()
        .unwrap()
        .ids()
        .is_empty());
    let mut value = source();
    value["post_processor"]["special_tokens"] = json!({});
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    assert!(!legacy.encode("hi", false).unwrap().get_ids().is_empty());
    assert!(
        TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
            .is_err()
    );
}
