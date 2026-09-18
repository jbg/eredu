use super::*;
use crate::{AddedVocabularyRefreshPlan, TokenizerCompilePlan};
use serde_json::{json, Value};

fn source(normalizer: Value, scheme: &str, replacement: &str, split: bool, nested: bool) -> Value {
    let meta =
        json!({"type":"Metaspace","replacement":replacement,"prepend_scheme":scheme,"split":split});
    let pre = if nested {
        json!({"type":"Sequence","pretokenizers":[{"type":"Sequence","pretokenizers":[]},{"type":"Sequence","pretokenizers":[meta]}]})
    } else {
        meta
    };
    json!({"version":"1.0","truncation":null,"padding":null,"normalizer":normalizer,"pre_tokenizer":pre,"post_processor":null,"decoder":null,
        "added_tokens":[{"id":99,"content":"<R>","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true},{"id":98,"content":"̈","normalized":true,"single_word":false,"lstrip":false,"rstrip":false,"special":false}],
        "model":{"type":"BPE","vocab":{"a":0,"b":1,"▁":2,"▁a":3,"_":4,"_a":5,"é":6,"e":7,"́":8,"x":9,"z":10,"̈":11,"Ω":12," ":13,"ë":14},"merges":[["▁","a"],["_","a"]]}})
}
fn remove_normalizer(value: &mut Value) {
    match value["type"].as_str() {
        Some("Prepend") => *value = Value::Null,
        Some("Sequence") => {
            let values = value["normalizers"].as_array_mut().unwrap();
            for value in values.iter_mut() {
                remove_normalizer(value);
            }
            values.retain(|v| !v.is_null());
            if values.is_empty() {
                *value = Value::Null;
            }
        }
        _ => {}
    }
}
fn remove_pre(value: &mut Value) {
    match value["type"].as_str() {
        Some("Metaspace") => value["prepend_scheme"] = json!("never"),
        Some("Sequence") => {
            for value in value["pretokenizers"].as_array_mut().unwrap() {
                remove_pre(value);
            }
        }
        _ => {}
    }
}
#[test]
fn metaspace_original_and_prefix_projection_preserve_first_origin_and_added_boundaries() {
    for normalizer in [
        Value::Null,
        json!({"type":"NFC"}),
        json!({"type":"Prepend","prepend":"x"}),
        json!({"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"e"},{"type":"NFC"}]}),
        json!({"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Prepend","prepend":"z"}]}),
    ] {
        for scheme in ["always", "first", "never"] {
            for replacement in ["_", "▁", " "] {
                for split in [false, true] {
                    for nested in [false, true] {
                        let value = source(normalizer.clone(), scheme, replacement, split, nested);
                        let actual =
                            TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
                                .unwrap()
                                .compile()
                                .unwrap();
                        let ordinary = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
                        assert!(actual.matches_compiled_configuration(&ordinary));
                        let added = AddedVocabularyRefreshPlan::for_input_prefix_removal(&actual)
                            .unwrap()
                            .map(|plan| plan.compile().unwrap());
                        let view = actual.input_prefix_view(
                            added
                                .as_ref()
                                .unwrap_or_else(|| actual.get_added_vocabulary()),
                        );
                        let mut normalized = value.clone();
                        remove_normalizer(&mut normalized["normalizer"]);
                        remove_pre(&mut normalized["pre_tokenizer"]);
                        let expected_normalized =
                            Tokenizer::from_bytes(normalized.to_string().as_bytes()).unwrap();
                        assert!(view.matches_compiled_configuration(&expected_normalized));
                        for text in [
                            "", "a", "a b", " a  b ", "a<R>b", "<R>a<R>b", "̈a", "\u{344}a", "́a",
                            "▁a_b", "é<R>Ω",
                        ] {
                            let ids = EncodeIdsPlan::prepare(&actual, text, false)
                                .unwrap()
                                .encode()
                                .unwrap();
                            let expected = ordinary.encode(text, false).unwrap();
                            assert_eq!(ids.ids(), expected.get_ids(), "{} {:?}", value, text);
                            let projected = EncodeIdsPlan::prepare(view, text, false)
                                .unwrap()
                                .encode()
                                .unwrap();
                            let expected = expected_normalized.encode(text, false).unwrap();
                            assert_eq!(
                                projected.ids(),
                                expected.get_ids(),
                                "projected {} {:?}",
                                value,
                                text
                            );
                        }
                    }
                }
            }
        }
    }
}
