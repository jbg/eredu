use super::*;
use crate::{Encoding, TokenizerCompilePlan};
use fancy_regex::workspace::{Buffer, PrepareFailure};
use serde_json::{json, Value};

fn source(pattern: usize, template: bool) -> Value {
    let mut value = crate::tokenizer::compile::template_tests::source();
    value["normalizer"] = json!({"type":"NFC"});
    value["model"]["ignore_merges"] = json!(false);
    value["model"]["continuing_subword_prefix"] = json!("");
    value["model"]["end_of_word_suffix"] = json!("");
    value["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
        json!(fancy_regex::workspace::construction::patterns()
            .nth(pattern)
            .unwrap());
    value["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] =
        json!(pattern == 4);
    if !template {
        value["post_processor"] = json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false});
    }
    let e = value["model"]["vocab"]["e"].clone();
    value["added_tokens"].as_array_mut().unwrap().extend([
        json!({"id":261,"content":"e\u{301}","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":false}),
        json!({"id":262,"content":"<S>x","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":false}),
        json!({"id":e,"content":"e","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true}),
    ]);
    value
}
fn compiled(value: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
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
    assert!(a.get_overflowing().is_empty() && b.get_overflowing().is_empty());
}
fn counts(text: &str) -> [usize; 3] {
    let plan =
        unicode_normalization_alignments::workspace::Plan::new(text).unwrap();
    let r = plan.requirements();
    [r.scalar_capacity(), r.scalar_capacity(), r.text_capacity()]
}
fn reserve_cause(
    mut error: Option<&(dyn std::error::Error + 'static)>,
) -> bool {
    while let Some(e) = error {
        if e.is::<TryReserveError>() {
            return true;
        }
        error = e.source();
    }
    false
}
#[test]
fn actual_nfc_patterns_raw_barriers_and_template_single_pair_keep_ordinary_encoding(
) {
    let bytes: String = (0u8..=255).map(char::from).collect();
    let combining =
        format!("a{}", "\u{315}\u{300}\u{301}\u{0323}".repeat(257));
    for (pattern, template) in [(3, false), (4, false), (4, true)] {
        let value = source(pattern, template);
        let mut actual = compiled(&value);
        let mut legacy =
            Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
        for skip in [false, true] {
            actual.set_encode_special_tokens(skip);
            legacy.set_encode_special_tokens(skip);
            for text in [
                "hi e\u{301} e\u{300}<S>x<S>hi",
                "\u{344}",
                "é e\u{301} \u{212b} \u{2126}",
                "\u{1100}\u{1161}\u{11a8} \u{301}a\u{315}\u{300}\u{315}",
                "العربية עברית हिन्दी 🦀\u{200c}\u{200d}",
                " can't 123456\r\n ",
                "",
                &bytes,
                &combining,
            ] {
                for special in [false, true] {
                    let expected = legacy.encode(text, special).unwrap();
                    compare(&actual.encode(text, special).unwrap(), &expected);
                    compare(
                        &actual
                            .encode((text, "hi e\u{300}"), special)
                            .unwrap(),
                        &legacy
                            .encode((text, "hi e\u{300}"), special)
                            .unwrap(),
                    );
                    let plan = EncodeIdsPlan::prepare(&actual, text, special)
                        .unwrap();
                    let limits = plan.requirements();
                    let geometry = counts(text);
                    assert_eq!(limits.normalization_capacities(), geometry);
                    assert_eq!(limits.symbol_capacity(), geometry[2]);
                    assert_eq!(limits.mapped_capacity(), 2 * geometry[2]);
                    assert_eq!(
                        limits.id_capacity(),
                        geometry[2] + if template && special { 3 } else { 0 }
                    );
                    let output = plan.encode().unwrap();
                    assert_eq!(output.ids(), expected.get_ids(), "{text:?}");
                    assert_eq!(output.normalization_capacities(), geometry);
                    assert_eq!(
                        output.capacities(),
                        [
                            limits.symbol_capacity(),
                            limits.merge_capacity(),
                            limits.id_capacity()
                        ]
                    );
                    assert_eq!(
                        output.mapped_capacity(),
                        limits.mapped_capacity()
                    );
                }
            }
        }
        let cloned = actual.clone();
        assert_eq!(
            cloned.get_encode_special_tokens(),
            actual.get_encode_special_tokens()
        );
        let mut restored = Tokenizer::from_bytes(
            serde_json::to_string(&actual).unwrap().as_bytes(),
        )
        .unwrap();
        // HF serialization stores added-token declarations, not this mutable
        // encoding policy. Restore its default before explicitly matching policy.
        assert!(!restored.get_encode_special_tokens());
        restored.set_encode_special_tokens(actual.get_encode_special_tokens());
        compare(
            &cloned.encode("e\u{300} hi", true).unwrap(),
            &legacy.encode("e\u{300} hi", true).unwrap(),
        );
        compare(
            &restored.encode("e\u{300} hi", true).unwrap(),
            &legacy.encode("e\u{300} hi", true).unwrap(),
        );
        assert!(EncodeIdsPlan::prepare(&restored, "hi", true).is_err());
    }
}
#[test]
fn expansion_empty_affixes_and_noop_option_identity_are_explicit() {
    let value = source(3, false);
    let actual = compiled(&value);
    let plan = EncodeIdsPlan::prepare(&actual, "\u{344}", false).unwrap();
    assert_eq!(plan.requirements().normalization_capacities(), [2, 2, 4]);
    assert_eq!(plan.requirements().symbol_capacity(), 4);
    assert_eq!(plan.requirements().mapped_capacity(), 8);
    let ids = plan.encode().unwrap();
    assert_eq!(ids.ids().len(), 4); // Four normalized UTF-8 bytes, no matching merge.
    let stored = serde_json::to_value(&actual).unwrap();
    assert_eq!(stored["model"]["continuing_subword_prefix"], json!(""));
    assert_eq!(stored["model"]["end_of_word_suffix"], json!(""));
    for prefix in [Value::Null, json!("")] {
        for suffix in [Value::Null, json!("")] {
            let mut options = value.clone();
            options["model"]["continuing_subword_prefix"] = prefix.clone();
            options["model"]["end_of_word_suffix"] = suffix.clone();
            let packed = compiled(&options);
            let legacy =
                Tokenizer::from_bytes(options.to_string().as_bytes()).unwrap();
            assert_eq!(
                EncodeIdsPlan::prepare(&packed, "\u{344}", false)
                    .unwrap()
                    .encode()
                    .unwrap()
                    .ids(),
                ids.ids()
            );
            compare(
                &packed.encode("hi e\u{300}", false).unwrap(),
                &legacy.encode("hi e\u{300}", false).unwrap(),
            );
            let stored = serde_json::to_value(&packed).unwrap();
            assert_eq!(stored["model"]["continuing_subword_prefix"], prefix);
            assert_eq!(stored["model"]["end_of_word_suffix"], suffix);
        }
    }
    for fuse in [false, true] {
        let mut sparse = value.clone();
        sparse["model"]["vocab"] = json!({"h":0,"i":1,"hi":2,"?":3});
        sparse["model"]["merges"] = json!([["h", "i"]]);
        sparse["model"]["unk_token"] = json!("?");
        sparse["model"]["fuse_unk"] = json!(fuse);
        sparse["added_tokens"] = json!([]);
        let packed = compiled(&sparse);
        let legacy =
            Tokenizer::from_bytes(sparse.to_string().as_bytes()).unwrap();
        for text in ["hi é", "\u{344}", "zz\u{301}hizz"] {
            compare(
                &packed.encode(text, false).unwrap(),
                &legacy.encode(text, false).unwrap(),
            );
            assert_eq!(
                EncodeIdsPlan::prepare(&packed, text, false)
                    .unwrap()
                    .encode()
                    .unwrap()
                    .ids(),
                legacy.encode(text, false).unwrap().get_ids()
            );
        }
    }
    for field in ["continuing_subword_prefix", "end_of_word_suffix"] {
        let mut nonempty = value.clone();
        nonempty["model"][field] = json!("#");
        nonempty["model"]["merges"] = json!([]);
        nonempty["model"]["unk_token"] = json!("?");
        let packed = compiled(&nonempty);
        let legacy =
            Tokenizer::from_bytes(nonempty.to_string().as_bytes()).unwrap();
        compare(
            &packed.encode("hi é", false).unwrap(),
            &legacy.encode("hi é", false).unwrap(),
        );
        assert!(matches!(
            EncodeIdsPlan::prepare(&packed, "hi", false),
            Err(EncodeIdsError::ModelProfile)
        ));
    }
}
#[test]
fn every_actual_nfc_reserve_and_later_regex_bpe_error_retain_the_real_prefix()
{
    let value = source(4, true);
    let actual = compiled(&value);
    let input = String::from("hi e\u{300}\u{344}<S>x");
    for stage in 0..4 {
        let failure = EncodeIdsPlan::prepare(&actual, &input, true)
            .unwrap()
            .fail_reservation(stage)
            .encode()
            .unwrap_err();
        assert_eq!(failure.normalization_capacities(), [0; 3]);
        assert!(reserve_cause(Some(&failure)));
    }
    for (stage, target) in [
        NormalizationBuffer::Decomposition,
        NormalizationBuffer::Recomposition,
        NormalizationBuffer::Text,
    ]
    .iter()
    .enumerate()
    {
        let failure = EncodeIdsPlan::prepare(&actual, &input, true)
            .unwrap()
            .fail_normalization_reservation(*target)
            .unwrap()
            .encode()
            .unwrap_err();
        assert!(failure.capacities().iter().all(|&n| n > 0));
        assert!(failure.mapped_capacity() > 0);
        let caps = failure.normalization_capacities();
        assert!(caps[..stage].iter().all(|&n| n > 0));
        assert!(caps[stage..].iter().all(|&n| n == 0));
        let EncodeIdsError::NormalizationPreparation(error) = failure.cause()
        else {
            panic!("actual NFC reserve");
        };
        assert_eq!(error.buffer(), *target);
        assert!(reserve_cause(Some(&failure)));
    }
    let later = EncodeIdsPlan::prepare(&actual, &input, true)
        .unwrap()
        .fail_regex_reservation(PrepareFailure::Outer(Buffer::Slots))
        .unwrap()
        .encode()
        .unwrap_err();
    assert_eq!(later.normalization_capacities(), counts(&input));
    assert!(matches!(later.cause(), EncodeIdsError::RegexPreparation(_)));
    assert!(reserve_cause(Some(&later)));
    let mut missing = value;
    missing["model"]["vocab"] = json!({"h":0,"i":1,"hi":2});
    missing["model"]["merges"] = json!([["h", "i"]]);
    missing["model"]["unk_token"] = json!("absent");
    missing["added_tokens"] = json!([{ "id":258,"content":"<S>","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true }]);
    let broken = compiled(&missing);
    let late = EncodeIdsPlan::prepare(&broken, "hi<S>\u{344}", true)
        .unwrap()
        .encode()
        .unwrap_err();
    assert!(matches!(late.cause(), EncodeIdsError::MissingUnknown));
    assert_eq!(late.partial_id_count(), 4); // Two template IDs, hi and raw <S> before the failed next word.
    assert_eq!(late.normalization_capacities(), counts("hi<S>\u{344}"));
    drop((input, actual, broken));
    assert!(
        later.normalization_capacities()[2] > 0
            && late.normalization_capacities()[2] > 0
    );
}
