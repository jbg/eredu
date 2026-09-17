use super::*;
use crate::{
    EncodeIdsPlan, Encoding, OffsetReferential, OffsetType, Tokenizer,
    TokenizerCompilePlan,
};
use serde_json::{json, Value};

fn source() -> Value {
    let mut value = crate::tokenizer::compile::template_tests::source();
    #[cfg(feature = "fancy-regex")]
    {
        value["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
            json!(fancy_regex::workspace::construction::patterns()
                .nth(1)
                .unwrap());
    }
    value["model"]["ignore_merges"] = json!(false);
    for (id, word, normalized, special) in [
        (261, "Mathias", true, false),
        (262, "python", true, false),
        (263, "bc", false, false),
        (264, "abcd", true, false),
        (265, "uvw", false, true),
        (266, "uv", false, false),
        (267, "uvwx", true, false),
        (268, "qrs", true, true),
        (269, "qr", true, false),
        (270, "éx", true, false),
    ] {
        value["added_tokens"].as_array_mut().unwrap().push(json!({"id":id,"content":word,"single_word":false,"lstrip":false,"rstrip":false,"normalized":normalized,"special":special}));
    }
    value
}
fn compiled(value: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
fn old_order(source: &AddedVocabulary, input: &str) -> PreTokenizedString {
    // Independent compatibility composition of the existing single-phase workers.
    // The new helper does not call these two PreTokenizedString passes.
    let mut result: PreTokenizedString = input.into();
    result
        .split(|_, text| Ok(source.split_with_indices(text, false)))
        .unwrap();
    result
        .split(|_, text| Ok(source.split_with_indices(text, true)))
        .unwrap();
    result
}
fn compare_encoding(a: &Encoding, b: &Encoding) {
    assert_eq!(a.get_ids(), b.get_ids());
    assert_eq!(a.get_tokens(), b.get_tokens());
    assert_eq!(a.get_offsets(), b.get_offsets());
    assert_eq!(a.get_type_ids(), b.get_type_ids());
    assert_eq!(a.get_word_ids(), b.get_word_ids());
    assert_eq!(a.get_sequence_ids(), b.get_sequence_ids());
    assert_eq!(a.get_special_tokens_mask(), b.get_special_tokens_mask());
    assert_eq!(a.get_attention_mask(), b.get_attention_mask());
    assert_eq!(a.get_overflowing().len(), b.get_overflowing().len());
    for (a, b) in a.get_overflowing().iter().zip(b.get_overflowing()) {
        compare_encoding(a, b);
    }
}
fn selected(source: &AddedVocabulary, text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    IdentityMatching::ordinary(source)
        .visit(text, |id, _| {
            if let Some(id) = id {
                ids.push(id);
            }
            Ok::<(), ()>(())
        })
        .unwrap();
    ids
}
#[test]
fn two_literal_phases_preserve_filtered_priority_offsets_and_single_pair_encoding(
) {
    let value = source();
    let mut actual = compiled(&value);
    let mut legacy =
        Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    for encode_special in [false, true] {
        actual.set_encode_special_tokens(encode_special);
        legacy.set_encode_special_tokens(encode_special);
        assert_eq!(selected(&actual.added_vocabulary, "abcd"), [263]);
        assert_eq!(
            selected(&actual.added_vocabulary, "uvwx"),
            if encode_special { vec![267] } else { vec![265] }
        );
        assert_eq!(
            selected(&actual.added_vocabulary, "qrs"),
            if encode_special { vec![] } else { vec![268] }
        );
        for text in [
            "Mathiaspython",
            "Mathias<S>python",
            "abcd uvwx qrs",
            "éxé e\u{301}🦀",
            " can't 1234\r\n",
            "",
        ] {
            for source in [&actual, &legacy] {
                let got = source
                    .added_vocabulary
                    .extract_and_normalize::<NormalizerWrapper>(None, text);
                let old = old_order(&source.added_vocabulary, text);
                for referential in [
                    OffsetReferential::Original,
                    OffsetReferential::Normalized,
                ] {
                    for unit in [OffsetType::Byte, OffsetType::Char] {
                        assert_eq!(
                            got.get_splits(referential, unit),
                            old.get_splits(referential, unit)
                        );
                    }
                }
            }
            for special in [false, true] {
                compare_encoding(
                    &actual.encode(text, special).unwrap(),
                    &legacy.encode(text, special).unwrap(),
                );
                compare_encoding(
                    &actual
                        .encode((text, "python<S>Mathias"), special)
                        .unwrap(),
                    &legacy
                        .encode((text, "python<S>Mathias"), special)
                        .unwrap(),
                );
                let plan =
                    EncodeIdsPlan::prepare(&actual, text, special).unwrap();
                let facts = plan.requirements();
                let output = plan.encode().unwrap();
                assert_eq!(
                    output.ids(),
                    legacy.encode(text, special).unwrap().get_ids()
                );
                assert_eq!(
                    facts.id_capacity(),
                    text.len() + if special { 3 } else { 0 }
                );
                assert_eq!(output.capacities()[2], facts.id_capacity());
            }
        }
        let clone = actual.clone();
        let restored =
            Tokenizer::from_bytes(serde_json::to_vec(&clone).unwrap())
                .unwrap();
        compare_encoding(
            &clone.encode(("Mathias", "python"), true).unwrap(),
            &restored.encode(("Mathias", "python"), true).unwrap(),
        );
    }
}
#[test]
fn exhaustive_small_raw_normalized_spans_match_the_existing_two_pass_composition(
) {
    let value = source();
    let mut packed = compiled(&value);
    let mut legacy =
        Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    for skip in [false, true] {
        packed.set_encode_special_tokens(skip);
        legacy.set_encode_special_tokens(skip);
        for width in 0..=5u32 {
            for mut n in 0..4usize.pow(width) {
                let mut text = String::new();
                for _ in 0..width {
                    text.push(['a', 'b', 'c', 'd'][n % 4]);
                    n /= 4;
                }
                for source in [&packed, &legacy] {
                    let expected = old_order(&source.added_vocabulary, &text);
                    let got = source
                        .added_vocabulary
                        .extract_and_normalize::<NormalizerWrapper>(
                            None, &text,
                        );
                    assert_eq!(
                        got.get_splits(
                            OffsetReferential::Original,
                            OffsetType::Byte
                        ),
                        expected.get_splits(
                            OffsetReferential::Original,
                            OffsetType::Byte
                        )
                    );
                }
            }
        }
    }
}
#[test]
fn actual_normalizer_and_word_strip_profiles_stay_rejected_and_callback_errors_propagate(
) {
    let value = source();
    let mut actual = compiled(&value);
    let matching = IdentityMatching::checked(
        &actual.added_vocabulary,
        actual.normalizer.as_ref(),
    )
    .unwrap();
    let mut seen = Vec::new();
    let result = matching.visit("Mathias python", |id, offsets| {
        seen.push((id, offsets));
        if id == Some(262) {
            Err(17)
        } else {
            Ok(())
        }
    });
    assert_eq!(result, Err(17));
    assert_eq!(
        seen,
        [(Some(261), (0, 7)), (None, (7, 8)), (Some(262), (8, 14))]
    );
    let n: NormalizerWrapper = crate::normalizers::NFC.into();
    assert!(
        IdentityMatching::checked(&actual.added_vocabulary, Some(&n))
            .is_none()
    );
    actual.with_normalizer(Some(n)).unwrap();
    assert!(matches!(
        EncodeIdsPlan::prepare(&actual, "Mathias", false),
        Err(crate::EncodeIdsError::PipelineProfile)
    ));
    for flag in ["single_word", "lstrip", "rstrip"] {
        let mut value = source();
        value["added_tokens"][3][flag] = json!(true);
        assert!(TokenizerCompilePlan::prepare_json(
            value.to_string().as_bytes()
        )
        .is_err());
    }
}
#[test]
fn canonical_duplicate_spelling_phase_updates_and_raw_only_fast_path_keep_semantics(
) {
    let mut value = source();
    // Updating one existing spelling changes its phase, not its ID or table priority.
    value["added_tokens"].as_array_mut().unwrap().push(json!({"id":261,"content":"Mathias","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}));
    let actual = compiled(&value);
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    compare_encoding(
        &actual.encode("Mathiaspython", false).unwrap(),
        &legacy.encode("Mathiaspython", false).unwrap(),
    );
    assert_eq!(
        EncodeIdsPlan::prepare(&actual, "Mathiaspython", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [261, 262]
    );
    let raw = compiled(&crate::tokenizer::compile::template_tests::source());
    let one = IdentityMatching::checked(&raw.added_vocabulary, None).unwrap();
    let two =
        IdentityMatching::checked(&actual.added_vocabulary, None).unwrap();
    assert!(!one.normalized);
    assert!(two.normalized);
    assert!(
        one.control_bytes::<crate::EncodeIdsError>().unwrap()
            < two.control_bytes::<crate::EncodeIdsError>().unwrap()
    );
}
