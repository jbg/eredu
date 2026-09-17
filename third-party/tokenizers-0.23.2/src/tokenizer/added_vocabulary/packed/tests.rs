use super::*;
use crate::{AddedToken, NormalizerWrapper, OffsetReferential, OffsetType, Tokenizer};
use serde_json::json;

fn model() -> BPE {
    crate::models::bpe::BpeCompilePlan::prepare_model_json(br#"{"vocab":{"h":0,"i":1,"hi":2," ":3,"a":4,"b":5,"c":6,"x":7,"y":8,"z":9,"e":10,"!":11,"\u00e9":12,"\ud83d\ude03":13},"merges":[["h","i"]]}"#).unwrap().compile().unwrap()
}
fn value(content: &str, normalized: bool, special: bool) -> serde_json::Value {
    json!({"id":900,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":normalized,"special":special})
}
fn source(values: Vec<serde_json::Value>) -> String {
    serde_json::to_string(&values).unwrap()
}
fn legacy(input: &str, model: &BPE, normalizer: Option<&NormalizerWrapper>) -> AddedVocabulary {
    let values: Vec<serde_json::Value> = serde_json::from_str(input).unwrap();
    let tokens: Vec<AddedToken> = values
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
    let mut v = AddedVocabulary::new();
    v.add_tokens(tokens, model, normalizer).unwrap();
    v
}
fn packed(input: &str, model: &BPE, normalizer: Option<&NormalizerWrapper>) -> AddedVocabulary {
    AddedVocabularyCompilePlan::prepare_json(input.as_bytes(), model, normalizer)
        .unwrap()
        .compile()
        .unwrap()
}
fn extraction(
    v: &AddedVocabulary,
    normalizer: Option<&NormalizerWrapper>,
    input: &str,
) -> (
    Vec<(String, (usize, usize), Vec<u32>)>,
    Vec<(String, (usize, usize), Vec<u32>)>,
) {
    let p = v.extract_and_normalize(normalizer, input);
    let collect = |kind| {
        p.get_splits(kind, OffsetType::Byte)
            .into_iter()
            .map(|(s, o, t)| {
                (
                    s.to_owned(),
                    o,
                    t.as_ref()
                        .map(|v| v.iter().map(|t| t.id).collect())
                        .unwrap_or_default(),
                )
            })
            .collect()
    };
    (
        collect(OffsetReferential::Original),
        collect(OffsetReferential::Normalized),
    )
}
#[test]
fn packed_nonzero_extraction_and_encoding_share_the_actual_pipeline() {
    let model = model();
    let input = source(vec![
        value("<|end|>", false, true),
        value("hi!", true, false),
        value("😃", false, false),
    ]);
    let expected = legacy(&input, &model, None);
    let actual = packed(&input, &model, None);
    assert!(matches!(actual.storage, Storage::Packed(_)));
    let mut first = Tokenizer::new(model.clone());
    first.with_added_vocabulary(expected);
    let mut second = Tokenizer::new(model);
    second.with_added_vocabulary(actual);
    for input in [
        "hi hi! <|end|>😃 hi",
        "<|end|><|end|>hi!",
        "hé e\u{301} hi!",
        "",
        "xyz",
    ] {
        assert_eq!(
            extraction(first.get_added_vocabulary(), None, input),
            extraction(second.get_added_vocabulary(), None, input)
        );
        let a = first.encode(input, false).unwrap();
        let b = second.encode(input, false).unwrap();
        assert_eq!(a.get_ids(), b.get_ids());
        assert_eq!(a.get_tokens(), b.get_tokens());
        assert_eq!(a.get_offsets(), b.get_offsets());
        if input == "hi hi! <|end|>😃 hi" {
            assert!(b.get_ids().contains(&2));
            assert!(b.get_ids().iter().any(|&id| id >= 14));
        }
    }
}
#[test]
fn raw_longest_match_is_consumed_before_special_filtering() {
    let model = model();
    let input = source(vec![
        value("ab", false, true),
        value("a", false, false),
        value("bc", false, false),
    ]);
    let mut expected = legacy(&input, &model, None);
    let mut actual = packed(&input, &model, None);
    let ab = actual.token_to_id("ab", &model).unwrap();
    assert_eq!(
        actual.raw_matches("abc", false).collect::<Vec<_>>(),
        vec![(ab, (0, 2))]
    );
    expected.set_encode_special_tokens(true);
    actual.set_encode_special_tokens(true);
    assert_eq!(actual.find_matches("abc", false), vec![(None, (0, 3))]);
    assert_eq!(
        extraction(&expected, None, "abc ab a"),
        extraction(&actual, None, "abc ab a")
    );
    // Neither shorter a at position 0 nor overlapping bc at position 1 is recovered.
    actual.set_encode_special_tokens(false);
    expected.set_encode_special_tokens(false);
    assert_eq!(
        extraction(&expected, None, "abc ab a"),
        extraction(&actual, None, "abc ab a")
    );
}
#[test]
fn source_order_updates_keep_first_id_final_flags_and_sticky_special_membership() {
    let model = model();
    let input = source(vec![
        value("", false, true),
        value("hi", false, true),
        value("new", false, false),
        value("hi", true, false),
        value("new", false, false),
    ]);
    let expected = legacy(&input, &model, None);
    let actual = packed(&input, &model, None);
    assert_eq!(actual.get_vocab(), expected.get_vocab());
    assert_eq!(
        actual.get_added_tokens_decoder(),
        expected.get_added_tokens_decoder()
    );
    assert_eq!(actual.token_to_id("hi", &model), Some(2));
    assert!(actual.is_special_token("hi"));
    assert!(!actual.token_ref(2).unwrap().special);
    assert!(actual.token_ref(2).unwrap().normalized);
    assert_eq!(actual.len(), 2);
    assert!(!actual.is_special_token(""));
    assert_eq!(
        extraction(&expected, None, "hi new hi"),
        extraction(&actual, None, "hi new hi")
    );
}
#[test]
fn cardinality_assignment_and_added_first_model_shadowing_preserve_sparse_ids() {
    let model = crate::models::bpe::BpeCompilePlan::prepare_model_json(
        br#"{"vocab":{"a":0,"b":2},"merges":[]}"#,
    )
    .unwrap()
    .compile()
    .unwrap();
    let input = source(vec![value("new", false, true)]);
    let actual = packed(&input, &model, None);
    let expected = legacy(&input, &model, None);
    assert_eq!(actual.token_to_id("new", &model), Some(2));
    assert_eq!(actual.decode_token_ref(2), Some("new"));
    assert_eq!(actual.get_vocab(), expected.get_vocab());
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_added_vocabulary(actual);
    assert_eq!(tokenizer.decode_vocabulary().id_to_token(2), Some("new"));
    assert_eq!(
        tokenizer
            .decode_vocabulary()
            .ids()
            .filter(|&id| id == 2)
            .count(),
        2
    );
}
#[test]
fn actual_normalizer_borrow_controls_profiles_and_nfc_unmatched_text() {
    let model = model();
    let normalizer: NormalizerWrapper = crate::normalizers::NFC.into();
    let input = source(vec![value("hi", false, true)]);
    let plan =
        AddedVocabularyCompilePlan::prepare_json(input.as_bytes(), &model, Some(&normalizer))
            .unwrap();
    assert!(std::ptr::eq(plan.normalizer.unwrap(), &normalizer));
    assert!(std::ptr::eq(plan.model, &model));
    let actual = plan.compile().unwrap();
    let expected = legacy(&input, &model, Some(&normalizer));
    assert_eq!(
        extraction(&expected, Some(&normalizer), "e\u{301} hi é"),
        extraction(&actual, Some(&normalizer), "e\u{301} hi é")
    );
    let normalized = source(vec![value("hi", true, false)]);
    assert_eq!(
        AddedVocabularyCompilePlan::prepare_json(normalized.as_bytes(), &model, Some(&normalizer))
            .unwrap_err()
            .kind,
        Kind::NormalizationProfile
    );
    let actual = packed(&normalized, &model, None);
    assert_eq!(actual.decode_token_ref(2), Some("hi"));
}
#[test]
fn four_real_capacity_frontiers_retain_every_allocated_prefix_after_input_drop() {
    let model = model();
    let mut previous = 0;
    for stage in 0..4 {
        let input = source(vec![
            value("<first>", false, true),
            value("😃hi", true, false),
            value("last", false, false),
        ]);
        let plan = AddedVocabularyCompilePlan::prepare_json(input.as_bytes(), &model, None)
            .unwrap()
            .fail_reservation(stage);
        let error = plan.compile().unwrap_err();
        drop(input);
        assert!(error.allocation_error().is_some());
        assert!(error.source_error().is_none());
        let capacities = error.buffer_capacities();
        assert!(capacities[..stage].iter().all(|&c| c > 0));
        assert!(capacities[stage..].iter().all(|&c| c == 0));
        let bytes = error.allocated_bytes().unwrap();
        assert_eq!(bytes == 0, stage == 0);
        if stage > 0 {
            assert!(bytes > previous);
        }
        previous = bytes;
        assert!(std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<TryReserveError>()
            .is_some());
    }
}
#[test]
fn late_reverse_id_collision_keeps_completed_raw_destination_without_retry() {
    let model = crate::models::bpe::BpeCompilePlan::prepare_model_json(
        br#"{"vocab":{"a":0,"b":2},"merges":[]}"#,
    )
    .unwrap()
    .compile()
    .unwrap();
    let input = source(vec![value("new", false, true), value("b", false, false)]);
    let plan = AddedVocabularyCompilePlan::prepare_json(input.as_bytes(), &model, None).unwrap();
    let facts = plan.requirements();
    let error = plan.compile().unwrap_err();
    drop(input);
    drop(model);
    assert_eq!(error.source_error().unwrap().kind, Kind::AmbiguousId);
    assert_eq!(error.allocated_bytes(), Some(facts.buffer_bytes()));
    assert_eq!(error.partial.entries.len(), 2);
    assert_eq!(error.partial.content(0), "new");
    assert_eq!(error.partial.content(1), "b");
}
#[test]
fn borrowed_and_explicit_map_compatibility_clone_serde_and_mutation_keep_values() {
    let model = model();
    let input = source(vec![value("HI", true, false), value("<s>", false, true)]);
    let actual = packed(&input, &model, None);
    assert!(matches!(actual.get_vocab(), std::borrow::Cow::Owned(_)));
    let expected = legacy(&input, &model, None);
    assert!(matches!(
        expected.get_vocab(),
        std::borrow::Cow::Borrowed(_)
    ));
    let alias = actual.clone();
    let hi = actual.token_to_id("HI", &model).unwrap();
    assert_eq!(actual.decode_token_ref(hi), Some("HI"));
    assert_eq!(alias.get_vocab(), actual.get_vocab());
    assert_eq!(
        serde_json::to_value(&actual).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    let mut changed = actual;
    let normalizer: NormalizerWrapper = crate::normalizers::Lowercase.into();
    changed
        .refresh_normalized_tokens(Some(&normalizer))
        .unwrap();
    assert!(matches!(changed.storage, Storage::Legacy(_)));
    assert_eq!(changed.decode_token_ref(hi), Some("hi"));
    assert_eq!(alias.decode_token_ref(hi), Some("HI"));
    changed
        .add_tokens([AddedToken::from("new", false)], &model, Some(&normalizer))
        .unwrap();
    assert!(changed.token_to_id("new", &model).is_some());
    assert_eq!(alias.token_to_id("new", &model), None);
}
#[test]
fn escaped_content_empty_input_and_complete_schema_validation_precede_reserves() {
    let model = model();
    let input = r#"[{"id":88,"content":"\ud83d\ude03","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]"#;
    let actual = packed(input, &model, None);
    assert_eq!(actual.token_to_id("😃", &model), Some(13));
    assert_eq!(actual.find_matches("", false), vec![(None, (0, 0))]);
    for key in ["single_word", "lstrip", "rstrip"] {
        let mut t = value("x", false, false);
        t[key] = json!(true);
        let input = source(vec![t]);
        assert_eq!(
            AddedVocabularyCompilePlan::prepare_json(input.as_bytes(), &model, None)
                .unwrap_err()
                .kind,
            Kind::WordOrStripProfile
        );
    }
    for input in [b"[".as_slice(), b"[] true", b"[{}]", b"[null]", b"[\xff]"] {
        assert!(AddedVocabularyCompilePlan::prepare_json(input, &model, None).is_err());
    }
    let empty = packed("[]", &model, None);
    assert!(empty.is_empty());
    assert_eq!(empty.find_matches("hi", false), vec![(None, (0, 2))]);
    assert_eq!(
        requirements(usize::MAX, 1).unwrap_err().kind,
        Kind::Overflow
    );
    assert_eq!(
        requirements(1, usize::MAX).unwrap_err().kind,
        Kind::Overflow
    );
}

#[test]
fn overlapping_unicode_and_phase_prefixes_match_legacy_on_exhaustive_short_inputs() {
    let model = model();
    let input = source(
        [
            "a", "ab", "aba", "abc", "b", "ba", "é", "éa", "é😃", "😃", "😃a", "<a>", "<ab>",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, text)| value(text, i % 3 == 1, i % 2 == 0))
        .collect(),
    );
    let mut actual = packed(&input, &model, None);
    let mut expected = legacy(&input, &model, None);
    let alphabet = ["a", "b", "é", "😃", " ", "<"];
    let mut corpus = vec![String::new()];
    let mut frontier = vec![String::new()];
    for _ in 0..4 {
        frontier = frontier
            .iter()
            .flat_map(|prefix| {
                alphabet
                    .iter()
                    .map(move |suffix| format!("{prefix}{suffix}"))
            })
            .collect();
        corpus.extend(frontier.iter().cloned());
    }
    corpus.extend(["abc aba <a><ab>é😃", "é😃a😃aba", "ababa", "<ab><a>"].map(str::to_owned));
    let mut hits = 0;
    for text in &corpus {
        for normalized in [false, true] {
            let found = actual.raw_matches(text, normalized).collect::<Vec<_>>();
            hits += found.len();
            assert_eq!(
                found,
                expected.raw_matches(text, normalized).collect::<Vec<_>>(),
                "phase={normalized}, input={text:?}"
            );
        }
    }
    assert!(hits > corpus.len());
    for encode_special in [false, true] {
        actual.set_encode_special_tokens(encode_special);
        expected.set_encode_special_tokens(encode_special);
        for text in &corpus {
            assert_eq!(
                extraction(&actual, None, text),
                extraction(&expected, None, text),
                "special={encode_special}, input={text:?}"
            );
        }
    }
}
