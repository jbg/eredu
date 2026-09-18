use super::*;
use eredu_text::tokenizer_storage::{RegexBuffer, RegexConstructionFailure, RegexWorkspaceFailure};

fn json() -> String {
    let mut value: serde_json::Value = serde_json::from_str(JSON).unwrap();
    value["normalizer"] = serde_json::json!({"type":"Sequence","normalizers":[
        {"type":"Prepend","prepend":"A"},{"type":"Lowercase"},{"type":"NFC"}]});
    value["pre_tokenizer"] = serde_json::json!({"type":"Sequence","pretokenizers":[
        {"type":"Whitespace"},
        {"type":"Sequence","pretokenizers":[
            {"type":"Digits","individual_digits":false},
            {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            {"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}]},
        {"type":"Whitespace"}]});
    value["model"]["vocab"] = serde_json::json!({"h":0,"i":1,"hi":2,"Ġ":3,"?":4,
        "a":6,"1":7,"2":8,"12":9,"▁":10});
    value["model"]["merges"] = serde_json::json!([["h", "i"], ["1", "2"]]);
    value.to_string()
}

fn reserve_leaf(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut cause = Some(error);
    while let Some(error) = cause {
        if error.is::<std::collections::TryReserveError>() {
            return true;
        }
        cause = error.source();
    }
    false
}

#[test]
fn ordered_pretokenizers_match_ordinary_with_exact_c_e_and_source_authentication() {
    let input = json();
    let ordinary = eredu_text::tokenizer::Tokenizer::from_bytes(input.as_bytes()).unwrap();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    assert!(original.matches_configuration(&ordinary));
    let c = original.original_bytes();
    let short = WorkingMemoryPool::new(c - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_with(
            TokenizerPlan::prepare_json(input.as_bytes()).unwrap(),
            || panic!("C must precede every source producer"),
        )
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    for text in ["", "HI12<S>hi ?", "hi 12", "é\n٣¼🙂", "İΣẞ", "\u{301}hi"] {
        let expected = ordinary.encode(text, false).unwrap();
        if text == "hi 12" {
            assert!(expected.get_ids().contains(&9));
        }
        let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&original, text, false).unwrap();
        for one_short in [true, false] {
            let pool = WorkingMemoryPool::new(c + e - u64::from(one_short), 0).unwrap();
            let source = source(&pool, &input);
            let result = pool.encode_tokenizer_ids_with(
                &source,
                text,
                false,
                |p| p,
                || {
                    assert!(!one_short);
                    assert_eq!(pool.used_bytes().unwrap(), c + e);
                },
                || {},
            );
            if one_short {
                let error = result.unwrap_err();
                assert_eq!(error.retained_bytes(), 0);
                assert!(matches!(
                    error.accounting_failure(),
                    Some(WorkingMemoryError::BudgetExceeded { .. })
                ));
                drop(source);
            } else {
                let output = result.unwrap();
                assert_eq!(output.ids(), expected.get_ids(), "{text:?}");
                assert!(output.matches_source(&source));
                assert!(!output.matches_source(&original));
                drop(source);
                assert_eq!(pool.used_bytes().unwrap(), c + e);
                drop(output);
            }
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
    // The whole ordered configuration remains authenticated, including repeats,
    // regex flags, and pre-tokenizer order; equivalent flattening is accepted.
    let mut changed: serde_json::Value = serde_json::from_str(&input).unwrap();
    changed["pre_tokenizer"]["pretokenizers"][1]["pretokenizers"][1]["add_prefix_space"] =
        false.into();
    let changed =
        eredu_text::tokenizer::Tokenizer::from_bytes(changed.to_string().as_bytes()).unwrap();
    assert!(!original.matches_configuration(&changed));
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn ordered_pretokenizers_refuse_each_destination_and_late_regex_with_exact_custody() {
    let input = json();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let c = original.original_bytes();
    let text = "HI12<S>hi ?";
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&original, text, false).unwrap();
    // Actual stage table, first text/splits, second text/splits, prefix text.
    for stage in 11..17 {
        let error = pool
            .encode_tokenizer_ids_with(
                &original,
                text,
                false,
                |p| p.fail_reservation(stage),
                || {},
                || {},
            )
            .unwrap_err();
        assert!(reserve_leaf(&error));
        assert!(error.matches_source(&original));
        assert_eq!(error.retained_bytes(), e);
        assert_eq!(error.encoding_failure().unwrap().partial_id_count(), 0);
        assert_eq!(pool.used_bytes().unwrap(), c + e);
        drop(error.into_backend_failure());
        assert_eq!(pool.used_bytes().unwrap(), c);
    }
    for ordinal in 0..3 {
        for buffer in [RegexBuffer::Saves, RegexBuffer::Branches, RegexBuffer::Undo] {
            let error = pool
                .encode_tokenizer_ids_with(
                    &original,
                    text,
                    false,
                    |p| {
                        p.fail_regex_reservation_at(ordinal, RegexWorkspaceFailure::Outer(buffer))
                            .unwrap()
                    },
                    || {},
                    || {},
                )
                .unwrap_err();
            assert!(reserve_leaf(&error));
            assert_eq!(error.retained_bytes(), e);
            assert_eq!(pool.used_bytes().unwrap(), c + e);
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), c);
        }
    }
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // Fail after earlier regex sources have really completed, including the
    // final constructor's complete-program transport.
    for ordinal in 0..3 {
        for target in [
            RegexConstructionFailure::Pattern,
            RegexConstructionFailure::Instructions,
            RegexConstructionFailure::Completed,
        ] {
            let error = pool
                .compile_tokenizer(
                    TokenizerPlan::prepare_json(input.as_bytes())
                        .unwrap()
                        .fail_regex_construction_at(ordinal, target),
                )
                .unwrap_err();
            assert_eq!(error.retained_bytes(), c);
            assert!(error.compiler_failure().is_some());
            if !matches!(target, RegexConstructionFailure::Completed) {
                assert!(reserve_leaf(&error));
            }
            assert_eq!(pool.used_bytes().unwrap(), c);
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn ordered_pretokenizer_prefix_derivative_preserves_bytelevel_policy_and_root() {
    let input = json();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let derived = original.input_prefix_normalized_source().unwrap();
    let mut changed: serde_json::Value = serde_json::from_str(&input).unwrap();
    changed["normalizer"]["normalizers"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    changed["pre_tokenizer"]["pretokenizers"][1]["pretokenizers"][2]["prepend_scheme"] =
        "never".into();
    let ordinary =
        eredu_text::tokenizer::Tokenizer::from_bytes(changed.to_string().as_bytes()).unwrap();
    assert!(derived.matches_configuration(&ordinary));
    assert!(derived.matches_semantic_root(&original));
    for text in ["HI12<S>hi ?", "hi 12", "é\n٣¼🙂", "İΣẞ", "\u{301}hi"] {
        let output = pool.encode_tokenizer_ids(&derived, text, false).unwrap();
        assert_eq!(
            output.ids(),
            ordinary.encode(text, false).unwrap().get_ids(),
            "{text:?}"
        );
    }
    let bytes = pool.used_bytes().unwrap();
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(derived);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
