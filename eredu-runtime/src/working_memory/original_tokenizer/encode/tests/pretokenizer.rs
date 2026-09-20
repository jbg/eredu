use super::*;

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
