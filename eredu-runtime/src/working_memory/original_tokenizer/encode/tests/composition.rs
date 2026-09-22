use super::*;

fn json() -> String {
    serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "normalizer":{"type":"Sequence","normalizers":[
            {"type":"Prepend","prepend":"É"},{"type":"Lowercase"},
            {"type":"Replace","pattern":{"String":"éa"},"content":"a"},{"type":"NFC"}]},
        "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
        "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
        "post_processor":null,
        "added_tokens":[{"id":6,"content":"<R>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":7,"content":"É","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"▁a":1,"▁é":2,"a":3,"é":4,"▁éé":5},"unk_token":"[UNK]"}}).to_string()
}

#[test]
fn ordered_normalization_matches_ordinary_and_refuses_before_source_and_operation_work() {
    let input = json();
    let ordinary = eredu_text::tokenizer::Tokenizer::from_bytes(input.as_bytes()).unwrap();
    let measure = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let original = source(&measure, &input);
    assert!(original.matches_configuration(&ordinary));
    let c = original.original_bytes();
    let short = crate::working_memory::memory_fixture::host_ledger(c - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_with(
            TokenizerPlan::prepare_json(input.as_bytes()).unwrap(),
            || panic!("cold refusal must precede source producers"),
        )
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    for text in ["", "A E\u{301}<R>A", "É", "İΣẞ", "\u{301}\u{300}A"] {
        let e = MemoryLedger::tokenizer_encode_required_bytes(&original, text, false).unwrap();
        let expected = ordinary.encode(text, false).unwrap();
        for short in [true, false] {
            let pool =
                crate::working_memory::memory_fixture::host_ledger(c + e - u64::from(short), 0)
                    .unwrap();
            let source = source(&pool, &input);
            let result = pool.encode_tokenizer_ids_with(
                &source,
                text,
                false,
                |p| p,
                || {
                    assert!(!short);
                    assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
                },
                || {},
            );
            if short {
                let error = result.unwrap_err();
                assert!(matches!(
                    error.accounting_failure(),
                    Some(WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    ))
                ));
                assert_eq!(error.retained_bytes(), 0);
                drop(source);
                assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            } else {
                let output = result.unwrap();
                assert_eq!(output.ids(), expected.get_ids(), "{text:?}");
                assert!(output.matches_source(&source));
                drop(source);
                assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
                drop(output);
                assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            }
        }
    }
    drop(original);
    assert_eq!(measure.payload_used_bytes().unwrap(), 0);
}

#[test]
fn ordered_normalization_prefix_derivative_refreshes_actual_added_patterns() {
    let input = json();
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let derived = original.input_prefix_normalized_source().unwrap();
    assert!(!derived.same_source(&original));
    assert!(derived.matches_semantic_root(&original));
    let mut ordinary_json: serde_json::Value = serde_json::from_str(&input).unwrap();
    ordinary_json["normalizer"]["normalizers"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    ordinary_json["pre_tokenizer"]["prepend_scheme"] = "never".into();
    let ordinary =
        eredu_text::tokenizer::Tokenizer::from_bytes(ordinary_json.to_string().as_bytes()).unwrap();
    for text in ["A", "É", "E\u{301}", "<R>A É", "İΣẞ"] {
        let output = pool.encode_tokenizer_ids(&derived, text, false).unwrap();
        assert_eq!(
            output.ids(),
            ordinary.encode(text, false).unwrap().get_ids(),
            "{text:?}"
        );
    }
    let bytes = original.original_bytes() + derived.original_bytes();
    drop(original);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(derived);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
