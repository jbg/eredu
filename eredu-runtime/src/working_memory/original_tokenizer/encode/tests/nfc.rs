use super::*;
const TEXT_NFC: &str = "hi e\u{301} \u{344}<S>";
fn json() -> String {
    let mut value: serde_json::Value = serde_json::from_str(&super::template::json()).unwrap();
    value["normalizer"] = serde_json::json!({"type":"NFC"});
    value["model"]["continuing_subword_prefix"] = serde_json::json!("");
    value["model"]["end_of_word_suffix"] = serde_json::json!("");
    // Keep the added boundary at its original ID when extending model vocab.
    value["model"]["vocab"]["<S>"] = serde_json::json!(5);
    value["model"]["vocab"]["Ã"] = serde_json::json!(6);
    value["model"]["vocab"]["©"] = serde_json::json!(7);
    value.to_string()
}
#[test]
fn nfc_expansion_exact_one_short_and_alias_retirement_keep_source_and_all_destinations() {
    let input = json();
    let sizing = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let measured = source(&sizing, &input);
    assert_eq!(measured.token_id("<S>"), Some(5));
    assert_eq!(measured.spelling(5), Some("<S>"));
    assert!(measured.is_special("<S>"));
    assert_eq!(measured.token_id("Ã"), Some(6));
    assert_eq!(measured.token_id("©"), Some(7));
    let c = measured.original_bytes();
    for text in [TEXT_NFC, "", "\u{344}"] {
        for special in [false, true] {
            let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&measured, text, special)
                .unwrap();
            for short in [true, false] {
                let pool = WorkingMemoryPool::new(c + e - u64::from(short), 0).unwrap();
                let source = source(&pool, &input);
                let result = pool.encode_tokenizer_ids_with(
                    &source,
                    text,
                    special,
                    |p| p,
                    || {
                        assert!(!short);
                        assert_eq!(pool.used_bytes().unwrap(), c + e);
                        assert!(matches!(
                            pool.acquire_unquoted(),
                            Err(WorkingMemoryError::ReservedWorkActive)
                        ));
                    },
                    || {},
                );
                if short {
                    let error = result.unwrap_err();
                    assert_eq!(error.retained_bytes(), 0);
                    assert!(!error.matches_source(&source));
                    assert!(
                        matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==e && *available_bytes==e-1)
                    );
                    drop(source);
                    assert_eq!(pool.used_bytes().unwrap(), 0);
                } else {
                    let output = result.unwrap();
                    let mut expected = if special { vec![5] } else { vec![] };
                    if text == TEXT_NFC {
                        expected.extend([2, 3, 6, 7, 3, 4, 4, 4, 4, 5]);
                    } else if !text.is_empty() {
                        expected.extend([4, 4, 4, 4]);
                    }
                    assert_eq!(output.ids(), expected);
                    assert!(output.matches_source(&source));
                    drop(pool.acquire_unquoted().unwrap());
                    let alias = source.clone();
                    drop(source);
                    drop(alias);
                    assert_eq!(pool.used_bytes().unwrap(), c + e);
                    drop(output);
                    assert_eq!(pool.used_bytes().unwrap(), 0);
                }
            }
        }
    }
    drop(measured);
    assert_eq!(sizing.used_bytes().unwrap(), 0);
}
#[test]
fn nfc_foreign_operations_cannot_retain_source_or_admission() {
    let input = json();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, &input);
    let c = source.original_bytes();
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let rejects: Vec<_> = (0..3)
        .map(|_| {
            foreign
                .encode_tokenizer_ids(&source, TEXT_NFC, true)
                .unwrap_err()
        })
        .collect();
    assert!(
        rejects
            .iter()
            .all(|error| error.retained_bytes() == 0 && !error.matches_source(&source))
    );
    assert_eq!(pool.used_bytes().unwrap(), c);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(rejects);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}

#[test]
fn nfc_missing_unknown_retains_source_and_operation_admission() {
    let input = json().replace("\"unk_token\":\"?\"", "\"unk_token\":\"missing\"");
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, &input);
    let c = source.original_bytes();
    let text = "hi<S>\u{344}";
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&source, text, true).unwrap();
    let failure = pool.encode_tokenizer_ids(&source, text, true).unwrap_err();
    assert!(matches!(
        failure.encoding_failure().unwrap().cause(),
        EncodeIdsError::Upstream(_)
    ));
    assert!(failure.matches_source(&source));
    assert_eq!(failure.retained_bytes(), e);
    assert!(failure.encoding_failure().unwrap().source().is_some());
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn independent_nfc_operations_share_only_immutable_source_and_one_atomic_capacity() {
    super::concurrent_case(&json(), TEXT_NFC, true);
}
#[test]
fn nfc_unwind_and_completed_poison_keep_actual_storage_under_existing_original_allowance() {
    super::unwind_case(&json(), TEXT_NFC, true);
}
