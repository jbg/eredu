use super::*;
use eredu_text::tokenizer_storage::{
    NormalizationBuffer as N, RegexBuffer as B, RegexWorkspaceFailure as F,
};
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
            let plan = EncodeIdsPlan::prepare(&measured.payload().model, text, special).unwrap();
            let caps = plan.normalization_capacities();
            let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&measured, text, special)
                .unwrap();
            if text == TEXT_NFC {
                assert_eq!(caps, [11, 11, 14]);
            }
            if text == "\u{344}" {
                assert_eq!(caps, [2, 2, 4]);
            }
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
                    assert_eq!(output.normalization_capacities(), caps);
                    assert_eq!(output.mapped_capacity(), 2 * caps[2]);
                    assert_eq!(output.capacities()[2], caps[2] + usize::from(special));
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
fn nfc_actual_seven_frontiers_and_later_regex_failure_preserve_original_c_e_and_causes() {
    let input = json();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, &input);
    let c = source.original_bytes();
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&source, TEXT_NFC, true).unwrap();
    let targets = [N::Decomposition, N::Recomposition, N::Text];
    for stage in 0..8 {
        let error = pool
            .encode_tokenizer_ids_with(
                &source,
                TEXT_NFC,
                true,
                |p| {
                    if stage < 4 {
                        p.fail_reservation(stage)
                    } else if stage < 7 {
                        p.fail_normalization_reservation(targets[stage - 4])
                            .unwrap()
                    } else {
                        p.fail_regex_reservation(F::Outer(B::Undo)).unwrap()
                    }
                },
                || {},
                || {},
            )
            .unwrap_err();
        assert!(error.matches_source(&source));
        assert_eq!(error.retained_bytes(), e);
        let failure = error.encoding_failure().unwrap();
        let caps = failure.normalization_capacities();
        if stage < 4 {
            assert_eq!(caps, [0; 3]);
        } else if stage < 7 {
            assert!(caps[..stage - 4].iter().all(|&n| n > 0));
            assert!(caps[stage - 4..].iter().all(|&n| n == 0));
            assert!(matches!(
                failure.cause(),
                EncodeIdsError::NormalizationPreparation(_)
            ));
        } else {
            assert_eq!(caps, [11, 11, 14]);
        }
        let mut next: Option<&(dyn std::error::Error + 'static)> = Some(failure);
        let mut reserve = false;
        while let Some(cause) = next {
            reserve |= cause.is::<std::collections::TryReserveError>();
            next = cause.source();
        }
        assert!(reserve);
        drop(pool.acquire_unquoted().unwrap());
        let erased = error.into_backend_failure();
        assert_eq!(pool.used_bytes().unwrap(), c + e);
        drop(erased);
        assert_eq!(pool.used_bytes().unwrap(), c);
    }
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
            .all(|e| e.retained_bytes() == 0 && !e.matches_source(&source))
    );
    let error = pool
        .encode_tokenizer_ids_with(
            &source,
            TEXT_NFC,
            true,
            |p| p.fail_normalization_reservation(N::Text).unwrap(),
            || {},
            || {},
        )
        .unwrap_err();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    assert_eq!(
        error.encoding_failure().unwrap().normalization_capacities(),
        [11, 11, 0]
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(rejects);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}
#[test]
fn nfc_missing_unknown_after_real_ids_retains_completed_normalization_storage() {
    let input = json().replace("\"unk_token\":\"?\"", "\"unk_token\":\"missing\"");
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let source = source(&pool, &input);
    let c = source.original_bytes();
    let text = "hi<S>\u{344}";
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&source, text, true).unwrap();
    let failure = pool.encode_tokenizer_ids(&source, text, true).unwrap_err();
    assert!(matches!(
        failure.encoding_failure().unwrap().cause(),
        EncodeIdsError::MissingUnknown
    ));
    assert_eq!(failure.encoding_failure().unwrap().partial_id_count(), 3); // BOS, hi, then the raw <S> boundary.
    assert_eq!(
        failure
            .encoding_failure()
            .unwrap()
            .normalization_capacities(),
        [7, 7, 9]
    );
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
