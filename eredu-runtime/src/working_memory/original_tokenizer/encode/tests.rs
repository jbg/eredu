use super::*;
use eredu_text::tokenizer_storage::TokenizerPlan;
use std::error::Error as _;
const JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[{"id":5,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2," ":3,"?":4},"merges":[["h","i"]],"unk_token":"?"}}"#;
const TEXT: &str = "hi<S>hi ?";
fn source(pool: &WorkingMemoryPool, json: &str) -> OriginalTokenizer {
    pool.compile_tokenizer(TokenizerPlan::prepare_json(json.as_bytes()).unwrap())
        .unwrap()
}
fn sizes() -> (u64, u64) {
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let source = source(&pool, JSON);
    (
        source.original_bytes(),
        WorkingMemoryPool::tokenizer_encode_required_bytes(&source, TEXT, false).unwrap(),
    )
}
#[test]
fn exact_original_e_and_one_short_preserve_c_and_end_only_operation_activity() {
    let (c, e) = sizes();
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(c + e - u64::from(short), 0).unwrap();
        let source = source(&pool, JSON);
        let input = TEXT.to_owned();
        let result = pool.encode_tokenizer_ids_with(
            &source,
            &input,
            false,
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
        drop(input);
        if short {
            let error = result.unwrap_err();
            assert_eq!(error.retained_bytes(), 0);
            assert!(!error.matches_source(&source));
            assert!(
                matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == e && *available_bytes == e-1)
            );
            assert_eq!(pool.used_bytes().unwrap(), c);
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            drop(error);
        } else {
            let ids = result.unwrap();
            assert_eq!(ids.ids(), [2, 5, 2, 3, 4]);
            assert!(ids.matches_source(&source));
            assert_eq!(ids.original_bytes(), e);
            drop(pool.acquire_unquoted().unwrap());
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), c + e);
            let ptr = ids.ids().as_ptr();
            assert_eq!(ids.ids().as_ptr(), ptr);
            drop(ids);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn real_partial_reserves_keep_exact_source_and_e_through_core_envelope_retirement() {
    let (c, e) = sizes();
    for stage in 0..3 {
        let pool = WorkingMemoryPool::new(c + e, 0).unwrap();
        let source = source(&pool, JSON);
        let error = pool
            .encode_tokenizer_ids_with(
                &source,
                TEXT,
                false,
                |p| p.fail_reservation(stage),
                || {},
                || {},
            )
            .unwrap_err();
        assert!(error.matches_source(&source));
        assert_eq!(error.retained_bytes(), e);
        let failure = error.encoding_failure().unwrap();
        for (index, capacity) in failure.capacities().iter().enumerate() {
            assert_eq!(*capacity > 0, index < stage);
        }
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(failure);
        let mut reserve = false;
        while let Some(error) = cause {
            reserve |= error.is::<std::collections::TryReserveError>();
            cause = error.source();
        }
        assert!(reserve);
        drop(pool.acquire_unquoted().unwrap());
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), c + e);
        let error = error.into_backend_failure();
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<OriginalTokenizerEncodeError>()
                .unwrap()
                .retained_bytes(),
            e
        );
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn foreign_and_profile_rejections_do_not_clone_source_or_consume_any_allowance() {
    let (c, e) = sizes();
    let pool = WorkingMemoryPool::new(c * 2 + e, 0).unwrap();
    let source = source(&pool, JSON);
    let foreign = WorkingMemoryPool::new(c + e, 0).unwrap();
    let rejects: Vec<_> = (0..32)
        .map(|_| {
            foreign
                .encode_tokenizer_ids(&source, TEXT, false)
                .unwrap_err()
        })
        .collect();
    assert!(rejects.iter().all(|e| matches!(
        e.accounting_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ) && e.retained_bytes() == 0));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let transformed = JSON.replace(
        "\"post_processor\":null",
        "\"post_processor\":{\"type\":\"ByteLevel\",\"trim_offsets\":true}",
    );
    let transformed = source_fn(&pool, &transformed);
    let rejects2: Vec<_> = (0..32)
        .map(|_| {
            pool.encode_tokenizer_ids(&transformed, TEXT, false)
                .unwrap_err()
        })
        .collect();
    assert!(rejects2.iter().all(|e| matches!(
        e.profile_failure(),
        Some(EncodeIdsError::PipelineProfile)
    ) && e.retained_bytes() == 0));
    drop(transformed);
    let ids = pool.encode_tokenizer_ids(&source, TEXT, true).unwrap();
    assert_eq!(ids.ids(), [2, 5, 2, 3, 4]);
    drop((source, ids));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop((rejects, rejects2));
}
fn source_fn(pool: &WorkingMemoryPool, json: &str) -> OriginalTokenizer {
    source(pool, json)
}
#[test]
fn unknown_token_terminal_failure_and_empty_success_keep_actual_operation_lifetimes() {
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let json = JSON.replace("\"unk_token\":\"?\"", "\"unk_token\":\"missing\"");
    let source = source(&pool, &json);
    let c = source.original_bytes();
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&source, "<S>hz", false).unwrap();
    let error = pool
        .encode_tokenizer_ids(&source, "<S>hz", false)
        .unwrap_err();
    assert!(matches!(
        error.encoding_failure().unwrap().cause(),
        EncodeIdsError::MissingUnknown
    ));
    assert_eq!(error.encoding_failure().unwrap().partial_id_count(), 1);
    let empty = pool.encode_tokenizer_ids(&source, "", true).unwrap();
    assert!(empty.ids().is_empty());
    assert_eq!(empty.capacities(), [0, 0, 0]);
    let empty_e = empty.original_bytes();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), c + e + empty_e);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), c + empty_e);
    drop(empty);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn source_and_actual_destinations_retire_before_e_on_unwind_and_poison_is_conservative() {
    unwind_case(JSON, TEXT, false);
}
fn unwind_case(json: &str, text: &str, special: bool) {
    let sizing = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let measured = source(&sizing, json);
    let c = measured.original_bytes();
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&measured, text, special).unwrap();
    let normalization = EncodeIdsPlan::prepare(&measured.payload().model, text, special)
        .unwrap()
        .normalization_capacities();
    drop(measured);
    for phase in 0..2 {
        let pool = WorkingMemoryPool::new(c + e, 0).unwrap();
        let source = source(&pool, json);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.encode_tokenizer_ids_with(
                &source,
                text,
                special,
                |p| p,
                || {
                    if phase == 0 {
                        panic!("admitted operation unwind")
                    }
                },
                || {
                    if phase == 1 {
                        panic!("completed operation unwind")
                    }
                },
            )
        }));
        assert!(result.is_err());
        assert_eq!(pool.used_bytes().unwrap(), c);
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let pool = WorkingMemoryPool::new(c + e, 0).unwrap();
    let source = source(&pool, json);
    let error = pool
        .encode_tokenizer_ids_with(
            &source,
            text,
            special,
            |p| p,
            || {},
            || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _guard = pool.0.usage.lock().unwrap();
                    panic!("terminal usage poison");
                }));
            },
        )
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Poisoned)
    ));
    assert!(error.completed_capacities().unwrap().iter().all(|n| *n > 0));
    assert_eq!(
        error.completed_normalization_capacities(),
        Some(normalization)
    );
    drop((source, error));
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(usage.reserved, c + e);
    assert_eq!(usage.reservations, 1);
}
#[test]
fn concurrent_original_operations_share_atomic_capacity_and_keep_independent_results() {
    concurrent_case(JSON, TEXT, false);
}
fn concurrent_case(json: &str, text: &str, special: bool) {
    let sizing = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let measured = source(&sizing, json);
    let c = measured.original_bytes();
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&measured, text, special).unwrap();
    drop(measured);
    let pool = WorkingMemoryPool::new(c + 2 * e, 0).unwrap();
    let source = source(&pool, json);
    let (a, b, arrivals, observation) = std::thread::scope(|scope| {
        let (notify, notifications) = std::sync::mpsc::channel();
        let (send_a, resume_a) = std::sync::mpsc::channel::<()>();
        let (send_b, resume_b) = std::sync::mpsc::channel::<()>();
        let make = |notify: std::sync::mpsc::Sender<bool>,
                    resume: std::sync::mpsc::Receiver<()>| {
            let mut entered = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.encode_tokenizer_ids_with(
                    &source,
                    text,
                    special,
                    |p| p,
                    || {
                        entered = true;
                        let _ = notify.send(true);
                        let _ = resume.recv();
                    },
                    || {},
                )
            }));
            if !entered {
                let _ = notify.send(false);
            }
            result
        };
        let peer = notify.clone();
        let a = scope.spawn(move || make(peer, resume_a));
        let b = scope.spawn(move || make(notify, resume_b));
        let arrivals = [notifications.recv(), notifications.recv()];
        let observation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let used = pool.used_bytes();
            let excluded = matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            );
            let rejected = pool
                .encode_tokenizer_ids(&source, text, special)
                .unwrap_err();
            (used, excluded, rejected)
        }));
        drop(send_a);
        drop(send_b);
        (a.join(), b.join(), arrivals, observation)
    });
    assert!(arrivals.iter().all(|x| matches!(x, Ok(true))));
    let (used, excluded, rejected) = observation.unwrap();
    assert_eq!(used.unwrap(), c + 2 * e);
    assert!(excluded);
    assert_eq!(rejected.retained_bytes(), 0);
    let a = a.unwrap().unwrap().unwrap();
    let b = b.unwrap().unwrap().unwrap();
    assert_eq!(a.ids(), b.ids());
    let normalization = EncodeIdsPlan::prepare(&source.payload().model, text, special)
        .unwrap()
        .normalization_capacities();
    assert_eq!(a.normalization_capacities(), normalization);
    assert_eq!(b.normalization_capacities(), normalization);
    assert_ne!(a.ids().as_ptr(), b.ids().as_ptr());
    assert!(a.matches_source(&source) && b.matches_source(&source));
    drop(source);
    drop(a);
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    drop(b);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(rejected);
}

mod regex;

mod template;

mod lfm;

mod nfc;
mod composition;

#[test]
fn wordlevel_whitespace_source_and_encoding_have_exact_original_custody() {
    const SOURCE: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"added_tokens":[],"model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{"[UNK]":0,"alpha":2,"é":5,"中文":9}}}"#;
    let c=WorkingMemoryPool::tokenizer_required_bytes(&TokenizerPlan::prepare_json(SOURCE.as_bytes()).unwrap()).unwrap();
    let short=WorkingMemoryPool::new(c-1,0).unwrap();
    let refusal=short.compile_tokenizer(TokenizerPlan::prepare_json(SOURCE.as_bytes()).unwrap()).unwrap_err();
    assert!(matches!(refusal.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{..})));
    assert_eq!(short.used_bytes().unwrap(),0);
    let pool=WorkingMemoryPool::new(64*1024*1024,0).unwrap();
    let tokenizer=source(&pool,SOURCE);
    assert_eq!(tokenizer.original_bytes(),c);
    let input="alpha\u{a0}é 中文 missing";
    let e=WorkingMemoryPool::tokenizer_encode_required_bytes(&tokenizer,input,false).unwrap();
    let ids=pool.encode_tokenizer_ids(&tokenizer,input,false).unwrap();
    assert_eq!(ids.ids(),[2,5,9,0]);
    assert!(ids.matches_source(&tokenizer));
    assert_eq!(pool.used_bytes().unwrap(),c+e);
    drop(tokenizer);
    assert_eq!(pool.used_bytes().unwrap(),c+e);
    drop(ids);
    assert_eq!(pool.used_bytes().unwrap(),0);
    let limited=WorkingMemoryPool::new(c+e-1,0).unwrap();
    let tokenizer=source(&limited,SOURCE);
    let error=limited.encode_tokenizer_ids(&tokenizer,input,false).unwrap_err();
    assert!(matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{..})));
    assert_eq!(limited.used_bytes().unwrap(),c);
}

#[test]
fn unigram_source_and_path_storage_retain_one_original_allowance() {
    const SOURCE: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"added_tokens":[],"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",-9.0],["a",-1.0],["ab",-1.0],["é",-1.0],[" ",-1.0]]}}"#;
    let c = WorkingMemoryPool::tokenizer_required_bytes(&TokenizerPlan::prepare_json(SOURCE.as_bytes()).unwrap()).unwrap();
    let short = WorkingMemoryPool::new(c - 1, 0).unwrap();
    let refusal = short.compile_tokenizer(TokenizerPlan::prepare_json(SOURCE.as_bytes()).unwrap()).unwrap_err();
    assert!(matches!(refusal.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { .. })));
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(64 * 1024 * 1024, 0).unwrap();
    let tokenizer = source(&pool, SOURCE);
    let input = "ab aé🙂";
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&tokenizer, input, false).unwrap();
    let ids = pool.encode_tokenizer_ids(&tokenizer, input, false).unwrap();
    assert_eq!(ids.ids(), [2, 4, 1, 3, 0]);
    assert!(ids.matches_source(&tokenizer));
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    drop(tokenizer);
    assert_eq!(pool.used_bytes().unwrap(), c + e);
    drop(ids);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let limited = WorkingMemoryPool::new(c + e - 1, 0).unwrap();
    let tokenizer = source(&limited, SOURCE);
    let refusal = limited.encode_tokenizer_ids(&tokenizer, input, false).unwrap_err();
    assert!(matches!(refusal.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { .. })));
    assert_eq!(limited.used_bytes().unwrap(), c);
}

mod pretokenizer;
