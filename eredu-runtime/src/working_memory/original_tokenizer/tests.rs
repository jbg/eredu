use super::*;
use std::error::Error as _;
const INPUT: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"post_processor":{"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"added_tokens":[{"id":4,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2,"Ġ":3},"merges":[["h","i"]]}}"#;
mod derivative;
mod metaspace_decoder;
fn plan(input: &str) -> TokenizerPlan<'_> {
    TokenizerPlan::prepare_json(input.as_bytes()).unwrap()
}
fn values(source: &OriginalTokenizer) {
    assert_eq!(source.token_id("hi"), Some(2));
    assert_eq!(source.spelling(4), Some("<S>"));
    assert!(source.is_special("<S>"));
    assert_eq!(source.token_count(), 5);
}
#[test]
fn one_original_comparison_precedes_all_reserves_and_idle_aliases_keep_exact_source_charge() {
    let input = INPUT.to_owned();
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(&input)).unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_with(plan(&input), || panic!("short admission entered compiler"))
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes}) if *required_bytes==bytes&&*available_bytes==bytes-1)
    );
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let source = pool
        .compile_tokenizer_with(plan(&input), || {
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    drop(input);
    values(&source);
    assert_eq!(source.original_bytes(), bytes);
    drop(pool.acquire_unquoted().unwrap());
    let alias = source.clone();
    assert!(source.same_source(&alias));
    let rejected: Vec<_> = (0..32)
        .map(|_| pool.compile_tokenizer(plan(INPUT)).unwrap_err())
        .collect();
    assert!(rejected.iter().all(|e| e.retained_bytes() == 0));
    let foreign = WorkingMemoryPool::new(bytes, 0).unwrap();
    assert!(matches!(
        alias.validate_pool(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let witness = pool.clone();
    drop((source, pool));
    values(&alias);
    assert_eq!(witness.used_bytes().unwrap(), bytes);
    let peer = alias.clone();
    std::thread::scope(|s| {
        s.spawn(move || drop(alias));
        s.spawn(move || drop(peer));
    });
    assert_eq!(witness.used_bytes().unwrap(), 0);
    assert_eq!(witness.peak_bytes().unwrap(), bytes);
    drop(rejected);
}
#[test]
fn decoder_allocation_and_upstream_config_errors_retain_source_admission() {
    for stage in 0..3 {
        let input = INPUT.to_owned();
        let plan = plan(&input).fail_decode_reservation(stage);
        let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let error = pool.compile_tokenizer(plan).unwrap_err();
        drop(input);
        assert_eq!(error.retained_bytes(), bytes);
        let failure = error.compiler_failure().unwrap();
        assert!(failure.completed_tokenizer());
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(failure);
        let mut found = false;
        while let Some(e) = cause {
            found |= e.is::<std::collections::TryReserveError>();
            cause = e.source();
        }
        assert!(found, "stage {stage}");
        drop(pool.acquire_unquoted().unwrap());
        let erased = BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
        assert_eq!(
            erased
                .source()
                .unwrap()
                .downcast_ref::<OriginalTokenizerError>()
                .unwrap()
                .retained_bytes(),
            bytes
        );
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        drop(erased);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let input = INPUT.replace(
        "\"h\":0,\"i\":1,\"hi\":2,\"Ġ\":3",
        "\"h\":0,\"i\":0,\"hi\":2,\"Ġ\":3",
    );
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(&input)).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let error = pool.compile_tokenizer(plan(&input)).unwrap_err();
    drop(input);
    let failure = error.compiler_failure().unwrap();
    assert!(failure.completed_tokenizer());
    assert!(matches!(
        std::error::Error::source(failure)
            .unwrap()
            .downcast_ref::<eredu_text::decoder_storage::DecodeSourceError>(),
        Some(eredu_text::decoder_storage::DecodeSourceError::DuplicateModelId(0))
    ));
    assert_eq!(error.retained_bytes(), bytes);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn independent_active_compilers_share_one_atomic_capacity_comparison() {
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes * 2, 0).unwrap();
    let (first, second, observations, arrivals) = std::thread::scope(|s| {
        let (notify, notifications) = std::sync::mpsc::channel();
        let (release_first, resume_first) = std::sync::mpsc::channel::<()>();
        let (release_second, resume_second) = std::sync::mpsc::channel::<()>();
        let make = |notify: std::sync::mpsc::Sender<bool>,
                    resume: std::sync::mpsc::Receiver<()>| {
            let mut admitted = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.compile_tokenizer_with(plan(INPUT), || {
                    admitted = true;
                    let _ = notify.send(true);
                    // Disconnection releases the worker even if the parent unwinds.
                    let _ = resume.recv();
                })
            }));
            if !admitted {
                // Admission rejection and a pre-hook panic must both wake the parent.
                let _ = notify.send(false);
            }
            result
        };
        let peer_notify = notify.clone();
        let first = s.spawn(move || make(peer_notify, resume_first));
        let second = s.spawn(move || make(notify, resume_second));
        let arrivals = [notifications.recv(), notifications.recv()];
        let observations = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let used = pool.used_bytes();
            let excluded = matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            );
            let reservations = pool.0.usage.lock().unwrap().reservations;
            (used, excluded, reservations)
        }));
        // Release and join both workers before any assertion, including observations.
        drop(release_first);
        drop(release_second);
        (first.join(), second.join(), observations, arrivals)
    });
    assert!(arrivals.iter().all(|result| matches!(result, Ok(true))));
    let (used, excluded, reservations) = observations.unwrap();
    assert_eq!(used.unwrap(), bytes * 2);
    assert!(excluded);
    assert_eq!(reservations, 2);
    let first = first.unwrap().unwrap().unwrap();
    let second = second.unwrap().unwrap().unwrap();
    assert!(!first.same_source(&second));
    values(&first);
    values(&second);
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.reservations, 0);
    }
    drop(pool.acquire_unquoted().unwrap());
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unknown_host_ownership_and_unwind_do_not_reach_an_unguarded_constructor() {
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let host = pool.acquire_unquoted().unwrap();
    let error = pool
        .compile_tokenizer_with(plan(INPUT), || panic!("unknown host compiled"))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(error.retained_bytes(), 0);
    drop(host);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.compile_tokenizer_with(plan(INPUT), || {
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            panic!("after actual original admission");
        })
    }));
    assert!(panic.is_err());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
    let model = pool.compile_tokenizer(plan(INPUT)).unwrap();
    values(&model);
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn poisoned_settlement_keeps_actual_completed_or_partial_model_and_unsettled_charge() {
    for partial in [false, true] {
        let plan = if partial {
            plan(INPUT).fail_decode_reservation(2)
        } else {
            plan(INPUT)
        };
        let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let error = pool
            .compile_tokenizer_with(plan, || {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _usage = pool.0.usage.lock().unwrap();
                        panic!("poison exact source account");
                    }))
                    .is_err()
                );
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), bytes);
        if partial {
            assert!(
                error
                    .compiler_failure()
                    .unwrap()
                    .decode_failure()
                    .unwrap()
                    .retained_buffer_bytes()
                    > 0
            );
            assert!(error._completed.is_none());
        } else {
            assert_eq!(error._completed.as_ref().unwrap().spelling(2), Some("hi"));
        }
        {
            let usage = pool.0.usage.lock().unwrap_err().into_inner();
            assert_eq!(usage.reservations, 1);
            assert_eq!(usage.reserved, bytes);
        }
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(usage.reserved, bytes);
    }
}

mod generation_domain;

#[test]
fn forbidden_lexical_source_uses_exact_tokenizer_domain_and_independent_original_copy() {
    let tokenizer_bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let tokenizer = pool.compile_tokenizer(plan(INPUT)).unwrap();
    let packed = tokenizer.token_byte_vocabulary().unwrap();
    let source_bytes =
        WorkingMemoryPool::forbidden_tokenizer_source_required_bytes(&packed, 2).unwrap();
    drop(packed);
    let source = pool
        .compile_forbidden_tokenizer_source(&tokenizer, b"hi")
        .unwrap();
    assert_eq!(source.inputs().token_bytes(0), Some(b"h".as_slice()));
    assert_eq!(source.inputs().token_bytes(2), Some(b"hi".as_slice()));
    assert_eq!(source.inputs().token_bytes(3), Some(b" ".as_slice()));
    assert_eq!(source.inputs().token_bytes(4), Some(b"<S>".as_slice()));
    assert_eq!(source.inputs().trigger(), b"hi");
    assert_eq!(pool.used_bytes().unwrap(), tokenizer_bytes + source_bytes);
    let foreign = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let error = foreign
        .compile_forbidden_tokenizer_source(&tokenizer, b"hi")
        .unwrap_err();
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    drop(error);
    let short = WorkingMemoryPool::new(tokenizer_bytes + source_bytes - 1, 0).unwrap();
    let other = short.compile_tokenizer(plan(INPUT)).unwrap();
    let error = short
        .compile_forbidden_tokenizer_source(&other, b"hi")
        .unwrap_err();
    assert_eq!(short.used_bytes().unwrap(), tokenizer_bytes);
    drop((error, other));
    assert_eq!(short.used_bytes().unwrap(), 0);
    let escaped = source.inputs().clone();
    drop((source, tokenizer));
    assert_eq!(pool.used_bytes().unwrap(), source_bytes);
    assert_eq!(escaped.token_bytes(4), Some(b"<S>".as_slice()));
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_fallback_literal_sources_hold_exact_custody_on_success_and_partial_allocation() {
    let input = INPUT.replace(
        r#""decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}"#,
        r#""decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"\u2581"},"content":" "}]}"#);
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(&input)).unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_with(plan(&input), || panic!("short source entered constructor"))
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    for failure in [None, Some(0), Some(1), Some(2)] {
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let selected = match failure {
            Some(stage) => plan(&input).fail_decode_reservation(stage),
            None => plan(&input),
        };
        let result = pool.compile_tokenizer(selected);
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        match result {
            Ok(source) => {
                values(&source);
                let escaped = source.clone();
                drop(source);
                assert_eq!(pool.used_bytes().unwrap(), bytes);
                drop(escaped);
            }
            Err(error) => {
                let failure = error.compiler_failure().unwrap();
                assert!(failure.completed_tokenizer());
                assert!(failure.decode_failure().is_some());
                assert_eq!(error.retained_bytes(), bytes);
                let escaped =
                    BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
                assert_eq!(pool.used_bytes().unwrap(), bytes);
                drop(escaped);
            }
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), bytes);
    }
}

#[test]
fn original_trie_keeps_exact_tokenizer_bytes_metadata_and_failed_prefix_custody() {
    use eredu_text::token_trie_storage::TokRxInfo;
    let tokenizer_bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    let probe_pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let probe = probe_pool.compile_tokenizer(plan(INPUT)).unwrap();
    let mut info = TokRxInfo::new(5, 4);
    info.tok_pad = Some(3);
    info.tok_end_of_turn = Some(4);
    let eos = [4, 3, 4];
    let bytes = WorkingMemoryPool::token_trie_source_required_bytes(&probe, &info, &eos).unwrap();
    drop(probe);
    assert_eq!(probe_pool.used_bytes().unwrap(), 0);
    let short = WorkingMemoryPool::new(tokenizer_bytes + bytes - 1, 0).unwrap();
    let source = short.compile_tokenizer(plan(INPUT)).unwrap();
    let error = source.compile_token_trie_source(&info, &eos).unwrap_err();
    assert!(
        matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == bytes && *available_bytes == bytes - 1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), tokenizer_bytes);
    drop((source, error));
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(tokenizer_bytes + bytes, 0).unwrap();
    let source = pool.compile_tokenizer(plan(INPUT)).unwrap();
    let trie = source.compile_token_trie_source(&info, &eos).unwrap();
    assert_eq!(trie.original_bytes(), bytes);
    assert_eq!(pool.used_bytes().unwrap(), tokenizer_bytes + bytes);
    trie.validate(&pool, &source, &info, &eos).unwrap();
    assert_eq!(trie.trie().info(), &info);
    assert_eq!(trie.trie().eos_tokens(), eos);
    for (id, expected) in [b"h".as_slice(), b"i", b"hi", b" ", b"\xff<S>"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(trie.trie().token(id as u32), expected);
    }
    assert_eq!(trie.trie().token_id(b"hi"), Some(2));
    let other_pool = WorkingMemoryPool::new(tokenizer_bytes, 0).unwrap();
    let other = other_pool.compile_tokenizer(plan(INPUT)).unwrap();
    assert!(matches!(
        trie.validate(&pool, &other, &info, &eos),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        trie.validate(&other_pool, &source, &info, &eos),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut changed = info;
    changed.tok_pad = None;
    assert!(matches!(
        trie.validate(&pool, &source, &changed, &eos),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        trie.validate(&pool, &source, &info, &[4, 4, 3]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let alias = trie.clone();
    assert!(alias.same_source(&trie));
    drop((trie, source, other));
    assert_eq!(other_pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.used_bytes().unwrap(), tokenizer_bytes + bytes);
    assert_eq!(alias.trie().token(4), b"\xff<S>");
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    // Unrepresentable parent-pop geometry is rejected before calling the stock
    // builder. Packed lexical buffers and source custody survive type erasure.
    let word = "x".repeat(1025);
    let input = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],
        "model":{"type":"BPE","vocab":{(word):0},"merges":[]}}).to_string();
    let tokenizer_bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(&input)).unwrap();
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let source = pool.compile_tokenizer(plan(&input)).unwrap();
    let info = TokRxInfo::new(1, 0);
    let policy = eredu_text::token_trie_storage::TokenTrieMemoryPolicy {
        max_token_bytes: 2048,
        ..Default::default()
    };
    let bytes = WorkingMemoryPool::token_trie_source_required_bytes_with_memory_policy(
        &source,
        &info,
        &[0],
        policy,
    )
    .unwrap();
    let error = pool
        .compile_token_trie_source_with_memory_policy(&source, &info, &[0], policy)
        .unwrap_err();
    assert_eq!(error.retained_bytes(), bytes);
    let failure = error.construction_failure().unwrap();
    assert!(failure.packed_capacity() > 1025);
    assert!(matches!(
        failure.source_error(),
        Some(eredu_text::token_trie_storage::TokenTrieSourceError::NodeEncoding)
    ));
    assert!(failure.retained_buffer_bytes() > failure.packed_capacity());
    drop((source, input));
    let error = BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
    assert_eq!(pool.used_bytes().unwrap(), tokenizer_bytes + bytes);
    assert!(
        error
            .source()
            .unwrap()
            .is::<super::super::OriginalTokenTrieSourceError>()
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
