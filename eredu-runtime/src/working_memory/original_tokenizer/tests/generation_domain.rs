use super::*;
fn generation(input: &str) -> TokenizerPlan<'_> {
    plan(input).with_generation_domain().unwrap()
}
#[test]
fn overlapping_added_token_retains_paid_capacity_without_extending_logical_domain() {
    let input = INPUT.replace("\"Ġ\":3", "\"Ġ\":3,\"<S>\":4");
    let plan = generation(&input);
    assert_eq!(plan.generation_domain_extent(), Some(6));
    let bytes = MemoryLedger::tokenizer_required_bytes(&plan).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let source = pool.compile_tokenizer(plan).unwrap();
    let Some(TokenFilter::Allowed(mask)) = source.generation_domain() else {
        panic!("original mask")
    };
    assert_eq!(mask.as_slice(), &[true; 5]);
    assert_eq!(
        mask.capacity(),
        6,
        "logical canonicalization does not refund storage"
    );
    assert_eq!(source.original_bytes(), bytes);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn generation_domain_is_originally_admitted_and_all_strong_exits_retain_its_full_c() {
    let bytes = MemoryLedger::tokenizer_required_bytes(&generation(INPUT)).unwrap();
    let plain = MemoryLedger::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    // The dense vocabulary adds five mask entries and its source-operation
    // controls. Admission/retirement below use the complete published quote;
    // later encode errors do not determine source-compiler storage.
    assert!(bytes >= plain + 5);
    let short = crate::working_memory::memory_fixture::host_ledger(bytes - 1, 0).unwrap();
    let error = short
        .compile_tokenizer_inner(
            generation(INPUT),
            || panic!("one-short entered construction"),
            false,
            None,
        )
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let source = pool
        .compile_tokenizer_with(generation(INPUT), || {
            assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    assert_eq!(
        source.generation_domain(),
        Some(&TokenFilter::Allowed(vec![true; 5]))
    );
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    let alias = source.clone();
    let object = source.generation_domain().unwrap() as *const TokenFilter;
    assert!(std::ptr::eq(alias.generation_domain().unwrap(), object));
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    let peer = alias.clone();
    std::thread::scope(|scope| {
        scope.spawn(move || drop(alias));
        scope.spawn(move || drop(peer));
    });
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn actual_domain_reserve_failure_retains_completed_compiler_until_error_retirement() {
    let bytes = MemoryLedger::tokenizer_required_bytes(&generation(INPUT)).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let error = pool
        .compile_tokenizer_inner(generation(INPUT), || {}, true, None)
        .unwrap_err();
    assert!(error.domain_failure().is_some());
    assert_eq!(error.domain_capacity(), 0);
    assert!(error._completed.is_some());
    assert!(error._completed.as_ref().unwrap().spelling(2).is_some());
    assert_eq!(error.retained_bytes(), bytes);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    let error = BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    assert!(
        error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<TryReserveError>()
    );
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn sparse_domain_and_declared_added_ids_use_actual_fresh_assignment() {
    for declared in [0, 1, 4, u32::MAX] {
        let input = INPUT.replace("\"id\":4", &format!("\"id\":{declared}"));
        let bytes = MemoryLedger::tokenizer_required_bytes(&generation(&input)).unwrap();
        let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
        let source = pool.compile_tokenizer(generation(&input)).unwrap();
        assert_eq!(source.token_id("<S>"), Some(4));
        assert_eq!(
            source.generation_domain(),
            Some(&TokenFilter::Allowed(vec![true; 5]))
        );
        drop(source);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    let input = INPUT.replace("\"Ġ\":3", "\"Ġ\":100");
    let bytes = MemoryLedger::tokenizer_required_bytes(&generation(&input)).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let source = pool.compile_tokenizer(generation(&input)).unwrap();
    let TokenFilter::Allowed(mask) = source.generation_domain().unwrap() else {
        panic!()
    };
    assert_eq!(mask.len(), 101);
    assert_eq!(mask.iter().filter(|bit| **bit).count(), 5);
    for id in [0, 1, 2, 4, 100] {
        assert!(mask[id]);
    }
    assert!(!mask[3]);
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let largest = INPUT.replace("\"Ġ\":3", "\"Ġ\":4294967295");
    let base = MemoryLedger::tokenizer_required_bytes(&plan(&largest)).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(base, 0).unwrap();
    let decoder_only = pool.compile_tokenizer(plan(&largest)).unwrap();
    assert!(decoder_only.generation_domain().is_none());
    assert_eq!(decoder_only.spelling(u32::MAX), Some("Ġ"));
    drop(decoder_only);
    // Merely selecting/planning the dense source never reserves its huge mask.
    let selected = plan(&largest).with_generation_domain();
    if let Ok(extent) = usize::try_from(u64::from(u32::MAX) + 1) {
        if std::alloc::Layout::array::<bool>(extent).is_ok() {
            let selected = selected.unwrap();
            assert_eq!(selected.generation_domain_extent(), Some(extent));
            assert!(pool.compile_tokenizer(selected).is_err());
        } else {
            assert!(selected.is_err());
        }
    } else {
        assert!(selected.is_err());
    }
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn generation_domain_construction_unwind_retires_only_after_actual_compiler_scope() {
    let bytes = MemoryLedger::tokenizer_required_bytes(&generation(INPUT)).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.compile_tokenizer_with(generation(INPUT), || {
            assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
            panic!("after actual admission");
        });
    }));
    assert!(panic.is_err());
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
}

#[test]
fn original_text_error_wrappers_retain_actual_stop_reserve_prefix_and_late_encoding_cause() {
    use eredu_text::{stop_storage::StopCompilePlan, tokenizer_storage::EncodeIdsError};
    for stage in 0..2 {
        let values = ["first", "é!", "first", ""];
        let plan = StopCompilePlan::prepare_refs(&values)
            .unwrap()
            .fail_reservation(stage);
        let packed = plan.requirements().buffer_bytes();
        let bytes = MemoryLedger::stop_source_required_bytes(&plan).unwrap();
        let pool = crate::working_memory::memory_fixture::host_ledger(bytes, 0).unwrap();
        let error = OriginalTextSourceError::from(pool.compile_stop_source(plan).unwrap_err());
        let OriginalTextSourceError::Stop(cause) = &error else {
            panic!()
        };
        assert_eq!(cause.retained_bytes(), bytes);
        let failed = cause.compiler_failure().unwrap();
        assert_eq!(
            failed.retained_buffer_bytes(),
            if stage == 0 {
                0
            } else {
                packed - "firsté!".len()
            }
        );
        let mut at: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut allocation = false;
        while let Some(e) = at {
            allocation |= e.is::<TryReserveError>();
            at = e.source();
        }
        assert!(allocation);
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        drop(error);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
    // INPUT also tests a construction-only Digits profile; use the supported
    // identity encoding path to reach the real late missing-unknown failure.
    let json = INPUT
        .replace(
            r#""pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}"#,
            r#""pre_tokenizer":null"#,
        )
        .replace(
            "\"merges\":[[\"h\",\"i\"]]",
            "\"merges\":[[\"h\",\"i\"]],\"unk_token\":\"missing\"",
        );
    let pool = crate::working_memory::memory_fixture::host_ledger(10_000_000, 0).unwrap();
    let source = pool.compile_tokenizer(generation(&json)).unwrap();
    let c = source.original_bytes();
    let e = MemoryLedger::tokenizer_encode_required_bytes(&source, "<S>hz", false).unwrap();
    let error = OriginalTextSourceError::from(
        pool.encode_tokenizer_ids(&source, "<S>hz", false)
            .unwrap_err(),
    );
    let OriginalTextSourceError::Encode(cause) = &error else {
        panic!()
    };
    assert_eq!(cause.retained_bytes(), e);
    assert!(cause.matches_source(&source));
    assert!(matches!(
        cause.encoding_failure().unwrap().cause(),
        EncodeIdsError::Upstream(_)
    ));
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
