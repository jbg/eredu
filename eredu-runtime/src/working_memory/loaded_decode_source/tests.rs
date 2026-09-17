use super::*;
use eredu_text::tokenizer::{Tokenizer, TokenizerSnapshot};
fn snapshot() -> TokenizerSnapshot {
    Tokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"é":8,"🦀":90,"[UNK]":1},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap().snapshot()
}
fn plan(snapshot: &TokenizerSnapshot) -> DecodeCompilePlan<'_> {
    DecodeCompilePlan::prepare(snapshot).unwrap()
}

#[test]
fn exact_cold_admission_precedes_compiler_and_retains_full_idle_allowance() {
    let snapshot = snapshot();
    let required = WorkingMemoryPool::decode_source_required_bytes(&plan(&snapshot)).unwrap();
    let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
    let error = short
        .compile_decode_source_with(plan(&snapshot), || {
            panic!("short admission reached compiler")
        })
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == required && *available_bytes == required - 1)
    );
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let source = pool
        .compile_decode_source_with(plan(&snapshot), || {
            assert_eq!(pool.used_bytes().unwrap(), required);
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        })
        .unwrap();
    assert_eq!(source.original_bytes(), required);
    assert_eq!(pool.used_bytes().unwrap(), required);
    // Terminal compilation ends the active operation, not its byte allowance.
    drop(pool.acquire_unquoted().unwrap());
    let second = pool.compile_decode_source(plan(&snapshot)).unwrap_err();
    assert!(matches!(
        second.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), required);
}

#[test]
fn every_real_partial_compiler_failure_retains_its_original_allowance() {
    let snapshot = snapshot();
    let mut previous_prefix = 0;
    for stage in 0..3 {
        let plan = plan(&snapshot).fail_reservation(stage);
        let required = WorkingMemoryPool::decode_source_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let error = pool.compile_decode_source(plan).unwrap_err();
        assert_eq!(error.retained_bytes(), required);
        let failure = error.compiler_failure().unwrap();
        assert!(matches!(
            failure.cause(),
            eredu_text::decoder_storage::DecodeSourceError::Allocation(_)
        ));
        let prefix = failure.retained_buffer_bytes();
        assert_eq!(prefix == 0, stage == 0);
        if stage > 0 {
            assert!(prefix > previous_prefix);
        }
        previous_prefix = prefix;
        assert_eq!(pool.used_bytes().unwrap(), required);
        drop(pool.acquire_unquoted().unwrap());
        // The actual owning error enters core's closed source Box directly.
        let error = BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error);
        assert_eq!(pool.used_bytes().unwrap(), required);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn unknown_host_owner_and_panicking_active_compiler_preserve_original_exclusion() {
    let snapshot = snapshot();
    let required = WorkingMemoryPool::decode_source_required_bytes(&plan(&snapshot)).unwrap();
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let unknown = pool.acquire_unquoted().unwrap();
    let error = pool
        .compile_decode_source_with(plan(&snapshot), || panic!("unknown bound invoked compiler"))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(unknown);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.compile_decode_source_with(plan(&snapshot), || {
            assert_eq!(pool.used_bytes().unwrap(), required);
            panic!("injected before compile, under actual allowance");
        })
    }));
    assert!(panic.is_err());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
    drop(pool.compile_decode_source(plan(&snapshot)).unwrap());
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn independent_sources_and_concurrent_last_aliases_preserve_exact_domain_bytes() {
    let snapshot = snapshot();
    let required = WorkingMemoryPool::decode_source_required_bytes(&plan(&snapshot)).unwrap();
    let pool = WorkingMemoryPool::new(2 * required, 0).unwrap();
    let first = pool.compile_decode_source(plan(&snapshot)).unwrap();
    let second = pool.compile_decode_source(plan(&snapshot)).unwrap();
    assert!(!first.same_source(&second));
    let alias = first.clone();
    assert!(first.same_source(&alias));
    assert!(matches!(
        first.validate_pool(&WorkingMemoryPool::new(u64::MAX, 0).unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    first.validate_pool(&pool).unwrap();
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), 2 * required);
    let peer = alias.clone();
    std::thread::scope(|scope| {
        scope.spawn(move || drop(alias));
        scope.spawn(move || drop(peer));
    });
    assert_eq!(pool.used_bytes().unwrap(), required);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn poisoned_terminal_accounting_keeps_completed_source_and_active_allowance_unsettled() {
    let snapshot = snapshot();
    let required = WorkingMemoryPool::decode_source_required_bytes(&plan(&snapshot)).unwrap();
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let error = pool
        .compile_decode_source_with(plan(&snapshot), || {
            let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _usage = pool.0.usage.lock().unwrap();
                panic!("poison only this actual compiler account");
            }));
            assert!(poisoned.is_err());
        })
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::Poisoned)
    ));
    assert!(error._completed.is_some());
    assert_eq!(error.retained_bytes(), required);
    {
        let usage = pool
            .0
            .usage
            .lock()
            .err()
            .expect("poisoned account")
            .into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(usage.reserved, required);
    }
    drop(error);
    let usage = pool
        .0
        .usage
        .lock()
        .err()
        .expect("poisoned account")
        .into_inner();
    assert_eq!(usage.reservations, 1);
    assert_eq!(usage.reserved, required);
}
