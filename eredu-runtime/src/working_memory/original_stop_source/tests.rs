use super::*;
fn inputs() -> Vec<String> {
    ["ENDtail", "é!", "END", "", "é!"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}
fn plan(strings: &[String]) -> StopCompilePlan<'_> {
    StopCompilePlan::prepare(strings).unwrap()
}

#[test]
fn exact_cold_admission_precedes_compiler_and_retains_full_idle_allowance() {
    let strings = inputs();
    let required = WorkingMemoryPool::stop_source_required_bytes(&plan(&strings)).unwrap();
    let short = WorkingMemoryPool::new(required - 1, 0).unwrap();
    let error = short
        .compile_stop_source_with(plan(&strings), || {
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
        .compile_stop_source_with(plan(&strings), || {
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
    let second = pool.compile_stop_source(plan(&strings)).unwrap_err();
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
    let strings = inputs();
    let mut previous_prefix = 0;
    for stage in 0..2 {
        let plan = plan(&strings).fail_reservation(stage);
        let required = WorkingMemoryPool::stop_source_required_bytes(&plan).unwrap();
        let pool = WorkingMemoryPool::new(required, 0).unwrap();
        let error = pool.compile_stop_source(plan).unwrap_err();
        assert_eq!(error.retained_bytes(), required);
        let failure = error.compiler_failure().unwrap();
        assert!(matches!(
            failure.cause(),
            eredu_text::stop_storage::StopSourceError::Allocation(_)
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
    let strings = inputs();
    let required = WorkingMemoryPool::stop_source_required_bytes(&plan(&strings)).unwrap();
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let unknown = pool.acquire_unquoted().unwrap();
    let error = pool
        .compile_stop_source_with(plan(&strings), || panic!("unknown bound invoked compiler"))
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(unknown);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.compile_stop_source_with(plan(&strings), || {
            assert_eq!(pool.used_bytes().unwrap(), required);
            panic!("injected before compile, under actual allowance");
        })
    }));
    assert!(panic.is_err());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(pool.acquire_unquoted().unwrap());
    drop(pool.compile_stop_source(plan(&strings)).unwrap());
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn independent_sources_and_concurrent_last_aliases_preserve_exact_domain_bytes() {
    let strings = inputs();
    let required = WorkingMemoryPool::stop_source_required_bytes(&plan(&strings)).unwrap();
    let pool = WorkingMemoryPool::new(2 * required, 0).unwrap();
    let first = pool.compile_stop_source(plan(&strings)).unwrap();
    let second = pool.compile_stop_source(plan(&strings)).unwrap();
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
    let strings = inputs();
    let required = WorkingMemoryPool::stop_source_required_bytes(&plan(&strings)).unwrap();
    let pool = WorkingMemoryPool::new(required, 0).unwrap();
    let error = pool
        .compile_stop_source_with(plan(&strings), || {
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
