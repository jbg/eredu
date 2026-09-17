use super::*;
thread_local! { static DIRECT: Cell<bool> = const { Cell::new(false) }; }
pub(super) fn requested() -> bool {
    DIRECT.get()
}
struct Direct;
impl Direct {
    fn enter() -> Self {
        assert!(!DIRECT.replace(true));
        Self
    }
}
impl Drop for Direct {
    fn drop(&mut self) {
        DIRECT.set(false);
        FAIL_AT.set(None);
    }
}

#[test]
fn borrowed_existing_entries_use_one_original_comparison_and_exact_retry() {
    let _direct = Direct::enter();
    let (mut runtime, data, source_bytes, required) = fixture(None, 3);
    let pool = data.borrow().pool.clone();
    let key = data
        .borrow()
        .state
        .layers
        .metadata()
        .identity()
        .registry_key()
        .clone();
    for _ in 0..3 {
        let error = runtime
            .reset_admitted(SessionResetLimits::new(source_bytes + required - 1))
            .unwrap_err();
        let error = std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<ResidentResetError<State>>()
            .unwrap();
        assert_eq!(error.retained_bytes(), 0);
        assert_eq!(FILLS.get(), 0);
        assert_eq!(pool.used_bytes().unwrap(), source_bytes);
        assert_eq!(
            data.borrow()
                .state
                .layers
                .metadata()
                .identity()
                .registry_key(),
            &key
        );
        assert!(data
            .borrow()
            .state
            .layers
            .slots()
            .iter()
            .all(|s| s.position == 19 && s.values[1] == 7));
    }
    runtime
        .reset_admitted(SessionResetLimits::new(source_bytes + required))
        .unwrap();
    assert_eq!(FILLS.get(), 3);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes + required);
    let (pool, layout_bytes) = sources::release_ordinary_source(&data);
    assert_eq!(pool.used_bytes().unwrap(), layout_bytes + required);
    drop(runtime);
    drop(data);
    assert_eq!(
        pool.used_bytes().unwrap(),
        0,
        "no pin from failed admission survived"
    );
}

#[test]
fn borrowed_source_partial_failure_keeps_both_entries_after_all_callers_drop() {
    let _direct = Direct::enter();
    let (mut runtime, data, source_bytes, required) = fixture(None, 3);
    let pool = data.borrow().pool.clone();
    FAIL_AT.set(Some(2));
    let error = runtime
        .reset_admitted(SessionResetLimits::new(source_bytes + required))
        .unwrap_err();
    FAIL_AT.set(None);
    let typed = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert_eq!(typed.initialized_count(), 2);
    assert_eq!(typed.retained_bytes(), required);
    drop(runtime);
    drop(data);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes + required);
    KEY_DROP_CHECK.with_borrow_mut(|check| *check = Some((pool.clone(), required, 0)));
    drop(error);
    let (_, _, drops) = KEY_DROP_CHECK.with_borrow_mut(Option::take).unwrap();
    assert_eq!(
        drops, 2,
        "canonical table/layout keys retire outside Usage before refund"
    );
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn borrowed_entries_cannot_replace_the_actual_genuine_claim_source() {
    let _direct = Direct::enter();
    let (mut runtime, data, source_bytes, _) = fixture(None, 2);
    let (_, foreign, _, _) = fixture(None, 2);
    data.borrow_mut().foreign_source = Some(foreign);
    let error = runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap_err();
    let error = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ResidentResetError<State>>()
        .unwrap();
    assert!(matches!(
        error.cause,
        ResetCause::Memory(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(error.retained_bytes(), 0);
    assert_eq!(FILLS.get(), 0);
    assert_eq!(data.borrow().pool.used_bytes().unwrap(), source_bytes);
    data.borrow_mut().foreign_source = None;
    runtime
        .reset_admitted(SessionResetLimits::new(10_000_000))
        .unwrap();
    assert_eq!(FILLS.get(), 2);
}
