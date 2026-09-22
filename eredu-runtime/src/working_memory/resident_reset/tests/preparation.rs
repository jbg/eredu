//! Genuine reset claims share prospective application policy with independent
//! source producers; policy checks do not duplicate physical reservations.
use super::*;
use crate::working_memory::SessionResetPreparationFunding;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static REQUIRED: Cell<u64> = const { Cell::new(0) };
    static REACHED: Cell<bool> = const { Cell::new(false) };
    static ESCAPED: RefCell<Option<SessionResetPreparationFunding>> = const { RefCell::new(None) };
}
struct Scope;
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.set(false);
        ESCAPED.with_borrow_mut(|value| *value = None);
    }
}
pub(super) fn prepare(
    session: &Session,
    claim: &SessionResetClaim<'_>,
    pool: &MemoryLedger,
    execution: &InferenceExecutionIdentity,
    state: u64,
) -> Result<Option<SessionResetPreparationFunding>, BackendFailure> {
    if !ACTIVE.get() {
        return Ok(None);
    }
    let before = pool.payload_used_bytes().unwrap();
    let funding = pool
        .prepare_reset_metadata(session, claim, execution, state)
        .map_err(BackendFailure::from_error)?;
    funding
        .metadata()
        .reserve_metadata(37)
        .map_err(BackendFailure::from_error)?;
    let held = pool.payload_used_bytes().unwrap() - before;
    REQUIRED.set(state + reset_headroom(claim.limits(), pool) + held);
    let foreign = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    assert_eq!(
        funding.validate_ledger(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    funding
        .validate_ledger(pool)
        .map_err(BackendFailure::from_error)?;
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        before + held,
        "policy does not reserve native bytes twice"
    );
    REACHED.set(true);
    ESCAPED.with_borrow_mut(|value| *value = Some(funding.clone()));
    Ok(Some(funding))
}
#[test]
fn reset_preparation_policy_is_prospective_and_retains_its_real_host_account() {
    ACTIVE.set(true);
    let _scope = Scope;
    let (mut runtime, data, _, _) = fixture(None, 3);
    runtime
        .reset_admitted(SessionResetLimits::new(
            crate::working_memory::memory_fixture::host_limits(10_000_000),
        ))
        .unwrap();
    let required = REQUIRED.get();
    drop(runtime);
    drop(data);
    ESCAPED.with_borrow_mut(|value| *value = None);
    for short in [true, false] {
        REACHED.set(false);
        let (mut runtime, data, baseline, _) = fixture(None, 3);
        let pool = data.borrow().pool.clone();
        let mut limits = SessionResetLimits::new(
            crate::working_memory::memory_fixture::host_limits(10_000_000),
        );
        limits.additional_headroom =
            eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 19)]);
        limits.memory_limits = crate::working_memory::memory_fixture::host_limits(
            baseline + required + 19 + registry_controls(&pool) - u64::from(short),
        );
        let result = runtime.reset_admitted(limits);
        assert_eq!(result.is_err(), short);
        assert!(
            REACHED.get(),
            "preparation is admitted before the final state transaction"
        );
        if short {
            assert_eq!(FILLS.get(), 0);
            let held = REQUIRED.get() - data.borrow().plan().required_bytes() - 19;
            assert_eq!(
                pool.payload_used_bytes().unwrap(),
                baseline + held,
                "escaped preparation retains its paid metadata after rejection"
            );
            assert!(
                data.borrow()
                    .state
                    .layers
                    .slots()
                    .iter()
                    .all(|slot| slot.position == 19)
            );
        }
        drop(result);
        drop(runtime);
        drop(data);
        if !short {
            assert!(
                pool.payload_used_bytes().unwrap() > 0,
                "escaped policy handle retains its actual shared shell and account"
            );
        }
        ESCAPED.with_borrow_mut(|value| *value = None);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
