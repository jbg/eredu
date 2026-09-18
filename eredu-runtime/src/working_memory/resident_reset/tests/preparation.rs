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
impl Drop for Scope { fn drop(&mut self) { ACTIVE.set(false); ESCAPED.with_borrow_mut(|value| *value=None); } }
pub(super) fn prepare(session: &Session, claim: &SessionResetClaim<'_>, pool: &WorkingMemoryPool,
    execution: &InferenceExecutionIdentity, state: u64) -> Result<Option<SessionResetPreparationFunding>, BackendFailure> {
    if !ACTIVE.get() { return Ok(None); }
    let before = pool.used_bytes().unwrap();
    let funding = pool.prepare_reset_metadata(session, claim, execution, state).map_err(BackendFailure::from_error)?;
    funding.metadata().reserve_metadata(37).map_err(BackendFailure::from_error)?;
    let held = pool.used_bytes().unwrap() - before;
    REQUIRED.set(state + claim.limits().safety_reserve_bytes + held + 91);
    let foreign = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    assert_eq!(funding.charge_communication(&foreign, 91), Err(WorkingMemoryError::IdentityMismatch));
    funding.charge_communication(pool, 91).map_err(BackendFailure::from_error)?;
    assert_eq!(pool.used_bytes().unwrap(), before + held, "policy does not reserve native bytes twice");
    REACHED.set(true);
    ESCAPED.with_borrow_mut(|value| *value=Some(funding.clone()));
    Ok(Some(funding))
}
#[test]
fn reset_preparation_policy_is_prospective_and_retains_its_real_host_account() {
    ACTIVE.set(true);
    let _scope=Scope;
    let (mut runtime,data,_,_)=fixture(None,3);
    runtime.reset_admitted(SessionResetLimits::new(10_000_000)).unwrap();
    let required=REQUIRED.get();
    drop(runtime);drop(data);
    ESCAPED.with_borrow_mut(|value| *value=None);
    for short in [true,false] {
        REACHED.set(false);
        let (mut runtime,data,baseline,_)=fixture(None,3);
        let pool=data.borrow().pool.clone();
        let mut limits=SessionResetLimits::new(10_000_000);
        limits.safety_reserve_bytes=19;
        limits.application_memory_budget_bytes=Some(required+19-u64::from(short));
        let result=runtime.reset_admitted(limits);
        assert_eq!(result.is_err(),short);
        assert_eq!(REACHED.get(),!short);
        if short {
            assert_eq!(FILLS.get(),0);
            assert_eq!(pool.used_bytes().unwrap(),baseline);
            assert!(data.borrow().state.layers.slots().iter().all(|slot|slot.position==19));
        }
        drop(result);drop(runtime);drop(data);
        if !short { assert!(pool.used_bytes().unwrap()>0,"escaped policy handle retains its actual shared shell and account"); }
        ESCAPED.with_borrow_mut(|value| *value=None);
        assert_eq!(pool.used_bytes().unwrap(),0);
    }
}
