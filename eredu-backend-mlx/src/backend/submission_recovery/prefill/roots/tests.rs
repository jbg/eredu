use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
#[test]
fn partial_root_carrier_restores_same_owner_before_unwind_and_expired_views_reject() {
    let runtime = PrefillRootsRuntime::prepare();
    let (owner, view) = RootsOwner::new(&runtime, 1, None, None, None).unwrap();
    let array = Array::from_slice(&[2.0f32, 5.0], &[2]);
    view.append(&array).unwrap();
    assert!(matches!(
        view.append(&array),
        Err(Error::PrefillRoots(safemlx::PrefillRootsError::Refused(
            safemlx::PrefillRootsCause::Capacity
        )))
    ));
    let payload = owner.0.as_ref().unwrap();
    let before = payload.value.borrow().as_ref().unwrap().len();
    let failure = catch_unwind(AssertUnwindSafe(|| {
        let value = payload.value.borrow_mut().take().unwrap();
        let _active = Active {
            value: Some(value),
            destination: &payload.value,
        };
        assert!(matches!(view.complete(), Err(Error::PrefillScopeReentrant)));
        panic!("after taking populated collector, before native submit");
    }));
    assert!(failure.is_err());
    assert_eq!(payload.value.borrow().as_ref().unwrap().len(), before);
    drop(owner);
    assert!(matches!(
        view.append(&array),
        Err(Error::PrefillScopeUnavailable)
    ));
    drop(view);
}
#[test]
fn ordinary_preparation_is_once_only_and_fixed_capacity_never_refills() {
    let runtime = PrefillRootsRuntime::prepare();
    let (owner, view) = RootsOwner::ordinary();
    view.prepare_ordinary(&runtime, 1).unwrap();
    assert!(matches!(
        view.prepare_ordinary(&runtime, 2),
        Err(Error::PrefillScopeUnavailable)
    ));
    let array = Array::from_slice(&[7i32], &[1]);
    view.append(&array).unwrap();
    assert!(matches!(
        view.append(&array),
        Err(Error::PrefillRoots(safemlx::PrefillRootsError::Refused(
            safemlx::PrefillRootsCause::Capacity
        )))
    ));
    assert_eq!(
        owner
            .0
            .as_ref()
            .unwrap()
            .value
            .borrow()
            .as_ref()
            .unwrap()
            .len(),
        1
    );
    drop((owner, view));
}

#[test]
fn capture_projection_rejects_ordinary_and_expired_owners_without_retaining_payload() {
    let (owner, view) = RootsOwner::ordinary();
    let capture = owner.capture_projection();
    assert!(matches!(
        capture.observer(),
        Err(Error::PrefillScopeUnavailable)
    ));
    drop(owner);
    // Both projections are weak. Neither can preserve or recreate an execution
    // role after its sole owning Recovery payload has retired.
    assert!(matches!(
        capture.observer(),
        Err(Error::PrefillScopeUnavailable)
    ));
    assert!(matches!(
        view.has_prepared_completion(),
        Err(Error::PrefillScopeUnavailable)
    ));
}
