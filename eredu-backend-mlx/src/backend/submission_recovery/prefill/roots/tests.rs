use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind};
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
#[test]
fn invocation_projection_retains_paid_weak_header_without_retaining_payload() {
    use eredu_core::{
        DomainMemoryRequirements, MemoryDomainDescription, MemoryLimit, MemoryLimits,
        MemoryLocation, MemoryTopology,
    };
    use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext};
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, MemoryLedger};
    use std::sync::Arc;

    struct Source {
        dropped: Rc<Cell<usize>>,
        charge_on_drop: Rc<Cell<u64>>,
        ledger: MemoryLedger,
        funding: HostMetadataFunding,
    }
    impl Drop for Source {
        fn drop(&mut self) {
            self.charge_on_drop
                .set(self.ledger.snapshot().unwrap().domains[0].current_charge_bytes);
            self.dropped.set(self.dropped.get().checked_add(1).unwrap());
        }
    }
    impl InvocationRootSource for Source {
        fn metadata_funding(&self) -> &HostMetadataFunding {
            &self.funding
        }
        fn observer(&self) -> Result<safemlx::OriginalScopeObserver, Error> {
            Err(Error::PrefillScopeUnavailable)
        }
        fn completion_recipe(
            &self,
        ) -> Result<crate::backend::nn::workspace::ResidentCompletionRecipe, Error> {
            Err(Error::PrefillScopeUnavailable)
        }
        fn append(&self, _: &Array) -> Result<(), Error> {
            Err(Error::PrefillScopeUnavailable)
        }
        fn retire_completed(&self, _: &Array) -> Result<(), Error> {
            Err(Error::PrefillScopeUnavailable)
        }
        fn close_construction(&self) -> Result<(), Error> {
            Err(Error::PrefillScopeUnavailable)
        }
    }

    let topology = Arc::new(
        MemoryTopology::new(vec![MemoryDomainDescription {
            name: "host".into(),
            locations: vec![MemoryLocation::Host],
        }])
        .unwrap(),
    );
    let limits = MemoryLimits::resolve(
        &topology,
        [(topology.host_domain(), MemoryLimit::Finite(1 << 20))],
    )
    .unwrap();
    let ledger = MemoryLedger::new(
        topology.clone(),
        limits.clone(),
        DomainMemoryRequirements::zero(&topology),
    )
    .unwrap();
    let baseline = ledger.snapshot().unwrap().domains[0].current_charge_bytes;
    let funding = ledger
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), limits)
        .unwrap();
    let before_source = ledger.snapshot().unwrap().domains[0].current_charge_bytes;
    let bytes = WorkspaceContext::metadata_rc_bytes::<Source>().unwrap();
    funding.reserve_metadata(bytes).unwrap();
    let charged = before_source
        .checked_add(u64::try_from(bytes).unwrap())
        .unwrap();
    assert_eq!(
        ledger.snapshot().unwrap().domains[0].current_charge_bytes,
        charged
    );
    let dropped = Rc::new(Cell::new(0));
    let charge_on_drop = Rc::new(Cell::new(0));
    let source = Rc::new(Source {
        dropped: dropped.clone(),
        charge_on_drop: charge_on_drop.clone(),
        ledger: ledger.clone(),
        funding,
    });
    // The constructor obtains the payer from this exact source. Erasure and
    // projection cloning lend the existing Rc allocation without a new payload.
    let projection = TransientRootsProjection::from_invocation(&source);
    let final_projection = projection.clone();
    let loan = projection.owner().unwrap();
    assert!(matches!(
        loan.observer(),
        Err(Error::PrefillScopeUnavailable)
    ));
    drop(source);
    drop(projection);
    assert_eq!(
        dropped.get(),
        0,
        "the active loan retains the source payload"
    );
    assert!(final_projection.owner().is_ok());
    assert_eq!(
        ledger.snapshot().unwrap().domains[0].current_charge_bytes,
        charged
    );

    drop(loan);
    assert_eq!(
        dropped.get(),
        1,
        "weak projections do not retain the payload"
    );
    assert_eq!(
        charge_on_drop.get(),
        charged,
        "payload retires before its payer"
    );
    assert!(matches!(
        final_projection.owner(),
        Err(Error::PrefillScopeUnavailable)
    ));
    assert_eq!(
        ledger.snapshot().unwrap().domains[0].current_charge_bytes,
        charged,
        "the surviving weak header retains the source account"
    );
    drop(final_projection);
    let retired = ledger.snapshot().unwrap();
    assert_eq!(retired.domains[0].current_charge_bytes, baseline);
    assert_eq!(retired.reservations, 0);
    assert_eq!(retired.funding_accounts, 0);
}
