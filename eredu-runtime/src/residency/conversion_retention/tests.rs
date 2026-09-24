use super::*;
use eredu_core::residency::ParameterConversionRetentionEligibility;
use std::sync::Barrier;

fn id(key: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: "test".into(),
        key: key.into(),
    }
}

fn budget(
    registry: &ConversionRetentionRegistry,
    key: &str,
    bytes: u64,
) -> ConversionRetentionBudget {
    registry
        .budget(
            ParameterConversionRetentionGroup(id(key)),
            ParameterConversionRetentionPolicyReport::resolve(
                Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: bytes }),
                ParameterConversionRetentionEligibility::Eligible,
            ),
        )
        .unwrap()
}

fn usage(budget: &ConversionRetentionBudget) -> (u64, u64, u64) {
    let Observed::Available { value, .. } = budget.report().usage else {
        panic!("missing usage")
    };
    (
        value.retained_claims,
        value.retained_payload_bytes,
        value.reserved_payload_bytes,
    )
}

fn reserve(
    parameter: &ConversionRetentionParameter,
    budget: &ConversionRetentionBudget,
) -> ConversionRetentionReservation {
    let ConversionRetentionAdmission::Reserved(reservation) = parameter.reserve(budget).unwrap()
    else {
        panic!("expected new reservation")
    };
    reservation
}

fn reused(
    parameter: &ConversionRetentionParameter,
    budget: &ConversionRetentionBudget,
) -> ConversionRetentionClaim {
    let ConversionRetentionAdmission::Retained(claim) = parameter.reserve(budget).unwrap() else {
        panic!("expected admitted existing allocation")
    };
    claim
}

fn allocation(
    registry: &ConversionRetentionRegistry,
    key: &str,
    bytes: u64,
) -> ConversionRetentionAllocation {
    registry.allocation(id(key), bytes).unwrap()
}

#[test]
fn exact_cap_first_admitted_reservation_publication_and_final_owner_release() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    let same = budget(&registry, "execution", 16);
    let a = registry.parameter(id("a"), 12).unwrap();
    let b = registry.parameter(id("b"), 4).unwrap();
    let c = registry.parameter(id("c"), 1).unwrap();
    let first = reserve(&a, &group);
    let second = reserve(&b, &same);
    assert_eq!(usage(&group), (0, 0, 16));
    assert!(matches!(
        c.reserve(&group),
        Err(ConversionRetentionError::Capacity)
    ));
    let first = first.publish(allocation(&registry, "a-f32", 12)).unwrap();
    assert_eq!(usage(&group), (1, 12, 4));
    let second = second.publish(allocation(&registry, "b-f32", 4)).unwrap();
    assert_eq!(usage(&same), (2, 16, 0));
    let alias = first.clone();
    drop(first);
    assert_eq!(usage(&group), (2, 16, 0));
    drop(alias);
    assert_eq!(usage(&group), (1, 4, 0));
    drop(second);
    assert_eq!(usage(&group), (0, 0, 0));
    assert!(matches!(
        a.reserve(&group),
        Ok(ConversionRetentionAdmission::Reserved(_))
    ));
}

#[test]
fn disabled_oversized_and_checked_overflow_fail_without_changing_usage() {
    let registry = ConversionRetentionRegistry::default();
    let small = budget(&registry, "small", 3);
    let disabled = budget(&registry, "disabled", 0);
    let a = registry.parameter(id("a"), 4).unwrap();
    assert!(matches!(
        a.reserve(&small),
        Err(ConversionRetentionError::Capacity)
    ));
    assert!(matches!(
        a.reserve(&disabled),
        Err(ConversionRetentionError::Disabled)
    ));
    let empty = registry.parameter(id("empty"), 0).unwrap();
    assert!(matches!(
        empty.reserve(&disabled),
        Err(ConversionRetentionError::Disabled)
    ));
    assert_eq!(
        conversion_retention_payload_bytes(u64::MAX, 4),
        Err(ConversionRetentionError::Overflow)
    );
    assert_eq!(conversion_retention_payload_bytes(3, 4), Ok(12));
    let unlimited = registry
        .budget(
            ParameterConversionRetentionGroup(id("unlimited")),
            ParameterConversionRetentionPolicyReport::resolve(
                Some(ParameterConversionRetentionPolicy::Unlimited),
                ParameterConversionRetentionEligibility::Eligible,
            ),
        )
        .unwrap();
    let huge = registry.parameter(id("huge"), u64::MAX).unwrap();
    let pending = reserve(&huge, &unlimited);
    assert!(matches!(
        a.reserve(&unlimited),
        Err(ConversionRetentionError::Overflow)
    ));
    let retained = pending
        .publish(allocation(&registry, "huge-f32", u64::MAX))
        .unwrap();
    assert_eq!(usage(&unlimited), (1, u64::MAX, 0));
    assert!(matches!(
        a.reserve(&unlimited),
        Err(ConversionRetentionError::Overflow)
    ));
    drop(retained);
    assert_eq!(usage(&unlimited), (0, 0, 0));
    assert_eq!(usage(&small), (0, 0, 0));
}

#[test]
fn aliases_share_one_claim_and_two_groups_admit_independently() {
    let registry = ConversionRetentionRegistry::default();
    let target = budget(&registry, "target", 16);
    let draft = budget(&registry, "external-draft", 16);
    let denied = budget(&registry, "denied", 15);
    let source = registry.parameter(id("source"), 16).unwrap();
    let alias = registry.parameter(id("source"), 16).unwrap();
    let claim = reserve(&source, &target)
        .publish(allocation(&registry, "f32", 16))
        .unwrap();
    let tied = reused(&alias, &target);
    let external = reused(&alias, &draft);
    assert_eq!(claim.parameter(), &id("source"));
    assert_eq!(claim.allocation(), external.allocation());
    assert_ne!(claim.group(), external.group());
    assert_eq!(claim.payload_bytes(), 16);
    assert_eq!(usage(&target), (1, 16, 0));
    assert_eq!(usage(&draft), (1, 16, 0));
    assert!(matches!(
        source.reserve(&denied),
        Err(ConversionRetentionError::Capacity)
    ));
    assert_eq!(usage(&denied), (0, 0, 0));
    drop(claim);
    drop(tied);
    assert_eq!(usage(&target), (0, 0, 0));
    assert!(external.is_active());
    assert_eq!(usage(&draft), (1, 16, 0));
    drop(external);
    assert_eq!(usage(&draft), (0, 0, 0));
}

#[test]
fn distinct_parameter_bindings_to_one_backing_release_only_their_own_claims() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 32);
    let a = registry.parameter(id("a"), 16).unwrap();
    let b = registry.parameter(id("b"), 16).unwrap();
    let backing = allocation(&registry, "shared", 16);
    let first = reserve(&a, &group).publish(backing.clone()).unwrap();
    let second = reserve(&b, &group).publish(backing).unwrap();
    assert_eq!(usage(&group), (1, 16, 0));
    a.invalidate();
    assert!(!first.is_active());
    assert!(second.is_active());
    assert_eq!(usage(&group), (1, 16, 0));
    drop(first);
    drop(second);
    assert_eq!(usage(&group), (0, 0, 0));
}

#[test]
fn cancellation_failure_and_unwind_restore_capacity_once() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    let source = registry.parameter(id("source"), 16).unwrap();
    {
        let _failed_conversion = reserve(&source, &group);
        assert_eq!(usage(&group), (0, 0, 16));
    }
    assert_eq!(usage(&group), (0, 0, 0));
    let result = std::panic::catch_unwind(|| {
        let _abandoned = reserve(&source, &group);
        panic!("native conversion failure");
    });
    assert!(result.is_err());
    assert_eq!(usage(&group), (0, 0, 0));
    let wrong_size = allocation(&registry, "wrong", 8);
    assert!(matches!(
        reserve(&source, &group).publish(wrong_size),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    assert_eq!(usage(&group), (0, 0, 0));
    let claim = reserve(&source, &group)
        .publish(allocation(&registry, "valid", 16))
        .unwrap();
    drop(claim);
    assert_eq!(usage(&group), (0, 0, 0));
}

#[test]
fn invalidation_cancels_late_publication_and_stale_drop_cannot_touch_replacement() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    for publish_stale in [false, true] {
        let old = registry.parameter(id("source"), 16).unwrap();
        let stale = reserve(&old, &group);
        old.invalidate();
        old.invalidate();
        assert_eq!(usage(&group), (0, 0, 0));
        let replacement = registry.parameter(id("source"), 16).unwrap();
        let pending = reserve(&replacement, &group);
        if publish_stale {
            assert!(matches!(
                stale.publish(allocation(&registry, "stale", 16)),
                Err(ConversionRetentionError::Invalidated)
            ));
        } else {
            drop(stale);
        }
        assert_eq!(usage(&group), (0, 0, 16));
        assert!(matches!(
            old.reserve(&group),
            Err(ConversionRetentionError::Invalidated)
        ));
        let current = pending
            .publish(allocation(&registry, "current", 16))
            .unwrap();
        old.invalidate();
        assert!(current.is_active());
        assert_eq!(usage(&group), (1, 16, 0));
        replacement.invalidate();
        assert!(!current.is_active());
        drop(current);
        assert_eq!(usage(&group), (0, 0, 0));
    }
}

#[test]
fn invalidation_revokes_all_alias_group_claims_without_reclaiming_other_entries() {
    let registry = ConversionRetentionRegistry::default();
    let a = budget(&registry, "a", 16);
    let b = budget(&registry, "b", 16);
    let source = registry.parameter(id("source"), 16).unwrap();
    let first = reserve(&source, &a)
        .publish(allocation(&registry, "shared", 16))
        .unwrap();
    let second = reused(&source, &b);
    source.invalidate();
    assert!(!first.is_active());
    assert!(!second.is_active());
    assert_eq!(usage(&a), (0, 0, 0));
    assert_eq!(usage(&b), (0, 0, 0));
    drop((first, second));
    assert_eq!(usage(&a), (0, 0, 0));
}

#[test]
fn concurrent_reservations_cannot_exceed_shared_execution_ceiling() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 32);
    let rendezvous = Arc::new(Barrier::new(17));
    std::thread::scope(|scope| {
        for n in 0..16 {
            let parameter = registry
                .parameter(id(&format!("parameter-{n}")), 8)
                .unwrap();
            let gate = rendezvous.clone();
            let shared = group.clone();
            scope.spawn(move || {
                let result = parameter.reserve(&shared);
                gate.wait();
                gate.wait();
                drop(result);
            });
        }
        rendezvous.wait();
        assert_eq!(usage(&group), (0, 0, 32));
        rendezvous.wait();
    });
    assert_eq!(usage(&group), (0, 0, 0));
}

#[test]
fn simultaneous_aliases_get_one_publisher_and_invalidation_wins_during_evaluation() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    let source = registry.parameter(id("source"), 16).unwrap();
    let rendezvous = Arc::new(Barrier::new(9));
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let parameter = source.clone();
            let group = group.clone();
            let gate = rendezvous.clone();
            let registry = &registry;
            scope.spawn(move || {
                let result = parameter.reserve(&group);
                assert!(matches!(
                    &result,
                    Ok(ConversionRetentionAdmission::Reserved(_))
                        | Err(ConversionRetentionError::Busy)
                ));
                gate.wait();
                gate.wait();
                if let Ok(ConversionRetentionAdmission::Reserved(reservation)) = result {
                    assert!(matches!(
                        reservation.publish(allocation(registry, "late", 16)),
                        Err(ConversionRetentionError::Invalidated)
                    ));
                }
            });
        }
        rendezvous.wait();
        assert_eq!(usage(&group), (0, 0, 16));
        source.invalidate();
        assert_eq!(usage(&group), (0, 0, 0));
        rendezvous.wait();
    });
    assert_eq!(usage(&group), (0, 0, 0));
}

#[test]
fn registries_are_weak_and_conflicting_live_identities_are_rejected() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    let source = registry.parameter(id("source"), 16).unwrap();
    let claim = reserve(&source, &group)
        .publish(allocation(&registry, "f32", 16))
        .unwrap();
    assert!(matches!(
        registry.parameter(id("source"), 8),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    assert!(matches!(
        registry.allocation(id("f32"), 8),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    let wrong_policy = ParameterConversionRetentionPolicyReport::resolve(
        Some(ParameterConversionRetentionPolicy::Unlimited),
        ParameterConversionRetentionEligibility::Eligible,
    );
    assert!(matches!(
        registry.budget(group.group().clone(), wrong_policy),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    let parameter_weak = Arc::downgrade(&source.0);
    let budget_weak = Arc::downgrade(&group.0);
    let allocation_weak = Arc::downgrade(&claim.0.allocation.0);
    drop(source);
    drop(group);
    assert!(parameter_weak.upgrade().is_some());
    drop(claim);
    assert!(parameter_weak.upgrade().is_none());
    assert!(budget_weak.upgrade().is_none());
    assert!(allocation_weak.upgrade().is_none());
}

#[test]
fn residency_controllers_share_the_selected_execution_budget() {
    use crate::residency::{OffloadUnit, ResidencyController};
    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        store::{StoreError, TensorMetadata},
    };
    use eredu_core::residency::{OffloadConfig, OffloadPlan};
    struct EmptyCatalog;
    impl RecipeCatalog for EmptyCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            Err(StoreError::UnknownTensor { key: key.into() })
        }
    }
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    let partition = || {
        ResidencyController::new(
            &EmptyCatalog,
            OffloadPlan::new(OffloadConfig::default(), []).unwrap(),
            [] as [OffloadUnit; 0],
        )
        .unwrap()
    };
    assert!(partition().conversion_retention().is_none());
    let target = partition().with_conversion_retention(group.clone());
    let embedded = partition().with_conversion_retention(group.clone());
    let source = registry.parameter(id("source"), 16).unwrap();
    let pending = reserve(&source, target.conversion_retention().unwrap());
    assert_eq!(usage(embedded.conversion_retention().unwrap()), (0, 0, 16));
    drop(pending);
    assert_eq!(usage(embedded.conversion_retention().unwrap()), (0, 0, 0));
}

#[test]
fn malformed_identity_and_inconsistent_policy_cannot_create_authority() {
    use eredu_core::residency::ParameterConversionRetentionPolicySource;
    let registry = ConversionRetentionRegistry::default();
    let invalid = ResourceIdentity {
        scope: " ".into(),
        key: "key".into(),
    };
    let managed = ParameterConversionRetentionPolicyReport::resolve(
        None,
        ParameterConversionRetentionEligibility::Eligible,
    );
    assert!(matches!(
        registry.budget(
            ParameterConversionRetentionGroup(invalid.clone()),
            managed.clone()
        ),
        Err(ConversionRetentionError::InvalidIdentity)
    ));
    assert!(matches!(
        registry.parameter(invalid.clone(), 16),
        Err(ConversionRetentionError::InvalidIdentity)
    ));
    assert!(matches!(
        registry.allocation(invalid, 16),
        Err(ConversionRetentionError::InvalidIdentity)
    ));
    let mut inconsistent = managed.clone();
    inconsistent.effective = ParameterConversionRetentionPolicy::Unlimited;
    assert!(matches!(
        registry.budget(
            ParameterConversionRetentionGroup(id("bad-effective")),
            inconsistent
        ),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    let mut unlimited = ParameterConversionRetentionPolicyReport::resolve(
        Some(ParameterConversionRetentionPolicy::Unlimited),
        ParameterConversionRetentionEligibility::Eligible,
    );
    unlimited.source = ParameterConversionRetentionPolicySource::ManagedDefault;
    assert!(matches!(
        registry.budget(
            ParameterConversionRetentionGroup(id("bad-source")),
            unlimited
        ),
        Err(ConversionRetentionError::ConflictingIdentity)
    ));
    let accepted = registry
        .budget(
            ParameterConversionRetentionGroup(id("managed")),
            managed.clone(),
        )
        .unwrap();
    assert_eq!(
        accepted.report().policy,
        Observed::exact(managed, "retained execution selection")
    );
}

#[test]
fn publication_racing_invalidation_never_leaves_an_obsolete_live_claim() {
    let registry = ConversionRetentionRegistry::default();
    let group = budget(&registry, "execution", 16);
    for n in 0..32 {
        let source = registry.parameter(id(&format!("source-{n}")), 16).unwrap();
        let pending = reserve(&source, &group);
        let backing = allocation(&registry, &format!("backing-{n}"), 16);
        let gate = Barrier::new(2);
        let result = std::thread::scope(|scope| {
            let publisher = scope.spawn(|| {
                gate.wait();
                pending.publish(backing)
            });
            gate.wait();
            source.invalidate();
            publisher.join().unwrap()
        });
        match result {
            Ok(claim) => assert!(!claim.is_active()),
            Err(error) => assert_eq!(error, ConversionRetentionError::Invalidated),
        }
        assert_eq!(usage(&group), (0, 0, 0));
    }
}
