use super::*;
fn id(value: &str) -> OffloadUnitId {
    OffloadUnitId::new(value).unwrap()
}
fn ledger(policy: ResidencyPolicy, resident_tier: MemoryTier) -> ResidencyLedger {
    let initial_tier = if policy == ResidencyPolicy::Pinned {
        resident_tier
    } else {
        MemoryTier::Disk
    };
    ResidencyLedger::new(
        OffloadPlan::new(
            OffloadConfig::new(Some(32), Some(32), 1).unwrap(),
            [OffloadUnitSpec::new(id("actual"), 16, policy, initial_tier).unwrap()],
        )
        .unwrap(),
    )
}
fn publish(ledger: &mut ResidencyLedger, tier: MemoryTier, generation: Option<u64>) {
    ledger
        .reserve_copy(&id("actual"), tier, 16, &BTreeSet::new())
        .unwrap();
    ledger
        .publish_reserved(&id("actual"), tier, 16, generation)
        .unwrap();
}
#[test]
fn borrowed_eviction_preserves_policy_pin_absence_accounting_and_ordinary_telemetry() {
    for tier in [MemoryTier::Host, MemoryTier::Device] {
        for policy in [
            ResidencyPolicy::Pinned,
            ResidencyPolicy::Windowed,
            ResidencyPolicy::Cacheable,
        ] {
            for pinned in [false, true] {
                let mut ordinary = ledger(policy, tier);
                let mut original = ledger(policy, tier);
                publish(&mut ordinary, tier, None);
                publish(&mut original, tier, None);
                if pinned {
                    ordinary.pin(&id("actual"), tier, 1).unwrap();
                    original.pin(&id("actual"), tier, 1).unwrap();
                }
                let before = original.telemetry();
                let expected = ordinary
                    .evict(&id("actual"), tier)
                    .map(|row| row.map(|row| row.bytes));
                let actual = original
                    .evict_settled_borrowed(&id("actual"), tier)
                    .map_err(|cause| cause.into_owned(&id("actual"), tier));
                assert_eq!(actual, expected);
                assert_eq!(original.telemetry(), ordinary.telemetry());
                if actual.is_err() {
                    assert_eq!(original.telemetry(), before);
                } else {
                    assert_eq!(actual, Ok(Some(16)));
                    assert_eq!(
                        original.evict_settled_borrowed(&id("actual"), tier),
                        Ok(None)
                    );
                }
            }
        }
    }
    let mut original = ledger(ResidencyPolicy::Cacheable, MemoryTier::Device);
    let before = original.telemetry();
    assert_eq!(
        original.evict_settled_borrowed(&id("foreign"), MemoryTier::Disk),
        Err(ResidencyEvictionError::InvalidTargetTier)
    );
    assert_eq!(
        original.evict_settled_borrowed(&id("foreign"), MemoryTier::Device),
        Err(ResidencyEvictionError::UnknownUnit)
    );
    assert_eq!(original.telemetry(), before);
    publish(&mut original, MemoryTier::Device, None);
    original.set_tier_bytes(MemoryTier::Device, 15);
    let before = original.telemetry();
    assert_eq!(
        original.evict_settled_borrowed(&id("actual"), MemoryTier::Device),
        Err(ResidencyEvictionError::Accounting)
    );
    assert!(
        original
            .is_resident(&id("actual"), MemoryTier::Device)
            .unwrap()
    );
    assert_eq!(original.telemetry(), before);
}
#[test]
fn borrowed_eviction_never_treats_pending_or_wrong_generation_as_completion() {
    let mut value = ledger(ResidencyPolicy::Cacheable, MemoryTier::Device);
    let generation = value.next_transfer_generation().unwrap();
    publish(&mut value, MemoryTier::Device, Some(generation));
    let before = value.telemetry();
    assert_eq!(
        value.evict_settled_borrowed(&id("actual"), MemoryTier::Device),
        Err(ResidencyEvictionError::PendingTransfer)
    );
    assert!(
        value
            .resolve_transfer(&[id("actual")], MemoryTier::Device, generation + 1, true)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        value.evict_settled_borrowed(&id("actual"), MemoryTier::Device),
        Err(ResidencyEvictionError::PendingTransfer)
    );
    assert_eq!(value.telemetry(), before);
    value
        .resolve_transfer(&[id("actual")], MemoryTier::Device, generation, true)
        .unwrap();
    assert_eq!(
        value.evict_settled_borrowed(&id("actual"), MemoryTier::Device),
        Ok(Some(16))
    );
    assert_eq!(value.telemetry().evictions().count(), 1);
    assert_eq!(
        value.telemetry().resident_bytes().get(MemoryTier::Device),
        0
    );
}
