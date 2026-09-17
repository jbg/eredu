use super::*;

fn id(name: &str) -> OffloadUnitId {
    OffloadUnitId::new(name).unwrap()
}
fn ledger(names: &[&str]) -> ResidencyLedger {
    ResidencyLedger::new(
        OffloadPlan::new(
            OffloadConfig::default(),
            names.iter().map(|name| {
                OffloadUnitSpec::new(id(name), 1, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                    .unwrap()
            }),
        )
        .unwrap(),
    )
}
fn reference(
    ledger: &ResidencyLedger,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), ResidencyLedgerError> {
    validate_ledger_tier(tier, "residency batch")?;
    let mut seen = BTreeSet::new();
    for actual in ids {
        if !seen.insert(actual) {
            return Err(ResidencyLedgerError::DuplicateBatchUnit);
        }
        ledger.spec(actual)?;
    }
    Ok(())
}

#[test]
fn prepared_batch_preserves_every_short_ordered_unknown_and_duplicate_failure() {
    let ledger = ledger(&["a", "b"]);
    let alphabet = [id("a"), id("b"), id("unknown")];
    let mut scratch = [usize::MAX; 6];
    for len in 0..=6 {
        for mut code in 0..3usize.pow(len as u32) {
            let mut input = Vec::new();
            for _ in 0..len {
                input.push(alphabet[code % 3].clone());
                code /= 3;
            }
            let original = input.clone();
            for tier in [MemoryTier::Host, MemoryTier::Device, MemoryTier::Disk] {
                let expected = reference(&ledger, &input, tier);
                let actual = ledger
                    .validate_batch_in(&input, tier, &mut scratch)
                    .map_err(|failure| match failure {
                        ResidencyBatchFailure::Ledger(cause) => cause.into_owned(),
                        other => panic!("complete destination refused: {other}"),
                    });
                assert_eq!(actual, expected);
                assert_eq!(ledger.validate_batch(&input, tier), expected);
                assert_eq!(input, original);
            }
        }
    }
}

#[test]
fn batch_refusal_keeps_actual_unknown_loan_and_target_precedes_short_destination() {
    let ledger = ledger(&["a"]);
    let unknown = [id("genuine-unknown"), id("genuine-unknown")];
    let before = ledger.telemetry();
    let mut scratch = [0; 2];
    let ResidencyBatchFailure::Ledger(ResidencyBatchError::UnknownUnit { id: actual }) = ledger
        .validate_batch_in(&unknown, MemoryTier::Device, &mut scratch)
        .unwrap_err()
    else {
        panic!("first unknown must precede its later duplicate")
    };
    assert!(std::ptr::eq(actual, &unknown[0]));
    assert!(matches!(
        ledger.validate_batch_in(&unknown, MemoryTier::Disk, &mut []),
        Err(ResidencyBatchFailure::Ledger(
            ResidencyBatchError::InvalidTargetTier { .. }
        ))
    ));
    assert_eq!(
        ledger.validate_batch_in(&unknown, MemoryTier::Device, &mut []),
        Err(ResidencyBatchFailure::Destination {
            required: 2,
            available: 0
        })
    );
    assert_eq!(ledger.telemetry(), before);
    assert_eq!(
        ledger.validate_batch_in(&[id("a")], MemoryTier::Device, &mut scratch),
        Ok(())
    );
}

#[test]
fn retained_plan_is_exact_ledger_identity_and_survives_its_mutable_owner() {
    let first = ledger(&["a", "b"]);
    let second = ledger(&["a", "b"]);
    assert_eq!(first.plan(), second.plan());
    let source = first.plan_source();
    let alias = source.clone();
    assert!(first.matches_plan_source(&source));
    assert!(!second.matches_plan_source(&source));
    assert!(alias.same_source(&source));
    assert!(!source.same_source(&second.plan_source()));
    let pointer = source.id(0).unwrap().as_str().as_ptr();
    drop(first);
    assert_eq!(source.id(0).unwrap().as_str(), "a");
    assert_eq!(source.id(0).unwrap().as_str().as_ptr(), pointer);
    assert_eq!(source.id(1).unwrap().as_str(), "b");
    assert_eq!(source.ordinal(&id("b")), Some(1));
    assert_eq!(source.ordinal(&id("unknown")), None);
    assert_eq!(alias.len(), 2);
}

#[test]
fn large_reverse_batch_uses_same_independent_input_and_scratch_storage() {
    let specs = (0..8192)
        .map(|n| {
            OffloadUnitSpec::new(
                id(&format!("unit.{n:05}")),
                1,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let ledger = ResidencyLedger::new(OffloadPlan::new(OffloadConfig::default(), specs).unwrap());
    let input = ledger
        .plan()
        .units()
        .iter()
        .rev()
        .map(|unit| unit.id().clone())
        .collect::<Vec<_>>();
    let mut scratch = vec![0; input.len()];
    let pointers = (input.as_ptr(), scratch.as_ptr(), input[0].as_str().as_ptr());
    ledger
        .validate_batch_in(&input, MemoryTier::Device, &mut scratch)
        .unwrap();
    assert_eq!(
        (input.as_ptr(), scratch.as_ptr(), input[0].as_str().as_ptr()),
        pointers
    );
    assert_eq!(input.first().unwrap().as_str(), "unit.08191");
    assert_eq!(input.last().unwrap().as_str(), "unit.00000");
}

fn capacity_ledger(policy: CacheEvictionPolicy) -> ResidencyLedger {
    let mut value = ResidencyLedger::new(
        OffloadPlan::new(
            OffloadConfig::new(Some(24), None, 1)
                .unwrap()
                .with_eviction_policy(policy),
            [
                ("a", ResidencyPolicy::Cacheable),
                ("b", ResidencyPolicy::Windowed),
                ("c", ResidencyPolicy::Cacheable),
                ("d", ResidencyPolicy::Cacheable),
                ("e", ResidencyPolicy::Cacheable),
            ]
            .map(|(name, policy)| {
                OffloadUnitSpec::new(id(name), 8, policy, MemoryTier::Disk).unwrap()
            }),
        )
        .unwrap(),
    );
    for name in ["a", "b", "c"] {
        value
            .reserve_copy(&id(name), MemoryTier::Device, 8, &BTreeSet::new())
            .unwrap();
        value
            .publish_reserved(&id(name), MemoryTier::Device, 8, None)
            .unwrap();
    }
    value
}
fn prepared_storage(ledger: &ResidencyLedger, batch: usize) -> ResidencyAdmissionStorage {
    ResidencyAdmissionStorage::try_new(ledger.plan_source(), batch, 32).unwrap()
}
#[test]
fn ordinary_and_prepared_choose_same_policy_victims_preserve_ticks_and_partial_rollback() {
    for policy in [
        CacheEvictionPolicy::LeastRecentlyUsed,
        CacheEvictionPolicy::LeastFrequentlyUsed,
    ] {
        let mut ordinary = capacity_ledger(policy);
        let mut prepared = capacity_ledger(policy);
        for value in [&mut ordinary, &mut prepared] {
            for _ in 0..2 {
                value.pin(&id("a"), MemoryTier::Device, 1).unwrap();
                value.unpin(&id("a"), MemoryTier::Device);
            }
            value.pin(&id("c"), MemoryTier::Device, 1).unwrap();
            value.unpin(&id("c"), MemoryTier::Device);
            // Existing eviction policy does not exclude in-flight copies here.
            value
                .units
                .get_mut(&id("c"))
                .unwrap()
                .device
                .as_mut()
                .unwrap()
                .in_flight = Some(17);
            // Cross the real renormalization boundary during the two commits.
            value.tick = u64::MAX;
        }
        let ids = [id("d"), id("e")];
        let rows = [
            ResidencyReservationRow { input: 0, bytes: 8 },
            ResidencyReservationRow { input: 1, bytes: 8 },
        ];
        let storage = prepared_storage(&prepared, 2);
        let pointers = (
            storage.order.as_ptr(),
            storage.candidates.as_ptr(),
            storage.evicted.as_ptr(),
            storage.blockers.as_ptr(),
        );
        let actual = prepared
            .reserve_copies_in(
                &ids,
                &rows,
                MemoryTier::Device,
                ResidencyProtection::sorted(&ids).unwrap(),
                storage,
            )
            .unwrap();
        let expected = ordinary
            .reserve_copies(
                &[(ids[0].clone(), 8), (ids[1].clone(), 8)],
                MemoryTier::Device,
                &BTreeSet::from(ids.clone()),
            )
            .unwrap();
        assert_eq!(
            actual
                .evicted()
                .iter()
                .map(|row| actual.source().id(row.plan).unwrap().clone())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|row| row.id.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .source()
                .id(actual.evicted()[0].plan)
                .unwrap()
                .as_str(),
            "b"
        );
        assert_eq!(
            actual
                .source()
                .id(actual.evicted()[1].plan)
                .unwrap()
                .as_str(),
            match policy {
                CacheEvictionPolicy::LeastRecentlyUsed => "a",
                CacheEvictionPolicy::LeastFrequentlyUsed => "c",
            }
        );
        assert_eq!(
            (
                actual.order.as_ptr(),
                actual.candidates.as_ptr(),
                actual.evicted.as_ptr(),
                actual.blockers.as_ptr()
            ),
            pointers
        );
        for value in [&mut ordinary, &mut prepared] {
            value
                .publish_reserved(&ids[0], MemoryTier::Device, 6, None)
                .unwrap();
            value
                .rollback_reserved(&ids[0], MemoryTier::Device)
                .unwrap();
            value
                .rollback_reserved(&ids[1], MemoryTier::Device)
                .unwrap();
            assert!(value.is_resident(&ids[0], MemoryTier::Device).unwrap());
            assert!(value.units[&ids[1]].device.is_none());
        }
        assert_eq!(ordinary.telemetry(), prepared.telemetry());
        assert_eq!(ordinary.tick, prepared.tick);
        for (a, b) in ordinary.units.values().zip(prepared.units.values()) {
            assert_eq!(a.device, b.device);
        }
    }
}
#[test]
fn prepared_capacity_refusal_retains_actual_blockers_and_source_after_ledger_drop() {
    let mut value = capacity_ledger(CacheEvictionPolicy::LeastRecentlyUsed);
    for _ in 0..3 {
        value.pin(&id("a"), MemoryTier::Device, 1).unwrap();
    }
    value
        .set_group_window("active", &[id("b")], MemoryTier::Device)
        .unwrap();
    let protected = [id("c"), id("d"), id("e")];
    let rows = [
        ResidencyReservationRow { input: 1, bytes: 8 },
        ResidencyReservationRow { input: 2, bytes: 8 },
    ];
    let storage = prepared_storage(&value, 3);
    let source = storage.source().clone();
    let before = value.telemetry();
    let error = value
        .reserve_copies_in(
            &protected,
            &rows,
            MemoryTier::Device,
            ResidencyProtection::sorted(&protected).unwrap(),
            storage,
        )
        .unwrap_err();
    assert_eq!(value.telemetry(), before);
    assert_eq!(value.units[&id("d")].device, None);
    assert_eq!(value.units[&id("e")].device, None);
    assert!(source.same_source(error.blockers().0));
    drop(value);
    let cause = error.capacity().unwrap();
    assert_eq!(
        (
            cause.requested.as_str(),
            cause.required_bytes,
            cause.budget_bytes,
            cause.resident_bytes
        ),
        ("d", 16, 24, 24)
    );
    let rows = cause.blockers().collect::<Vec<_>>();
    assert_eq!(
        rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    assert_eq!(rows[0].in_use, 3);
    assert!(rows[1].active_window);
    assert!(rows[2].request_protected);
}
#[test]
fn short_final_output_and_corrupt_removal_accounting_fail_before_any_eviction() {
    let ids = [id("d"), id("e")];
    let rows = [
        ResidencyReservationRow { input: 0, bytes: 8 },
        ResidencyReservationRow { input: 1, bytes: 8 },
    ];
    let mut value = capacity_ledger(CacheEvictionPolicy::LeastRecentlyUsed);
    let before = value.telemetry();
    let mut storage = prepared_storage(&value, 2);
    storage.evicted = Vec::new();
    let error = value
        .reserve_copies_in(
            &ids,
            &rows,
            MemoryTier::Device,
            ResidencyProtection::sorted(&ids).unwrap(),
            storage,
        )
        .unwrap_err();
    assert!(matches!(
        error.cause,
        Failure::Destination {
            field: "eviction",
            ..
        }
    ));
    assert_eq!(value.telemetry(), before);
    assert!(value.is_resident(&id("b"), MemoryTier::Device).unwrap());
    // Test-private corruption proves the newly atomic removal preflight. No
    // public mutation API is added, and normal backend paths cannot forge this.
    value.set_tier_bytes(MemoryTier::Device, 1);
    let state = value.units[&id("b")].device;
    let error = value
        .remove_copy(&id("b"), MemoryTier::Device, true)
        .unwrap_err();
    assert!(matches!(
        error,
        ResidencyLedgerError::StateInconsistent {
            operation: "copy removal accounting",
            ..
        }
    ));
    assert_eq!(value.units[&id("b")].device, state);
    assert_eq!(value.tier_bytes(MemoryTier::Device), 1);
}
#[test]
fn foreign_prepared_source_and_full_unknown_text_refuse_without_changing_either_ledger() {
    let mut first = ledger(&["a"]);
    let second = ledger(&["a"]);
    let before = first.telemetry();
    let error = first
        .reserve_copies_in(
            &[id("a")],
            &[ResidencyReservationRow { input: 0, bytes: 1 }],
            MemoryTier::Device,
            ResidencyProtection::sorted(&[]).unwrap(),
            prepared_storage(&second, 1),
        )
        .unwrap_err();
    assert_eq!(error.cause, Failure::Source);
    assert_eq!(first.telemetry(), before);
    let actual = [id("actual unknown owned text")];
    let storage = prepared_storage(&first, 1);
    let error = storage
        .validate_batch(&first, &actual, MemoryTier::Device)
        .unwrap_err();
    drop(actual);
    drop(first);
    drop(second);
    assert_eq!(
        error.to_string(),
        "unknown residency unit: actual unknown owned text"
    );
    assert!(error.capacity().is_none());
}

#[test]
fn prepared_invalid_reservations_and_incomplete_victim_plans_leave_all_state_unchanged() {
    let inputs = [
        (
            vec![id("d")],
            vec![ResidencyReservationRow { input: 0, bytes: 0 }],
        ),
        (
            vec![id("a")],
            vec![ResidencyReservationRow { input: 0, bytes: 8 }],
        ),
        (
            vec![id("d"), id("e")],
            vec![
                ResidencyReservationRow {
                    input: 0,
                    bytes: u64::MAX,
                },
                ResidencyReservationRow { input: 1, bytes: 1 },
            ],
        ),
        (
            vec![id("d")],
            vec![ResidencyReservationRow {
                input: 0,
                bytes: u64::MAX,
            }],
        ),
        (
            vec![id("d")],
            vec![
                ResidencyReservationRow { input: 0, bytes: 0 },
                ResidencyReservationRow { input: 0, bytes: 8 },
            ],
        ),
        (
            vec![id("unknown")],
            vec![ResidencyReservationRow { input: 0, bytes: 8 }],
        ),
        (
            vec![id("d")],
            vec![ResidencyReservationRow { input: 1, bytes: 8 }],
        ),
    ];
    for (ids, rows) in inputs {
        let mut value = capacity_ledger(CacheEvictionPolicy::LeastRecentlyUsed);
        let telemetry = value.telemetry();
        let copies = value
            .units
            .values()
            .map(|unit| unit.device)
            .collect::<Vec<_>>();
        let tick = value.tick;
        let storage = prepared_storage(&value, 2);
        let error = value
            .reserve_copies_in(
                &ids,
                &rows,
                MemoryTier::Device,
                ResidencyProtection::sorted(&[]).unwrap(),
                storage,
            )
            .unwrap_err();
        assert!(error.capacity().is_none());
        assert_eq!(value.telemetry(), telemetry);
        assert_eq!(value.tick, tick);
        assert_eq!(
            value
                .units
                .values()
                .map(|unit| unit.device)
                .collect::<Vec<_>>(),
            copies
        );
    }
    let mut value = capacity_ledger(CacheEvictionPolicy::LeastRecentlyUsed);
    value.pin(&id("a"), MemoryTier::Device, 1).unwrap();
    value.pin(&id("b"), MemoryTier::Device, 1).unwrap();
    let before = value.telemetry();
    let storage = prepared_storage(&value, 2);
    let ids = [id("d"), id("e")];
    let error = value
        .reserve_copies_in(
            &ids,
            &[
                ResidencyReservationRow { input: 0, bytes: 8 },
                ResidencyReservationRow { input: 1, bytes: 8 },
            ],
            MemoryTier::Device,
            ResidencyProtection::sorted(&ids).unwrap(),
            storage,
        )
        .unwrap_err();
    assert!(error.capacity().is_some());
    assert_eq!(value.telemetry(), before);
    assert!(
        value.is_resident(&id("c"), MemoryTier::Device).unwrap(),
        "one eligible victim is insufficient; it must remain resident"
    );
    assert!(value.units[&id("d")].device.is_none());
    assert!(value.units[&id("e")].device.is_none());
}
