use super::*;
struct Publication(ResidentResetPublicationCustody);
impl ResidentResetPublicationProfile for Publication {
    fn control_bytes() -> Option<u64> {
        u64::try_from(size_of::<Self>()).ok()
    }
    fn prepare(custody: ResidentResetPublicationCustody) -> Self {
        Self(custody)
    }
}
fn current(pool: &MemoryLedger) -> u64 {
    pool.snapshot()
        .unwrap()
        .domains
        .iter()
        .map(|row| row.current_charge_bytes)
        .sum()
}

#[test]
fn parameter_reset_uses_same_exact_host_account_under_finite_and_unlimited_limits() {
    for limit_mode in [0, 1, 2] {
        let (_runtime, data, _, _) = fixture(None, 3);
        let session = Session(data.clone());
        let source = data.borrow();
        let pool = &source.pool;
        let before = current(pool);
        let required = source
            .plan()
            .parameter_publication_required_bytes::<Publication>()
            .unwrap();
        let host = pool.topology().host_domain();
        let limit = match limit_mode {
            0 => eredu_core::MemoryLimit::Finite(before + required),
            1 => eredu_core::MemoryLimit::Unlimited,
            _ => eredu_core::MemoryLimit::Finite(before + required - 1),
        };
        let limits = eredu_core::MemoryLimits::resolve(pool.topology(), [(host, limit)]).unwrap();
        let result = source
            .plan()
            .construct_for_parameter_publication::<_, Publication>(
                &session,
                &source.execution,
                &limits,
                pool,
            );
        if limit_mode == 2 {
            assert!(matches!(
                result,
                Err(ResidentResetError {
                    cause: ResetCause::Memory(WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    )),
                    ..
                })
            ));
            assert_eq!(FILLS.get(), 0);
            assert_eq!(current(pool), before);
        } else {
            let (installation, owner) = result.unwrap();
            assert_eq!(FILLS.get(), 3);
            assert_eq!(current(pool), before + required);
            assert!(
                source
                    .state
                    .layers
                    .slots()
                    .iter()
                    .all(|row| row.position == 19)
            );
            assert!(
                installation
                    .state
                    .layers
                    .slots()
                    .iter()
                    .all(|row| row.position == 0)
            );
            let slot = PreparedParameterStateReset::from_installation(installation);
            assert!(slot.matches(&source.plan().source));
            drop(slot);
            assert_eq!(
                current(pool),
                before + required,
                "prepared retirement owner keeps the allowance"
            );
            drop(owner);
            assert_eq!(current(pool), before);
        }
    }
}

#[test]
fn parameter_reset_rejects_foreign_execution_and_selected_source_before_construction() {
    let (_runtime, data, _, _) = fixture(None, 2);
    let (_foreign_runtime, foreign, _, _) = fixture(None, 2);
    let session = Session(data.clone());
    let foreign_session = Session(foreign.clone());
    let source = data.borrow();
    let before = current(&source.pool);
    let limits = eredu_core::MemoryLimits::unlimited(source.pool.topology());
    let before_id = source.pool.0.usage.lock().unwrap().next_reset;
    for (actual, execution) in [
        (&session, foreign.borrow().execution.clone()),
        (&foreign_session, source.execution.clone()),
    ] {
        let error = source
            .plan()
            .construct_for_parameter_publication::<_, Publication>(
                actual,
                &execution,
                &limits,
                &source.pool,
            )
            .err()
            .unwrap();
        assert!(matches!(
            error.cause,
            ResetCause::Memory(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!(current(&source.pool), before);
        assert_eq!(source.pool.0.usage.lock().unwrap().next_reset, before_id);
        assert_eq!(FILLS.get(), 0);
    }
}

#[test]
fn reset_domain_bookkeeping_is_host_charged_when_device_limits_are_zero() {
    use eredu_core::{
        DomainMemoryRequirements, MemoryDeviceId, MemoryDomainDescription, MemoryLimit,
        MemoryLimits, MemoryLocation, MemoryTopology,
    };
    let mut requirements = Vec::new();
    for separate in [false, true] {
        let gpu = |ordinal| {
            MemoryLocation::Device(MemoryDeviceId {
                backend: "reset-conformance",
                ordinal,
            })
        };
        let domains = if separate {
            vec![
                MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![MemoryLocation::Host],
                },
                MemoryDomainDescription {
                    name: "gpu0".into(),
                    locations: vec![gpu(0)],
                },
                MemoryDomainDescription {
                    name: "gpu1".into(),
                    locations: vec![gpu(1)],
                },
            ]
        } else {
            vec![MemoryDomainDescription {
                name: "shared".into(),
                locations: vec![MemoryLocation::Host, gpu(0), gpu(1)],
            }]
        };
        let topology = Arc::new(MemoryTopology::new(domains).unwrap());
        let limits = MemoryLimits::resolve(
            &topology,
            topology.domains().map(|(domain, _)| {
                (
                    domain,
                    if domain == topology.host_domain() {
                        MemoryLimit::Unlimited
                    } else {
                        MemoryLimit::Finite(0)
                    },
                )
            }),
        )
        .unwrap();
        let pool = MemoryLedger::new(
            topology.clone(),
            limits.clone(),
            DomainMemoryRequirements::zero(&topology),
        )
        .unwrap();
        let (_runtime, data, _, _) = fixture_in_pool(pool.clone(), 2);
        let source = data.borrow();
        let session = Session(data.clone());
        let required = source
            .plan()
            .parameter_publication_required_bytes::<Publication>()
            .unwrap();
        let before = pool.snapshot().unwrap();
        let (destination, owner) = source
            .plan()
            .construct_for_parameter_publication::<_, Publication>(
                &session,
                &source.execution,
                &limits,
                &pool,
            )
            .unwrap();
        let after = pool.snapshot().unwrap();
        for (old, new) in before.domains.iter().zip(&after.domains) {
            assert_eq!(
                new.current_charge_bytes - old.current_charge_bytes,
                if new.domain == topology.host_domain() {
                    required
                } else {
                    0
                }
            );
        }
        drop((destination, owner));
        assert_eq!(
            current(&pool),
            before
                .domains
                .iter()
                .map(|row| row.current_charge_bytes)
                .sum::<u64>()
        );
        requirements.push(required);
    }
    assert!(
        requirements[1] > requirements[0],
        "separate physical domains require additional paid host vectors"
    );
}
