use super::*;
use std::error::Error as _;
fn isolated(name: &str, body: impl FnOnce()) {
    if std::env::var("EREDU_DEVICE_COMPONENT_CASE").as_deref() == Ok(name) {
        body();
        println!("DEVICE_COMPONENT_OK:{name}");
        return;
    }
    let test = format!("backend::managed_memory::metal_device::tests::{name}");
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture"])
        .env("EREDU_DEVICE_COMPONENT_CASE", name)
        .output()
        .unwrap();
    let output = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{output}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        output.contains(&format!("DEVICE_COMPONENT_OK:{name}")),
        "{output}"
    );
}
fn qualified() -> bool {
    let layout = PreparedMetalDevice::<SharedNativeInitializationCustody>::layout();
    if std::env::var_os("EREDU_REQUIRE_METAL_DEVICE_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(layout.is_ok(), "{layout:?}");
    }
    match layout {
        Ok(_) => {
            let plan = Initializer::prepare().unwrap();
            let bound = MemoryLedger::shared_native_initialization_required_bytes(&plan);
            if std::env::var_os("EREDU_REQUIRE_METAL_DEVICE_INITIALIZATION_QUALIFICATION").is_some()
            {
                assert!(bound.is_ok(), "{bound:?}");
            }
            match bound {
                Ok(_) => true,
                Err(WorkingMemoryError::UnknownBound) => {
                    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
                    let error = pool.initialize_shared_native(plan).unwrap_err();
                    assert!(matches!(
                        error.accounting_failure(),
                        Some(WorkingMemoryError::UnknownBound)
                    ));
                    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
                    false
                }
                Err(error) => panic!("unexpected account qualification: {error}"),
            }
        }
        Err(MetalDeviceCause::UnknownLayout | MetalDeviceCause::UnqualifiedSource) => {
            assert!(matches!(
                Initializer::prepare(),
                Err(MetalDeviceCause::UnknownLayout | MetalDeviceCause::UnqualifiedSource)
            ));
            false
        }
        Err(error) => panic!("unexpected source qualification: {error}"),
    }
}
#[test]
fn actual_device_exact_admission_and_one_short_preserve_permanent_owner() {
    isolated(
        "actual_device_exact_admission_and_one_short_preserve_permanent_owner",
        || {
            if !qualified() {
                return;
            }
            let plan = Initializer::prepare().unwrap();
            let bytes = MemoryLedger::shared_native_initialization_required_bytes(&plan).unwrap();
            let short = crate::memory_fixture::ledger(bytes - 1, 0).unwrap();
            let error = short.initialize_shared_native(plan).unwrap_err();
            assert!(matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::Domain(
                    eredu_core::MemoryDomainError::BudgetExceeded { .. }
                ))
            ));
            assert!(error.rejected_plan().is_some());
            assert_eq!(short.fixture_host_charge().unwrap(), 0);
            drop(error);
            let exact = crate::memory_fixture::ledger(bytes, 0).unwrap();
            let owner = exact.initialize_shared_native(plan).unwrap();
            assert_eq!(owner.original_bytes(), bytes);
            assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
            owner.validate_pool(&exact).unwrap();
            assert!(matches!(
                owner.validate_pool(&short),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            owner.output().try_borrow().unwrap();
            // Public wrappers do not retire a permanent live Device/default library.
            drop(owner);
            assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
        },
    );
}
#[test]
fn actual_device_scheduler_allocator_join_share_the_exact_domain_once() {
    isolated(
        "actual_device_scheduler_allocator_join_share_the_exact_domain_once",
        || {
            if !qualified() {
                return;
            }
            let pool = super::super::ledger();
            let before = pool.fixture_host_charge().unwrap();
            assert!(super::super::scheduler::initialized_original_bytes(&pool).is_none());
            let scheduler_layout =
                safemlx::PreparedScheduler::<SharedNativeInitializationCustody>::layout();
            if std::env::var_os("EREDU_REQUIRE_SCHEDULER_INITIALIZATION_QUALIFICATION").is_some() {
                assert!(scheduler_layout.is_ok(), "{scheduler_layout:?}");
            }
            if matches!(
                scheduler_layout,
                Err(safemlx::SchedulerCause::UnknownLayout)
            ) {
                let error = super::super::input_allocator::prepare_admitted(&pool).unwrap_err();
                let cause = error
                    .source()
                    .unwrap()
                    .source()
                    .unwrap()
                    .downcast_ref::<safemlx::SchedulerCause>()
                    .unwrap();
                assert_eq!(*cause, safemlx::SchedulerCause::UnknownLayout);
                assert!(super::super::scheduler::initialized_original_bytes(&pool).is_none());
                drop(error);
                return;
            }
            scheduler_layout.unwrap();
            let foreign = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
            let error = prepare_admitted(&foreign).unwrap_err();
            assert!(matches!(
                error.0,
                Failure::Policy(WorkingMemoryError::IdentityMismatch)
            ));
            assert_eq!(pool.fixture_host_charge().unwrap(), before);
            let (runtime, coverage) =
                super::super::input_allocator::prepare_admitted(&pool).unwrap();
            assert!(matches!(
                coverage,
                super::super::input_allocator::InputAllocatorCoverage::Admitted {
                    requires_device: true,
                    device_admitted: true
                }
            ));
            let after = pool.fixture_host_charge().unwrap();
            let device = INITIALIZED.get().unwrap();
            let scheduler_bytes =
                super::super::scheduler::initialized_original_bytes(&pool).unwrap();
            assert!(scheduler_bytes > 0);
            assert!(after > before + device.original_bytes() + scheduler_bytes); // third, allocator account
            assert!(super::super::scheduler::initialized_original_bytes(&foreign).is_none());
            device.validate_pool(&pool).unwrap();
            device.output().try_borrow().unwrap();
            let (again, again_coverage) =
                super::super::input_allocator::prepare_admitted(&pool).unwrap();
            assert_eq!(coverage, again_coverage);
            assert_eq!(pool.fixture_host_charge().unwrap(), after);
            drop((runtime, again));
            assert_eq!(pool.fixture_host_charge().unwrap(), after);
        },
    );
}
#[test]
fn ordinary_device_predecessor_is_not_promoted_into_the_source_account() {
    isolated(
        "ordinary_device_predecessor_is_not_promoted_into_the_source_account",
        || {
            if !qualified() {
                return;
            }
            let runtime = safemlx::PreparedInputRuntime::prepare().unwrap();
            let pool = super::super::ledger();
            let before = pool.fixture_host_charge().unwrap();
            let error = prepare_admitted(&pool).unwrap_err();
            let Failure::Constructor(ref failure) = error.0 else {
                panic!("{error:?}")
            };
            let Some(ConstructorFailure::Native(native)) = failure.constructor_failure() else {
                panic!("{error:?}")
            };
            assert_eq!(native.cause(), MetalDeviceCause::OrdinaryPredecessor);
            assert!(pool.fixture_host_charge().unwrap() > before); // failed node still owns its account
            assert!(INITIALIZED.get().is_none());
            drop(error);
            assert_eq!(pool.fixture_host_charge().unwrap(), before);
            drop(runtime);
        },
    );
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
