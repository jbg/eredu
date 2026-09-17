use super::*;
fn isolated(name: &str, body: impl FnOnce()) {
    if std::env::var("EREDU_SCHEDULER_COMPONENT_CASE").as_deref() == Ok(name) {
        body();
        println!("SCHEDULER_COMPONENT_OK:{name}");
        return;
    }
    let test = format!("backend::managed_memory::scheduler::tests::{name}");
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture"])
        .env("EREDU_SCHEDULER_COMPONENT_CASE", name)
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{out}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        out.contains(&format!("SCHEDULER_COMPONENT_OK:{name}")),
        "{out}"
    );
}
fn qualified() -> bool {
    let plan = Initializer::prepare();
    let required =
        std::env::var_os("EREDU_REQUIRE_SCHEDULER_INITIALIZATION_QUALIFICATION").is_some();
    if required {
        assert!(plan.is_ok(), "{plan:?}");
    }
    match plan {
        Ok(plan) => match WorkingMemoryPool::shared_native_initialization_required_bytes(&plan) {
            Ok(_) => true,
            Err(WorkingMemoryError::UnknownBound) => {
                assert!(!required, "positive source-account qualification required");
                let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
                let error = pool.initialize_shared_native(plan).unwrap_err();
                assert!(matches!(
                    error.accounting_failure(),
                    Some(WorkingMemoryError::UnknownBound)
                ));
                assert_eq!(pool.used_bytes().unwrap(), 0);
                false
            }
            Err(error) => panic!("unexpected source-account qualification: {error}"),
        },
        Err(SchedulerCause::UnknownLayout) => false,
        Err(error) => panic!("unexpected constructor qualification: {error}"),
    }
}
#[test]
fn scheduler_exact_and_one_short_admission_preserve_permanent_source_owner() {
    isolated(
        "scheduler_exact_and_one_short_admission_preserve_permanent_source_owner",
        || {
            if !qualified() {
                return;
            }
            let plan = Initializer::prepare().unwrap();
            let bytes =
                WorkingMemoryPool::shared_native_initialization_required_bytes(&plan).unwrap();
            let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
            let error = short.initialize_shared_native(plan).unwrap_err();
            assert!(matches!(
                error.accounting_failure(),
                Some(WorkingMemoryError::BudgetExceeded { .. })
            ));
            assert!(error.rejected_plan().is_some());
            assert!(error.constructor_failure().is_none());
            assert_eq!(short.used_bytes().unwrap(), 0);
            drop(error);
            let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
            let owner = exact.initialize_shared_native(plan).unwrap();
            assert_eq!(owner.original_bytes(), bytes);
            assert_eq!(exact.used_bytes().unwrap(), bytes);
            owner.validate_pool(&exact).unwrap();
            owner.output().try_borrow().unwrap();
            assert!(matches!(
                owner.validate_pool(&short),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            drop(owner);
            assert_eq!(exact.used_bytes().unwrap(), bytes); // actual permanent native raw custody
        },
    );
}
#[test]
fn ordinary_scheduler_predecessor_refuses_chain_and_keeps_admitted_device_prefix() {
    use std::error::Error as _;
    isolated(
        "ordinary_scheduler_predecessor_refuses_chain_and_keeps_admitted_device_prefix",
        || {
            if !qualified() {
                return;
            }
            let pool = super::super::domain();
            // Establish the genuine Device prefix first, so the intended next
            // refusal is the ordinary Scheduler and never an ordinary Device fallback.
            super::super::metal_device::prepare_admitted(&pool).unwrap();
            let before = pool.used_bytes().unwrap();
            let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
            let stream = safemlx::Stream::new_with_device(&device);
            let runtime =
                safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
            assert!(INITIALIZED.get().is_none());
            let error = super::super::input_allocator::prepare_admitted(&pool).unwrap_err();
            let scheduler = error
                .source()
                .unwrap()
                .downcast_ref::<MlxSchedulerInitializationError>()
                .unwrap();
            let Failure::Constructor(failure) = &scheduler.0 else {
                panic!("{error:?}");
            };
            let Some(ConstructorFailure::Native(native)) = failure.constructor_failure() else {
                panic!("{error:?}");
            };
            assert_eq!(native.cause(), SchedulerCause::OrdinaryPredecessor);
            assert!(INITIALIZED.get().is_none());
            assert!(pool.used_bytes().unwrap() > before); // intact failed Scheduler node/account
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), before); // Device prefix is still owned
            super::super::metal_device::prepare_admitted(&pool).unwrap();
            assert_eq!(pool.used_bytes().unwrap(), before);
            drop((runtime, stream, device));
        },
    );
}
