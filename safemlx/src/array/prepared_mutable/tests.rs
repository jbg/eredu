use super::*;
use crate::{reclaim_allocation_owners, PreparedOriginalBufferBudget};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

// Pinned native/Rust validation must set this flag; an unknown layout then
// fails the gate rather than silently replacing its required positive behavior.
fn qualified_plan(runtime: &PreparedInputRuntime) -> Option<OriginalMutablePairPlan<'_>> {
    match OriginalMutablePairPlan::inspect(runtime) {
        Ok(plan) => Some(plan),
        Err(cause) => {
            assert_eq!(cause, OriginalMutablePairCause::Layout);
            assert_eq!(empty_context(), Ok(()));
            assert_ne!(
                std::env::var_os("EREDU_REQUIRE_MUTABLE_PAIR_QUALIFICATION"),
                Some("1".into()),
                "pinned mutable-pair validation requires positive native qualification"
            );
            eprintln!("mutable-pair qualification=native-layout-unknown; typed refusal before owner preparation");
            None
        }
    }
}

struct Custody {
    retired: Arc<[AtomicUsize; 6]>,
    index: usize,
}
impl Drop for Custody {
    fn drop(&mut self) {
        self.retired[self.index].fetch_add(1, Ordering::SeqCst);
    }
}
fn counts() -> Arc<[AtomicUsize; 6]> {
    Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)))
}
fn custody(retired: &Arc<[AtomicUsize; 6]>, index: usize) -> Custody {
    Custody {
        retired: retired.clone(),
        index,
    }
}
fn owners(retired: &Arc<[AtomicUsize; 6]>) -> OriginalMutablePairCustodies<Custody> {
    OriginalMutablePairCustodies::new(
        custody(retired, 0),
        custody(retired, 1),
        custody(retired, 2),
        custody(retired, 3),
        custody(retired, 4),
    )
}
fn budget(
    runtime: &PreparedInputRuntime,
    bytes: usize,
    retired: &Arc<[AtomicUsize; 6]>,
) -> OriginalBufferBudget {
    PreparedOriginalBufferBudget::try_new(runtime, bytes, custody(retired, 5))
        .unwrap()
        .try_allocate()
        .unwrap()
}
fn observed(retired: &Arc<[AtomicUsize; 6]>) -> [usize; 6] {
    std::array::from_fn(|i| retired[i].load(Ordering::SeqCst))
}

#[test]
fn mutable_pair_exact_layout_nonzero_birth_and_alias_outlive_private_scope() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(plan) = qualified_plan(&runtime) else {
        return;
    };
    let facts = plan.facts();
    assert_eq!(facts.graph_requests(), 5);
    assert_eq!(facts.copy_bytes(), 8);
    assert!(facts.backing_bytes() >= facts.copy_bytes());
    assert!(facts.metadata_bytes() > 0);
    assert!(facts.module_bytes() > 0 && facts.thread_bytes() > 0);
    assert!(facts.control_bytes::<Custody>().unwrap() > facts.metadata_bytes());
    assert_eq!(
        facts.record_minimum_capacity(),
        PreparedSubmissionRecordQuota::<Custody>::minimum_layout()
            .unwrap()
            .capacity
    );
    let retired = counts();
    let budget = budget(&runtime, facts.backing_bytes(), &retired);
    let inspect = budget.clone();
    let prepared = plan
        .prepare([17, 0xf123_4567], budget, owners(&retired))
        .unwrap();
    assert_eq!(observed(&retired), [0; 6]);
    let array = prepared.try_construct().unwrap();
    assert_eq!(empty_context(), Ok(()));
    let allocation = inspect.inspect_array(&array).unwrap().unwrap().allocation();
    assert_eq!(allocation.bytes(), facts.backing_bytes());
    assert_eq!(inspect.occupied_bytes(), facts.backing_bytes());
    assert_eq!(
        array.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &[17, 0xf123_4567]
    );
    let alias = array.clone();
    drop((array, inspect));
    reclaim_allocation_owners();
    let before_alias = observed(&retired);
    assert_eq!(
        before_alias,
        [0, 1, 1, 1, 1, 0],
        "Graph and physical budget survive their originating Scope and capsule"
    );
    assert_eq!(
        alias
            .inspect_original_buffer_alias()
            .unwrap()
            .unwrap()
            .allocation()
            .identity(),
        allocation.identity()
    );
    drop(alias);
    reclaim_allocation_owners();
    assert_eq!(observed(&retired), [1; 6]);
}

#[test]
fn mutable_pair_one_short_backing_preserves_typed_failure_input_and_every_owner() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(plan) = qualified_plan(&runtime) else {
        return;
    };
    let retired = counts();
    let budget = budget(&runtime, plan.facts().backing_bytes() - 1, &retired);
    let inspect = budget.clone();
    let prepared = plan.prepare([31, 47], budget, owners(&retired)).unwrap();
    let error = prepared.try_construct().unwrap_err();
    assert_eq!(error.cause(), OriginalMutablePairCause::Native(9));
    assert_eq!(error.values(), &[31, 47]);
    assert!(
        error.native_source().is_some(),
        "actual C++ allocator exception retained"
    );
    assert_eq!(empty_context(), Ok(()));
    assert_eq!(
        inspect.occupied_bytes(),
        0,
        "no physical debit survives synchronous rollback"
    );
    assert_eq!(
        observed(&retired),
        [0; 6],
        "all exact owner custody survives the failed call"
    );
    drop(inspect);
    drop(error);
    reclaim_allocation_owners();
    assert_eq!(observed(&retired), [1; 6]);
}

#[test]
fn mutable_pair_native_snapshot_and_failed_capsule_retire_in_either_order() {
    for snapshot_first in [false, true] {
        let runtime = PreparedInputRuntime::prepare().unwrap();
        let Some(plan) = qualified_plan(&runtime) else {
            return;
        };
        let retired = counts();
        let budget = budget(&runtime, plan.facts().backing_bytes() - 1, &retired);
        let mut error = plan
            .prepare([59, 61], budget, owners(&retired))
            .unwrap()
            .try_construct()
            .unwrap_err();
        let snapshot = error.take_native_source().unwrap();
        assert!(error.native_source().is_none());
        assert!(error.take_native_source().is_none());
        if snapshot_first {
            drop(snapshot);
            reclaim_allocation_owners();
            assert_eq!(observed(&retired), [0; 6]);
            drop(error);
        } else {
            drop(error);
            reclaim_allocation_owners();
            assert_eq!(
                observed(&retired),
                [1, 1, 1, 0, 1, 1],
                "only the actual snapshot's carrier remains"
            );
            assert!(snapshot.source_type_bytes().is_some());
            drop(snapshot);
        }
        reclaim_allocation_owners();
        assert_eq!(observed(&retired), [1; 6]);
    }
}

#[test]
fn mutable_pair_active_scope_refuses_before_setup_and_before_prepared_construction() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(plan) = qualified_plan(&runtime) else {
        return;
    };
    for after_prepare in [false, true] {
        let retired = counts();
        let budget = budget(&runtime, plan.facts().backing_bytes(), &retired);
        let inspect = budget.clone();
        let (error, mut scope) = if after_prepare {
            let prepared = plan.prepare([71, 73], budget, owners(&retired)).unwrap();
            let scope = SubmissionScope::try_begin().unwrap();
            (prepared.try_construct().unwrap_err(), scope)
        } else {
            let scope = SubmissionScope::try_begin().unwrap();
            (
                plan.prepare([71, 73], budget, owners(&retired))
                    .err()
                    .expect("active context must refuse"),
                scope,
            )
        };
        assert_eq!(error.cause(), OriginalMutablePairCause::Context);
        assert_eq!(error.values(), &[71, 73]);
        assert_eq!(inspect.occupied_bytes(), 0);
        assert_eq!(observed(&retired), [0; 6]);
        assert!(scope.status().is_settled());
        scope.seal();
        drop((scope, error, inspect));
        reclaim_allocation_owners();
        assert_eq!(observed(&retired), [1; 6]);
    }
}

#[test]
fn mutable_pair_runtime_busy_returns_the_same_prepared_prefix_without_native_entry() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(plan) = qualified_plan(&runtime) else {
        return;
    };
    let retired = counts();
    let budget = budget(&runtime, plan.facts().backing_bytes(), &retired);
    let inspect = budget.clone();
    let prepared = plan.prepare([79, 83], budget, owners(&retired)).unwrap();
    let (ready, began) = std::sync::mpsc::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _loan = runtime_lock::coordinate_entry();
        let _ = ready.send(());
        let _ = wait.recv(); // sender destruction also releases on test unwind
    });
    struct Release {
        sender: Option<std::sync::mpsc::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }
    impl Drop for Release {
        fn drop(&mut self) {
            drop(self.sender.take());
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
    let release = Release {
        sender: Some(release),
        thread: Some(thread),
    };
    began
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let error = prepared.try_construct().unwrap_err();
    assert_eq!(error.cause(), OriginalMutablePairCause::Busy);
    assert_eq!(error.values(), &[79, 83]);
    assert!(error.native_source().is_none());
    assert_eq!(observed(&retired), [0; 6]);
    assert_eq!(inspect.occupied_bytes(), 0);
    assert_eq!(empty_context(), Ok(()));
    drop(release);
    drop((error, inspect));
    reclaim_allocation_owners();
    assert_eq!(observed(&retired), [1; 6]);
}
