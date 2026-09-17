use super::*;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, WorkingMemoryError, WorkingMemoryFundingRun, WorkingMemoryPool,
        WorkingMemoryReservation,
    },
    HostSlotAttachmentError, HostSlotTable,
};
use std::{
    num::NonZeroU8,
    panic::{catch_unwind, AssertUnwindSafe},
};

// This publication-only fixture prices the actual completed backing/boxed
// extents. It performs no model submission and makes no whole-native bound claim.
fn funding(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::StateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let zero = || WorkspaceBound::bounded(0, "completed storage publication fixture");
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: zero(),
        attention: zero(),
        vocabulary: zero(),
        state_update: zero(),
        materialization: zero(),
        retained: WorkspaceBound::bounded(bytes, "exact completed storage extents"),
    })
    .unwrap();
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &Admission {
            requested_positions: 1,
            state,
            incremental_required_bytes: bytes,
            available_memory_bytes: None,
        },
        bytes,
    )
    .unwrap()
    .into_funding()
    .unwrap()
}
fn inventory(array: &Array) -> RetainedStorage {
    let mut storage = RetainedStorage::default();
    storage.include_array(array).unwrap();
    storage
}
fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return Some(value);
        }
        error = error.source()?;
    }
}
fn fenced(result: Result<(), Error>) -> bool {
    result
        .as_ref()
        .err()
        .and_then(|e| cause::<WorkingMemoryError>(e))
        == Some(&WorkingMemoryError::ExecutionFenced)
}
fn reclaim(pool: &WorkingMemoryPool) -> u64 {
    safemlx::reclaim_allocation_owners();
    crate::backend::ordinary_retirement::reclaim_all();
    pool.used_bytes().unwrap()
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Observation {
    calls: usize,
    unlocked: bool,
    certify_fenced: bool,
    publish_fenced: bool,
}
thread_local! {
    static REENTRY:RefCell<Option<(FundedWorkOwner,Rc<Cell<Observation>>)>>=const { RefCell::new(None) };
}
struct InstalledProbe;
impl Drop for InstalledProbe {
    fn drop(&mut self) {
        // Release the test cell loan before the last closed Work owner can drop.
        let owner = REENTRY.with(|slot| slot.borrow_mut().take());
        drop(owner);
    }
}
struct RetiredProbe {
    panic: bool,
}
impl Drop for RetiredProbe {
    fn drop(&mut self) {
        let active = REENTRY
            .with(|slot| slot.borrow().clone())
            .expect("scoped retirement probe");
        let (work, observed) = active;
        let prior = observed.get();
        let unlocked = work.scope.try_borrow_mut().is_ok();
        let certify_fenced = fenced(work.certify());
        let publish_fenced = fenced(work.publish(RetainedStorage::default()));
        observed.set(Observation {
            calls: prior.calls + 1,
            unlocked,
            certify_fenced,
            publish_fenced,
        });
        if self.panic {
            std::panic::panic_any(731_u32);
        }
    }
}
fn queue_retirement(
    work: &FundedWorkOwner,
    panic: bool,
) -> (InstalledProbe, Rc<Cell<Observation>>) {
    let earlier = Array::from_slice(&[17_i32, -19], &[2]);
    earlier
        .retain_allocation_owner(RetiredProbe { panic })
        .unwrap();
    let observed = Rc::new(Cell::new(Observation::default()));
    REENTRY.with(|slot| {
        assert!(slot
            .borrow_mut()
            .replace((work.clone(), observed.clone()))
            .is_none())
    });
    // Array deletion queues this real native sidecar. The next ordinary native
    // entry is the actual publication attachment, which reclaims it unlocked.
    drop(earlier);
    (InstalledProbe, observed)
}

#[test]
fn real_native_attachment_reentry_is_fenced_and_unwind_restores_the_same_scope() {
    for panic in [false, true] {
        let array = Array::from_slice(&[3_i32, 5, 7, 11], &[4]);
        let storage = inventory(&array);
        let bytes = storage.byte_bound().unwrap().unwrap();
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let (reservation, run) = funding(&pool, bytes);
        let work = FundedWork::new(run.scope().unwrap());
        let (probe, observed) = queue_retirement(&work, panic);
        let result = catch_unwind(AssertUnwindSafe(|| work.publish(storage)));
        let observation = observed.get();
        drop(probe);
        assert_eq!(
            observation,
            Observation {
                calls: 1,
                unlocked: true,
                certify_fenced: true,
                publish_fenced: true
            }
        );
        assert!(work
            .scope
            .borrow()
            .as_ref()
            .unwrap()
            .pool()
            .same_domain(&pool));
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        if panic {
            assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 731);
            assert!(!work.published.get());
            // Actual registration cleanup returned credit to this same live
            // envelope. The activity and scope restore permits a real retry.
            work.publish(inventory(&array)).unwrap();
        } else {
            result.unwrap().unwrap();
        }
        assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), [3, 5, 7, 11]);
        work.certify().unwrap();
        assert!(work.scope.borrow().is_none());
        drop((work, reservation, run));
        assert_eq!(
            reclaim(&pool),
            bytes,
            "escaped array retains its exact charge"
        );
        drop(array);
        assert_eq!(reclaim(&pool), 0);
    }
}

#[test]
fn failed_decoder_publication_keeps_prior_native_attachment_and_uncertified_credit() {
    let array = Array::from_slice(&[23_i32, 29], &[2]);
    let nonstate = inventory(&array);
    let a = HostSlotTable::new(Box::new([3_u32, 5, 7, 11]));
    let b = HostSlotTable::new(Box::new([3_u32, 5, 7, 11]));
    let (live, dead) = if a.metadata().identity() < b.metadata().identity() {
        (a, b)
    } else {
        (b, a)
    };
    let token = dead.metadata().clone();
    drop(dead);
    let mut decoder = RetainedStorage::default();
    decoder
        .include_slot_metadata(live.metadata().clone())
        .unwrap();
    decoder.include_slot_metadata(token.clone()).unwrap();
    let bytes = nonstate.byte_bound().unwrap().unwrap() + decoder.byte_bound().unwrap().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (reservation, run) = funding(&pool, bytes);
    let work = FundedWork::new(run.scope().unwrap());
    let (probe, observed) = queue_retirement(&work, false);
    let error = work.publish_model(nonstate, decoder).unwrap_err();
    let observation = observed.get();
    drop(probe);
    assert_eq!(
        observation,
        Observation {
            calls: 1,
            unlocked: true,
            certify_fenced: true,
            publish_fenced: true
        }
    );
    assert!(matches!(
        cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
        Some(HostSlotAttachmentError::Retired)
    ));
    assert!(!work.published.get());
    work.certify().unwrap();
    assert!(
        work.scope.borrow().is_some(),
        "partial publication grants no certification"
    );
    assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), [23, 29]);
    assert_eq!(live.slots(), [3, 5, 7, 11]);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop((work, reservation, run, array, live, token, error));
    assert_eq!(
        reclaim(&pool),
        bytes,
        "uncertified work quarantines the full envelope"
    );
}

#[test]
fn publication_budget_failure_restores_scope_without_certification_or_refund() {
    let source = HostSlotTable::new(Box::new([31_u32, 37]));
    let mut storage = RetainedStorage::default();
    storage
        .include_slot_metadata(source.metadata().clone())
        .unwrap();
    let bytes = source.metadata().capacity_bytes().unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (reservation, run) = funding(&pool, bytes - 1);
    let work = FundedWork::new(run.scope().unwrap());
    let error = work.publish(storage).unwrap_err();
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    work.certify().unwrap();
    assert!(work.scope.borrow().is_some());
    assert_eq!(pool.used_bytes().unwrap(), bytes - 1);
    drop((work, reservation, run, source, error));
    assert_eq!(reclaim(&pool), bytes - 1);
}
