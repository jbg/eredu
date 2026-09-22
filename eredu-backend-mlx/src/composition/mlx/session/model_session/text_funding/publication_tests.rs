use super::*;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, MemoryLedger, WorkingMemoryError, WorkingMemoryFundingRun,
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
fn funding(pool: &MemoryLedger, bytes: u64) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
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
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: WorkspaceBound::bounded(bytes, "exact completed storage extents"),
        },
    ))
    .unwrap();
    let admission = crate::memory_fixture::admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(bytes),
    });
    let before = pool.snapshot().unwrap();
    let required = pool
        .reservation_requirements(&admission, None)
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
        .checked_add(pool.fixture_host_current().unwrap())
        .unwrap();
    let execution = InferenceExecutionIdentity::default();
    assert!(matches!(
        pool.reserve_with_capacity(
            &execution,
            &admission,
            crate::memory_fixture::physical_host_limits(pool, required - 1)
        ),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    pool.reserve_with_capacity(
        &execution,
        &admission,
        crate::memory_fixture::physical_host_limits(pool, required),
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
fn assigned_charge(pool: &MemoryLedger) -> u64 {
    let snapshot = pool.snapshot().unwrap();
    let host = &snapshot.domains[0];
    host.current_charge_bytes
        .checked_sub(host.fixed_baseline.total().unwrap())
        .unwrap()
        .checked_sub(host.reservation_control_bytes)
        .unwrap()
}
fn reclaim(pool: &MemoryLedger) -> u64 {
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    crate::backend::ordinary_retirement::reclaim_all();
    pool.fixture_host_charge().unwrap()
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
    // Array deletion queues this real native sidecar. Prepared attachment does
    // not reclaim it; an explicit retirement boundary does.
    drop(earlier);
    (InstalledProbe, observed)
}

fn retire_while_publishing(work: &FundedWork) {
    let _activity = publication_scope::Activity::begin(&work.publishing).unwrap();
    let _scope = publication_scope::OwnedScope::take(&work.scope).unwrap();
    // The earlier physical allocation may still belong to the native cache.
    // Eviction establishes its retirement before explicitly draining owners.
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
}

#[test]
fn prepared_attachment_defers_retirement_and_explicit_reentry_restores_the_same_scope() {
    for panic in [false, true] {
        let array = Array::from_slice(&[3_i32, 5, 7, 11], &[4]);
        let storage = inventory(&array);
        let bytes = storage.byte_bound().unwrap().unwrap()
            + u64::try_from(
                array
                    .try_allocation_info()
                    .unwrap()
                    .unwrap()
                    .host_control_bytes(),
            )
            .unwrap();
        let (plan, controls) = OrdinaryPublicationPlan::fixture(2);
        let required = bytes.checked_add(controls).unwrap();
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (reservation, run) = funding(&pool, required);
        let work = FundedWork::new_ordinary(run.scope().unwrap(), plan).unwrap();
        drop(storage);
        let mut storage = work.prepare_inventory().unwrap();
        storage.include_array(&array).unwrap();
        let (probe, observed) = queue_retirement(&work, panic);
        work.publish(storage).unwrap();
        assert_eq!(observed.get(), Observation::default());
        let result = catch_unwind(AssertUnwindSafe(|| retire_while_publishing(&work)));
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
            .same_ledger(&pool));
        assert_eq!(assigned_charge(&pool), required);
        if panic {
            assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 731);
            assert!(work.published.get());
            // The explicit retirement unwind restores this same scope and
            // leaves its successful publication valid for an idempotent retry.
            let mut retry = work.prepare_inventory().unwrap();
            retry.include_array(&array).unwrap();
            work.publish(retry).unwrap();
        } else {
            result.unwrap();
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
        let retired = pool.snapshot().unwrap();
        assert_eq!(retired.funding_accounts, 0);
        assert_eq!(retired.reservations, 0);
        assert_eq!(
            retired.domains[0].current_charge_bytes,
            retired.domains[0].fixed_baseline.total().unwrap()
        );
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
    let returned_native = nonstate.byte_bound().unwrap().unwrap()
        + u64::try_from(
            array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .host_control_bytes(),
        )
        .unwrap();
    let bytes = returned_native + decoder.byte_bound().unwrap().unwrap();
    let (plan, controls) = OrdinaryPublicationPlan::fixture(4);
    let required = bytes.checked_add(controls).unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (reservation, run) = funding(&pool, required);
    let work = FundedWork::new_ordinary(run.scope().unwrap(), plan).unwrap();
    drop((nonstate, decoder));
    let mut nonstate = work.prepare_inventory().unwrap();
    nonstate.include_array(&array).unwrap();
    let mut decoder = work.prepare_inventory().unwrap();
    decoder
        .include_slot_metadata(live.metadata().clone())
        .unwrap();
    decoder.include_slot_metadata(token.clone()).unwrap();
    let (probe, observed) = queue_retirement(&work, false);
    let error = work.publish_model(nonstate, decoder).unwrap_err();
    assert_eq!(observed.get(), Observation::default());
    retire_while_publishing(&work);
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
    assert!(
        matches!(
            cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
            Some(HostSlotAttachmentError::Retired)
        ),
        "{error:?}"
    );
    assert!(!work.published.get());
    work.certify().unwrap();
    assert!(
        work.scope.borrow().is_some(),
        "partial publication grants no certification"
    );
    assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), [23, 29]);
    assert_eq!(live.slots(), [3, 5, 7, 11]);
    assert_eq!(assigned_charge(&pool), required);
    let before = pool.snapshot().unwrap();
    let unconverted = before.domains[0]
        .outstanding_reservation_bytes
        .checked_sub(before.domains[0].reservation_control_bytes)
        .unwrap();
    assert!(unconverted > 0);
    // Attached payloads return to this uncertified reservation. Other paid
    // host constructors retire with their own allocation; they are not native
    // payload credit. The first live decoder slot was attached before refusal.
    let returned_host = u64::try_from(std::mem::size_of_val(live.slots())).unwrap();
    let quarantined = unconverted
        .checked_add(before.domains[0].registry_metadata_bytes)
        .and_then(|bytes| bytes.checked_add(returned_native))
        .and_then(|bytes| bytes.checked_add(returned_host))
        .unwrap();
    drop((work, reservation, run, array, live, token, error));
    reclaim(&pool);
    let after = pool.snapshot().unwrap();
    assert_eq!(
        after.domains[0]
            .outstanding_reservation_bytes
            .checked_sub(after.domains[0].reservation_control_bytes)
            .unwrap(),
        quarantined,
        "uncertified work retains unused and returned metadata allowances after completed payloads retire"
    );
    assert_eq!(after.funding_accounts, before.funding_accounts);
    assert_eq!(after.domains[0].registered_storage_bytes, 0);
}

#[test]
fn publication_budget_failure_restores_scope_without_certification_or_refund() {
    let (plan, controls) = OrdinaryPublicationPlan::fixture(1);
    let quoted_payload = 8u64;
    let required = quoted_payload.checked_add(controls).unwrap();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let (reservation, run) = funding(&pool, required);
    let work = FundedWork::new_ordinary(run.scope().unwrap(), plan).unwrap();
    // A new actual source exceeds the complete admitted allowance by one byte,
    // even if every unused constructor allowance were available for payload.
    let source =
        HostSlotTable::new(vec![31_u8; usize::try_from(required + 1).unwrap()].into_boxed_slice());
    let mut storage = work.prepare_inventory().unwrap();
    storage
        .include_slot_metadata(source.metadata().clone())
        .unwrap();
    let error = work.publish(storage).unwrap_err();
    assert!(
        matches!(
            cause::<WorkingMemoryError>(&error),
            Some(WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. })
                if required_bytes > available_bytes
        ),
        "{error:?}"
    );
    work.certify().unwrap();
    assert!(work.scope.borrow().is_some());
    assert_eq!(assigned_charge(&pool), required);
    drop((work, reservation, run, source, error));
    assert_eq!(reclaim(&pool), required);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
