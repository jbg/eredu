use super::*;
use crate::working_memory::{
    HostSlotStorageKey, InferenceExecutionIdentity, InferenceRequest, InferenceTextPreparation,
    RegisteredDecoderHostCopy, WorkingMemoryPool,
};
use crate::{HostMetadataKey, HostSlotTable};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig, StateMemoryLayout,
    TextGenerationConfig, WorkspaceBound, cache::LayerCachePolicy,
};
use std::{
    cell::Cell,
    num::NonZeroU8,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicUsize, Ordering},
};

const CAPACITY: u64 = 65536;
thread_local! {
    static CONSTRUCTIONS: Cell<usize> = const { Cell::new(0) };
    static PANIC_CONSTRUCTION: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn before_construct() {
    CONSTRUCTIONS.set(CONSTRUCTIONS.get() + 1);
    assert!(
        !PANIC_CONSTRUCTION.replace(false),
        "injected pending-part construction failure"
    );
}
fn held(pool: &WorkingMemoryPool) -> u64 {
    pool.0
        .usage
        .lock()
        .unwrap()
        .funding
        .values()
        .map(|state| state.host_held)
        .sum()
}
fn state(pool: &WorkingMemoryPool) -> (u64, u64, u64, usize) {
    let u = pool.0.usage.lock().unwrap();
    (
        u.reserved,
        u.registered,
        u.peak,
        u.funding.values().map(|s| s.scopes).sum(),
    )
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(HostMetadataKey);
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.0)
    }
}
fn fresh(
    pool: &WorkingMemoryPool,
    bytes: u64,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        0,
        1,
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "closed scalar host preparation fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1,
                state,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            CAPACITY,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: false,
        temperature: 0.0,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(0),
    });
    (
        request.prepare_text(&execution, geometry, config).unwrap(),
        run,
        config,
    )
}
// A real zero-slot dense worker provides the authentic original prompt
// completion. The test prices only this closed host component, not inference.
fn completion(
    preparation: &InferenceTextPreparation,
    run: &WorkingMemoryFundingRun,
) -> InferencePromptCompletion {
    let source = HostSlotTable::new(Vec::<u8>::new().into_boxed_slice());
    let key = Key(source.metadata().identity().registry_key().clone());
    let charge = run.pool().register_storage([(key.clone(), 0)]).unwrap();
    let plan =
        RegisteredDecoderHostCopy::bind(run.pool(), source.prepare_copy_slots().unwrap(), key)
            .unwrap()
            .for_dense_destination::<u8>()
            .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(plan, run, charge)
        .unwrap();
    let finished = slots.finish().unwrap();
    let destination = Key(finished.metadata().identity().registry_key().clone());
    let (table, completion) = finished.publish(destination).unwrap();
    native.certify().unwrap();
    drop(table);
    completion
}
struct Payload {
    value: u32,
    pool: WorkingMemoryPool,
    drops: Arc<AtomicUsize>,
    expected_hold: u64,
}
impl Drop for Payload {
    fn drop(&mut self) {
        assert_eq!(
            held(&self.pool),
            self.expected_hold,
            "payload must retire before its host hold"
        );
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn payload(pool: &WorkingMemoryPool, drops: &Arc<AtomicUsize>, expected_hold: u64) -> Payload {
    Payload {
        value: 37,
        pool: pool.clone(),
        drops: drops.clone(),
        expected_hold,
    }
}

#[test]
fn exact_host_plan_constructs_one_nonclone_payload_and_aliases_keep_custody() {
    let plan = PendingTokenInputHostPlan::<Payload>::prepare().unwrap();
    let d = size_of::<PreparedInputPart<Payload>>() as u64;
    assert_eq!(plan.retained_bytes(), d);
    let h = plan.initialization_peak_bytes();
    assert_eq!(h, 3 * d + size_of::<Payload>() as u64);
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let completion = completion(&preparation, &run);
    let prior = CONSTRUCTIONS.get();
    let prepared = completion
        .prepare_pending_token_input::<Payload>(&run)
        .unwrap();
    assert_eq!(CONSTRUCTIONS.get(), prior);
    assert_eq!(held(&pool), h);
    let drops = Arc::new(AtomicUsize::new(0));
    let (input, complete) = prepared.construct(payload(&pool, &drops, h)).unwrap();
    let alias = input.clone();
    assert!(std::ptr::eq(input.parts(), alias.parts()));
    assert_eq!(input.parts().len(), 1);
    assert_eq!(input.parts()[0].modality(), InputModality::Text);
    assert!(input.parts()[0].metadata().is_empty());
    assert!(input.parts()[0].extents().is_empty());
    let PreparedInputPayload::TokenIds(actual) = input.parts()[0].payload() else {
        panic!("closed token role")
    };
    assert_eq!(actual.value, 37);
    assert_eq!(input.request().geometry().input_positions, 1);
    complete.finish().unwrap();
    preparation.bind_prompt().unwrap();
    drop(preparation);
    run.close().unwrap();
    drop(input);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), h);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(pool.acquire_unquoted().is_ok());
}

#[test]
fn one_byte_short_rejects_before_part_construction_without_refunding_claim() {
    let h = PendingTokenInputHostPlan::<u32>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h - 1);
    let completion = completion(&preparation, &run);
    let before = state(&pool);
    let attempts = CONSTRUCTIONS.get();
    assert!(
        matches!(completion.prepare_pending_token_input::<u32>(&run),
        Err(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes})
        if required_bytes==h && available_bytes==h-1)
    );
    assert_eq!(state(&pool), before);
    assert_eq!(held(&pool), 0);
    assert_eq!(CONSTRUCTIONS.get(), attempts);
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    drop(preparation);
    run.close().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn foreign_pool_and_same_pool_account_fail_before_host_hold() {
    let h = PendingTokenInputHostPlan::<u32>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    for same_pool in [false, true] {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let other = if same_pool {
            pool.clone()
        } else {
            WorkingMemoryPool::new(CAPACITY, 0).unwrap()
        };
        let (preparation, run, _) = fresh(&pool, h);
        let (foreign, foreign_run, _) = fresh(&other, h);
        let complete = completion(&preparation, &run);
        let before = (state(&pool), state(&other), CONSTRUCTIONS.get());
        assert!(matches!(
            complete.prepare_pending_token_input::<u32>(&foreign_run),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        assert_eq!((state(&pool), state(&other), CONSTRUCTIONS.get()), before);
        assert_eq!(held(&pool), 0);
        assert_eq!(held(&other), 0);
        foreign_run.scope().unwrap().certify().unwrap();
        drop((preparation, foreign));
        run.close().unwrap();
        foreign_run.close().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(other.used_bytes().unwrap(), 0);
    }
}

#[test]
fn protected_host_payload_cannot_be_spent_by_native_publication() {
    let h = PendingTokenInputHostPlan::<u32>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let prepared = completion(&preparation, &run)
        .prepare_pending_token_input::<u32>(&run)
        .unwrap();
    let native = run.scope().unwrap();
    let before = state(&pool);
    assert!(matches!(
        native.adopt_storage_individually([(17u32, 1)]),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(state(&pool), before);
    let (input, complete) = prepared.construct(11).unwrap();
    native.certify().unwrap();
    complete.finish().unwrap();
    drop(preparation);
    run.close().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(input);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unwind_drops_incoming_payload_before_closed_hold_and_does_not_quarantine() {
    let h = PendingTokenInputHostPlan::<Payload>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let prepared = completion(&preparation, &run)
        .prepare_pending_token_input::<Payload>(&run)
        .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    PANIC_CONSTRUCTION.set(true);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || prepared.construct(payload(&pool, &drops, h))
        ))
        .is_err()
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(held(&pool), 0);
    run.scope().unwrap().certify().unwrap();
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    drop(preparation);
    run.close().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn accounting_overflow_and_quarantine_reject_without_partial_host_hold() {
    let h = PendingTokenInputHostPlan::<u32>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let complete = completion(&preparation, &run);
    {
        let mut usage = pool.0.usage.lock().unwrap();
        let account = usage.funding.values_mut().next().unwrap();
        assert_eq!(account.scopes, 0);
        account.scopes = usize::MAX;
    }
    let before = state(&pool);
    assert!(matches!(
        complete.prepare_pending_token_input::<u32>(&run),
        Err(WorkingMemoryError::Overflow)
    ));
    assert_eq!(state(&pool), before);
    assert_eq!(held(&pool), 0);
    pool.0
        .usage
        .lock()
        .unwrap()
        .funding
        .values_mut()
        .next()
        .unwrap()
        .scopes = 0;
    drop(preparation);
    run.close().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let complete = completion(&preparation, &run);
    drop(run.scope().unwrap());
    let before = state(&pool);
    assert!(matches!(
        complete.prepare_pending_token_input::<u32>(&run),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(state(&pool), before);
    assert_eq!(held(&pool), 0);
}

#[test]
fn prepared_hold_does_not_permit_fill_after_quarantine_or_run_close() {
    let h = PendingTokenInputHostPlan::<Payload>::prepare()
        .unwrap()
        .initialization_peak_bytes();
    for quarantine in [false, true] {
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let (preparation, run, _) = fresh(&pool, h);
        let prepared = completion(&preparation, &run)
            .prepare_pending_token_input::<Payload>(&run)
            .unwrap();
        let mut open_run = Some(run);
        if quarantine {
            drop(open_run.as_ref().unwrap().scope().unwrap());
        } else {
            open_run.take().unwrap().close().unwrap();
        }
        let attempts = CONSTRUCTIONS.get();
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(matches!(
            prepared.construct(payload(&pool, &drops, h)),
            Err(PendingTokenInputConstructionError::Memory(
                WorkingMemoryError::ExecutionFenced
            ))
        ));
        assert_eq!(CONSTRUCTIONS.get(), attempts);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(held(&pool), 0);
        if let Some(run) = open_run {
            run.close().unwrap();
        }
        drop(preparation);
        assert_eq!(pool.used_bytes().unwrap(), if quarantine { h } else { 0 });
    }
}
