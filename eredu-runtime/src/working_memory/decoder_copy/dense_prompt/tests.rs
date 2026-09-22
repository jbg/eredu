use super::*;
use crate::ConfiguredTextSampler;
use crate::working_memory::{
    InferenceRequest, InferenceTextPreparation, RegisteredWorkspaceCopy,
    RegisteredWorkspaceStorage, RunOwnedTextSampler, WorkspaceCopyLimits,
};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    SharedStorageAccountingId, SharedStorageAttachmentError, StateMemoryLayout,
    TextGenerationConfig, WorkspaceBound, cache::LayerCachePolicy,
};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    convert::Infallible,
    mem::size_of,
    num::NonZeroU8,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

const CAPACITY: u64 = 65_536;
thread_local! {
    static INITIALIZATIONS: Cell<usize> = const { Cell::new(0) };
    static FAIL_INITIALIZATION: Cell<bool> = const { Cell::new(false) };
}
pub(super) fn before_initialize() {
    INITIALIZATIONS.set(INITIALIZATIONS.get() + 1);
    assert!(
        !FAIL_INITIALIZATION.replace(false),
        "injected before dense allocation"
    );
}
fn attempts() -> usize {
    INITIALIZATIONS.get()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Host(HostMetadataKey),
    Extra(u32),
}
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        match self {
            Self::Host(id) => Some(id),
            Self::Extra(_) => None,
        }
    }
}
fn key<T>(table: &HostSlotTable<T>) -> Key {
    Key::Host(table.metadata().identity().registry_key().clone())
}
fn registered<T>(pool: &MemoryLedger, table: &HostSlotTable<T>) -> WorkingMemoryStorage<Key> {
    pool.register_host_storage([(key(table), table.metadata().capacity_bytes().unwrap())])
        .unwrap()
}
fn empty<K: Clone + Ord + Send + 'static>(pool: &MemoryLedger) -> WorkingMemoryStorage<K> {
    pool.pin_registered_storage(std::iter::empty::<(K, u64)>())
        .unwrap()
}
fn plan<'a, S, D>(
    pool: &MemoryLedger,
    table: &'a HostSlotTable<S>,
) -> RegisteredDenseDecoderInitialization<'a, S, D, Key> {
    RegisteredDecoderHostCopy::bind(pool, table.prepare_copy_slots().unwrap(), key(table))
        .unwrap()
        .for_dense_destination()
        .unwrap()
}
fn usage(pool: &MemoryLedger) -> (u64, u64, u64, usize, usize, u64) {
    let usage = pool.0.usage.lock().unwrap();
    (
        usage.reserved,
        usage.registered - usage.registry_metadata,
        usage.peak,
        usage.funding.values().map(|s| s.scopes).sum(),
        usage.funding.len(),
        usage
            .funding
            .values()
            .map(|s| s.host_held.checked_sub(s.control_floor).unwrap())
            .sum(),
    )
}
fn held(pool: &MemoryLedger) -> u64 {
    usage(pool).5
}
fn total(pool: &MemoryLedger) -> u64 {
    pool.payload_used_bytes().unwrap()
}

// A real fresh request whose only allocated destination here is the closed
// inline host table (and an empty canonical sampler when requested). No native
// model, tensor workspace or complete resumed continuation is asserted.
fn fresh(
    pool: &MemoryLedger,
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
    let mut state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            physical_domains: Some(crate::working_memory::memory_fixture::host_workspace(
                pool, geometry, bytes,
            )),
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    state.physical_domains = Some(crate::working_memory::memory_fixture::empty_state(
        pool, geometry,
    ));
    let reservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1,
                state,
                incremental_required_bytes: Some(bytes),
                additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
                memory_limits: eredu_core::MemoryLimitDeclarations::unlimited(),
            },
            crate::working_memory::memory_fixture::resolved_host_limits(pool, CAPACITY),
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
        request
            .prepare_text(&execution, geometry, config.clone())
            .unwrap(),
        run,
        config,
    )
}
fn fresh_dense<S, D, K: HostSlotStorageKey>(
    pool: &MemoryLedger,
    bytes: u64,
    source: &RegisteredDenseDecoderInitialization<'_, S, D, K>,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    fresh(
        pool,
        bytes
            .checked_add(source.ordinary_preparation_control_bytes().unwrap())
            .unwrap(),
    )
}
fn publication_controls() -> u64 {
    crate::working_memory::storage::ordinary_dense_preparation_bytes::<u32, u64, Key>().unwrap()
}
fn sampler(
    preparation: &InferenceTextPreparation,
    run: &WorkingMemoryFundingRun,
    config: TextGenerationConfig,
) -> RunOwnedTextSampler {
    let (sampler, complete) = preparation
        .claim_sampling(config.clone())
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    complete.finish().unwrap();
    sampler
}
fn assert_claim_consumed(preparation: &InferenceTextPreparation) {
    assert!(matches!(
        preparation.claim_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    assert!(matches!(
        preparation.bind_prompt(),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
}
fn handoff_memory(error: &HostSlotAttachmentError<WorkingMemoryError>) -> &WorkingMemoryError {
    match error {
        HostSlotAttachmentError::Attachment(SharedStorageAttachmentError::Provider(error)) => error,
        other => panic!("expected typed provider cause, got {other:?}"),
    }
}

#[test]
fn exact_prompt_hold_and_transfer_preserve_account_pointer_and_ready_order() {
    for short in [true, false] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([11_u32, 23, 47]));
        let charge = registered(&pool, &source);
        let plan = plan::<_, u64>(&pool, &source);
        let (d, p) = (plan.retained_bytes(), plan.initialization_peak_bytes());
        assert_eq!((d, p), (24, 40));
        let h = size_of::<ConfiguredTextSampler>() as u64;
        let (preparation, run, config) = fresh_dense(&pool, h + p - u64::from(short), &plan);
        let sampler = sampler(&preparation, &run, config);
        let complete = empty::<Key>(&pool);
        let before = (usage(&pool), attempts());
        let result = preparation.claim_prompt().unwrap().construct_dense_decoder(
            plan,
            &run,
            complete.clone(),
        );
        if short {
            assert!(matches!(result,Err(DecoderCopyAdmissionError::Memory(
                WorkingMemoryError::MetadataConstruction(eredu_nn::workspace::WorkspaceMetadataError::Funding(
                    eredu_core::HostMetadataFundingError::DomainAllowance { required, available, .. }))))
                if required == available + 1));
            assert_eq!((usage(&pool), attempts()), before);
            assert_claim_consumed(&preparation);
            drop((sampler, preparation, run, charge, source));
            assert_eq!(total(&pool), 0);
            continue;
        }
        let (mut slots, native) = result.unwrap();
        assert_eq!(attempts(), before.1 + 1);
        assert_eq!(held(&pool), h + p);
        assert_eq!(usage(&pool).3, before.0.3 + 2);
        assert_eq!(
            usage(&pool).4,
            before.0.4,
            "bootstrap creates no new account"
        );
        assert_claim_consumed(&preparation);
        assert!(matches!(
            native.publish_host_storage_fixture([(Key::Extra(91), 1)]),
            Err(WorkingMemoryError::DomainAllowanceExceeded {
                required_bytes: 1,
                available_bytes: 0,
                ..
            })
        ));
        for index in 0..slots.len() {
            let value = *slots.source_at(index).unwrap() as u64;
            slots.push(value).unwrap();
        }
        let completed = slots.finish().unwrap();
        let address = completed.get(0).unwrap() as *const u64;
        let alias = completed.metadata().clone();
        let destination_key = Key::Host(alias.identity().registry_key().clone());
        let before_publish = usage(&pool);
        assert_claim_consumed(&preparation);
        let (table, completion) = completed.publish(destination_key.clone()).unwrap();
        let after = usage(&pool);
        assert_eq!(
            (after.0, after.1, after.2),
            (before_publish.0 - d, before_publish.1 + d, before_publish.2)
        );
        assert_eq!(held(&pool), h, "only the new table's host hold retired");
        assert_eq!(table.slots().as_ptr(), address);
        assert!(table.metadata().same_storage(&alias));
        assert_eq!(table.slots(), &[11_u64, 23, 47]);
        assert_eq!(source.slots(), &[11_u32, 23, 47]);
        assert_claim_consumed(&preparation);
        completion.finish().unwrap(); // CONSTRUCTED; prefill still cannot run.
        let request = preparation.request();
        let execution = &request.memory_reservation().0.execution;
        assert!(matches!(
            request.begin_prefill(execution, request.geometry()),
            Err(WorkingMemoryError::PreparationNotReady)
        ));
        preparation.bind_prompt().unwrap(); // Existing binding makes READY.
        request
            .begin_prefill(execution, request.geometry())
            .unwrap();
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        let ordinary_publisher = crate::working_memory::StoragePublicationLayout::<Key>::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap();
        let pin_publisher = crate::working_memory::StoragePublicationLayout::<Key>::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap();
        let duplicate_publisher = crate::working_memory::StoragePublicationLayout::<Key>::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap();
        let stable = usage(&pool);
        let ordinary = ordinary_publisher
            .register_storage([(
                destination_key.clone(),
                crate::working_memory::StorageAllocation::new(d, pool.host_placement_handle()),
            )])
            .unwrap();
        let pin = pin_publisher
            .pin_registered_storage([(destination_key.clone(), d)])
            .unwrap();
        assert_eq!(
            (usage(&pool).0, usage(&pool).1, usage(&pool).2),
            (stable.0, stable.1, stable.2)
        );
        drop((ordinary, pin));
        let duplicate = duplicate_publisher
            .adopt_storage_individually(
                &native,
                [(
                    destination_key,
                    crate::working_memory::StorageAllocation::new(d, pool.host_placement_handle()),
                )],
            )
            .unwrap();
        assert_eq!(
            (usage(&pool).0, usage(&pool).1, usage(&pool).2),
            (stable.0, stable.1, stable.2)
        );
        drop(duplicate);
        native.certify().unwrap();
        drop((sampler, preparation, run, charge, source));
        assert_eq!(total(&pool), d);
        drop(table);
        assert_eq!(
            total(&pool),
            d,
            "preexisting metadata alias retains the attached charge"
        );
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        drop(alias);
        assert_eq!(total(&pool), 0);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn wrong_stage_run_pool_and_source_inventory_reject_before_initializer() {
    for case in 0..4 {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let other = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let charge = registered(&pool, &source);
        let plan = plan::<_, u64>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, config) = fresh_dense(&pool, p, &plan);
        let (other_preparation, other_run, _) = fresh(if case == 2 { &other } else { &pool }, p);
        let inventory = if case == 3 {
            empty::<Key>(&other)
        } else {
            empty::<Key>(&pool)
        };
        let stage = if case == 0 {
            preparation.claim_sampling(config.clone()).unwrap()
        } else {
            preparation.claim_prompt().unwrap()
        };
        let before = (usage(&pool), usage(&other), attempts());
        let error = stage
            .construct_dense_decoder(
                plan,
                if case == 1 || case == 2 {
                    &other_run
                } else {
                    &run
                },
                inventory,
            )
            .unwrap_err();
        assert!(matches!(error,DecoderCopyAdmissionError::Memory(error)
            if error==if case==0 {WorkingMemoryError::InvocationPhaseMismatch} else {WorkingMemoryError::IdentityMismatch}));
        assert_eq!((usage(&pool), usage(&other), attempts()), before);
        if case != 0 {
            assert_claim_consumed(&preparation);
        } else {
            assert!(matches!(
                preparation.claim_sampling(config.clone()),
                Err(WorkingMemoryError::PreparationAlreadyStarted)
            ));
        }
        drop((
            preparation,
            run,
            other_preparation,
            other_run,
            charge,
            source,
        ));
        assert_eq!((total(&pool), total(&other)), (0, 0));
    }
}

#[test]
fn registered_source_requires_exact_table_and_quarantine_is_rechecked_after_plan() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let other = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let source = HostSlotTable::new(Box::new([3_u32, 7]));
    let replacement = HostSlotTable::new(Box::new([3_u32, 7]));
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(&pool, source.prepare_copy_slots().unwrap(), key(&source)),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let (origin_preparation, origin, _) = fresh(&pool, 512);
    let origin_scope = origin.scope().unwrap();
    let charge = origin_scope
        .publish_host_storage_fixture([(key(&source), 8)])
        .unwrap();
    origin_scope.certify().unwrap();
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(
            &pool,
            source.prepare_copy_slots().unwrap(),
            key(&replacement)
        ),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert!(matches!(
        RegisteredDecoderHostCopy::bind(&other, source.prepare_copy_slots().unwrap(), key(&source)),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let plan = plan::<_, u64>(&pool, &source);
    let (preparation, run, _) = fresh_dense(&pool, plan.initialization_peak_bytes(), &plan);
    drop(origin.scope().unwrap());
    let complete = empty::<Key>(&pool);
    let before = (usage(&pool), attempts());
    assert!(matches!(
        preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, complete),
        Err(DecoderCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!((usage(&pool), attempts()), before);
    assert_claim_consumed(&preparation);
    drop((
        preparation,
        run,
        origin_preparation,
        origin,
        charge,
        source,
        replacement,
    ));
    assert_eq!(total(&pool), 512);
}

#[test]
fn partial_recovery_keeps_source_claim_capacity_and_payload_before_hold() {
    struct Value {
        number: u32,
        pool: MemoryLedger,
        hold: u64,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            assert!(
                held(&self.pool) >= self.hold,
                "payload must precede its host custody"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    for recover in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let replacement = HostSlotTable::new(Box::new([3_u32, 7]));
        let charge = registered(&pool, &source);
        let plan = plan::<_, Value>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<Key>(&pool))
            .unwrap();
        let original = source
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination::<Value>()
            .unwrap();
        let substitute = replacement
            .prepare_copy_slots()
            .unwrap()
            .for_dense_destination::<Value>()
            .unwrap();
        assert_eq!(
            slots.validate_source(&substitute),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        slots.validate_source(&original).unwrap();
        assert!(std::ptr::eq(
            slots.source_at(1).unwrap(),
            &source.slots()[1]
        ));
        let drops = Arc::new(AtomicUsize::new(0));
        slots
            .push(Value {
                number: 17,
                pool: pool.clone(),
                hold: p,
                drops: drops.clone(),
            })
            .unwrap();
        assert_eq!(
            slots.validate_source(&original),
            Err(WorkingMemoryError::IdentityMismatch)
        );
        let error = slots.finish().unwrap_err();
        assert_eq!((error.expected(), error.initialized()), (2, 1));
        assert_eq!(held(&pool), p);
        assert_claim_consumed(&preparation);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        if recover {
            let mut slots = error.into_builder();
            assert_eq!((slots.len(), slots.initialized_count()), (2, 1));
            slots
                .push(Value {
                    number: 19,
                    pool: pool.clone(),
                    hold: p,
                    drops: drops.clone(),
                })
                .unwrap();
            let extra = slots
                .push(Value {
                    number: 23,
                    pool: pool.clone(),
                    hold: 0,
                    drops: drops.clone(),
                })
                .unwrap_err();
            assert_eq!(extra.capacity(), 2);
            assert_eq!(extra.into_value().number, 23);
            let completed = slots.finish().unwrap();
            assert_eq!(completed.get(0).unwrap().number, 17);
            assert_eq!(completed.get(1).unwrap().number, 19);
            let error = completed.publish(Key::Extra(9)).unwrap_err();
            assert_eq!(
                handoff_memory(error.error()),
                &WorkingMemoryError::IdentityMismatch
            );
            assert_eq!(held(&pool), p);
            drop(error);
            assert_eq!(drops.load(Ordering::SeqCst), 3);
        } else {
            drop(error);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
        assert_eq!(held(&pool), 0);
        assert_claim_consumed(&preparation);
        native.certify().unwrap();
        drop((preparation, run, charge, source, replacement));
        assert_eq!(total(&pool), 0);
    }
}

#[test]
fn failed_handoff_preserves_owner_and_hold_for_exact_recovery_or_retirement() {
    for case in 0..4 {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let charge = registered(&pool, &source);
        let plan = plan::<_, u64>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p + 64, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<Key>(&pool))
            .unwrap();
        slots.push(31).unwrap();
        slots.push(37).unwrap();
        let completed = slots.finish().unwrap();
        let address = completed.get(0).unwrap() as *const u64;
        let alias = completed.metadata().clone();
        let target = Key::Host(alias.identity().registry_key().clone());
        let mut existing = None;
        match case {
            1 => {
                alias
                    .try_attach(pool.shared_storage_accounting_id(), || {
                        Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
                    })
                    .unwrap();
            }
            2 => {
                existing = Some(pool.register_host_storage([(target.clone(), 16)]).unwrap());
            }
            3 => {
                let failed = catch_unwind(AssertUnwindSafe(|| {
                    let _ = alias
                        .try_attach::<Infallible>(&SharedStorageAccountingId::default(), || {
                            panic!("poison destination custody")
                        });
                }));
                assert!(failed.is_err());
            }
            _ => {}
        }
        let before = (usage(&pool), attempts());
        let error = completed
            .publish(if case == 0 {
                Key::Extra(9)
            } else {
                target.clone()
            })
            .unwrap_err();
        if case == 3 {
            assert_eq!(handoff_memory(error.error()), &WorkingMemoryError::Poisoned);
        } else {
            assert_eq!(
                handoff_memory(error.error()),
                &WorkingMemoryError::IdentityMismatch
            );
        }
        assert_eq!((usage(&pool), attempts()), before);
        assert_claim_consumed(&preparation);
        let (completed, cause) = error.into_parts();
        drop(cause);
        assert_eq!(completed.get(0).unwrap() as *const u64, address);
        assert_eq!((completed.get(0), completed.get(1)), (Some(&31), Some(&37)));
        assert_eq!(held(&pool), p);
        if case == 0 || case == 2 {
            drop(existing.take());
            let (table, completion) = completed.publish(target).unwrap();
            assert_eq!(table.slots().as_ptr(), address);
            completion.finish().unwrap();
            preparation.bind_prompt().unwrap();
            native.certify().unwrap();
            drop((preparation, run, charge, source, table));
            assert_eq!(total(&pool), 16);
            drop(alias);
        } else {
            drop(completed);
            assert_eq!(held(&pool), 0);
            native.certify().unwrap();
            drop((preparation, run, charge, source, alias, existing));
        }
        assert_eq!(total(&pool), 0);
    }
}

#[test]
fn native_abandonment_remains_independent_before_and_after_host_publication() {
    for publish in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let source_key = key(&source);
        let charge = registered(&pool, &source);
        let plan = plan::<_, u64>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<Key>(&pool))
            .unwrap();
        slots.push(13).unwrap();
        slots.push(17).unwrap();
        let completed = slots.finish().unwrap();
        let mut destination = None;
        if publish {
            let target = Key::Host(completed.metadata().identity().registry_key().clone());
            let (table, completion) = completed.publish(target).unwrap();
            drop(completion); // Publication does not finish the full prompt.
            destination = Some(table);
        } else {
            drop(completed);
        }
        assert_eq!(held(&pool), 0);
        assert_claim_consumed(&preparation);
        drop(native); // A host-only retirement cannot certify this scope.
        drop((preparation, run, charge, source));
        assert_eq!(
            total(&pool),
            p + 8 + if publish { 0 } else { publication_controls() }
        );
        let pin = pool.pin_registered_storage([(source_key, 8)]).unwrap();
        drop(pin);
        drop(destination);
        assert_eq!(
            total(&pool),
            p + 8 + publication_controls(),
            "retired payload and constructor credit return to quarantine"
        );
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
    }
}

#[test]
fn panic_after_scope_commit_retains_full_source_pin_without_host_hold_leak() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let source = HostSlotTable::new(Box::new([3_u32, 7]));
    let source_key = key(&source);
    let charge = registered(&pool, &source);
    let plan = plan::<_, u64>(&pool, &source);
    let p = plan.initialization_peak_bytes();
    let (preparation, run, _) = fresh_dense(&pool, p, &plan);
    let before = attempts();
    FAIL_INITIALIZATION.set(true);
    let failed = catch_unwind(AssertUnwindSafe(|| {
        let _ = preparation.claim_prompt().unwrap().construct_dense_decoder(
            plan,
            &run,
            empty::<Key>(&pool),
        );
    }));
    assert!(failed.is_err());
    assert_eq!(attempts(), before + 1);
    assert_eq!(held(&pool), 0);
    assert_claim_consumed(&preparation);
    drop((preparation, run, charge, source));
    assert_eq!(total(&pool), p + 8);
    drop(pool.pin_registered_storage([(source_key, 8)]).unwrap());
}

#[test]
fn zero_and_zst_tables_keep_logical_scope_and_registration_origin_lifetimes() {
    for count in [0_usize, 3] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(vec![(); count].into_boxed_slice());
        let charge = registered(&pool, &source);
        let plan = plan::<_, ()>(&pool, &source);
        assert_eq!(
            (plan.retained_bytes(), plan.initialization_peak_bytes()),
            (0, 0)
        );
        let (preparation, run, _) = fresh_dense(&pool, 0, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<Key>(&pool))
            .unwrap();
        for _ in 0..count {
            slots.push(()).unwrap();
        }
        let completed = slots.finish().unwrap();
        let alias = completed.metadata().clone();
        let target = Key::Host(alias.identity().registry_key().clone());
        let (table, completion) = completed.publish(target).unwrap();
        assert_eq!(
            (table.len(), table.metadata().capacity_bytes()),
            (count, Some(0))
        );
        completion.finish().unwrap();
        preparation.bind_prompt().unwrap();
        native.certify().unwrap();
        drop((preparation, run, charge, source, table));
        assert_eq!(total(&pool), 0);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        drop(alias);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

fn no_native_copy(pool: &MemoryLedger) -> RegisteredWorkspaceCopy<Key> {
    use eredu_nn::workspace::{
        WorkspaceContext, WorkspaceExistingStorage, WorkspaceIsolatedCopyPlan, WorkspaceMechanisms,
        WorkspaceOperation, WorkspaceOperationBound,
    };
    #[derive(Debug)]
    struct NoNative(MemoryLedger);
    impl WorkspaceMechanisms for NoNative {
        fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
            Some(self.0.topology())
        }
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
            panic!("scalar source fixture has no native operations")
        }
    }
    let context = WorkspaceContext::new(NoNative(pool.clone()));
    let source = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        std::iter::empty::<(Key, WorkspaceExistingStorage)>(),
    )
    .unwrap();
    let plan =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &[]).unwrap();
    assert_eq!(plan.incremental_bytes(), Some(0));
    RegisteredWorkspaceCopy::bind(plan, source).unwrap()
}

#[test]
fn actual_funded_saved_source_survives_original_retirement_and_rechecks_health() {
    for quarantine in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let charge = registered(&pool, &source);
        let (old_preparation, old_run, config) = fresh(&pool, 1024);
        let old_sampler = sampler(&old_preparation, &old_run, config);
        let decoder = RegisteredDecoderHostCopy::bind(
            &pool,
            source.prepare_copy_slots().unwrap(),
            key(&source),
        )
        .unwrap();
        let joint =
            RegisteredSamplingCopy::prepare(old_sampler.borrow_funded(), no_native_copy(&pool))
                .unwrap()
                .with_decoder_slots(decoder, empty::<Key>(&pool))
                .unwrap();
        let (copied_sampler, mut slots, native) = pool
            .copy_text_components(
                joint,
                WorkspaceCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                    CAPACITY,
                )),
            )
            .unwrap();
        slots.push(11).unwrap();
        slots.push(23).unwrap();
        let saved = slots.finish().unwrap();
        let old_account = copied_sampler.bytes() + saved.protected_bytes();
        let (custody, scope) = native.into_parts();
        drop((
            copied_sampler,
            custody,
            old_sampler,
            old_preparation,
            old_run,
            charge,
            source,
        ));
        assert_eq!(saved.iter().copied().collect::<Vec<_>>(), [11, 23]);
        let plan = saved
            .prepare_copy::<Key>()
            .unwrap()
            .for_dense_destination::<u64>()
            .unwrap();
        assert_eq!(plan.source_at(0), Some(&11));
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        if quarantine {
            drop(scope);
        } else {
            scope.certify().unwrap();
        }
        let complete = empty::<Key>(&pool);
        let before = (usage(&pool), attempts());
        let result = preparation.claim_prompt().unwrap().construct_dense_decoder(
            plan,
            &run,
            complete.clone(),
        );
        if quarantine {
            assert!(matches!(
                result,
                Err(DecoderCopyAdmissionError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert_eq!((usage(&pool), attempts()), before);
            assert_claim_consumed(&preparation);
            drop((saved, preparation, run));
            assert_eq!(total(&pool), old_account + 8);
        } else {
            let (mut slots, native) = result.unwrap();
            assert_eq!(attempts(), before.1 + 1);
            for i in 0..slots.len() {
                let value = *slots.source_at(i).unwrap() as u64;
                slots.push(value).unwrap();
            }
            let finished = slots.finish().unwrap();
            let key = Key::Host(finished.metadata().identity().registry_key().clone());
            let (table, completion) = finished.publish(key).unwrap();
            drop(saved);
            completion.finish().unwrap();
            preparation.bind_prompt().unwrap();
            native.certify().unwrap();
            drop((preparation, run));
            assert_eq!(total(&pool), 16);
            assert_eq!(table.slots(), &[11_u64, 23]);
            drop(table);
            assert_eq!(total(&pool), 0);
        }
    }
}

#[test]
fn target_health_and_transfer_counter_failure_leave_full_hold_and_owned_table() {
    for quarantine in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let charge = registered(&pool, &source);
        let plan = plan::<_, u64>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<Key>(&pool))
            .unwrap();
        slots.push(29).unwrap();
        slots.push(31).unwrap();
        let finished = slots.finish().unwrap();
        let id = preparation
            .request()
            .memory_reservation()
            .0
            .funding
            .unwrap();
        let original_allocations = pool
            .0
            .usage
            .lock()
            .unwrap()
            .funding
            .get(&id)
            .unwrap()
            .allocations;
        if quarantine {
            drop(run.scope().unwrap());
        } else {
            pool.0
                .usage
                .lock()
                .unwrap()
                .funding
                .get_mut(&id)
                .unwrap()
                .allocations = usize::MAX;
        }
        let before = (usage(&pool), attempts());
        let target = Key::Host(finished.metadata().identity().registry_key().clone());
        let error = finished.publish(target.clone()).unwrap_err();
        assert_eq!(
            handoff_memory(error.error()),
            &if quarantine {
                WorkingMemoryError::ExecutionFenced
            } else {
                WorkingMemoryError::Overflow
            }
        );
        assert_eq!((usage(&pool), attempts()), before);
        assert_eq!(held(&pool), p);
        assert_claim_consumed(&preparation);
        let (finished, cause) = error.into_parts();
        drop(cause);
        assert_eq!(finished.get(0), Some(&29));
        if !quarantine {
            pool.0
                .usage
                .lock()
                .unwrap()
                .funding
                .get_mut(&id)
                .unwrap()
                .allocations = original_allocations;
            let (table, completion) = finished.publish(target).unwrap();
            completion.finish().unwrap();
            preparation.bind_prompt().unwrap();
            native.certify().unwrap();
            drop((preparation, run, charge, source, table));
            assert_eq!(total(&pool), 0);
        } else {
            drop(finished);
            native.certify().unwrap();
            drop((preparation, run, charge, source));
            assert_eq!(
                total(&pool),
                p + publication_controls(),
                "host cleanup and explicit native finish never clear quarantine"
            );
        }
    }
}

#[derive(Clone)]
struct LockProbe {
    pool: MemoryLedger,
    token: HostSlotMetadata,
    domain: SharedStorageAccountingId,
    clones: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    failures: Arc<AtomicUsize>,
}
thread_local! {
    static PROBES: RefCell<BTreeMap<u32,LockProbe>> = const { RefCell::new(BTreeMap::new()) };
    static PANIC_ORD: Cell<Option<usize>> = const { Cell::new(None) };
}
fn probe_locks(id: u32, cloning: bool) {
    let probe = PROBES.with(|probes| probes.borrow().get(&id).cloned());
    let Some(probe) = probe else { return };
    if cloning {
        probe.clones.fetch_add(1, Ordering::SeqCst);
    } else {
        probe.drops.fetch_add(1, Ordering::SeqCst);
    }
    // A blocked worker times out here, making an under-lock regression fail
    // rather than deadlocking this test thread inside reentrant attachment.
    let failures = probe.failures.clone();
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let unlocked = !matches!(
            probe.pool.0.usage.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        );
        let attached = probe.token.try_attach(&probe.domain, || {
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
        });
        let acceptable = matches!(
            attached,
            Ok(_)
                | Err(HostSlotAttachmentError::Retired)
                | Err(HostSlotAttachmentError::Attachment(
                    SharedStorageAttachmentError::Poisoned
                ))
        );
        let _ = send.send(unlocked && acceptable);
    });
    if receive.recv_timeout(std::time::Duration::from_secs(2)) == Ok(true) {
        worker.join().unwrap();
    } else {
        // Do not double-panic inside a key destructor during the deliberate
        // comparison unwind. Record failure; unwinding releases blocked locks.
        failures.fetch_add(1, Ordering::SeqCst);
        drop(worker);
    }
}
#[derive(Debug)]
struct ProbedKey {
    identity: HostMetadataKey,
    probe: u32,
}
impl Clone for ProbedKey {
    fn clone(&self) -> Self {
        probe_locks(self.probe, true);
        Self {
            identity: self.identity.clone(),
            probe: self.probe,
        }
    }
}
impl Drop for ProbedKey {
    fn drop(&mut self) {
        probe_locks(self.probe, false);
    }
}
impl PartialEq for ProbedKey {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}
impl Eq for ProbedKey {}
impl PartialOrd for ProbedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ProbedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if let Some(remaining) = PANIC_ORD.get() {
            if remaining == 1 {
                PANIC_ORD.set(None);
                panic!("injected key comparison");
            }
            PANIC_ORD.set(Some(remaining - 1));
        }
        self.identity.cmp(&other.identity)
    }
}
impl HostSlotStorageKey for ProbedKey {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.identity)
    }
}

#[test]
fn transfer_key_clone_and_rejection_drop_run_outside_both_table_and_pool_locks() {
    for duplicate in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let source_key = ProbedKey {
            identity: source.metadata().identity().registry_key().clone(),
            probe: 0,
        };
        let charge = pool
            .register_host_storage([(source_key.clone(), 8)])
            .unwrap();
        let plan = RegisteredDecoderHostCopy::bind(
            &pool,
            source.prepare_copy_slots().unwrap(),
            source_key,
        )
        .unwrap()
        .for_dense_destination::<u64>()
        .unwrap();
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        let (mut slots, native) = preparation
            .claim_prompt()
            .unwrap()
            .construct_dense_decoder(plan, &run, empty::<ProbedKey>(&pool))
            .unwrap();
        slots.push(37).unwrap();
        slots.push(41).unwrap();
        let complete = slots.finish().unwrap();
        let alias = complete.metadata().clone();
        let clones = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let failures = Arc::new(AtomicUsize::new(0));
        if duplicate {
            alias
                .try_attach(pool.shared_storage_accounting_id(), || {
                    Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(()))
                })
                .unwrap();
        }
        PROBES.with(|probes| {
            probes.borrow_mut().insert(
                1,
                LockProbe {
                    pool: pool.clone(),
                    token: alias.clone(),
                    domain: SharedStorageAccountingId::default(),
                    clones: clones.clone(),
                    drops: drops.clone(),
                    failures: failures.clone(),
                },
            )
        });
        let target = ProbedKey {
            identity: alias.identity().registry_key().clone(),
            probe: 1,
        };
        let result = complete.publish(target);
        assert!(clones.load(Ordering::SeqCst) > 0);
        if duplicate {
            let error = result.unwrap_err();
            assert_eq!(
                handoff_memory(error.error()),
                &WorkingMemoryError::IdentityMismatch
            );
            assert!(
                drops.load(Ordering::SeqCst) >= 2,
                "both staged keys retire after skipped provider"
            );
            assert_eq!(held(&pool), p);
            drop(error);
        } else {
            let (table, completion) = result.unwrap();
            drop(completion);
            drop(table);
        }
        native.certify().unwrap();
        drop((preparation, run, charge, source));
        // Keys retain only payload-free identities. Probe target custody is
        // external test state and is removed before final alias retirement.
        let removed = PROBES.with(|probes| probes.borrow_mut().remove(&1));
        drop(removed);
        drop(alias);
        assert_eq!(
            failures.load(Ordering::SeqCst),
            0,
            "all key callbacks must run outside both locks"
        );
        assert_eq!(total(&pool), 0);
    }
}

#[test]
fn comparison_panic_does_not_commit_transfer_or_destroy_keys_under_locks() {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let source = HostSlotTable::new(Box::new([3_u32, 7]));
    let source_key = ProbedKey {
        identity: source.metadata().identity().registry_key().clone(),
        probe: 0,
    };
    let charge = pool
        .register_host_storage([(source_key.clone(), 8)])
        .unwrap();
    let plan =
        RegisteredDecoderHostCopy::bind(&pool, source.prepare_copy_slots().unwrap(), source_key)
            .unwrap()
            .for_dense_destination::<u64>()
            .unwrap();
    let p = plan.initialization_peak_bytes();
    let (preparation, run, _) = fresh_dense(&pool, p, &plan);
    let (mut slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(plan, &run, empty::<ProbedKey>(&pool))
        .unwrap();
    slots.push(43).unwrap();
    slots.push(47).unwrap();
    let completed = slots.finish().unwrap();
    let alias = completed.metadata().clone();
    let clones = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    PROBES.with(|probes| {
        probes.borrow_mut().insert(
            2,
            LockProbe {
                pool: pool.clone(),
                token: alias.clone(),
                domain: SharedStorageAccountingId::default(),
                clones: clones.clone(),
                drops: drops.clone(),
                failures: failures.clone(),
            },
        )
    });
    let target = ProbedKey {
        identity: alias.identity().registry_key().clone(),
        probe: 2,
    };
    let before = usage(&pool);
    let original_allocations = pool
        .0
        .usage
        .lock()
        .unwrap()
        .funding
        .values()
        .next()
        .unwrap()
        .allocations;
    assert_eq!(
        original_allocations, 1,
        "ordinary constructor grant is already retained"
    );
    // Provider identity comparisons precede every published accounting update.
    PANIC_ORD.set(Some(1));
    let failed = catch_unwind(AssertUnwindSafe(|| {
        let _ = completed.publish(target);
    }));
    assert!(failed.is_err());
    assert_eq!(
        PANIC_ORD.get(),
        None,
        "the intended identity comparison was reached"
    );
    {
        let usage = pool
            .0
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        assert_eq!(
            (
                usage.reserved,
                usage.registered - usage.registry_metadata,
                usage.peak
            ),
            (before.0, before.1, before.2)
        );
        let state = usage.funding.values().next().unwrap();
        assert_eq!(
            (state.allocations, state.registrations),
            (original_allocations, 0)
        );
        assert_eq!(
            (state.host_held - state.control_floor, state.scopes),
            (0, 1),
            "unwind retires only the actual host owner"
        );
    }
    assert!(clones.load(Ordering::SeqCst) > 0);
    assert!(drops.load(Ordering::SeqCst) >= 2);
    assert_eq!(failures.load(Ordering::SeqCst), 0);
    assert_claim_consumed(&preparation);
    drop(native);
    drop((preparation, run, charge, source));
    let removed = PROBES.with(|probes| probes.borrow_mut().remove(&2));
    drop(removed);
    drop(alias);
    let usage = pool
        .0
        .usage
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let controls = usage.funding.control_bytes().unwrap();
    assert_eq!(
        usage.reserved + usage.registered - usage.registry_metadata - controls,
        p + 8 + crate::working_memory::storage::ordinary_dense_preparation_bytes::<u32,u64,ProbedKey>().unwrap()
    );
    assert_eq!(failures.load(Ordering::SeqCst), 0);
}

#[test]
fn nonempty_complete_source_checks_origin_health_and_survives_native_abandonment() {
    for quarantine_origin in [true, false] {
        let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
        let source = HostSlotTable::new(Box::new([3_u32, 7]));
        let source_key = key(&source);
        let charge = registered(&pool, &source);
        // This real disjoint source payload is not a decoder slot/copy operand.
        // Only complete_source supplies its key to the new operation.
        let (origin_preparation, origin_run, _) = fresh(&pool, 512);
        let origin_scope = origin_run.scope().unwrap();
        let uncopied = Box::new([11_u32, 17, 23, 31]);
        let uncopied_key = Key::Extra(73);
        let mut inventory = origin_scope
            .publish_host_storage_fixture([(uncopied_key.clone(), 16)])
            .unwrap();
        let complete_source = inventory.remove(&uncopied_key).unwrap();
        drop(inventory);
        origin_scope.certify().unwrap();
        let plan = plan::<_, u64>(&pool, &source);
        let p = plan.initialization_peak_bytes();
        let (preparation, run, _) = fresh_dense(&pool, p, &plan);
        if quarantine_origin {
            drop(origin_run.scope().unwrap());
        }
        let before = (usage(&pool), attempts());
        let result = preparation.claim_prompt().unwrap().construct_dense_decoder(
            plan,
            &run,
            complete_source.clone(),
        );
        if quarantine_origin {
            assert!(matches!(
                result,
                Err(DecoderCopyAdmissionError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert_eq!((usage(&pool), attempts()), before);
            assert_claim_consumed(&preparation);
            drop((
                preparation,
                run,
                complete_source,
                origin_preparation,
                origin_run,
                charge,
                source,
                uncopied,
            ));
            assert_eq!(total(&pool), 512);
        } else {
            let (mut slots, native) = result.unwrap();
            assert_eq!(attempts(), before.1 + 1);
            assert_eq!(slots.source_at(1), Some(&7));
            slots.push(41).unwrap();
            let partial = slots.finish().unwrap_err();
            drop((complete_source, origin_preparation, origin_run));
            drop(partial); // Both actual source plan and destination host retire.
            assert_eq!(held(&pool), 0);
            drop(native); // Must quarantine the full complete_source bundle.
            drop((preparation, run, charge, source));
            assert_eq!(total(&pool), p + 8 + 16 + publication_controls());
            let pin = pool
                .pin_registered_storage([(source_key, 8), (uncopied_key, 16)])
                .unwrap();
            assert_eq!(&*uncopied, &[11_u32, 17, 23, 31]);
            drop((pin, uncopied));
            assert_eq!(total(&pool), p + 8 + 16 + publication_controls());
            assert!(matches!(
                pool.acquire_unquoted(),
                Err(WorkingMemoryError::ReservedWorkActive)
            ));
        }
    }
}
