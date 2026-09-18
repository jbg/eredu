use super::*;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    WorkspaceBound,
};
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, WorkingMemoryFundingRun, WorkingMemoryReservation,
    },
    HostSlotAttachmentError, HostSlotMetadata, HostSlotTable,
};
use std::{
    num::NonZeroU8,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

fn table() -> HostSlotTable<u32> {
    HostSlotTable::new(Box::new([3, 5, 7, 11]))
}

fn inventory(tokens: impl IntoIterator<Item = HostSlotMetadata>) -> RetainedStorage {
    let mut storage = RetainedStorage::default();
    for token in tokens {
        storage.include_slot_metadata(token).unwrap();
    }
    storage
}

fn reclaim(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.used_bytes().unwrap(), bytes);
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(cause) = error.downcast_ref::<T>() {
            return Some(cause);
        }
        error = error.source()?;
    }
}

#[test]
fn slot_inventory_deduplicates_tokens_but_preserves_independent_equal_tables() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let first = table();
    let second = table();
    assert_eq!(first.slots(), second.slots());
    let a = first.metadata().clone();
    let alias = a.clone();
    let b = second.metadata().clone();
    assert!(a.same_storage(&alias));
    assert!(!a.same_storage(&b));
    let bytes = a.capacity_bytes().unwrap();
    assert_eq!(bytes, 4 * std::mem::size_of::<u32>() as u64);
    let mut storage = inventory([a.clone(), alias.clone(), b.clone()]);
    storage
        .merge(inventory([alias.clone(), b.clone()]))
        .unwrap();
    assert_eq!(storage.slot_metadata.len(), 2);
    assert_eq!(storage.byte_bound().unwrap(), Some(bytes * 2));
    let entries = storage.storage_entries().unwrap();
    assert!(entries
        .iter()
        .all(|(key, _)| matches!(key, StorageIdentity::HostMetadata(_))));
    let registered = inventory([a.clone(), b.clone()]).register(&pool).unwrap();
    let publication = storage.publish_unquoted(&loading).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes * 2);
    drop((registered, publication, loading, alias));
    reclaim(&pool, bytes * 2);
    drop((first, a));
    reclaim(&pool, bytes);
    assert_eq!(second.slots(), [3, 5, 7, 11]);
    drop((second, b));
    reclaim(&pool, 0);
}

#[test]
fn a_slot_table_attaches_once_per_domain_and_does_not_cycle_through_publication() {
    let a = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let b = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading_a = NativeMemoryOwner::acquire(&a).unwrap();
    let loading_b = NativeMemoryOwner::acquire(&b).unwrap();
    let source = table();
    let bytes = source.metadata().capacity_bytes().unwrap();
    let key = source.metadata().identity().clone();
    let pa = inventory([source.metadata().clone()])
        .publish_unquoted(&loading_a)
        .unwrap();
    let pb = inventory([source.metadata().clone()])
        .publish_unquoted(&loading_b)
        .unwrap();
    let repeated = inventory([source.metadata().clone()])
        .publish_unquoted(&loading_a)
        .unwrap();
    assert_eq!(
        (a.used_bytes().unwrap(), b.used_bytes().unwrap()),
        (bytes, bytes)
    );
    drop((loading_a, loading_b, repeated));
    // Publication records keep neither this table nor its accounting token.
    // Actual table retirement must release both charges even while they survive.
    drop(source);
    reclaim(&a, 0);
    reclaim(&b, 0);
    // Payload-free registry keys can survive without keeping a charge alive.
    drop(key);
    drop((pa, pb));
}

#[test]
fn mutable_slots_and_immutable_layouts_share_registry_namespace_without_sharing_charges() {
    use eredu_runtime::{SharedHostMetadata, SharedStateLayout, StateLayout};

    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = table();
    let layout = SharedHostMetadata::Layout(SharedStateLayout::new(
        StateLayout::new(LayerSchedule::new(2, vec![LayerCachePolicy::NoState; 2]).unwrap())
            .unwrap(),
    ));
    assert_ne!(source.metadata().identity(), layout.identity());
    let slot_bytes = source.metadata().capacity_bytes().unwrap();
    let layout_bytes = layout.capacity_bytes().unwrap();
    assert!(layout_bytes > 0);
    let mut storage = inventory([source.metadata().clone()]);
    storage.include_metadata(layout.clone()).unwrap();
    storage.include_metadata(layout.clone()).unwrap();
    assert_eq!(storage.storage_entries().unwrap().len(), 2);
    assert_eq!(
        storage.byte_bound().unwrap(),
        Some(slot_bytes + layout_bytes)
    );
    let publication = storage.publish_unquoted(&loading).unwrap();
    drop(loading);
    reclaim(&pool, slot_bytes + layout_bytes);
    drop(source);
    reclaim(&pool, layout_bytes);
    assert_eq!(layout.capacity_bytes(), Some(layout_bytes));
    drop(layout);
    reclaim(&pool, 0);
    drop(publication);
}

#[test]
fn table_payload_retires_before_charge_and_escaped_tokens_cannot_attach_again() {
    struct Tracked {
        value: u32,
        pool: WorkingMemoryPool,
        observed: Arc<AtomicU64>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.observed
                .store(self.pool.used_bytes().unwrap(), Ordering::SeqCst);
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let other = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let loading_other = NativeMemoryOwner::acquire(&other).unwrap();
    let observed = Arc::new(AtomicU64::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    // The slot's pool/counter handles are test accounting metadata. This token
    // prices only the actual inline boxed Tracked extent, not nested owners.
    let source = HostSlotTable::new(Box::new([Tracked {
        value: 37,
        pool: pool.clone(),
        observed: observed.clone(),
        drops: drops.clone(),
    }]));
    assert_eq!(source.slots()[0].value, 37);
    let token = source.metadata().clone();
    let key = token.identity().clone();
    let bytes = token.capacity_bytes().unwrap();
    let publication = inventory([token.clone()])
        .publish_unquoted(&loading)
        .unwrap();
    drop(source);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(observed.load(Ordering::SeqCst), bytes);
    reclaim(&pool, bytes);
    for authority in [&loading, &loading_other] {
        let error = inventory([token.clone()])
            .publish_unquoted(authority)
            .unwrap_err();
        assert!(matches!(
            cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
            Some(HostSlotAttachmentError::Retired)
        ));
    }
    reclaim(&pool, bytes);
    reclaim(&other, 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop((publication, loading, loading_other, token));
    reclaim(&pool, 0);
    // Payload-free registry keys can survive without keeping a charge alive.
    drop(key);
}

#[test]
fn slot_registration_rejects_one_short_and_unknown_coverage_without_accounting_changes() {
    let source_pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let source_authority = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let source = table();
    let bytes = source.metadata().capacity_bytes().unwrap();
    let short = WorkingMemoryPool::new(bytes - 1, 0).unwrap();
    let loading_short = NativeMemoryOwner::acquire(&short).unwrap();
    let error = inventory([source.metadata().clone()])
        .publish_unquoted(&loading_short)
        .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::BudgetExceeded {
            required_bytes: bytes,
            available_bytes: bytes - 1
        })
    );
    assert_eq!(
        (short.used_bytes().unwrap(), short.peak_bytes().unwrap()),
        (0, 0)
    );
    assert_eq!(short.unquoted_owner_count().unwrap(), 1);
    let exact = WorkingMemoryPool::new(bytes, 0).unwrap();
    let loading_exact = NativeMemoryOwner::acquire(&exact).unwrap();
    let mut unknown = inventory([source.metadata().clone()]);
    unknown.mark_incomplete();
    assert_eq!(unknown.slot_metadata.len(), 1);
    assert_eq!(unknown.byte_bound().unwrap(), None);
    let error = unknown.publish_unquoted(&loading_exact).unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(
        (exact.used_bytes().unwrap(), exact.peak_bytes().unwrap()),
        (0, 0)
    );
    let publication = inventory([source.metadata().clone()])
        .publish_unquoted(&loading_exact)
        .unwrap();
    reclaim(&exact, bytes);
    assert_eq!(source.slots(), [3, 5, 7, 11]);
    drop((
        publication,
        loading_short,
        loading_exact,
        source_authority,
        source,
    ));
    reclaim(&exact, 0);
    reclaim(&short, 0);
}

fn ordered_tables() -> (HostSlotTable<u32>, HostSlotTable<u32>) {
    let first = table();
    let second = table();
    if first.metadata().identity() < second.metadata().identity() {
        (first, second)
    } else {
        (second, first)
    }
}

#[test]
fn failed_later_slot_attachment_preserves_earlier_charge_and_original_typed_error() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let (live, retired) = ordered_tables();
    let bytes = live.metadata().capacity_bytes().unwrap();
    let dead_token = retired.metadata().clone();
    drop(retired);
    let error = inventory([live.metadata().clone(), dead_token.clone()])
        .publish_unquoted(&loading)
        .unwrap_err();
    assert!(matches!(
        cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
        Some(HostSlotAttachmentError::Retired)
    ));
    reclaim(&pool, bytes);
    assert_eq!(pool.peak_bytes().unwrap(), bytes * 2);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(live.slots(), [3, 5, 7, 11]);
    drop((loading, live, dead_token));
    reclaim(&pool, 0);
}

#[test]
fn zero_length_slot_sources_keep_attachment_retirement_semantics_without_payload_charge() {
    let pool = WorkingMemoryPool::new(1024, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = HostSlotTable::<u32>::new(Vec::new().into_boxed_slice());
    let token = source.metadata().clone();
    assert_eq!(token.capacity_bytes(), Some(0));
    let publication = inventory([token.clone()])
        .publish_unquoted(&loading)
        .unwrap();
    reclaim(&pool, 0);
    drop(source);
    let error = inventory([token.clone()])
        .publish_unquoted(&loading)
        .unwrap_err();
    assert!(matches!(
        cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
        Some(HostSlotAttachmentError::Retired)
    ));
    drop((publication, token, loading));
    reclaim(&pool, 0);
}

// This host-only fixture has no native operation. Its sole payload contribution
// is the measured fixed boxed slot extent constructed under the returned scope.
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
    let zero = || WorkspaceBound::bounded(0, "host-only metadata fixture has no native operation");
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
        retained: WorkspaceBound::bounded(bytes, "exact retained boxed host slot extent"),
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

fn table_capacity() -> u64 {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = table();
    let bytes = source.metadata().capacity_bytes().unwrap();
    drop((source, loading));
    bytes
}

#[test]
fn funded_slot_publication_transfers_exact_credit_and_tokens_retain_it_after_table_retirement() {
    let bytes = table_capacity();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let (reservation, run) = funding(&pool, bytes);
    let scope = run.scope().unwrap();
    let source = table();
    let token = source.metadata().clone();
    let key = token.identity().clone();
    assert_eq!(token.capacity_bytes(), Some(bytes));
    let publication = inventory([source.metadata().clone(), token.clone()])
        .publish_funded(&scope)
        .unwrap();
    let duplicate = inventory([token.clone()]).publish_funded(&scope).unwrap();
    reclaim(&pool, bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    // All credit transferred to the actual boxed source. Repeated attachment
    // shares that charge rather than retaining a second slice of the envelope.
    assert_eq!(
        scope.adopt_storage_individually([(37_u32, 1)]).unwrap_err(),
        WorkingMemoryError::BudgetExceeded {
            required_bytes: 1,
            available_bytes: 0,
        }
    );
    scope.certify().unwrap();
    drop((reservation, run, publication, duplicate));
    reclaim(&pool, bytes);
    assert_eq!(source.slots(), [3, 5, 7, 11]);
    drop(source);
    reclaim(&pool, bytes);
    assert_eq!(token.identity(), &key);
    // This remaining token is accounting custody only: its table has retired.
    let attempts = AtomicUsize::new(0);
    let error = token
        .try_attach(pool.shared_storage_domain(), || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(()))
        })
        .unwrap_err();
    assert!(matches!(error, HostSlotAttachmentError::Retired));
    assert_eq!(attempts.load(Ordering::SeqCst), 0);
    drop(token);
    reclaim(&pool, 0);
    assert_eq!(pool.peak_bytes().unwrap(), bytes);
    drop(key);
}

#[test]
fn failed_funded_slot_publication_quarantines_the_complete_uncertified_envelope() {
    let bytes = table_capacity();
    let full = bytes * 2;
    let pool = WorkingMemoryPool::new(full, 0).unwrap();
    let (reservation, run) = funding(&pool, full);
    let scope = run.scope().unwrap();
    let (live, retired) = ordered_tables();
    assert_eq!(live.metadata().capacity_bytes(), Some(bytes));
    assert_eq!(retired.metadata().capacity_bytes(), Some(bytes));
    let retired_token = retired.metadata().clone();
    drop(retired);
    let error = inventory([live.metadata().clone(), retired_token.clone()])
        .publish_funded(&scope)
        .unwrap_err();
    assert!(matches!(
        cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
        Some(HostSlotAttachmentError::Retired)
    ));
    reclaim(&pool, full);
    assert_eq!(live.slots(), [3, 5, 7, 11]);
    // The first attachment is retained, and the failed second attachment has
    // returned its charge to the active scope. Failure grants no certification.
    drop(scope);
    drop((run, reservation, live, retired_token, error));
    reclaim(&pool, full);
    assert_eq!(pool.peak_bytes().unwrap(), full);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn ordinary_native_source_classification_preserves_pin_bytes_and_alias_lifetimes() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let first = table();
    let second = table();
    let a = first.metadata().clone();
    let b = second.metadata().clone();
    let mut storage = inventory([a.clone(), a.clone(), b.clone()]);
    storage.merge(inventory([a.clone(), b.clone()])).unwrap();
    let physical = storage.byte_bound().unwrap().unwrap();
    let (entries, original) = storage.source_entries(&pool, None).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(original.is_empty());
    assert!(
        storage.pin_registered(&pool).is_err(),
        "classification cannot register missing ordinary sources"
    );
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let registered = inventory([a.clone(), b.clone()]).register(&pool).unwrap();
    let pin = storage.pin_registered(&pool).unwrap();
    assert_eq!(pin.bytes(), physical);
    assert_eq!(registered.bytes(), physical);
    assert_eq!(pool.used_bytes().unwrap(), physical);
    assert_eq!(first.slots(), [3, 5, 7, 11]);
    assert_eq!(second.slots(), first.slots());
    drop((first, second, a, b, storage, registered));
    reclaim(&pool, physical);
    drop(pin);
    reclaim(&pool, 0);
}

#[test]
fn ordinary_source_error_retains_genuine_participant_after_callers_drop_their_lease() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let source_population = UnquotedOriginalSlotSources::prepare(&owner.unquoted_lease().unwrap());
    // The first fallible ordinary source step may fail before retaining a table.
    // Its concrete error still belongs to this actual ordinary participant.
    let error = original_source_failure(
        Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)),
        source_population,
    );
    assert!(matches!(error, Error::StorageSource(_)));
    assert!(matches!(
        cause::<WorkingMemoryError>(&error),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(!error.model_state_preserved());
    drop(owner);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(error);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}

#[test]
fn partial_native_attachment_returns_no_complete_receipt_and_preserves_loading_authority() {
    let pool = WorkingMemoryPool::new(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let array = safemlx::Array::from_slice(&[53u32, 59], &[2]);
    let bytes = array.allocation_info().unwrap().unwrap().bytes() as u64;
    let retired = table();
    let dead_token = retired.metadata().clone();
    let table_bytes = dead_token.capacity_bytes().unwrap();
    drop(retired);
    let mut storage = inventory([dead_token.clone()]);
    storage.include_array(&array).unwrap();
    // Native attachment precedes the deliberately retired host-slot source.
    // The failed complete publication can return only its typed error, never a
    // receipt authorizing omission of future attachments.
    let error = storage.publish_unquoted(&loading).unwrap_err();
    assert!(matches!(
        cause::<HostSlotAttachmentError<WorkingMemoryError>>(&error),
        Some(HostSlotAttachmentError::Retired)
    ));
    reclaim(&pool, bytes);
    assert_eq!(pool.peak_bytes().unwrap(), bytes + table_bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    assert_eq!(array.evaluated().unwrap().as_slice::<u32>(), &[53, 59]);
    drop((array, error, dead_token));
    safemlx::reclaim_allocation_owners();
    reclaim(&pool, 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    drop(loading);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
}
