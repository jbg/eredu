use super::*;
use eredu_core::checkpoint::TensorDtype;
use eredu_core::{
    cache::LayerCachePolicy, Admission, AttentionPolicy, EstimationCompleteness,
    ExecutionWorkspaceEstimate, InferenceGeometry, InputModality, InputPartDescriptor,
    InputPayloadKind, InputTensorIdentity, InputTokenCount, LayerSchedule, OutputDemand,
    PreparedInputIdentity, StateMemoryLayout, WorkspaceBound,
};
use eredu_runtime::{
    working_memory::{
        InferenceExecutionIdentity, WorkingMemoryFundingRun, WorkingMemoryReservation,
    },
    PreparedInputCacheIdentity, SharedHostMetadata, SharedPreparedInputCacheIdentity,
    SharedStateLayout, StateLayout, StateSegmentLifetime, StateSegmentSpec,
};
use std::num::NonZeroU8;

fn layout() -> SharedHostMetadata {
    let mut name = String::with_capacity(83);
    name.push_str("decoder-β");
    let layers = LayerSchedule::new(
        2,
        vec![
            LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 4).unwrap(),
            LayerCachePolicy::NoState,
        ],
    )
    .unwrap();
    let layout = StateLayout::segmented(
        layers,
        [StateSegmentSpec::new(name, 0..2, StateSegmentLifetime::Persistent, 0).unwrap()],
    )
    .unwrap();
    SharedHostMetadata::Layout(SharedStateLayout::new(layout))
}

fn input() -> SharedHostMetadata {
    let mut shape = Vec::with_capacity(7);
    shape.extend_from_slice(&[1, 5]);
    let prepared = PreparedInputIdentity::new(vec![InputPartDescriptor::new(
        InputModality::Text,
        InputPayloadKind::TokenIds,
        InputTensorIdentity::new(TensorDtype::U32, shape).unwrap(),
        [],
    )
    .unwrap()])
    .unwrap();
    let mut fingerprint = String::with_capacity(97);
    fingerprint.push_str("content-λ-123");
    SharedHostMetadata::Input(SharedPreparedInputCacheIdentity::new(
        PreparedInputCacheIdentity::new(prepared, fingerprint).unwrap(),
    ))
}

fn inventory(sources: impl IntoIterator<Item = SharedHostMetadata>) -> RetainedStorage {
    let mut storage = RetainedStorage::default();
    for source in sources {
        storage.include_metadata(source).unwrap();
    }
    storage
}

fn reclaim(pool: &MemoryLedger, expected: u64) {
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.fixture_host_charge().unwrap(), expected);
}

fn assert_payload(source: &SharedHostMetadata) {
    match source {
        SharedHostMetadata::ObservationPaths(_) => unreachable!("layout/input fixture"),
        SharedHostMetadata::Layout(layout) => {
            assert_eq!(layout.layout().len(), 2);
            assert_eq!(layout.layout().segments()[0].id().as_str(), "decoder-β");
        }
        SharedHostMetadata::Input(input) => {
            assert_eq!(input.prepared().parts()[0].payload().shape(), [1, 5]);
            assert_eq!(input.semantic_content_fingerprint(), "content-λ-123");
        }
    }
}

fn cause<'a, T: std::error::Error + 'static>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    loop {
        if let Some(source) = error.downcast_ref::<T>() {
            return Some(source);
        }
        error = error.source()?;
    }
}

#[test]
fn metadata_publication_keeps_preexisting_aliases_charged_after_enclosing_owner_drop() {
    let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let layout = layout();
    let input = input();
    let aliases = [layout.clone(), input.clone()];
    let keys = [layout.identity().clone(), input.identity().clone()];
    let bytes = layout.capacity_bytes().unwrap() + input.capacity_bytes().unwrap();
    assert!(bytes > 180);
    let publication = inventory([layout.clone(), input.clone(), layout.clone()])
        .publish_unquoted(&loading)
        .unwrap();
    let duplicate = inventory(aliases.clone())
        .publish_unquoted(&loading)
        .unwrap();
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    assert_eq!(
        pool.fixture_host_peak().unwrap(),
        bytes
            + 2 * crate::memory_fixture::publication_control_bytes(2)
            + crate::memory_fixture::native_owner_control_bytes()
    );
    // This tuple stands for the enclosing model-like owner; the escaped aliases
    // predate publication and must acquire custody through the shared payload.
    drop((loading, layout, input, publication, duplicate));
    reclaim(&pool, bytes);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    for source in &aliases {
        assert_payload(source);
    }
    let [layout, input] = aliases;
    let input_bytes = input.capacity_bytes().unwrap();
    drop(layout);
    reclaim(&pool, input_bytes);
    drop(input);
    reclaim(&pool, 0);
    // Registry/identity metadata must not form source -> charge -> source cycles.
    assert_ne!(keys[0], keys[1]);
    assert_eq!(
        pool.fixture_host_peak().unwrap(),
        bytes
            + 2 * crate::memory_fixture::publication_control_bytes(2)
            + crate::memory_fixture::native_owner_control_bytes()
    );
}

#[test]
fn equal_metadata_values_are_distinct_sources_while_aliases_deduplicate() {
    for make in [layout as fn() -> SharedHostMetadata, input] {
        let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let first = make();
        let second = make();
        match (&first, &second) {
            (SharedHostMetadata::Layout(a), SharedHostMetadata::Layout(b)) => assert_eq!(a, b),
            (SharedHostMetadata::Input(a), SharedHostMetadata::Input(b)) => assert_eq!(a, b),
            _ => unreachable!(),
        }
        assert_ne!(first.identity(), second.identity());
        let bytes = first.capacity_bytes().unwrap();
        assert_eq!(second.capacity_bytes(), Some(bytes));
        let storage = inventory([first.clone(), second.clone(), first.clone(), second.clone()]);
        assert_eq!(storage.byte_bound().unwrap(), Some(bytes * 2));
        let publication = storage.publish_unquoted(&loading).unwrap();
        drop((publication, loading));
        reclaim(&pool, bytes * 2);
        drop(first);
        reclaim(&pool, bytes);
        assert_payload(&second);
        drop(second);
        reclaim(&pool, 0);
    }
}

#[test]
fn metadata_attachments_keep_each_domain_charged_until_the_last_payload_alias() {
    for make in [layout as fn() -> SharedHostMetadata, input] {
        let a = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
        let b = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
        let loading_a = NativeMemoryOwner::acquire(&a).unwrap();
        let loading_b = NativeMemoryOwner::acquire(&b).unwrap();
        let source = make();
        let alias = source.clone();
        let bytes = source.capacity_bytes().unwrap();
        let pa = inventory([source.clone()])
            .publish_unquoted(&loading_a)
            .unwrap();
        let pb = inventory([alias.clone()])
            .publish_unquoted(&loading_b)
            .unwrap();
        let repeated = inventory([source.clone()])
            .publish_unquoted(&loading_a)
            .unwrap();
        assert_eq!(
            (
                a.fixture_host_charge().unwrap(),
                b.fixture_host_charge().unwrap()
            ),
            (bytes, bytes)
        );
        drop((pa, pb, repeated, source, loading_a, loading_b));
        reclaim(&a, bytes);
        reclaim(&b, bytes);
        assert_payload(&alias);
        drop(alias);
        reclaim(&a, 0);
        reclaim(&b, 0);
    }
}

#[test]
fn metadata_batch_registration_rejects_one_short_without_partial_charge_or_owner_loss() {
    let source_pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let source_authority = NativeMemoryOwner::acquire(&source_pool).unwrap();
    let sources = [layout(), input()];
    let bytes: u64 = sources
        .iter()
        .map(|source| source.capacity_bytes().unwrap())
        .sum();
    let controls = crate::memory_fixture::publication_control_bytes(2);
    let short = crate::memory_fixture::ledger(
        bytes + controls - 1 + crate::memory_fixture::native_owner_control_bytes(),
        0,
    )
    .unwrap();
    let loading_short = NativeMemoryOwner::acquire(&short).unwrap();
    let error = inventory(sources.clone())
        .publish_unquoted(&loading_short)
        .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error)
            .and_then(crate::tests::support::memory_error::host_budget_numbers),
        Some((bytes, bytes - 1))
    );
    assert_eq!(
        (
            short.fixture_host_charge().unwrap(),
            short.fixture_host_peak().unwrap()
        ),
        (
            0,
            controls + crate::memory_fixture::native_owner_control_bytes()
        )
    );
    assert_eq!(short.unquoted_owner_count().unwrap(), 1);
    for source in &sources {
        assert_payload(source);
    }
    let exact = crate::memory_fixture::ledger(
        bytes
            + crate::memory_fixture::publication_control_bytes(2)
            + crate::memory_fixture::native_owner_control_bytes(),
        0,
    )
    .unwrap();
    let loading_exact = NativeMemoryOwner::acquire(&exact).unwrap();
    let publication = inventory(sources.clone())
        .publish_unquoted(&loading_exact)
        .unwrap();
    assert_eq!(exact.fixture_host_charge().unwrap(), bytes);
    assert_eq!(exact.unquoted_owner_count().unwrap(), 1);
    drop((publication, loading_exact, loading_short, source_authority));
    reclaim(&exact, bytes);
    drop(sources);
    reclaim(&exact, 0);
    reclaim(&short, 0);
}

// A host-only accounting fixture: no model or native execution is claimed by
// these zero state/operator terms. Its only managed payload is the exact real
// metadata constructed below, and the retained contribution prices that payload.
fn funding(
    pool: &MemoryLedger,
    payload: u64,
    publication_rows: &[usize],
) -> (WorkingMemoryReservation, WorkingMemoryFundingRun) {
    let bytes = publication_rows
        .iter()
        .try_fold(payload, |bytes, rows| {
            bytes.checked_add(crate::memory_fixture::publication_control_bytes(*rows))
        })
        .unwrap();
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
    .with_execution_workspace(crate::memory_fixture::workspace(
        ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: zero(),
            attention: zero(),
            vocabulary: zero(),
            state_update: zero(),
            materialization: zero(),
            retained: WorkspaceBound::bounded(
                bytes,
                "exact retained immutable host metadata payload",
            ),
        },
    ))
    .unwrap();
    let admission = crate::memory_fixture::host_admission(Admission {
        additional_headroom: Default::default(),
        memory_limits: Default::default(),
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(bytes),
    });
    let requirements = pool
        .reservation_requirements(
            &admission,
            Some(&crate::memory_fixture::host_requirements(bytes)),
        )
        .unwrap();
    let physical_limit = pool
        .fixture_host_current()
        .unwrap()
        .checked_add(
            requirements
                .get(pool.topology().host_domain())
                .unwrap()
                .total()
                .unwrap(),
        )
        .unwrap();
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &admission,
        crate::memory_fixture::physical_host_limits(pool, physical_limit),
    )
    .unwrap()
    .into_funding()
    .unwrap()
}

#[test]
fn funded_metadata_publication_consumes_exact_credit_and_outlives_run_as_aliases() {
    for make in [layout as fn() -> SharedHostMetadata, input] {
        // Establish this deterministic test fixture's concrete capacity while
        // holding loading authority; no guessed numerical allocation is hidden.
        let probe_pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&probe_pool).unwrap();
        let probe = make();
        let bytes = probe.capacity_bytes().unwrap();
        drop((probe, loading));
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let (reservation, run) = funding(&pool, bytes, &[1, 1]);
        let accepted_peak = pool.fixture_host_peak().unwrap();
        let scope = run.scope().unwrap();
        let source = make();
        assert_eq!(source.capacity_bytes(), Some(bytes));
        let alias = source.clone();
        let key = source.identity().clone();
        let publication = inventory([source.clone(), alias.clone()])
            .publish_funded(&scope)
            .unwrap();
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            bytes + crate::memory_fixture::publication_control_bytes(1)
        );
        let duplicate = inventory([alias.clone()]).publish_funded(&scope).unwrap();
        assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
        // No remaining credit: every byte was transferred to this actual source.
        let error = scope
            .adopt_host_storage_individually([(37_u32, 1)])
            .unwrap_err();
        assert_eq!(
            error,
            WorkingMemoryError::DomainAllowanceExceeded {
                domain: pool.topology().host_domain(),
                required_bytes: MemoryLedger::storage_metadata_control_bytes().unwrap(),
                available_bytes: 0
            }
        );
        scope.certify().unwrap();
        drop((reservation, run, publication, duplicate, source));
        reclaim(&pool, bytes);
        assert_payload(&alias);
        assert_eq!(alias.identity(), &key);
        drop(alias);
        reclaim(&pool, 0);
        assert_eq!(pool.fixture_host_peak().unwrap(), accepted_peak);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
