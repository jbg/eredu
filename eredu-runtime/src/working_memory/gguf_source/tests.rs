use super::*;
use eredu_checkpoint::{
    gguf_store::GgufCatalogPlan,
    schema::{
        CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
        TensorOperation,
    },
    store::{CheckpointSource, ReadPolicy, TensorReadRequest, TensorSelection, WeightStore},
    validation::resolve_gguf_plan,
};
use eredu_gguf::{Checkpoint, GgmlType, TensorInput, Writer};
use std::{collections::BTreeMap, fs::File};
struct Fixture {
    _directory: tempfile::TempDir,
    checkpoint: Checkpoint,
    resolution: eredu_checkpoint::validation::ResolvedCheckpointPlan,
    mapping: Vec<eredu_gguf::TranslatedTensorLayout>,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("catalog.gguf");
        let bytes = [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat();
        Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "first",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &bytes,
                    },
                    TensorInput {
                        name: "second",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &bytes,
                    },
                ],
            )
            .unwrap();
        let checkpoint = Checkpoint::open(&path).unwrap();
        let schema = GgufCheckpointPlan::new(
            "catalog fixture",
            ["first", "second"]
                .into_iter()
                .map(|name| {
                    GgufTensorConstraint::required(
                        name,
                        vec![2],
                        GgufTypeConstraint::OperationClass(TensorOperation::Dense),
                    )
                })
                .collect(),
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap();
        let resolution = resolve_gguf_plan(&checkpoint, &schema).unwrap();
        let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
        Self {
            _directory: directory,
            checkpoint,
            resolution,
            mapping,
        }
    }
    fn plan(&self) -> GgufCatalogPlan<'_> {
        // Actual cold input owners are outside this catalog-only component.
        GgufCatalogPlan::new(self.checkpoint.clone(), &self.resolution, &self.mapping, 1)
    }
}

fn requirements(fixture: &Fixture) -> Option<(u64, u64)> {
    let catalog = fixture.plan().compile(()).unwrap();
    let result = WorkingMemoryPool::gguf_source_required_bytes(&catalog);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_GGUF_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(source) => Some((
            WorkingMemoryPool::gguf_catalog_required_bytes(&fixture.plan()).unwrap(),
            source,
        )),
        Err(WorkingMemoryError::UnknownBound) => {
            assert!(matches!(
                WorkingMemoryPool::gguf_source_required_bytes(&catalog),
                Err(WorkingMemoryError::UnknownBound)
            ));
            None
        }
        Err(e) => panic!("unexpected request error: {e}"),
    }
}
fn identity(source: &GgufWeightStore) -> eredu_checkpoint::store::SourceStorageIdentity {
    let mut identity = None;
    source
        .visit_source_storage(&mut |r| {
            assert!(identity.is_none());
            identity = Some(r.identity());
        })
        .unwrap();
    identity.unwrap()
}
#[test]
fn source_constructor_exact_one_short_preserves_catalog_and_last_weak_charge() {
    let fixture = Fixture::new();
    let Some((catalog_bytes, source_bytes)) = requirements(&fixture) else {
        return;
    };
    let short = WorkingMemoryPool::new(catalog_bytes + source_bytes - 1, 0).unwrap();
    let catalog = short
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared();
    let refused = short.compile_gguf_source(catalog).unwrap_err();
    assert!(
        matches!(refused.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
        if *required_bytes == source_bytes && *available_bytes == source_bytes - 1)
    );
    assert!(refused.rejected_catalog().is_some());
    assert!(refused.construction_failure().is_none());
    assert!(std::error::Error::source(&refused)
        .unwrap()
        .is::<WorkingMemoryError>());
    assert_eq!(short.used_bytes().unwrap(), catalog_bytes);
    drop(refused);
    assert_eq!(short.used_bytes().unwrap(), 0);

    let pool = WorkingMemoryPool::new(catalog_bytes + source_bytes, 0).unwrap();
    let catalog = pool
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared();
    let source = pool.compile_gguf_source(catalog).unwrap();
    pool.validate_gguf_source_controls(&source).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), catalog_bytes + source_bytes);
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    let key = identity(&source);
    let lease = source
        .acquire(TensorReadRequest {
            key: "first".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), catalog_bytes + source_bytes);
    let converted = lease.materialize_portable().unwrap();
    assert_eq!(converted.output_names(), ["first"]);
    drop(converted);
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes);
    let clones = [key.clone(), key.clone(), key.clone()];
    drop(key);
    std::thread::scope(|scope| {
        for key in clones {
            scope.spawn(move || drop(key));
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn source_weak_account_has_no_pool_cycle_and_origin_cannot_be_promoted() {
    let fixture = Fixture::new();
    let Some((catalog_bytes, source_bytes)) = requirements(&fixture) else {
        return;
    };
    let pool = WorkingMemoryPool::new(catalog_bytes + source_bytes, 0).unwrap();
    let other = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let ordinary = fixture.plan().compile(()).unwrap();
    let refused = pool.compile_gguf_source(ordinary).unwrap_err();
    assert!(matches!(
        refused.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(refused);
    let catalog = pool
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared();
    let refused = other.compile_gguf_source(catalog).unwrap_err();
    assert!(matches!(
        refused.accounting_failure(),
        Some(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(other.used_bytes().unwrap(), 0);
    drop(refused);
    let catalog = pool
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared();
    let source = pool.compile_gguf_source(catalog).unwrap();
    assert!(matches!(
        other.validate_gguf_source_controls(&source),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let key = identity(&source);
    let weak_pool = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak_pool.upgrade().is_some()); // Actual live source still retains its catalog account.
    drop(source);
    assert!(weak_pool.upgrade().is_none()); // Weak source key must not keep the ledger alive.
    drop(key); // No late credit or attempt to reconstitute a dead ledger.
    assert!(weak_pool.upgrade().is_none());
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Source(eredu_checkpoint::store::SourceStorageIdentity),
    Ordinary(u8),
}
impl GgufSourceStorageKey for Key {
    fn gguf_source_identity(&self) -> Option<&eredu_checkpoint::store::SourceStorageIdentity> {
        match self {
            Self::Source(id) => Some(id),
            Self::Ordinary(_) => None,
        }
    }
}
#[test]
fn source_inventory_registration_preserves_full_capacity_and_charges_only_same_pool_residual() {
    let fixture = Fixture::new();
    let Some((catalog_bytes, source_bytes)) = requirements(&fixture) else {
        return;
    };
    let catalog = fixture.plan().compile(()).unwrap();
    let (physical, prepaid) = catalog
        .source_storage_request::<SourcePayloadCustody>()
        .unwrap()
        .inventory_bytes();
    assert!(prepaid > 0 && physical >= prepaid);
    let residual = physical - prepaid;
    let pool = WorkingMemoryPool::new(catalog_bytes + source_bytes + residual + 19, 0).unwrap();
    let source = pool
        .compile_gguf_source(
            pool.compile_gguf_catalog(fixture.plan())
                .unwrap()
                .into_prepared(),
        )
        .unwrap();
    let key = Key::Source(identity(&source));
    let before = pool.used_bytes().unwrap();
    let bad = pool
        .register_storage_with_gguf_sources([(key.clone(), physical + 1)])
        .unwrap_err();
    assert!(matches!(
        bad,
        WorkingMemoryError::StorageCapacityMismatch { .. }
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let first = pool
        .register_storage_with_gguf_sources([(key.clone(), physical), (Key::Ordinary(1), 19)])
        .unwrap();
    assert_eq!(first.bytes(), physical + 19);
    assert_eq!(pool.used_bytes().unwrap(), before + residual + 19);
    let alias = pool
        .register_storage_with_gguf_sources([(key.clone(), physical)])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), before + residual + 19);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), before + residual);
    let raw_alias = pool.register_storage([(key.clone(), physical)]).unwrap();
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), before + residual);
    drop(raw_alias);
    assert_eq!(pool.used_bytes().unwrap(), before);
    // B cannot borrow A's accepted reader construction. It registers exactly
    // the full truthful physical inventory in its independent domain.
    let foreign = WorkingMemoryPool::new(physical, 0).unwrap();
    let foreign_pin = foreign
        .register_storage_with_gguf_sources([(key.clone(), physical)])
        .unwrap();
    assert_eq!(foreign.used_bytes().unwrap(), physical);
    drop(foreign_pin);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes);
    drop(key);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[derive(Clone, Debug)]
struct ObservedKey {
    id: eredu_checkpoint::store::SourceStorageIdentity,
    pool: Weak<crate::working_memory::Pool>,
    panic_in_lookup: Arc<AtomicBool>,
    drops: Arc<std::sync::atomic::AtomicUsize>,
}
impl PartialEq for ObservedKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for ObservedKey {}
impl PartialOrd for ObservedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ObservedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.panic_in_lookup.load(Ordering::SeqCst) {
            if let Some(pool) = self.pool.upgrade() {
                if matches!(
                    pool.usage.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ) {
                    self.panic_in_lookup.store(false, Ordering::SeqCst);
                    panic!("provider comparison during canonical lookup");
                }
            }
        }
        self.id.cmp(&other.id)
    }
}
impl Drop for ObservedKey {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            assert!(
                !matches!(
                    pool.usage.try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ),
                "provider key retired under Usage"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}
impl GgufSourceStorageKey for ObservedKey {
    fn gguf_source_identity(&self) -> Option<&eredu_checkpoint::store::SourceStorageIdentity> {
        Some(&self.id)
    }
}
#[test]
fn source_registration_comparison_unwind_retires_candidates_after_usage_without_credit() {
    let fixture = Fixture::new();
    let Some((catalog_bytes, source_bytes)) = requirements(&fixture) else {
        return;
    };
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let catalog = pool
        .compile_gguf_catalog(fixture.plan())
        .unwrap()
        .into_prepared();
    let (physical, prepaid) = catalog
        .source_storage_request::<SourcePayloadCustody>()
        .unwrap()
        .inventory_bytes();
    let source = pool.compile_gguf_source(catalog).unwrap();
    let key = ObservedKey {
        id: identity(&source),
        pool: Arc::downgrade(&pool.0),
        panic_in_lookup: Arc::new(AtomicBool::new(false)),
        drops: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    let registration = pool
        .register_storage_with_gguf_sources([(key.clone(), physical)])
        .unwrap();
    let before = pool.used_bytes().unwrap();
    assert_eq!(before, catalog_bytes + source_bytes + physical - prepaid);
    let drops_before = key.drops.load(Ordering::SeqCst);
    key.panic_in_lookup.store(true, Ordering::SeqCst);
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.register_storage_with_gguf_sources([(key.clone(), physical)])
    }));
    assert!(failed.is_err());
    assert!(key.drops.load(Ordering::SeqCst) > drops_before);
    assert!(pool.0.usage.is_poisoned());
    // Explicit test recovery permits inspection after the injected comparison;
    // production preserves poison and never treats it as permission to refund.
    pool.0.usage.clear_poison();
    assert_eq!(pool.used_bytes().unwrap(), before);
    drop(registration);
    assert_eq!(pool.used_bytes().unwrap(), catalog_bytes + source_bytes);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), source_bytes);
    drop(key);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn native_publication_reuses_actual_source_origin_and_rejects_absent_or_foreign_kind_rows() {
    use crate::working_memory::{
        funding::{native_partition::test_receipt, tests::reservation},
        storage::native_publication::PreparedNativePublication,
    };
    let fixture = Fixture::new();
    let Some((catalog_bytes, source_bytes)) = requirements(&fixture) else {
        return;
    };
    let catalog = fixture.plan().compile(()).unwrap();
    let (physical, _) = catalog
        .source_storage_request::<SourcePayloadCustody>()
        .unwrap()
        .inventory_bytes();
    let capacity = catalog_bytes + source_bytes + 2 * physical + 64;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = pool
        .compile_gguf_source(
            pool.compile_gguf_catalog(fixture.plan())
                .unwrap()
                .into_prepared(),
        )
        .unwrap();
    let id = identity(&source);
    let key = Key::Source(id.clone());
    // Pure publication-ledger fixture: no mutable native birth is fabricated.
    let (metadata, run) = reservation(&pool, 64, capacity).into_funding().unwrap();
    let partition = run.take_native_partition(test_receipt(&run, 0)).unwrap();
    let scope = run.scope().unwrap();
    let mut missing = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    missing
        .push_source(&key, physical, Some(&id), &pool)
        .unwrap();
    let before = pool.used_bytes().unwrap();
    assert_eq!(
        missing.publish(&scope).unwrap_err(),
        WorkingMemoryError::IdentityMismatch
    );
    assert!(missing.take_input(0).is_none());
    assert_eq!(pool.used_bytes().unwrap(), before);
    let existing = pool
        .register_storage_with_gguf_sources([(key.clone(), physical)])
        .unwrap();
    let ordinary = pool
        .register_storage([(Key::Ordinary(99), physical)])
        .unwrap();
    let mut wrong = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    wrong
        .push_source(&Key::Ordinary(99), physical, Some(&id), &pool)
        .unwrap();
    let before = pool.used_bytes().unwrap();
    assert_eq!(
        wrong.publish(&scope).unwrap_err(),
        WorkingMemoryError::IdentityMismatch
    );
    assert_eq!(pool.used_bytes().unwrap(), before);
    let foreign = WorkingMemoryPool::new(capacity, 0).unwrap();
    let mut wrong_pool = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    assert_eq!(
        wrong_pool
            .push_source(&key, physical, Some(&id), &foreign)
            .unwrap_err(),
        WorkingMemoryError::IdentityMismatch
    );
    let mut changed = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    assert!(matches!(
        changed.push_source(&key, physical + 1, Some(&id), &pool),
        Err(WorkingMemoryError::StorageCapacityMismatch { .. })
    ));
    let mut alias = PreparedNativePublication::prepare_slots(partition.clone(), 1);
    alias.push_source(&key, physical, Some(&id), &pool).unwrap();
    alias.publish(&scope).unwrap();
    assert_eq!(
        pool.used_bytes().unwrap(),
        before,
        "existing source is not recharged as mutable P"
    );
    let retained = alias.take_input(0).unwrap();
    assert_eq!(retained.bytes(), physical);
    scope.certify().unwrap();
    drop((
        missing, wrong, changed, wrong_pool, alias, ordinary, existing, partition, metadata, run,
        key, id, source,
    ));
    assert!(
        pool.used_bytes().unwrap() > 0,
        "the canonical source row retains the real original account"
    );
    drop(retained);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
