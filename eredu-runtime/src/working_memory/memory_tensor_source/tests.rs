use super::*;
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, EncodedRecipeRead},
    store::{
        CheckpointSource, EncodedTensorLease, MemoryWeightStore, ReadPolicy, SourceStorageIdentity,
        TensorReadRequest, TensorSelection, WeightStore,
    },
};
use std::sync::{Arc, Barrier};

const NAME: &str = "packed";
const SHAPE: &[usize] = &[3, 2];
const PAYLOAD: u64 = 24;
const POLICY: DependencyMemoryPolicy = DependencyMemoryPolicy {
    fixed_bytes: 1024,
    bytes_per_input_byte: 8,
};
fn quote() -> Option<MemoryTensorBufferQuote> {
    let result = WorkingMemoryPool::memory_tensor_buffer_quote(NAME, SHAPE, PAYLOAD, POLICY);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MEMORY_TENSOR_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(quote) => Some(quote),
        Err(WorkingMemoryError::UnknownBound) => None,
        other => panic!("unexpected quote: {other:?}"),
    }
}
fn buffer(pool: &WorkingMemoryPool) -> MemoryTensorBuffer {
    let mut buffer = pool
        .allocate_memory_tensor_buffer(NAME, Dtype::U32, SHAPE, PAYLOAD, POLICY)
        .unwrap();
    for (bytes, value) in buffer.bytes_mut().chunks_exact_mut(4).zip(1u32..=6) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    buffer
}
fn identity(store: &MemoryWeightStore) -> SourceStorageIdentity {
    let mut key = None;
    assert!(store
        .visit_source_storage(&mut |row| {
            assert!(key.is_none());
            assert_eq!(row.bytes(), PAYLOAD);
            key = Some(row.identity());
        })
        .unwrap());
    key.unwrap()
}
fn request(selection: TensorSelection) -> TensorReadRequest {
    TensorReadRequest {
        key: NAME.into(),
        selection,
        policy: ReadPolicy::RequireBounded,
    }
}

#[test]
fn exact_admission_precedes_construction_and_metadata_headroom_is_separate() {
    let Some(quote) = quote() else { return };
    assert_eq!(quote.payload_bytes(), PAYLOAD);
    assert_eq!(
        quote.total_bytes(),
        quote.payload_bytes() + quote.metadata_estimate_bytes() + quote.constructor_control_bytes()
    );
    let larger =
        WorkingMemoryPool::memory_tensor_buffer_quote(NAME, SHAPE, PAYLOAD + 1000, POLICY).unwrap();
    assert_eq!(larger.total_bytes(), quote.total_bytes() + 1000);
    assert_eq!(
        larger.metadata_estimate_bytes(),
        quote.metadata_estimate_bytes()
    );
    let headroom = WorkingMemoryPool::memory_tensor_buffer_quote(
        NAME,
        SHAPE,
        PAYLOAD,
        DependencyMemoryPolicy {
            fixed_bytes: POLICY.fixed_bytes + 77,
            ..POLICY
        },
    )
    .unwrap();
    assert_eq!(headroom.total_bytes(), quote.total_bytes() + 77);
    let short = WorkingMemoryPool::new(quote.total_bytes() - 1, 0).unwrap();
    // Invalid dtype/length would reach construction if the comparison were late.
    let error = short
        .allocate_memory_tensor_buffer(NAME, Dtype::U8, SHAPE, PAYLOAD, POLICY)
        .unwrap_err();
    assert!(
        matches!(error.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
        if *required_bytes == quote.total_bytes() && *available_bytes == quote.total_bytes() - 1)
    );
    assert!(error.construction_failure().is_none());
    assert_eq!(short.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(quote.total_bytes(), 0).unwrap();
    let buffer = buffer(&pool);
    assert_eq!(buffer.payload_capacity(), PAYLOAD as usize);
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    // Settled source custody permits the later ordinary native load owner.
    drop(pool.acquire_unquoted().unwrap());
    drop(buffer);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn published_aliases_and_detached_reads_keep_original_reservation() {
    let Some(quote) = quote() else { return };
    let pool = WorkingMemoryPool::new(quote.total_bytes(), 0).unwrap();
    let mut buffer = buffer(&pool);
    let pointer = buffer.bytes_mut().as_ptr();
    let store = MemoryWeightStore::from_buffers([buffer]).unwrap();
    let key = identity(&store);
    pool.validate_original_source_inventory(&key, PAYLOAD)
        .unwrap();
    let lease = store.acquire(request(TensorSelection::Full)).unwrap();
    assert_eq!(lease.encoded_bytes().unwrap().as_ptr(), pointer);
    let recipe = DerivedWeightRecipe::source(
        NAME,
        TensorSelection::Indices {
            axis: 0,
            indices: vec![2, 0, 2],
        },
    );
    let read = recipe.prepare_encoded_read(&store).unwrap().unwrap();
    let detached = EncodedRecipeRead::prepare_detached(std::iter::once(&read))
        .unwrap()
        .construct(())
        .unwrap();
    drop((read, store));
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    let expected: Vec<u8> = [5u32, 6, 1, 2, 5, 6]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let mut actual = [0u8; 24];
    detached.read_many_into(&mut [&mut actual]).unwrap();
    assert_eq!(actual.as_slice(), expected);
    drop((detached, lease));
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    let keys = [key.clone(), key.clone(), key];
    std::thread::scope(|scope| {
        for key in keys {
            scope.spawn(move || drop(key));
        }
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_inventory_authenticates_capacity_pool_and_constructor() {
    let Some(quote) = quote() else { return };
    let pool = WorkingMemoryPool::new(quote.total_bytes(), 0).unwrap();
    let store = MemoryWeightStore::from_buffers([buffer(&pool)]).unwrap();
    let key = identity(&store);
    let first = pool
        .register_storage_with_gguf_sources([(key.clone(), PAYLOAD)])
        .unwrap();
    let alias = pool
        .register_storage_with_gguf_sources([(key.clone(), PAYLOAD)])
        .unwrap();
    assert_eq!(first.bytes(), PAYLOAD);
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    pool.validate_retained_source_inventory(&key, PAYLOAD)
        .unwrap();
    assert!(matches!(
        pool.validate_original_source_inventory(&key, PAYLOAD + 1),
        Err(WorkingMemoryError::StorageCapacityMismatch { .. })
    ));
    let foreign = WorkingMemoryPool::new(PAYLOAD, 0).unwrap();
    assert!(matches!(
        foreign.validate_original_source_inventory(&key, PAYLOAD),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let pin = foreign
        .register_storage_with_gguf_sources([(key.clone(), PAYLOAD)])
        .unwrap();
    assert_eq!(foreign.used_bytes().unwrap(), PAYLOAD);
    drop(pin);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
    let ordinary = MemoryWeightStore::from_buffers([MemoryTensorBuffer::allocate_with_custody(
        NAME,
        Dtype::U32,
        SHAPE,
        PAYLOAD,
        (),
    )
    .unwrap()])
    .unwrap();
    assert!(matches!(
        pool.validate_original_source_inventory(&identity(&ordinary), PAYLOAD),
        Err(WorkingMemoryError::UnknownBound)
    ));
    drop((first, alias, store, key));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn constructor_and_publication_failures_retain_only_the_actual_prefix() {
    let Some(quote) = quote() else { return };
    let pool = WorkingMemoryPool::new(quote.total_bytes() * 3, 0).unwrap();
    let error = pool
        .allocate_memory_tensor_buffer(NAME, Dtype::U8, SHAPE, PAYLOAD, POLICY)
        .unwrap_err();
    assert!(error.accounting_failure().is_none());
    assert!(error.construction_failure().is_some());
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<MemoryTensorBufferError>());
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let error =
        MemoryWeightStore::from_buffers([buffer(&pool), buffer(&pool), buffer(&pool)]).unwrap_err();
    assert!(error.to_string().contains("duplicate"));
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes() * 2);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_birth_and_ordinary_work_preserve_admission_exclusion() {
    let Some(quote) = quote() else { return };
    let pool = WorkingMemoryPool::new(quote.total_bytes(), 0).unwrap();
    let ordinary = pool.acquire_unquoted().unwrap();
    let error = pool
        .allocate_memory_tensor_buffer(NAME, Dtype::U32, SHAPE, PAYLOAD, POLICY)
        .unwrap_err();
    assert!(matches!(
        error.accounting_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop((error, ordinary));
    let gate = Arc::new(Barrier::new(2));
    let results = std::thread::scope(|scope| {
        let mut jobs = Vec::new();
        for _ in 0..2 {
            let gate = gate.clone();
            let pool = &pool;
            jobs.push(scope.spawn(move || {
                gate.wait();
                pool.allocate_memory_tensor_buffer(NAME, Dtype::U32, SHAPE, PAYLOAD, POLICY)
            }));
        }
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(results
        .iter()
        .filter_map(|r| r.as_ref().err())
        .all(|e| matches!(
            e.accounting_failure(),
            Some(WorkingMemoryError::BudgetExceeded { .. })
        )));
    assert_eq!(pool.used_bytes().unwrap(), quote.total_bytes());
    drop(results);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn weak_source_custody_does_not_retain_a_dead_pool() {
    let Some(quote) = quote() else { return };
    let pool = WorkingMemoryPool::new(quote.total_bytes(), 0).unwrap();
    let store = MemoryWeightStore::from_buffers([buffer(&pool)]).unwrap();
    let key = identity(&store);
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    drop((store, key));
    assert!(weak.upgrade().is_none());
}

#[test]
fn oversized_quotes_fail_without_a_source_reservation() {
    let Some(_) = quote() else { return };
    assert!(matches!(
        WorkingMemoryPool::memory_tensor_buffer_quote(NAME, SHAPE, u64::MAX, POLICY),
        Err(WorkingMemoryError::Overflow)
    ));
    let policy = DependencyMemoryPolicy {
        fixed_bytes: usize::MAX,
        bytes_per_input_byte: 1,
    };
    assert!(matches!(
        WorkingMemoryPool::memory_tensor_buffer_quote(NAME, SHAPE, PAYLOAD, policy),
        Err(WorkingMemoryError::Overflow)
    ));
}
