use super::*;
use eredu_checkpoint::{
    safetensors::SafetensorsHeaderRequest,
    store::{CheckpointSource, WeightStore},
};
const POLICY: DependencyMemoryPolicy = DependencyMemoryPolicy {
    fixed_bytes: 1024,
    bytes_per_input_byte: 8,
};
const INDEX: &[u8] = br#"{"weight_map":{"weight":"weights.safetensors"}}"#;
const HEADER: &[u8] = br#"{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
fn fixture(index: Option<&[u8]>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut payload = (HEADER.len() as u64).to_le_bytes().to_vec();
    payload.extend_from_slice(HEADER);
    payload.extend_from_slice(&[11, 19]);
    std::fs::write(dir.path().join("weights.safetensors"), payload).unwrap();
    if let Some(index) = index {
        std::fs::write(dir.path().join("model.safetensors.index.json"), index).unwrap();
    }
    dir
}
fn qualified(path: &Path) -> bool {
    let result = WorkingMemoryPool::safetensors_source_initial_bytes(path, POLICY);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MEMORY_TENSOR_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(_) => true,
        Err(WorkingMemoryError::UnknownBound) => false,
        other => panic!("{other:?}"),
    }
}
fn open(
    pool: &WorkingMemoryPool,
    path: &Path,
) -> Result<SafetensorsWeightStore, OriginalSafetensorsSourceError> {
    pool.open_safetensors_source(path, 1, SafetensorsDiscoveryLimits::default(), POLICY)
}
fn memory<'a>(mut error: &'a (dyn Error + 'static)) -> &'a WorkingMemoryError {
    loop {
        if let Some(memory) = error.downcast_ref::<WorkingMemoryError>() {
            return memory;
        }
        error = error.source().expect("typed pool refusal");
    }
}
fn indexed_parts(dir: &tempfile::TempDir) -> (u64, u64, u64, u64, u64) {
    let initial = WorkingMemoryPool::safetensors_source_initial_bytes(dir.path(), POLICY).unwrap();
    let index = WorkingMemoryPool::safetensors_source_index_bytes(
        SafetensorsIndexRequest {
            encoded_bytes: INDEX.len(),
            path_bytes: dir.path().canonicalize().unwrap().as_os_str().len()
                + dir
                    .path()
                    .join("model.safetensors.index.json")
                    .as_os_str()
                    .len(),
        },
        POLICY,
    )
    .unwrap();
    let header_controls = WorkingMemoryPool::safetensors_header_policy_required_bytes(1).unwrap();
    let path_bytes = dir
        .path()
        .join("weights.safetensors")
        .canonicalize()
        .unwrap()
        .as_os_str()
        .len();
    let store = POLICY.estimate("weight".len() + 2 * path_bytes).unwrap() as u64;
    let header = WorkingMemoryPool::safetensors_header_quote(
        SafetensorsHeaderRequest {
            json_bytes: HEADER.len(),
            buffer_bytes: HEADER.len() + 8,
            path_bytes,
        },
        POLICY,
    )
    .unwrap()
    .total_bytes();
    (initial, index, header_controls, store, header)
}
#[test]
fn exact_source_budget_is_incremental_and_preserves_lazy_headers() {
    let dir = fixture(Some(INDEX));
    if !qualified(dir.path()) {
        return;
    }
    let (initial, index, headers, store, header) = indexed_parts(&dir);
    let total = initial + index + headers + store;
    let pool = WorkingMemoryPool::new(total - 1, 0).unwrap();
    let error = open(&pool, dir.path()).unwrap_err();
    assert!(
        matches!(memory(&error), WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }
        if *required_bytes == store && *available_bytes == store-1)
    );
    assert_eq!(pool.used_bytes().unwrap(), initial + index);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    drop(pool.acquire_unquoted().unwrap());
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let pool = WorkingMemoryPool::new(total + header, 0).unwrap();
    let source = open(&pool, dir.path()).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), total);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    assert_eq!(
        WeightStore::metadata(&source, "weight")
            .unwrap()
            .logical_shape,
        [2]
    );
    assert_eq!(pool.used_bytes().unwrap(), total + header);
    let batch = source
        .prepare_encoded_read(&["weight".into()])
        .unwrap()
        .unwrap();
    drop(source);
    // File identities and diagnostic owners keep source funding after the store.
    assert!(pool.used_bytes().unwrap() >= initial + index + store);
    let mut bytes = [0; 2];
    batch.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [11, 19]);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn initial_and_index_refusals_precede_filesystem_or_json_work() {
    let dir = fixture(Some(b"this is not JSON"));
    if !qualified(dir.path()) {
        return;
    }
    let missing = dir.path().join("missing.safetensors");
    let initial = WorkingMemoryPool::safetensors_source_initial_bytes(&missing, POLICY).unwrap();
    let pool = WorkingMemoryPool::new(initial - 1, 0).unwrap();
    let error = open(&pool, &missing).unwrap_err();
    assert!(matches!(
        error.memory_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert!(error.construction_failure().is_none());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let initial = WorkingMemoryPool::safetensors_source_initial_bytes(dir.path(), POLICY).unwrap();
    let pool = WorkingMemoryPool::new(initial, 0).unwrap();
    let error = open(&pool, dir.path()).unwrap_err();
    assert!(matches!(
        memory(&error),
        WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        }
    ));
    assert!(matches!(
        error.construction_failure(),
        Some(StoreError::SafetensorsSourceAdmission(_))
    ));
    assert_eq!(pool.used_bytes().unwrap(), initial);
    let alias = error.construction_failure().unwrap().clone();
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), initial);
    assert!(matches!(
        memory(&alias),
        WorkingMemoryError::BudgetExceeded { .. }
    ));
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn malformed_index_retains_accepted_storage_and_finishes_construction() {
    let dir = fixture(Some(b"xxxxxxxx"));
    if !qualified(dir.path()) {
        return;
    }
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let error = open(&pool, dir.path()).unwrap_err();
    assert!(matches!(
        error.construction_failure(),
        Some(StoreError::SafetensorsShards(
            eredu_checkpoint::safetensors::SafetensorsShardError::MalformedIndex { .. }
        ))
    ));
    let retained = pool.used_bytes().unwrap();
    assert!(
        retained > WorkingMemoryPool::safetensors_source_initial_bytes(dir.path(), POLICY).unwrap()
    );
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    drop(pool.acquire_unquoted().unwrap());
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn direct_source_keeps_pool_weak_and_allows_later_unquoted_work() {
    let dir = fixture(None);
    let path = dir.path().join("weights.safetensors");
    if !qualified(&path) {
        return;
    }
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    assert!(matches!(
        open(&pool, &path).unwrap_err().memory_failure(),
        Some(WorkingMemoryError::UnknownBound)
    ));
    drop(unquoted);
    let source = open(&pool, &path).unwrap();
    assert_eq!(
        WeightStore::metadata(&source, "weight")
            .unwrap()
            .encoded_byte_len,
        2
    );
    drop(pool.acquire_unquoted().unwrap());
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    let batch = source
        .prepare_encoded_read(&["weight".into()])
        .unwrap()
        .unwrap();
    let mut bytes = [0; 2];
    batch.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [11, 19]);
}

#[test]
fn concurrent_source_opening_and_retirement_preserve_shared_pool_totals() {
    let dir = fixture(Some(INDEX));
    if !qualified(dir.path()) {
        return;
    }
    let (initial, index, headers, store, _) = indexed_parts(&dir);
    let total = initial + index + headers + store;
    let pool = WorkingMemoryPool::new(total * 8, 0).unwrap();
    let gate = Arc::new(std::sync::Barrier::new(8));
    let joins: Vec<_> = (0..8)
        .map(|_| {
            let path = dir.path().to_owned();
            let pool = pool.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                gate.wait();
                open(&pool, &path).unwrap()
            })
        })
        .collect();
    let mut sources: Vec<_> = joins
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(pool.used_bytes().unwrap(), total * 8);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    while let Some(source) = sources.pop() {
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), total * sources.len() as u64);
    }
}
