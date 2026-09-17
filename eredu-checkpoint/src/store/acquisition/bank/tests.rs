use super::*;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    }
}
fn acquire(
    bank: &mut PreparedAcquisitionBank,
    source: &SharedCheckpointSource,
    key: &str,
) -> Result<CheckpointLease, PreparedAcquisitionBankError> {
    bank.acquire(
        source.as_ref(),
        key,
        &TensorSelection::Full,
        ReadPolicy::RequireBounded,
    )
}
#[test]
fn repeated_requests_and_skipped_rows_keep_exact_wrapper_identity_and_multiplicity() {
    let leaf: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            ("same".into(), Dtype::U8, vec![3], vec![2, 4, 6]),
            ("skip".into(), Dtype::U8, vec![3], vec![1, 3, 5]),
        ])
        .unwrap(),
    );
    let restricted = || -> SharedCheckpointSource {
        Arc::new(
            RestrictedCheckpointSource::including(
                leaf.clone(),
                "actual restricted source",
                BTreeSet::from(["same".into(), "skip".into()]),
            )
            .unwrap(),
        )
    };
    let a = restricted();
    let b = restricted();
    let foreign = restricted();
    let mut bank = PreparedAcquisitionBank::new(4).unwrap();
    for (source, key) in [
        (a.clone(), "same"),
        (a.clone(), "skip"),
        (a.clone(), "same"),
        (b.clone(), "same"),
    ] {
        bank.prepare_next(source, request(key)).unwrap();
    }
    let storage = bank.storage().unwrap();
    assert_eq!(storage.remaining, 4);
    assert!(storage.retained_slots.size() >= storage.requested_slots.size());
    assert!(storage.owned_payload_capacity_bytes > 0);
    assert_eq!(storage.destination_arc_allocations, 0);
    assert!(matches!(
        acquire(&mut bank, &a, "same"),
        Err(PreparedAcquisitionBankError::Population)
    ));
    bank.seal().unwrap();
    assert!(matches!(
        acquire(&mut bank, &foreign, "same"),
        Err(PreparedAcquisitionBankError::NoDestination)
    ));
    assert_eq!(bank.storage().unwrap().remaining, 4);
    for source in [&b, &a, &a] {
        let actual = acquire(&mut bank, source, "same").unwrap();
        let ordinary = source.acquire_lease(request("same")).unwrap();
        assert_eq!(actual.encoded_bytes(), ordinary.encoded_bytes());
        assert_eq!(actual.encoded_bytes().unwrap(), [2, 4, 6]);
    }
    assert!(matches!(
        acquire(&mut bank, &a, "same"),
        Err(PreparedAcquisitionBankError::NoDestination)
    ));
    assert_eq!(bank.storage().unwrap().remaining, 1);
    assert_eq!(
        acquire(&mut bank, &a, "skip")
            .unwrap()
            .encoded_bytes()
            .unwrap(),
        [1, 3, 5]
    );
}
fn shards() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    for (key, bytes) in [("a", [1, 2, 3]), ("b", [7, 8, 9])] {
        serialize_to_file(
            [(key, TensorView::new(Dtype::U8, vec![3], &bytes).unwrap())],
            None,
            &directory.path().join(format!("{key}.safetensors")),
        )
        .unwrap();
    }
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}"#,
    )
    .unwrap();
    let store =
        Arc::new(SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), 1).unwrap());
    // Real manager preflight has already retained each selected header.
    store.source_metadata("a").unwrap();
    store.source_metadata("b").unwrap();
    (directory, store)
}
#[test]
fn actual_cache_failure_consumes_only_its_destination_and_retry_uses_same_cache() {
    let (_dir, store) = shards();
    let source: SharedCheckpointSource = store.clone();
    let ordinary = store.acquire(request("a")).unwrap();
    let mut bank = PreparedAcquisitionBank::new(2).unwrap();
    for _ in 0..2 {
        bank.prepare_next(source.clone(), request("b")).unwrap();
    }
    bank.seal().unwrap();
    assert_eq!(store.diagnostics().unwrap().physical_reads, 1);
    assert_eq!(bank.storage().unwrap().destination_arc_allocations, 0);
    let failed = acquire(&mut bank, &source, "b").unwrap_err();
    assert!(matches!(
        failed.store_error(),
        Some(StoreError::CapacityExhausted { maximum: 1, .. })
    ));
    assert_eq!(bank.storage().unwrap().remaining, 1);
    assert_eq!(store.diagnostics().unwrap().evictions, 0);
    drop(failed);
    drop(ordinary);
    let lease = acquire(&mut bank, &source, "b").unwrap();
    assert_eq!(lease.encoded_bytes().unwrap(), [7, 8, 9]);
    assert_eq!(store.diagnostics().unwrap().evictions, 1);
    let ordinary = source.acquire_lease(request("b")).unwrap();
    assert_eq!(
        ordinary.encoded_bytes().unwrap().as_ptr(),
        lease.encoded_bytes().unwrap().as_ptr()
    );
}
#[test]
fn post_validation_failure_keeps_real_cache_pin_until_error_is_retired() {
    struct Uncached(Arc<SafetensorsWeightStore>);
    impl CheckpointSource for Uncached {
        fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
            self.0.prepared_acquisition_source()
        }
        fn source_keys(&self) -> Vec<String> {
            self.0.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let mut metadata = self.0.source_metadata(key)?;
            if key == "a" {
                metadata.encoded_byte_len += 1;
            }
            Ok(metadata)
        }
        fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
            self.0.source_provenance(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.0.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.0.source_diagnostics()
        }
    }
    let (_dir, store) = shards();
    let source: SharedCheckpointSource = Arc::new(Uncached(store.clone()));
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let metadata = source.source_metadata(&key).unwrap();
            let provenance = source.source_provenance(&key).unwrap();
            (
                key,
                PreparedTensorSource {
                    metadata,
                    provenance,
                },
            )
        })
        .collect();
    let root: SharedCheckpointSource =
        Arc::new(PreparedCheckpointSource::new(source, catalog).unwrap());
    let mut bank = PreparedAcquisitionBank::new(1).unwrap();
    bank.prepare_next(root.clone(), request("a")).unwrap();
    bank.seal().unwrap();
    let failed = acquire(&mut bank, &root, "a").unwrap_err();
    assert!(
        matches!(&failed, PreparedAcquisitionBankError::Acquisition(error) if error.retains_rejected_lease())
    );
    assert!(matches!(
        store.acquire(request("b")),
        Err(StoreError::CapacityExhausted { .. })
    ));
    drop(failed);
    assert_eq!(
        store
            .acquire(request("b"))
            .unwrap()
            .encoded_bytes()
            .unwrap(),
        [7, 8, 9]
    );
}

#[test]
fn future_tickets_prepare_only_metadata_and_payload_tracks_live_acquisitions() {
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors(["a", "b", "c"].into_iter().map(|key| {
            (
                key.into(),
                Dtype::U8,
                vec![64, 64],
                (0..4096).map(|n| (n % 251 + 1) as u8).collect(),
            )
        }))
        .unwrap(),
    );
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: (0..32).rev().collect(),
    };
    let windows: &[&[&str]] = &[&["a", "b"], &["b", "c"], &["c"]];
    let mut banks = Vec::new();
    for window in windows {
        for _ in 0..5 {
            let mut bank = PreparedAcquisitionBank::new(window.len() * 4).unwrap();
            for key in *window {
                for _ in 0..4 {
                    bank.prepare_next(
                        source.clone(),
                        TensorReadRequest {
                            key: (*key).into(),
                            selection: selection.clone(),
                            policy: ReadPolicy::RequireBounded,
                        },
                    )
                    .unwrap();
                }
            }
            bank.seal().unwrap();
            banks.push(bank);
        }
    }
    let arcs: usize = banks
        .iter()
        .map(|bank| bank.storage().unwrap().destination_arc_allocations)
        .sum();
    let bytes: usize = banks
        .iter()
        .map(|bank| bank.storage().unwrap().owned_payload_capacity_bytes)
        .sum();
    println!("3 windows, 5 forwards, 4 retry tickets: 100 destinations; source=12288 bytes, selected=2048 bytes, eager payload replicas={} bytes, observed named capacity={} bytes, destination Arcs={}", arcs * 2048, bytes, arcs);
    assert_eq!(
        arcs, 0,
        "future tickets must not retain future selected payload replicas"
    );
    let first = banks[0]
        .acquire(source.as_ref(), "a", &selection, ReadPolicy::RequireBounded)
        .unwrap();
    let second = banks[1]
        .acquire(source.as_ref(), "a", &selection, ReadPolicy::RequireBounded)
        .unwrap();
    assert_eq!(first.encoded_bytes().unwrap().len(), 2048);
    assert_eq!(first.encoded_bytes(), second.encoded_bytes());
    assert_ne!(
        first.encoded_bytes().unwrap().as_ptr(),
        second.encoded_bytes().unwrap().as_ptr()
    );
    let CheckpointLease::Memory(first_memory) = &first else {
        panic!("actual memory source")
    };
    let weak = Arc::downgrade(first_memory.selected_bytes.as_ref().unwrap());
    drop(first);
    assert!(weak.upgrade().is_none());
    assert_eq!(second.encoded_bytes().unwrap().len(), 2048);
    assert!(banks
        .iter()
        .all(|bank| bank.storage().unwrap().destination_arc_allocations == 0));
}
