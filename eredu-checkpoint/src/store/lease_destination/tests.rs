use super::*;
use safetensors::tensor::{TensorView, serialize_to_file};

impl SafetensorsWeightStore {
    fn old_lock_cache(&self) -> Result<MutexGuard<'_, CacheState>, StoreError> {
        self.cache
            .lock()
            .map_err(|_| StoreError::Internal("checkpoint shard cache is poisoned".into()))
    }
    fn old_acquire_shard(&self, entry: &CatalogEntry) -> Result<Arc<CachedShard>, StoreError> {
        let canonical_path = entry.shard.clone();
        let mut cache = self.old_lock_cache()?;
        cache.tick = cache.tick.saturating_add(1);
        let tick = cache.tick;
        if let Some(shard) = cache
            .entries
            .get(&canonical_path)
            .map(|entry| Arc::clone(&entry.shard))
        {
            cache.hits = cache.hits.saturating_add(1);
            cache.entries.get_mut(&canonical_path).unwrap().last_used = tick;
            return Ok(shard);
        }
        cache.misses = cache.misses.saturating_add(1);
        if cache.entries.len() >= self.max_cached_shards {
            let victim = cache
                .entries
                .iter()
                .filter(|(_, candidate)| Arc::strong_count(&candidate.shard) == 1)
                .min_by(|(left_path, left), (right_path, right)| {
                    (left.last_used, *left_path).cmp(&(right.last_used, *right_path))
                })
                .map(|(path, _)| path.clone());
            if let Some(victim) = victim {
                cache.entries.remove(&victim);
                cache.evictions = cache.evictions.saturating_add(1);
            } else {
                return Err(StoreError::CapacityExhausted {
                    maximum: self.max_cached_shards,
                    leased: cache
                        .entries
                        .values()
                        .map(|entry| entry.shard.path.clone())
                        .collect(),
                });
            }
        }
        let admission = Arc::clone(self.shards.admission(&canonical_path));
        admission.header(&canonical_path)?;
        let shard = Arc::new(CachedShard {
            path: canonical_path.clone(),
            admitted_file: Arc::clone(&admission.file),
            admission,
            full_tensors: Mutex::new(BTreeMap::new()),
        });
        cache.paths.mark_touched(&entry.shard);
        cache.entries.insert(
            canonical_path,
            CacheEntry {
                shard: Arc::clone(&shard),
                last_used: tick,
            },
        );
        Ok(shard)
    }
    fn old_cached_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let entry = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let metadata = self
            .shards
            .admission(&entry.shard)
            .header(&entry.shard)?
            .tensors
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        self.old_lock_cache()?.paths.mark_touched(&entry.shard);
        Ok(metadata)
    }
    fn old_acquire(&self, request: TensorReadRequest) -> Result<SafetensorsLease, StoreError> {
        let entry = self
            .catalog
            .get(&request.key)
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let shard = self.old_acquire_shard(entry)?;
        let metadata = self.old_cached_metadata(&request.key)?;
        let header = shard.admission.header(&shard.path)?;
        let info = header.metadata.info(&request.key).ok_or_else(|| {
            io_error(
                &entry.shard,
                format!("shard does not contain tensor {:?}", request.key),
            )
        })?;
        let output_shape =
            validate_selection(&request.key, &metadata.logical_shape, &request.selection)?;
        let payload_start = header
            .payload_offset
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("payload start for {:?}", request.key),
            })?;
        let tensor_len = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or_else(|| io_error(&shard.path, "tensor payload offsets descend"))?;
        let read = plan_safetensors_reads(
            &request.key,
            info.dtype,
            &info.shape,
            tensor_len,
            &request.selection,
            &output_shape,
            request.policy,
        )?;
        let cached = shard
            .full_tensors
            .lock()
            .map_err(|_| StoreError::Internal("checkpoint tensor cache is poisoned".into()))?
            .get(&request.key)
            .and_then(Weak::upgrade);
        let complete_tensor =
            read.ranges.len() == 1 && read.ranges[0].start == 0 && read.ranges[0].end == tensor_len;
        let cache_hit = cached.is_some();
        if cache_hit {
            drop(shard.admitted_file.open_validated(&shard.path)?);
        }
        let bytes = match cached {
            Some(bytes) if complete_tensor => bytes,
            Some(bytes) => Arc::new(copy_safetensors_ranges(
                &request.key,
                bytes.as_ref(),
                &read.ranges,
            )?),
            None => {
                let bytes = Arc::new(read_safetensors_ranges(
                    &shard.path,
                    &shard.admitted_file,
                    payload_start,
                    &read.ranges,
                    self.read_telemetry.as_ref(),
                )?);
                if complete_tensor {
                    shard
                        .full_tensors
                        .lock()
                        .map_err(|_| {
                            StoreError::Internal("checkpoint tensor cache is poisoned".into())
                        })?
                        .insert(request.key.clone(), Arc::downgrade(&bytes));
                }
                bytes
            }
        };
        let length = u64::try_from(bytes.len()).map_err(|_| StoreError::Overflow {
            context: format!("physical read length for {:?}", request.key),
        })?;
        self.old_lock_cache()?.paths.mark_payload(&shard.path);
        Ok(SafetensorsLease {
            metadata,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded: read.physically_bounded,
                offset_bytes: u64::try_from(read.ranges[0].start).map_err(|_| {
                    StoreError::Overflow {
                        context: "selection byte offset".into(),
                    }
                })?,
                length_bytes: length,
                physical_reads: if cache_hit {
                    0
                } else {
                    u64::try_from(read.ranges.len()).unwrap_or(u64::MAX)
                },
                physical_read_bytes: if cache_hit { 0 } else { length },
            },
            shard,
            bytes,
        })
    }
}

fn source(maximum: usize) -> (tempfile::TempDir, SafetensorsWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(
        [(
            "left",
            TensorView::new(Dtype::U8, vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
        )],
        None,
        &directory.path().join("left.safetensors"),
    )
    .unwrap();
    serialize_to_file(
        [(
            "right",
            TensorView::new(Dtype::U8, vec![2, 4], &[11, 12, 13, 14, 15, 16, 17, 18]).unwrap(),
        )],
        None,
        &directory.path().join("right.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"left":"left.safetensors","right":"right.safetensors"}}"#,
    )
    .unwrap();
    let store =
        SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), maximum).unwrap();
    // The fixed source loan is genuinely admitted first; this is metadata I/O,
    // not a cached payload acquisition performed by destination preparation.
    store.metadata("left").unwrap();
    store.metadata("right").unwrap();
    (directory, store)
}
fn request(key: &str, selection: TensorSelection) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection,
        policy: ReadPolicy::RequireBounded,
    }
}
fn prepared(
    store: &SafetensorsWeightStore,
    key: &str,
    selection: TensorSelection,
) -> PreparedSafetensorsLease {
    store
        .source_lease_controls(key)
        .unwrap()
        .safetensors_lease_source()
        .unwrap()
        .prepare(selection, ReadPolicy::RequireBounded)
        .unwrap()
}
fn statistics(
    store: &SafetensorsWeightStore,
) -> (u64, u64, u64, usize, u64, u64, Vec<PathBuf>, Vec<PathBuf>) {
    let d = store.diagnostics().unwrap();
    (
        d.cache_hits,
        d.cache_misses,
        d.evictions,
        d.currently_cached_shards,
        d.physical_reads,
        d.physical_read_bytes,
        d.touched_shard_paths,
        d.payload_shard_paths,
    )
}
fn gathered() -> TensorSelection {
    TensorSelection::Indices {
        axis: 1,
        indices: vec![0, 3],
    }
}

#[test]
fn preparation_owns_final_storage_without_acquiring_shards_or_reading_payloads() {
    let (directory, store) = source(1);
    let other = SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), 1).unwrap();
    other.metadata("left").unwrap();
    let before = statistics(&store);
    let destination = prepared(&store, "left", gathered());
    assert_eq!(destination.destination_layout().size(), 4);
    assert_eq!(statistics(&store), before);
    assert!(
        destination.matches_request(
            store
                .source_lease_controls("left")
                .unwrap()
                .safetensors_lease_source()
                .unwrap(),
            &gathered(),
            ReadPolicy::RequireBounded
        )
    );
    assert!(
        !destination.matches_request(
            other
                .source_lease_controls("left")
                .unwrap()
                .safetensors_lease_source()
                .unwrap(),
            &gathered(),
            ReadPolicy::RequireBounded
        )
    );
    assert!(
        !destination.matches_request(
            store
                .source_lease_controls("left")
                .unwrap()
                .safetensors_lease_source()
                .unwrap(),
            &TensorSelection::Full,
            ReadPolicy::RequireBounded
        )
    );
    let pointer = destination.0.bytes.as_ref().unwrap().as_ptr();
    let lease = destination.acquire_checkpoint().unwrap();
    assert_eq!(lease.encoded_bytes().unwrap(), [1, 4, 5, 8]);
    assert_eq!(lease.encoded_bytes().unwrap().as_ptr(), pointer);
    assert_eq!(lease.bounded_read_proof().physical_reads, 3);
    assert_eq!(store.diagnostics().unwrap().currently_cached_shards, 1);
}

#[test]
fn prepared_and_ordinary_leases_share_full_hits_and_actual_payload_publication() {
    let (_directory, store) = source(1);
    let ordinary = store
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    let before = statistics(&store);
    let full = prepared(&store, "left", TensorSelection::Full)
        .acquire()
        .unwrap();
    assert!(Arc::ptr_eq(&full.bytes, &ordinary.bytes));
    assert!(Arc::ptr_eq(&full.shard, &ordinary.shard));
    assert_eq!(full.bounded_read_proof().physical_reads, 0);
    let selected = prepared(&store, "left", gathered()).acquire().unwrap();
    assert_eq!(selected.encoded_bytes().unwrap(), [1, 4, 5, 8]);
    assert_eq!(selected.bounded_read_proof().physical_reads, 0);
    assert!(Arc::ptr_eq(&selected.shard, &ordinary.shard));
    assert_eq!(store.diagnostics().unwrap().physical_reads, before.4);
    drop(selected);
    drop(full);
    drop(ordinary);
    let published = prepared(&store, "left", TensorSelection::Full)
        .acquire()
        .unwrap();
    assert_eq!(published.bounded_read_proof().physical_reads, 1);
    let subsequent = store
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    assert!(Arc::ptr_eq(&published.bytes, &subsequent.bytes));
    assert_eq!(subsequent.bounded_read_proof().physical_reads, 0);
}

#[test]
fn preexisting_ordinary_pin_blocks_prepared_eviction_then_release_allows_same_cache_reuse() {
    let (_directory, store) = source(1);
    let left = store
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    let failure = prepared(&store, "right", TensorSelection::Full)
        .acquire()
        .unwrap_err();
    let StoreError::CapacityExhausted { maximum, leased } = failure.store_error().unwrap() else {
        panic!("pinned shared cache");
    };
    assert_eq!(*maximum, 1);
    assert_eq!(
        leased.as_slice(),
        [left.backing_path().unwrap().to_path_buf()]
    );
    assert!(!failure.retains_shard());
    assert!(!failure.retains_payload());
    assert_eq!(store.diagnostics().unwrap().evictions, 0);
    // Retaining the failed destination must not invent an extra shard pin.
    drop(left);
    let right = prepared(&store, "right", TensorSelection::Full)
        .acquire()
        .unwrap();
    assert_eq!(
        right.encoded_bytes().unwrap(),
        [11, 12, 13, 14, 15, 16, 17, 18]
    );
    assert_eq!(store.diagnostics().unwrap().evictions, 1);
    let ordinary_failure = store
        .acquire(request("left", TensorSelection::Full))
        .unwrap_err();
    assert!(matches!(
        ordinary_failure,
        StoreError::CapacityExhausted { maximum: 1, .. }
    ));
    drop(right);
    drop(failure);
    let left = store
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    assert_eq!(left.encoded_bytes().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(store.diagnostics().unwrap().evictions, 2);
}

#[test]
fn prepared_reordered_and_duplicate_selections_do_not_publish_false_full_payloads() {
    let (_directory, store) = source(1);
    for indices in [vec![0, 2, 3, 1], vec![0, 1, 1, 2]] {
        let selection = TensorSelection::Indices { axis: 1, indices };
        let lease = prepared(&store, "left", selection.clone())
            .acquire()
            .unwrap();
        let expected = store.acquire(request("left", selection)).unwrap();
        assert_eq!(lease.encoded_bytes(), expected.encoded_bytes());
        assert_ne!(lease.encoded_bytes().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(lease.bounded_read_proof().offset_bytes, 0);
        assert_eq!(lease.bounded_read_proof().length_bytes, 8);
        let full = store
            .acquire(request("left", TensorSelection::Full))
            .unwrap();
        assert_eq!(full.encoded_bytes().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(full.bounded_read_proof().physical_reads, 1);
    }
}

#[test]
fn failed_prepared_file_read_retains_final_storage_and_shard_until_error_retirement() {
    let (_directory, store) = source(1);
    let destination = prepared(&store, "left", gathered());
    let path = destination.0.path().clone();
    // Change after genuine metadata preparation, before its first payload read.
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(1)
        .unwrap();
    let failure = destination.acquire().unwrap_err();
    assert!(failure.to_string().contains("changed after preparation"));
    assert_eq!(failure.completed_bytes(), 0);
    assert_eq!(failure.initialized_bytes(), 0);
    assert_eq!(failure.destination(), [0; 4]);
    assert!(failure.retains_file());
    assert!(failure.retains_shard());
    assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
    assert!(matches!(
        store.acquire(request("right", TensorSelection::Full)),
        Err(StoreError::CapacityExhausted { .. })
    ));
    drop(failure);
    let right = store
        .acquire(request("right", TensorSelection::Full))
        .unwrap();
    assert_eq!(
        right.encoded_bytes().unwrap(),
        [11, 12, 13, 14, 15, 16, 17, 18]
    );
}

#[test]
fn ordinary_shared_cache_adapter_matches_untouched_cache_and_acquisition_bodies() {
    let (directory, current) = source(1);
    let old = SafetensorsWeightStore::open_with_max_cached_shards(directory.path(), 1).unwrap();
    old.metadata("left").unwrap();
    old.metadata("right").unwrap();
    let left_a = current
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    let left_b = old
        .old_acquire(request("left", TensorSelection::Full))
        .unwrap();
    assert_eq!(left_a.encoded_bytes(), left_b.encoded_bytes());
    assert_eq!(statistics(&current), statistics(&old));
    let selected_a = current.acquire(request("left", gathered())).unwrap();
    let selected_b = old.old_acquire(request("left", gathered())).unwrap();
    assert_eq!(selected_a.encoded_bytes(), selected_b.encoded_bytes());
    assert_eq!(statistics(&current), statistics(&old));
    let error_a = current
        .acquire(request("right", TensorSelection::Full))
        .unwrap_err();
    let error_b = old
        .old_acquire(request("right", TensorSelection::Full))
        .unwrap_err();
    assert_eq!(error_a.to_string(), error_b.to_string());
    assert_eq!(statistics(&current), statistics(&old));
    drop(selected_a);
    drop(selected_b);
    drop(left_a);
    drop(left_b);
    let right_a = current
        .acquire(request("right", TensorSelection::Full))
        .unwrap();
    let right_b = old
        .old_acquire(request("right", TensorSelection::Full))
        .unwrap();
    assert_eq!(right_a.encoded_bytes(), right_b.encoded_bytes());
    assert_eq!(statistics(&current), statistics(&old));
}

#[test]
fn routed_destination_retains_actual_cache_after_prepared_and_restricted_sources_retire() {
    let (_directory, store) = source(1);
    let leaf = Arc::new(store);
    let weak = Arc::downgrade(&leaf);
    let ordinary = leaf
        .acquire(request("left", TensorSelection::Full))
        .unwrap();
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "selected",
            BTreeSet::from(["left".into()]),
        )
        .unwrap(),
    );
    assert!(restricted.source_lease_controls("right").is_err());
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restricted]).unwrap());
    let catalog = BTreeMap::from([(
        "left".into(),
        PreparedTensorSource {
            metadata: composite.source_metadata("left").unwrap(),
            provenance: composite.source_provenance("left").unwrap(),
        },
    )]);
    let routed = PreparedCheckpointSource::new(composite, catalog).unwrap();
    let loan = routed.source_lease_controls("left").unwrap();
    let before = statistics(&leaf);
    let invalid = loan
        .safetensors_lease_source()
        .unwrap()
        .prepare(
            TensorSelection::Indices {
                axis: 1,
                indices: vec![4],
            },
            ReadPolicy::RequireBounded,
        )
        .unwrap_err();
    assert!(matches!(
        invalid.store_error(),
        Some(StoreError::InvalidSelection { .. })
    ));
    assert!(!invalid.retains_shard());
    assert_eq!(statistics(&leaf), before);
    drop(invalid);
    let destination = loan
        .safetensors_lease_source()
        .unwrap()
        .prepare(TensorSelection::Full, ReadPolicy::RequireBounded)
        .unwrap();
    drop(routed);
    drop(leaf);
    assert!(weak.upgrade().is_none());
    let lease = destination.acquire().unwrap();
    assert!(Arc::ptr_eq(&lease.shard, &ordinary.shard));
    assert!(Arc::ptr_eq(&lease.bytes, &ordinary.bytes));
    drop(ordinary);
    assert_eq!(lease.encoded_bytes().unwrap(), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(lease.bounded_read_proof().physical_reads, 0);
}
