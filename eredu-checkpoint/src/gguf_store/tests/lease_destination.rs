use super::*;
use crate::gguf_store::lease_destination::Prepared;
use crate::store::{
    CompositeCheckpointSource, PreparedCheckpointAcquisition, PreparedCheckpointSource,
    PreparedTensorSource, ResolvedCheckpointSource, RestrictedCheckpointSource,
    SharedCheckpointSource,
};
use std::sync::atomic::AtomicBool;

impl GgufWeightStore {
    fn old_acquire(&self, request: TensorReadRequest) -> Result<GgufLease, StoreError> {
        let entry = self
            .inner
            .catalog
            .get(&request.key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let output_shape = validate_selection(
            &request.key,
            &entry.metadata.logical_shape,
            &request.selection,
        )?;
        let read = match plan_bounded_selection(&request.key, &entry, &request.selection) {
            Ok(plan) => plan,
            Err(StoreError::BoundedSelectionUnavailable { .. })
                if request.policy == ReadPolicy::AllowFullTensorRead =>
            {
                ReadPlan {
                    physical_selection: None,
                    physical_offset: 0,
                    physical_byte_len: entry.physical_descriptor.byte_len,
                    selection_is_materialized: false,
                }
            }
            Err(error) => return Err(error),
        };
        let physically_bounded =
            matches!(request.selection, TensorSelection::Full) || read.physical_selection.is_some();
        Ok(GgufLease {
            store: self.inner.clone(),
            identity: GgufLeaseIdentity {
                source: self.inner.identity(),
                checkpoint: entry.checkpoint,
                physical_name: entry.physical_name.clone(),
                selection: read.physical_selection,
            },
            entry,
            selection: request.selection,
            output_shape,
            proof: BoundedReadProof {
                physically_bounded,
                offset_bytes: read.physical_offset,
                length_bytes: read.physical_byte_len,
                physical_reads: 1,
                physical_read_bytes: read.physical_byte_len,
            },
            selection_is_materialized: read.selection_is_materialized,
        })
    }
}

fn request(key: &str, selection: TensorSelection, policy: ReadPolicy) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection,
        policy,
    }
}
fn proof(lease: &GgufLease) -> (bool, u64, u64, u64, u64) {
    let p = lease.bounded_read_proof();
    (
        p.physically_bounded,
        p.offset_bytes,
        p.length_bytes,
        p.physical_reads,
        p.physical_read_bytes,
    )
}
fn stats(store: &GgufWeightStore) -> (u64, u64, u64, usize, u64, u64) {
    let d = store.diagnostics().unwrap();
    (
        d.cache_hits,
        d.cache_misses,
        d.evictions,
        d.currently_cached_shards,
        d.physical_reads,
        d.physical_read_bytes,
    )
}
fn same_lease(a: &GgufLease, b: &GgufLease) {
    assert_eq!(a.metadata(), b.metadata());
    assert_eq!(a.selection(), b.selection());
    assert_eq!(a.output_shape(), b.output_shape());
    assert_eq!(a.backing_path(), b.backing_path());
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.selection_is_materialized(), b.selection_is_materialized());
    assert_eq!(proof(a), proof(b));
    assert_eq!(a.encoded_bytes(), None);
    assert!(a.store.same(&b.store));
}
fn dense_source() -> (tempfile::TempDir, GgufWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.gguf");
    let values = [1.25_f32, -2.5, 3.75, -4.0, 5.5, -6.25, 7.0, 8.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_tensor(&path, "matrix.weight", &[4, 2], GgmlType::F32, &values);
    let store = test_store(&path);
    (directory, store)
}

#[test]
fn dense_physical_selectors_match_old_lazy_lease_and_reuse_the_prepared_box() {
    let (_directory, store) = dense_source();
    for policy in [ReadPolicy::RequireBounded, ReadPolicy::AllowFullTensorRead] {
        for selection in [
            TensorSelection::Full,
            TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![3, 1, 3],
            },
            TensorSelection::Contiguous {
                offset_elements: 1,
                shape: vec![2, 2],
            },
        ] {
            let read = request("matrix.weight", selection, policy);
            let before = stats(&store);
            let old = store.old_acquire(read.clone()).unwrap();
            let ordinary = store.acquire(read.clone()).unwrap();
            let prepared = Prepared::prepare(&store, &read).unwrap();
            assert!(prepared.matches(&store, &read));
            let pointer = std::ptr::from_ref(prepared.lease());
            same_lease(&old, prepared.lease());
            same_lease(&old, &ordinary);
            let CheckpointLease::Gguf(lease) = prepared.into_lease() else {
                panic!("actual GGUF")
            };
            assert_eq!(pointer, std::ptr::from_ref(lease.as_ref()));
            assert_eq!(
                stats(&store),
                before,
                "lease creation remains metadata-only"
            );
            let expected = old.materialize_portable().unwrap();
            assert_eq!(lease.materialize_portable().unwrap(), expected);
            assert_eq!(ordinary.materialize_portable().unwrap(), expected);
            assert_eq!(stats(&store).4, before.4 + 3);
        }
    }
}

fn packed_source(ty: GgmlType) -> (tempfile::TempDir, GgufWeightStore) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("packed.gguf");
    let (_, size) = ty.block_and_bytes().unwrap();
    let size = size as usize;
    let raw = (0..4)
        .flat_map(|block| {
            let mut bytes = vec![0u8; size];
            if ty == GgmlType::MxFp4 {
                bytes[0] = 127;
                for (i, b) in bytes.iter_mut().enumerate().skip(1) {
                    *b = ((i + block) % 15 + 1) as u8;
                }
            } else {
                bytes[0] = 0;
                bytes[1] = 60;
                for (i, b) in bytes.iter_mut().enumerate().skip(2) {
                    *b = ((i + block) % 63 + 1) as u8;
                }
            }
            bytes
        })
        .collect::<Vec<_>>();
    write_tensor(&path, "matrix.weight", &[64, 2], ty, &raw);
    let store = test_store(&path);
    (directory, store)
}
#[test]
fn native_affine_and_mxfp4_outputs_reuse_real_block_mappings_and_conversion_groups() {
    for ty in [GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::MxFp4] {
        let (_directory, store) = packed_source(ty);
        // Actual logical catalog includes quantization companion outputs.
        for key in store.keys() {
            let entry = &store.inner.catalog[&key];
            let units = entry.logical_last_units_per_block.unwrap();
            let last = *entry.metadata.logical_shape.last().unwrap();
            for selection in [
                TensorSelection::Full,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                },
                TensorSelection::Range {
                    axis: 1,
                    start: 0,
                    end: units,
                },
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![1, 0, 1],
                },
                TensorSelection::Indices {
                    axis: 1,
                    indices: (last - units..last).chain(0..units).collect(),
                },
                TensorSelection::Contiguous {
                    offset_elements: units,
                    shape: vec![1, units],
                },
            ] {
                let read = request(&key, selection, ReadPolicy::RequireBounded);
                let before = stats(&store);
                let old = store.old_acquire(read.clone()).unwrap();
                let prepared = Prepared::prepare(&store, &read).unwrap();
                same_lease(&old, prepared.lease());
                assert_eq!(stats(&store), before);
                let CheckpointLease::Gguf(lease) = prepared.into_lease() else {
                    panic!("GGUF")
                };
                let expected = old.materialize_portable().unwrap();
                let actual = lease.materialize_portable().unwrap();
                assert_eq!(actual, expected, "{ty:?} {read:?}");
                assert_eq!(actual.output_names(), expected.output_names());
                assert_eq!(lease.logical_output_name(), entry.original_name);
            }
        }
    }
}

#[test]
fn allow_full_fallback_matches_only_the_original_bounded_selection_failure() {
    let (_directory, store) = packed_source(GgmlType::Q4_0);
    let key = "matrix.weight";
    let selection = TensorSelection::Range {
        axis: 1,
        start: 1,
        end: 2,
    };
    let bounded = request(key, selection.clone(), ReadPolicy::RequireBounded);
    let old = store.old_acquire(bounded.clone()).unwrap_err();
    let fixed = Prepared::prepare(&store, &bounded).unwrap_err();
    assert!(matches!(
        old,
        StoreError::BoundedSelectionUnavailable { .. }
    ));
    assert_eq!(old.to_string(), fixed.to_string());
    let allow = request(key, selection, ReadPolicy::AllowFullTensorRead);
    let before = stats(&store);
    let old = store.old_acquire(allow.clone()).unwrap();
    let prepared = Prepared::prepare(&store, &allow).unwrap();
    same_lease(&old, prepared.lease());
    assert_eq!(stats(&store), before);
    assert!(!prepared.lease().selection_is_materialized());
    assert!(!prepared.lease().bounded_read_proof().physically_bounded);
    assert!(prepared.lease().identity().physical_selection().is_none());
    let CheckpointLease::Gguf(lease) = prepared.into_lease() else {
        panic!("GGUF")
    };
    assert_eq!(
        lease.materialize_portable().unwrap(),
        old.materialize_portable().unwrap()
    );
    for read in [
        request(
            key,
            TensorSelection::Range {
                axis: 4,
                start: 0,
                end: 1,
            },
            ReadPolicy::AllowFullTensorRead,
        ),
        request(
            key,
            TensorSelection::Indices {
                axis: 0,
                indices: vec![],
            },
            ReadPolicy::AllowFullTensorRead,
        ),
        request(
            "absent",
            TensorSelection::Full,
            ReadPolicy::AllowFullTensorRead,
        ),
    ] {
        let before = stats(&store);
        let old = store.old_acquire(read.clone()).unwrap_err();
        let fixed = Prepared::prepare(&store, &read).unwrap_err();
        assert!(!matches!(
            old,
            StoreError::BoundedSelectionUnavailable { .. }
        ));
        assert_eq!(old.to_string(), fixed.to_string());
        assert_eq!(stats(&store), before);
    }
}

fn snapshot(source: SharedCheckpointSource) -> PreparedCheckpointSource {
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let row = PreparedTensorSource {
                metadata: source.source_metadata(&key).unwrap(),
                provenance: source.source_provenance(&key).unwrap(),
            };
            (key, row)
        })
        .collect();
    PreparedCheckpointSource::new(source, catalog).unwrap()
}
#[test]
fn closed_gguf_route_keeps_authorization_and_lazy_store_after_all_source_views_retire() {
    let (directory, store) = dense_source();
    let inner = store.inner.ordinary_weak();
    let leaf: SharedCheckpointSource = Arc::new(store.clone());
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "selected",
            BTreeSet::from(["matrix.weight".into()]),
        )
        .unwrap(),
    );
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restricted.clone()]).unwrap());
    let prepared: SharedCheckpointSource = Arc::new(snapshot(composite));
    let checkpoint = Checkpoint::open(directory.path().join("dense.gguf")).unwrap();
    let resolved = resolve_gguf_plan(&checkpoint, &test_plan(&checkpoint)).unwrap();
    let root: SharedCheckpointSource = Arc::new(ResolvedCheckpointSource::new(prepared, resolved));
    let weak = Arc::downgrade(&root);
    let read = request(
        "matrix.weight",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0],
        },
        ReadPolicy::RequireBounded,
    );
    let destination = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(destination.matches_request(&root, &read));
    assert_eq!(stats(&store).4, 0);
    for denied in [root.clone(), restricted.clone()] {
        let bad = request("absent", TensorSelection::Full, ReadPolicy::RequireBounded);
        let old = denied.acquire_lease(bad.clone()).unwrap_err();
        let error = PreparedCheckpointAcquisition::prepare(denied, bad).unwrap_err();
        assert_eq!(old.to_string(), error.to_string());
    }
    drop((leaf, restricted, root, store, checkpoint));
    assert!(weak.upgrade().is_some());
    let CheckpointLease::Gguf(lease) = destination.acquire().unwrap() else {
        panic!("GGUF")
    };
    assert!(weak.upgrade().is_none());
    assert!(inner.upgrade().is_some());
    assert_eq!(
        lease
            .store
            .statistics
            .physical_reads
            .load(Ordering::Relaxed),
        0
    );
    let ConvertedTensor::Dense(converted) = lease.materialize_portable().unwrap().into_converted()
    else {
        panic!("dense")
    };
    assert_eq!(converted.shape, [2, 4]);
    let values = converted
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, [5.5, -6.25, 7.0, 8.5, 1.25, -2.5, 3.75, -4.0]);
    drop(lease);
    assert!(inner.upgrade().is_none());
}

struct Switching {
    first: GgufWeightStore,
    second: GgufWeightStore,
    switched: AtomicBool,
}
impl CheckpointSource for Switching {
    fn prepared_acquisition_source(&self) -> crate::store::PreparedAcquisitionSource<'_> {
        if self.switched.load(Ordering::Relaxed) {
            self.second.prepared_acquisition_source()
        } else {
            self.first.prepared_acquisition_source()
        }
    }
    fn source_keys(&self) -> Vec<String> {
        self.first.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.first.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("no ordinary fallback")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("no diagnostics fallback")
    }
}
#[test]
fn same_file_foreign_store_refuses_the_unconsumed_box_and_retains_source_until_failure_drop() {
    let (directory, first) = dense_source();
    let second = test_store(&directory.path().join("dense.gguf"));
    let read = request(
        "matrix.weight",
        TensorSelection::Full,
        ReadPolicy::RequireBounded,
    );
    let direct = Prepared::prepare(&first, &read).unwrap();
    assert!(!direct.matches(&second, &read));
    assert!(direct.matches(&first.clone(), &read));
    drop(direct);
    let inner = first.inner.ordinary_weak();
    let owner = Arc::new(Switching {
        first,
        second,
        switched: AtomicBool::new(false),
    });
    let weak = Arc::downgrade(&owner);
    let root: SharedCheckpointSource = owner.clone();
    let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    owner.switched.store(true, Ordering::Relaxed);
    let error = prepared.acquire().unwrap_err();
    assert_eq!(
        error.to_string(),
        "prepared checkpoint acquisition source changed"
    );
    assert!(error.matches_request(&root, &read));
    assert!(!error.retains_rejected_lease());
    assert_eq!(stats(&owner.first).4, 0);
    assert_eq!(stats(&owner.second).4, 0);
    drop((owner, root));
    assert!(weak.upgrade().is_some());
    assert!(inner.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
    assert!(inner.upgrade().is_none());
}

#[test]
fn prepared_box_preserves_deferred_reopen_failure_and_never_reads_during_acquisition() {
    let (directory, store) = dense_source();
    let read = request(
        "matrix.weight",
        TensorSelection::Full,
        ReadPolicy::RequireBounded,
    );
    let old = store.old_acquire(read.clone()).unwrap();
    let prepared = Prepared::prepare(&store, &read).unwrap();
    let inner = store.inner.ordinary_weak();
    File::options()
        .write(true)
        .open(directory.path().join("dense.gguf"))
        .unwrap()
        .set_len(1)
        .unwrap();
    let CheckpointLease::Gguf(lease) = prepared.into_lease() else {
        panic!("GGUF")
    };
    assert_eq!(stats(&store).4, 0);
    same_lease(&old, &lease);
    drop(store);
    let expected = old.materialize_portable().unwrap_err();
    let actual = lease.materialize_portable().unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(
        lease
            .store
            .statistics
            .physical_reads
            .load(Ordering::Relaxed),
        0
    );
    assert!(inner.upgrade().is_some());
    drop((old, lease));
    assert!(inner.upgrade().is_none());
}

#[path = "lease_destination/selected_acquisition.rs"]
mod selected_acquisition;
