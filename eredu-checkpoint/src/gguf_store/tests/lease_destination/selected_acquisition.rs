use super::*;
use crate::store::{
    PreparedAcquisitionBank, PreparedAcquisitionBankError, SelectedGgufConversionPlan,
};

fn capacity(actual: usize, requested: usize) {
    assert!(actual >= requested);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_G4").is_some() {
        assert_eq!(
            actual, requested,
            "pinned positive validation must exercise exact fresh-clone capacities"
        );
    }
}
#[test]
fn selected_dense_tickets_copy_lengths_preserve_final_box_and_do_no_io() {
    let (_directory, store) = dense_source();
    // Construct the ordinary fixture's immutable catalog before publishing it.
    // Deliberate source spare capacity must not become the requested copy size.
    let mut entry = store.inner.catalog.get("matrix.weight").unwrap().clone();
    entry.metadata.name.reserve(4096);
    entry.metadata.logical_shape.reserve(256);
    entry.physical_descriptor.dimensions.reserve(256);
    let mut rows = crate::gguf_store::catalog::CatalogRows::default();
    rows.insert("matrix.weight".into(), entry);
    let inner = store.inner.into_ordinary();
    let store = GgufWeightStore {
        inner: crate::store::SourceHandle::new(
            StoreInner {
                catalog: crate::gguf_store::catalog::CatalogHandle::new(
                    crate::gguf_store::catalog::CatalogData {
                        rows,
                        unclaimed: Default::default(),
                    },
                    (),
                ),
                ..inner
            },
            None,
        ),
    };
    let root: SharedCheckpointSource = Arc::new(store.clone());
    for selection in [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        TensorSelection::Contiguous {
            offset_elements: 2,
            shape: vec![2, 2],
        },
    ] {
        let mut read = request("matrix.weight", selection, ReadPolicy::RequireBounded);
        read.key.reserve(2048);
        let before = stats(&store);
        let plan = SelectedGgufConversionPlan::query(root.clone(), read.clone())
            .unwrap()
            .unwrap();
        let layout = plan.acquisition_storage().unwrap().unwrap();
        let old = store.old_acquire(read.clone()).unwrap();
        let prepared = Prepared::prepare_plan(&store, &read, plan.physical()).unwrap();
        let address = std::ptr::from_ref(prepared.lease());
        let CheckpointLease::Gguf(lease) = prepared.into_lease() else {
            panic!("GGUF")
        };
        assert_eq!(address, std::ptr::from_ref(lease.as_ref()));
        same_lease(&old, &lease);
        let mut bank = PreparedAcquisitionBank::new(2).unwrap();
        bank.prepare_selected_gguf(&plan).unwrap();
        bank.prepare_selected_gguf(&plan).unwrap();
        capacity(
            bank.storage().unwrap().owned_payload_capacity_bytes,
            2 * layout.retained_bytes(),
        );
        assert!(layout.retained_bytes() < 4096);
        bank.seal().unwrap();
        let CheckpointLease::Gguf(actual) = bank
            .acquire(root.as_ref(), &read.key, &read.selection, read.policy)
            .unwrap()
        else {
            panic!("GGUF")
        };
        same_lease(&old, &actual);
        assert_eq!(bank.storage().unwrap().remaining, 1);
        assert_eq!(stats(&store), before);
        assert_eq!(
            actual.materialize_portable().unwrap(),
            old.materialize_portable().unwrap()
        );
    }
}

#[test]
fn selected_affine_and_mxfp4_tickets_reuse_actual_companion_read_plans() {
    for ty in [GgmlType::Q4_0, GgmlType::MxFp4] {
        let (_directory, store) = packed_source(ty);
        let root: SharedCheckpointSource = Arc::new(store.clone());
        for key in store.keys() {
            let units = store.inner.catalog[&key]
                .logical_last_units_per_block
                .unwrap();
            let read = request(
                &key,
                TensorSelection::Contiguous {
                    offset_elements: units,
                    shape: vec![1, units],
                },
                ReadPolicy::RequireBounded,
            );
            let before = stats(&store);
            let plan = SelectedGgufConversionPlan::query(root.clone(), read.clone())
                .unwrap()
                .unwrap();
            let old = store.old_acquire(read.clone()).unwrap();
            let mut bank = PreparedAcquisitionBank::new(1).unwrap();
            bank.prepare_selected_gguf(&plan).unwrap();
            capacity(
                bank.storage().unwrap().owned_payload_capacity_bytes,
                plan.acquisition_storage()
                    .unwrap()
                    .unwrap()
                    .retained_bytes(),
            );
            bank.seal().unwrap();
            let CheckpointLease::Gguf(actual) = bank
                .acquire(root.as_ref(), &key, &read.selection, read.policy)
                .unwrap()
            else {
                panic!("GGUF")
            };
            same_lease(&old, &actual);
            assert_eq!(stats(&store), before);
            assert_eq!(
                actual.materialize_portable().unwrap(),
                old.materialize_portable().unwrap()
            );
        }
    }
}

#[test]
fn custom_root_and_hidden_child_remain_unknown_without_qualifying_callbacks() {
    let (directory, first) = dense_source();
    let second = test_store(&directory.path().join("dense.gguf"));
    let switching = Arc::new(Switching {
        first,
        second,
        switched: AtomicBool::new(false),
    });
    let direct: SharedCheckpointSource = switching.clone();
    let wrapped: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            direct.clone(),
            "original contract",
            BTreeSet::from(["matrix.weight".into()]),
        )
        .unwrap(),
    );
    let read = request(
        "matrix.weight",
        TensorSelection::Full,
        ReadPolicy::RequireBounded,
    );
    for root in [direct, wrapped] {
        let plan = SelectedGgufConversionPlan::query(root.clone(), read.clone())
            .unwrap()
            .unwrap();
        assert!(plan.acquisition_storage().unwrap().is_none());
        assert!(plan.prepare_acquisition().unwrap().is_none());
        let genuine_root: SharedCheckpointSource = Arc::new(switching.first.clone());
        let genuine = SelectedGgufConversionPlan::query(genuine_root.clone(), read.clone())
            .unwrap()
            .unwrap();
        let weak = Arc::downgrade(&genuine_root);
        let mut selected = PreparedAcquisitionBank::new(2).unwrap();
        selected.prepare_selected_gguf(&genuine).unwrap();
        drop((genuine, genuine_root));
        assert!(matches!(
            selected.prepare_selected_gguf(&plan),
            Err(PreparedAcquisitionBankError::Unavailable)
        ));
        assert_eq!(selected.storage().unwrap().remaining, 1);
        assert!(
            weak.upgrade().is_some(),
            "successful prefix remains owned after later refusal"
        );
        drop(selected);
        assert!(weak.upgrade().is_none());
        // Ordinary preparation retains its original dynamic recheck and cause.
        let mut ordinary = PreparedAcquisitionBank::new(1).unwrap();
        ordinary.prepare_next(root.clone(), read.clone()).unwrap();
        ordinary.seal().unwrap();
        switching.switched.store(true, Ordering::Relaxed);
        let error = ordinary
            .acquire(root.as_ref(), &read.key, &read.selection, read.policy)
            .unwrap_err();
        assert!(matches!(
            error,
            PreparedAcquisitionBankError::Acquisition(_)
        ));
        assert!(error.to_string().contains("source changed"));
        switching.switched.store(false, Ordering::Relaxed);
    }
    assert_eq!(stats(&switching.first).4, 0);
    assert_eq!(stats(&switching.second).4, 0);
}

#[test]
fn selected_concrete_wrappers_retain_authority_and_sources_through_final_ticket() {
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
    let plan = SelectedGgufConversionPlan::query(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(plan.acquisition_storage().unwrap().is_some());
    assert!(plan.acquisition_route_shared_layout().is_some());
    let ticket = plan.prepare_acquisition().unwrap().unwrap();
    assert!(ticket.matches_request(&root, &read));
    let expected = store.old_acquire(read.clone()).unwrap();
    for denied in [root.clone(), restricted.clone()] {
        let bad = request("absent", TensorSelection::Full, ReadPolicy::RequireBounded);
        let old = denied.acquire_lease(bad.clone()).unwrap_err();
        let error = SelectedGgufConversionPlan::query(denied, bad).unwrap_err();
        let GgufConversionPlanError::Store(error) = error else {
            panic!("actual wrapper error")
        };
        assert_eq!(old.to_string(), error.to_string());
    }
    drop((plan, root, leaf, restricted, store, checkpoint));
    assert!(weak.upgrade().is_some());
    let CheckpointLease::Gguf(lease) = ticket.acquire().unwrap() else {
        panic!("GGUF")
    };
    same_lease(&expected, &lease);
    drop(expected);
    assert!(weak.upgrade().is_none());
    assert_eq!(
        lease
            .store
            .statistics
            .physical_reads
            .load(Ordering::Relaxed),
        0
    );
    drop(lease);
    assert!(inner.upgrade().is_none());
}

struct ForwardedOwner {
    source: Arc<GgufWeightStore>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl CheckpointSource for ForwardedOwner {
    fn prepared_acquisition_owner(
        self: Arc<Self>,
    ) -> Option<crate::store::PreparedAcquisitionOwner> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.source.clone().prepared_acquisition_owner()
    }
    fn prepared_acquisition_source(&self) -> crate::store::PreparedAcquisitionSource<'_> {
        self.source.prepared_acquisition_source()
    }
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("no fallback")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("no diagnostics")
    }
}
#[test]
fn forwarding_a_concrete_owner_does_not_certify_custom_root_or_child() {
    let (_directory, store) = dense_source();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let custom: SharedCheckpointSource = Arc::new(ForwardedOwner {
        source: Arc::new(store.clone()),
        calls: calls.clone(),
    });
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            custom.clone(),
            "selected",
            BTreeSet::from(["matrix.weight".into()]),
        )
        .unwrap(),
    );
    for root in [custom, restricted] {
        let plan = SelectedGgufConversionPlan::query(
            root,
            request(
                "matrix.weight",
                TensorSelection::Full,
                ReadPolicy::RequireBounded,
            ),
        )
        .unwrap()
        .unwrap();
        let cold = calls.load(Ordering::Relaxed);
        assert!(cold > 0);
        assert!(plan.acquisition_storage().unwrap().is_none());
        assert!(plan.prepare_acquisition().unwrap().is_none());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            cold,
            "unknown companion never reenters provider"
        );
    }
    assert_eq!(stats(&store).4, 0);
}

struct ChangingCache {
    source: GgufWeightStore,
    present: AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
}
impl CheckpointSource for ChangingCache {
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.present
            .load(Ordering::Relaxed)
            .then_some(&self.source.inner.recipes)
    }
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<crate::store::TensorSourceProvenance, StoreError> {
        self.source.source_provenance(key)
    }
    fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        panic!("no acquisition")
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        panic!("no diagnostics")
    }
}
#[test]
fn off_route_custom_cache_dependency_keeps_prepared_post_validation_unknown() {
    let (directory, store) = dense_source();
    let path = directory.path().join("other.gguf");
    write_tensor(
        &path,
        "other.weight",
        &[1, 1],
        GgmlType::F32,
        &1.25f32.to_le_bytes(),
    );
    let unrelated = Arc::new(ChangingCache {
        source: test_store(&path),
        present: AtomicBool::new(true),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let children: Vec<SharedCheckpointSource> = vec![Arc::new(store.clone()), unrelated.clone()];
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new(children).unwrap());
    let root: SharedCheckpointSource = Arc::new(snapshot(composite));
    let read = request(
        "matrix.weight",
        TensorSelection::Full,
        ReadPolicy::RequireBounded,
    );
    let calls = unrelated.calls.load(Ordering::Relaxed);
    let plan = SelectedGgufConversionPlan::query(root, read)
        .unwrap()
        .unwrap();
    assert!(plan.acquisition_storage().unwrap().is_none());
    unrelated.present.store(false, Ordering::Relaxed);
    assert!(plan.prepare_acquisition().unwrap().is_none());
    assert_eq!(
        unrelated.calls.load(Ordering::Relaxed),
        calls,
        "custom cache is not used as stable proof"
    );
    assert_eq!(stats(&store).4, 0);
}

#[test]
fn off_route_builtin_memory_and_safetensors_keep_gguf_companion_known() {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    let (directory, store) = dense_source();
    let path = directory.path().join("other.safetensors");
    serialize_to_file(
        [(
            "safe.weight",
            TensorView::new(Dtype::U8, vec![3], &[2, 4, 6]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let memory: SharedCheckpointSource = Arc::new(
        crate::store::MemoryWeightStore::from_safetensors([(
            "memory.weight".into(),
            Dtype::U8,
            vec![3],
            vec![1, 3, 5],
        )])
        .unwrap(),
    );
    let safe: SharedCheckpointSource =
        Arc::new(crate::store::SafetensorsWeightStore::open(path).unwrap());
    let children: Vec<SharedCheckpointSource> = vec![Arc::new(store.clone()), memory, safe];
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new(children).unwrap());
    let root: SharedCheckpointSource = Arc::new(snapshot(composite));
    let read = request(
        "matrix.weight",
        TensorSelection::Full,
        ReadPolicy::RequireBounded,
    );
    let plan = SelectedGgufConversionPlan::query(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(plan.acquisition_storage().unwrap().is_some());
    let ticket = plan.prepare_acquisition().unwrap().unwrap();
    let CheckpointLease::Gguf(lease) = ticket.acquire().unwrap() else {
        panic!("GGUF");
    };
    same_lease(&store.old_acquire(read).unwrap(), &lease);
    for key in ["memory.weight", "safe.weight"] {
        assert!(SelectedGgufConversionPlan::query(
            root.clone(),
            request(key, TensorSelection::Full, ReadPolicy::RequireBounded)
        )
        .unwrap()
        .is_none());
    }
    assert_eq!(stats(&store).4, 0);
}
