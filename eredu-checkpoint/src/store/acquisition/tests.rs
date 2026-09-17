use super::*;
use crate::{schema::*, validation::resolve_safetensors_plan};
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use std::sync::atomic::{AtomicBool, Ordering};

fn source() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "selected",
                TensorView::new(Dtype::U8, vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap(),
            ),
            (
                "hidden",
                TensorView::new(Dtype::U8, vec![2, 3], &[9, 8, 7, 6, 5, 4]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(path).unwrap());
    (directory, source)
}
fn request(key: &str) -> TensorReadRequest {
    TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        policy: ReadPolicy::RequireBounded,
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
fn resolved(source: SharedCheckpointSource) -> SharedCheckpointSource {
    let plan = SafetensorsCheckpointPlan::new(
        "selected contract",
        vec![SafetensorsTensorConstraint::required(
            "selected",
            vec![2, 3],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let plan = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
    Arc::new(ResolvedCheckpointSource::new(source, plan))
}
// Untouched pre-extraction body: independent ordinary post-check/order oracle.
fn old_prepared_acquire(
    owner: &PreparedCheckpointSource,
    request: TensorReadRequest,
) -> Result<CheckpointLease, StoreError> {
    owner.expected(&request.key)?;
    let lease = owner.source.acquire_lease(request.clone())?;
    if owner.source.recipe_cache().is_none() {
        owner.validate_lease(&request, &lease)?;
    }
    Ok(lease)
}

#[test]
fn actual_nested_route_preserves_selected_bytes_request_identity_and_source_retirement() {
    let (_directory, leaf) = source();
    let selected: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "kept",
            BTreeSet::from(["selected".into()]),
        )
        .unwrap(),
    );
    let hidden: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "other",
            BTreeSet::from(["hidden".into()]),
        )
        .unwrap(),
    );
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([selected, hidden]).unwrap());
    let root = resolved(Arc::new(snapshot(composite)));
    let weak = Arc::downgrade(&root);
    let read = request("selected");
    let before = leaf.diagnostics().unwrap();
    let destination = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert!(destination.matches_request(&root, &read));
    let foreign: SharedCheckpointSource = Arc::new(snapshot(leaf.clone()));
    assert!(!destination.matches_request(&foreign, &read));
    for changed in [
        TensorReadRequest {
            selection: TensorSelection::Full,
            ..read.clone()
        },
        TensorReadRequest {
            policy: ReadPolicy::AllowFullTensorRead,
            ..read.clone()
        },
        request("hidden"),
    ] {
        assert!(!destination.matches_request(&root, &changed));
    }
    let after = leaf.diagnostics().unwrap();
    assert_eq!(
        after.currently_cached_shards,
        before.currently_cached_shards
    );
    assert_eq!(after.physical_reads, before.physical_reads);
    drop(root);
    assert!(weak.upgrade().is_some());
    let lease = destination.acquire().unwrap();
    assert!(weak.upgrade().is_none());
    assert_eq!(lease.encoded_bytes().unwrap(), [4, 5, 6, 1, 2, 3, 4, 5, 6]);
    let ordinary = foreign.acquire_lease(read).unwrap();
    assert_eq!(lease.metadata(), ordinary.metadata());
    assert_eq!(lease.selection(), ordinary.selection());
    assert_eq!(lease.output_shape(), ordinary.output_shape());
    assert_eq!(lease.backing_path(), ordinary.backing_path());
    assert_eq!(lease.encoded_bytes(), ordinary.encoded_bytes());
    assert_eq!(lease.bounded_read_proof().length_bytes, 9);
    assert!(lease.bounded_read_proof().physically_bounded);
    drop((ordinary, foreign, leaf));
    assert_eq!(lease.encoded_bytes().unwrap(), [4, 5, 6, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn authorized_wrappers_preserve_unknown_and_denied_first_errors_without_payload_access() {
    let (_directory, leaf) = source();
    let restriction: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "only selected",
            BTreeSet::from(["selected".into()]),
        )
        .unwrap(),
    );
    let prepared: SharedCheckpointSource = Arc::new(snapshot(restriction.clone()));
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restriction.clone()]).unwrap());
    let resolved = resolved(leaf.clone());
    for (root, key, denied) in [
        (restriction.clone(), "hidden", true),
        (restriction, "absent", true),
        (prepared, "hidden", false),
        (composite, "absent", false),
        (resolved.clone(), "hidden", true),
        (resolved, "absent", true),
    ] {
        let read = request(key);
        let ordinary = root.acquire_lease(read.clone()).unwrap_err();
        let fixed = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone()).unwrap_err();
        assert_eq!(fixed.to_string(), ordinary.to_string());
        assert_eq!(
            matches!(
                fixed.store_error(),
                Some(StoreError::UnauthorizedTensor { .. })
            ),
            denied
        );
        assert!(fixed.matches_request(&root, &read));
        assert!(!fixed.retains_rejected_lease());
    }
    assert_eq!(leaf.diagnostics().unwrap().physical_reads, 0);
    assert_eq!(leaf.diagnostics().unwrap().currently_cached_shards, 0);
}

// A real borrowed leaf route with no catalog-cache promise. Ordinary acquisition
// delegates unchanged. A deliberately inconsistent catalog tests its enclosing
// Prepared source's actual required validation, not an invented fit authority.
struct Uncached {
    leaf: Arc<SafetensorsWeightStore>,
    inconsistent: bool,
    materialized: BTreeSet<String>,
    events: Arc<Mutex<Vec<(&'static str, usize)>>>,
}
impl CheckpointSource for Uncached {
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        self.leaf.prepared_acquisition_source()
    }
    fn source_keys(&self) -> Vec<String> {
        self.leaf.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let mut metadata = self.leaf.source_metadata(key)?;
        if self.inconsistent {
            metadata.encoded_byte_len += 1;
        }
        Ok(metadata)
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.leaf.source_provenance(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.leaf.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.leaf.source_diagnostics()
    }
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        self.events.lock().unwrap().push((
            "post",
            self.leaf.diagnostics().unwrap().currently_cached_shards,
        ));
        None
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.events.lock().unwrap().push((
            "authority",
            self.leaf.diagnostics().unwrap().currently_cached_shards,
        ));
        self.materialized.contains(key)
    }
}

#[test]
fn actual_uncached_post_validation_matches_old_body_and_retains_rejected_lease() {
    for inconsistent in [false, true] {
        let (_directory, leaf) = source();
        let events = Arc::new(Mutex::new(Vec::new()));
        let owner = Arc::new(snapshot(Arc::new(Uncached {
            leaf: leaf.clone(),
            inconsistent,
            materialized: BTreeSet::new(),
            events: events.clone(),
        })));
        let read = request("selected");
        events.lock().unwrap().clear();
        let old = old_prepared_acquire(&owner, read.clone());
        let old_events = events.lock().unwrap().drain(..).collect::<Vec<_>>();
        let ordinary = owner.acquire_lease(read.clone());
        assert_eq!(*events.lock().unwrap(), old_events);
        events.lock().unwrap().clear();
        let root: SharedCheckpointSource = owner.clone();
        let weak = Arc::downgrade(&owner);
        let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
            .unwrap()
            .unwrap();
        assert!(
            events.lock().unwrap().is_empty(),
            "post-validation is not a preparation callback"
        );
        let fixed = prepared.acquire();
        assert_eq!(*events.lock().unwrap(), [("post", 1)]);
        if inconsistent {
            let old = old.unwrap_err();
            assert_eq!(ordinary.unwrap_err().to_string(), old.to_string());
            let failed = fixed.unwrap_err();
            assert_eq!(failed.to_string(), old.to_string());
            assert!(matches!(
                failed.store_error(),
                Some(StoreError::PreparedCatalogMismatch { .. })
            ));
            assert!(failed.retains_rejected_lease());
            assert!(failed.matches_request(&root, &read));
            let CheckpointLease::Safetensors(lease) = failed.rejected.as_ref().unwrap() else {
                panic!("actual SafeTensors")
            };
            let bytes = Arc::downgrade(&lease.bytes);
            let shard = Arc::downgrade(&lease.shard);
            let cache = Arc::clone(&leaf.cache);
            assert_eq!(
                shard.strong_count(),
                2,
                "actual cache entry and rejected lease"
            );
            drop((owner, root, leaf));
            assert!(weak.upgrade().is_some());
            assert!(bytes.upgrade().is_some());
            drop(failed);
            assert!(weak.upgrade().is_none());
            assert!(bytes.upgrade().is_none());
            assert_eq!(
                shard.strong_count(),
                1,
                "only the retained cache owns the shard"
            );
            drop(cache);
            assert!(shard.upgrade().is_none());
        } else {
            let fixed = fixed.unwrap();
            assert_eq!(fixed.encoded_bytes(), old.unwrap().encoded_bytes());
            assert_eq!(fixed.encoded_bytes(), ordinary.unwrap().encoded_bytes());
        }
    }
}

#[test]
fn resolved_materialized_exception_uses_actual_membership_before_child_and_post_validation() {
    let (_directory, leaf) = source();
    let events = Arc::new(Mutex::new(Vec::new()));
    let overlay: SharedCheckpointSource = Arc::new(Uncached {
        leaf: leaf.clone(),
        inconsistent: false,
        materialized: BTreeSet::from(["hidden".into()]),
        events: events.clone(),
    });
    let root = resolved(overlay);
    // The hidden key is outside the selected contract but owned by the actual
    // neutral fixture overlay, exercising the existing materialized exception.
    let read = request("hidden");
    events.lock().unwrap().clear();
    let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    assert_eq!(*events.lock().unwrap(), [("authority", 0)]);
    events.lock().unwrap().clear();
    let fixed = prepared.acquire().unwrap();
    assert_eq!(*events.lock().unwrap(), [("authority", 0)]);
    events.lock().unwrap().clear();
    let old = root.acquire_lease(read).unwrap();
    assert_eq!(*events.lock().unwrap(), [("authority", 1)]);
    assert_eq!(fixed.encoded_bytes(), old.encoded_bytes());
    assert_eq!(fixed.encoded_bytes().unwrap(), [6, 5, 4, 9, 8, 7, 6, 5, 4]);
}

struct Switching {
    first: Arc<SafetensorsWeightStore>,
    second: Arc<SafetensorsWeightStore>,
    switched: AtomicBool,
}
impl CheckpointSource for Switching {
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        if self.switched.load(Ordering::SeqCst) {
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
        panic!("no diagnostic fallback")
    }
}
#[test]
fn same_metadata_foreign_cache_is_refused_and_partial_destination_retains_original_source() {
    let (directory, first) = source();
    let second = Arc::new(
        SafetensorsWeightStore::open(directory.path().join("weights.safetensors")).unwrap(),
    );
    let owner = Arc::new(Switching {
        first: first.clone(),
        second: second.clone(),
        switched: AtomicBool::new(false),
    });
    let weak = Arc::downgrade(&owner);
    let root: SharedCheckpointSource = owner.clone();
    let read = request("selected");
    let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    owner.switched.store(true, Ordering::SeqCst);
    let failure = prepared.acquire().unwrap_err();
    assert!(matches!(failure.cause, Cause::SourceChanged));
    assert!(failure.pending.is_some());
    assert!(failure.matches_request(&root, &read));
    assert_eq!(first.diagnostics().unwrap().physical_reads, 0);
    assert_eq!(second.diagnostics().unwrap().physical_reads, 0);
    drop((root, owner));
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert!(weak.upgrade().is_none());
}

#[test]
fn changed_file_failure_keeps_actual_root_request_and_read_owners_until_retirement() {
    let (directory, leaf) = source();
    let root: SharedCheckpointSource = Arc::new(snapshot(leaf.clone()));
    let weak = Arc::downgrade(&root);
    let read = request("selected");
    let prepared = PreparedCheckpointAcquisition::prepare(root.clone(), read.clone())
        .unwrap()
        .unwrap();
    File::options()
        .write(true)
        .open(directory.path().join("weights.safetensors"))
        .unwrap()
        .set_len(1)
        .unwrap();
    let error = prepared.acquire().unwrap_err();
    assert!(error.matches_request(&root, &read));
    let destination = error.destination_error().unwrap();
    assert!(destination.retains_file());
    assert!(destination.retains_shard());
    assert_eq!(destination.completed_bytes(), 0);
    assert_eq!(leaf.diagnostics().unwrap().physical_reads, 0);
    drop(root);
    assert!(weak.upgrade().is_some());
    drop(error);
    assert!(weak.upgrade().is_none());
}

#[test]
fn unavailable_route_never_invokes_ordinary_source_or_metadata_fallback() {
    struct OrdinaryOnly;
    impl CheckpointSource for OrdinaryOnly {
        fn source_keys(&self) -> Vec<String> {
            panic!("ordinary keys")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("ordinary metadata")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("ordinary acquisition")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("ordinary diagnostics")
        }
    }
    assert!(
        PreparedCheckpointAcquisition::prepare(Arc::new(OrdinaryOnly), request("selected"))
            .unwrap()
            .is_none()
    );
}
