use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct Policy {
    calls: AtomicUsize,
    retired: Arc<AtomicUsize>,
    refuse: bool,
    completed: Arc<AtomicUsize>,
    refuse_completion: bool,
}
#[derive(Debug, thiserror::Error)]
#[error("header admission refused")]
struct Refused;
#[derive(Debug)]
struct Hold {
    retired: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    refuse_completion: bool,
}
impl Hold {
    fn complete(&self) -> Result<(), Arc<dyn std::error::Error + Send + Sync>> {
        self.completed.fetch_add(1, Ordering::SeqCst);
        if self.refuse_completion {
            Err(Arc::new(CompletionRefused))
        } else {
            Ok(())
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("header completion refused")]
struct CompletionRefused;
impl Drop for Hold {
    fn drop(&mut self) {
        self.retired.fetch_add(1, Ordering::SeqCst);
    }
}
impl SafetensorsHeaderAdmission for Policy {
    fn reserve(
        &self,
        request: SafetensorsHeaderRequest,
    ) -> Result<SafetensorsHeaderReservation, Arc<SafetensorsHeaderFailure>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.buffer_bytes, request.json_bytes + 8);
        assert!(request.path_bytes > 0);
        if self.refuse {
            Err(Arc::new(SafetensorsHeaderFailure::refused(Arc::new(
                Refused,
            ))))
        } else {
            Ok(SafetensorsHeaderReservation::with_completion(
                Hold {
                    retired: self.retired.clone(),
                    completed: self.completed.clone(),
                    refuse_completion: self.refuse_completion,
                },
                Hold::complete,
            ))
        }
    }
}

#[test]
fn refusal_precedes_body_read_and_keeps_the_typed_cause() {
    let policy = Policy {
        refuse: true,
        ..Default::default()
    };
    let mut input = std::io::Cursor::new(32_u64.to_le_bytes());
    let mut hold = None;
    let error = read_safetensors_metadata_from_admitted(
        Path::new("weights"),
        &mut input,
        40,
        32,
        Some(&policy),
        &mut hold,
    )
    .unwrap_err();
    assert_eq!(input.position(), 8);
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    assert!(hold.is_none());
    let mut cause: &dyn std::error::Error = &error;
    while !cause.is::<Refused>() {
        cause = cause.source().expect("typed refusal");
    }
}

#[test]
fn successful_header_hold_outlives_store_with_a_payload_lease() {
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize_to_file},
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::U8, vec![2], &[11, 19]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let policy = Arc::new(Policy::default());
    let store = SafetensorsWeightStore::open_with_header_admission(
        &path,
        1,
        SafetensorsDiscoveryLimits::default(),
        policy.clone(),
    )
    .unwrap();
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    let lease = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    drop(store);
    assert_eq!(policy.completed.load(Ordering::SeqCst), 1);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 0);
    assert_eq!(lease.encoded_bytes().unwrap(), [11, 19]);
    drop(lease);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_failed_headers_share_one_admission_and_errors_retain_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut malformed = 8_u64.to_le_bytes().to_vec();
    malformed.extend_from_slice(b"xxxxxxxx");
    std::fs::write(dir.path().join("bad.safetensors"), malformed).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"bad.safetensors"}}"#,
    )
    .unwrap();
    let policy = Arc::new(Policy::default());
    let store = Arc::new(
        SafetensorsWeightStore::open_with_header_admission(
            dir.path(),
            1,
            SafetensorsDiscoveryLimits::default(),
            policy.clone(),
        )
        .unwrap(),
    );
    assert_eq!(policy.calls.load(Ordering::SeqCst), 0);
    let joins: Vec<_> = (0..8)
        .map(|_| {
            let store = store.clone();
            std::thread::spawn(move || WeightStore::metadata(store.as_ref(), "weight").unwrap_err())
        })
        .collect();
    let mut errors: Vec<_> = joins
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    assert_eq!(policy.completed.load(Ordering::SeqCst), 1);
    let StoreError::SafetensorsHeader(first) = &errors[0] else {
        panic!("retained header failure")
    };
    for error in &errors[1..] {
        let StoreError::SafetensorsHeader(other) = error else {
            panic!("same shared failure")
        };
        assert!(Arc::ptr_eq(first, other));
    }
    drop(store);
    let last = errors.pop().unwrap();
    drop(errors);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 0);
    assert!(last.to_string().contains("malformed safetensors shard"));
    drop(last);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn index_consistency_failure_keeps_the_accepted_header_reservation() {
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize_to_file},
    };
    let dir = tempfile::tempdir().unwrap();
    serialize_to_file(
        [(
            "actual",
            TensorView::new(Dtype::U8, vec![1], &[13]).unwrap(),
        )],
        None,
        &dir.path().join("weights.safetensors"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"claimed":"weights.safetensors"}}"#,
    )
    .unwrap();
    let policy = Arc::new(Policy::default());
    let store = SafetensorsWeightStore::open_with_header_admission(
        dir.path(),
        1,
        SafetensorsDiscoveryLimits::default(),
        policy.clone(),
    )
    .unwrap();
    let error = WeightStore::metadata(&store, "claimed").unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    while !matches!(
        cause.downcast_ref::<StoreError>(),
        Some(StoreError::ContradictoryIndexMapping { .. })
    ) {
        cause = cause.source().expect("original index validation error");
    }
    drop(store);
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn strict_discovery_reuses_admitted_headers_and_preserves_refusal_sources() {
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize_to_file},
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights.safetensors");
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::U8, vec![1], &[23]).unwrap(),
        )],
        None,
        &path,
    )
    .unwrap();
    let policy = Arc::new(Policy::default());
    let catalog = crate::safetensors::SafetensorsMetadataCatalog::discover_with_header_admission(
        &path,
        SafetensorsDiscoveryLimits::default(),
        policy.clone(),
    )
    .unwrap();
    assert_eq!(catalog.tensor("weight").unwrap().encoded_byte_len, 1);
    let store = SafetensorsWeightStore::open_admitted(catalog.admitted_shards(), 1).unwrap();
    drop(catalog);
    assert_eq!(
        WeightStore::metadata(&store, "weight")
            .unwrap()
            .logical_shape,
        [1]
    );
    assert_eq!(policy.calls.load(Ordering::SeqCst), 1);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 0);
    drop(store);
    assert_eq!(policy.completed.load(Ordering::SeqCst), 1);
    assert_eq!(policy.retired.load(Ordering::SeqCst), 1);
    let refused = Arc::new(Policy {
        refuse: true,
        ..Default::default()
    });
    let error = SafetensorsShards::discover_with_header_admission(
        &path,
        SafetensorsDiscoveryLimits::default(),
        refused,
    )
    .unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    while !cause.is::<Refused>() {
        cause = cause.source().expect("strict typed refusal");
    }
}

#[test]
fn completion_failure_keeps_completed_metadata_and_original_parse_failure() {
    for malformed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let json: &[u8] = if malformed {
            b"xxxxxxxx"
        } else {
            br#"{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#
        };
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json);
        bytes.extend_from_slice(&[11, 19]);
        std::fs::write(dir.path().join("weights.safetensors"), bytes).unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"weight":"weights.safetensors"}}"#,
        )
        .unwrap();
        let policy = Arc::new(Policy {
            refuse_completion: true,
            ..Default::default()
        });
        let store = SafetensorsWeightStore::open_with_header_admission(
            dir.path(),
            1,
            SafetensorsDiscoveryLimits::default(),
            policy.clone(),
        )
        .unwrap();
        let error = WeightStore::metadata(&store, "weight").unwrap_err();
        let StoreError::SafetensorsHeader(failure) = &error else {
            panic!("shared failure")
        };
        assert!(
            failure
                .completion_failure()
                .unwrap()
                .is::<CompletionRefused>()
        );
        if malformed {
            assert!(
                std::error::Error::source(failure.as_ref())
                    .unwrap()
                    .is::<StoreError>()
            );
            assert!(error.to_string().contains("malformed safetensors shard"));
        } else {
            assert!(
                std::error::Error::source(failure.as_ref())
                    .unwrap()
                    .is::<CompletionRefused>()
            );
        }
        let again = WeightStore::metadata(&store, "weight").unwrap_err();
        let StoreError::SafetensorsHeader(other) = &again else {
            panic!("shared failure")
        };
        assert!(Arc::ptr_eq(failure, other));
        assert_eq!(policy.completed.load(Ordering::SeqCst), 1);
        drop(store);
        drop(again);
        assert_eq!(policy.retired.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(policy.retired.load(Ordering::SeqCst), 1);
    }
}
