use super::*;
use eredu_checkpoint::{
    safetensors::SafetensorsDiscoveryLimits,
    store::{
        EncodedTensorLease, ReadPolicy, SafetensorsWeightStore, TensorReadRequest, TensorSelection,
        WeightStore,
    },
};
use std::sync::Barrier;

const METADATA: DependencyMemoryPolicy = DependencyMemoryPolicy {
    fixed_bytes: 1024,
    bytes_per_input_byte: 8,
};
const JSON: &[u8] = br#"{"weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
fn qualified() -> bool {
    let result = MemoryLedger::safetensors_header_policy_required_bytes(1);
    if std::env::var_os("EREDU_REQUIRE_QUALIFIED_MEMORY_TENSOR_SOURCE").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(_) => true,
        Err(WorkingMemoryError::UnknownBound) => false,
        other => panic!("{other:?}"),
    }
}
fn fixture(json: &[u8]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json);
    bytes.extend_from_slice(&[11, 19]);
    std::fs::write(dir.path().join("weights.safetensors"), bytes).unwrap();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"weight":"weights.safetensors"}}"#,
    )
    .unwrap();
    dir
}
fn request(dir: &tempfile::TempDir, json: &[u8]) -> SafetensorsHeaderRequest {
    SafetensorsHeaderRequest {
        json_bytes: json.len(),
        buffer_bytes: json.len() + 8,
        path_bytes: dir
            .path()
            .join("weights.safetensors")
            .canonicalize()
            .unwrap()
            .as_os_str()
            .len(),
    }
}
fn open(
    dir: &tempfile::TempDir,
    policy: Arc<dyn SafetensorsHeaderAdmission>,
) -> SafetensorsWeightStore {
    SafetensorsWeightStore::open_with_header_admission(
        dir.path(),
        1,
        SafetensorsDiscoveryLimits::default(),
        policy,
    )
    .unwrap()
}
fn memory<'a>(error: &'a (dyn Error + 'static)) -> &'a WorkingMemoryError {
    let mut error = error;
    loop {
        if let Some(memory) = error.downcast_ref::<WorkingMemoryError>() {
            return memory;
        }
        error = error.source().expect("typed memory failure");
    }
}
#[test]
fn exact_and_short_admission_keep_headroom_separate_and_refusals_funded() {
    if !qualified() {
        return;
    }
    let dir = fixture(JSON);
    let request = request(&dir, JSON);
    let quote = MemoryLedger::safetensors_header_quote(request, METADATA).unwrap();
    assert_eq!(quote.encoded_buffer_bytes(), JSON.len() as u64 + 8);
    assert_eq!(
        quote.metadata_estimate_bytes(),
        METADATA
            .estimate(request.json_bytes + request.path_bytes)
            .unwrap() as u64
    );
    assert_eq!(
        quote.total_bytes(),
        quote.encoded_buffer_bytes()
            + quote.metadata_estimate_bytes()
            + quote.reservation_control_bytes()
    );
    let higher = MemoryLedger::safetensors_header_quote(
        request,
        DependencyMemoryPolicy {
            fixed_bytes: METADATA.fixed_bytes + 77,
            ..METADATA
        },
    )
    .unwrap();
    assert_eq!(higher.total_bytes(), quote.total_bytes() + 77);
    let controls = MemoryLedger::safetensors_header_policy_required_bytes(1).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(controls - 1, 0).unwrap();
    assert!(matches!(
        short.safetensors_header_admission(1, METADATA),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let short =
        crate::working_memory::memory_fixture::host_ledger(controls + quote.total_bytes() - 1, 0)
            .unwrap();
    let policy = short.safetensors_header_admission(1, METADATA).unwrap();
    let store = open(&dir, policy.clone());
    assert_eq!(short.payload_used_bytes().unwrap(), controls);
    let error = WeightStore::metadata(&store, "weight").unwrap_err();
    assert!(
        matches!(memory(&error), WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })
        if *required_bytes == quote.total_bytes() && (limit_bytes - existing_bytes) == quote.total_bytes()-1)
    );
    drop(store);
    drop(policy);
    assert_eq!(short.payload_used_bytes().unwrap(), controls);
    drop(error);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    let pool =
        crate::working_memory::memory_fixture::host_ledger(controls + quote.total_bytes(), 0)
            .unwrap();
    let policy = pool.safetensors_header_admission(1, METADATA).unwrap();
    let store = open(&dir, policy.clone());
    let lease = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        controls + quote.total_bytes()
    );
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    drop(store);
    drop(policy);
    assert!(pool.payload_used_bytes().unwrap() >= quote.total_bytes());
    assert_eq!(lease.encoded_bytes().unwrap(), [11, 19]);
    drop(lease);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn active_parsing_blocks_unquoted_work_and_completion_retains_bytes() {
    if !qualified() {
        return;
    }
    #[derive(Debug)]
    struct Gate {
        policy: Arc<dyn SafetensorsHeaderAdmission>,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }
    impl SafetensorsHeaderAdmission for Gate {
        fn reserve(
            &self,
            request: SafetensorsHeaderRequest,
        ) -> Result<SafetensorsHeaderReservation, Arc<SafetensorsHeaderFailure>> {
            let hold = self.policy.reserve(request)?;
            self.entered.wait();
            self.release.wait();
            Ok(hold)
        }
    }
    let dir = fixture(JSON);
    let controls = MemoryLedger::safetensors_header_policy_required_bytes(1).unwrap();
    let quote = MemoryLedger::safetensors_header_quote(request(&dir, JSON), METADATA).unwrap();
    let pool =
        crate::working_memory::memory_fixture::host_ledger(controls + quote.total_bytes(), 0)
            .unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let policy = Arc::new(Gate {
        policy: pool.safetensors_header_admission(1, METADATA).unwrap(),
        entered: entered.clone(),
        release: release.clone(),
    });
    let store = Arc::new(open(&dir, policy));
    let reader = store.clone();
    let join =
        std::thread::spawn(move || WeightStore::metadata(reader.as_ref(), "weight").unwrap());
    entered.wait();
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 1);
    let unquoted = pool.acquire_unquoted();
    release.wait();
    assert!(matches!(
        unquoted,
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    assert_eq!(join.join().unwrap().logical_shape, [2]);
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        controls + quote.total_bytes()
    );
    drop(store);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn concurrent_malformed_header_finishes_once_and_keeps_charge_through_error_aliases() {
    if !qualified() {
        return;
    }
    let dir = fixture(b"xxxxxxxx");
    let controls = MemoryLedger::safetensors_header_policy_required_bytes(1).unwrap();
    let quote =
        MemoryLedger::safetensors_header_quote(request(&dir, b"xxxxxxxx"), METADATA).unwrap();
    let pool =
        crate::working_memory::memory_fixture::host_ledger(controls + quote.total_bytes(), 0)
            .unwrap();
    let policy = pool.safetensors_header_admission(1, METADATA).unwrap();
    let store = Arc::new(open(&dir, policy.clone()));
    let barrier = Arc::new(Barrier::new(8));
    let joins: Vec<_> = (0..8)
        .map(|_| {
            let store = store.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                WeightStore::metadata(store.as_ref(), "weight").unwrap_err()
            })
        })
        .collect();
    let mut errors: Vec<_> = joins.into_iter().map(|j| j.join().unwrap()).collect();
    let StoreError::SafetensorsHeader(first) = &errors[0] else {
        panic!("shared failure")
    };
    for error in &errors[1..] {
        let StoreError::SafetensorsHeader(other) = error else {
            panic!("shared failure")
        };
        assert!(Arc::ptr_eq(first, other));
    }
    assert_eq!(pool.0.usage.lock().unwrap().reservations, 0);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        controls + quote.total_bytes()
    );
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    drop(store);
    drop(policy);
    let last = errors.pop().unwrap();
    drop(errors);
    assert_eq!(pool.payload_used_bytes().unwrap(), quote.total_bytes());
    assert!(last.to_string().contains("malformed safetensors shard"));
    drop(last);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn exhausted_policy_reuses_prepaid_failure_and_cannot_keep_pool_alive() {
    if !qualified() {
        return;
    }
    let controls = MemoryLedger::safetensors_header_policy_required_bytes(0).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(controls, 0).unwrap();
    let policy = pool.safetensors_header_admission(0, METADATA).unwrap();
    let request = SafetensorsHeaderRequest {
        json_bytes: 0,
        buffer_bytes: 8,
        path_bytes: 1,
    };
    let first = policy.reserve(request).unwrap_err();
    let second = policy.reserve(request).unwrap_err();
    assert!(Arc::ptr_eq(&first, &second));
    let cause = first
        .source()
        .unwrap()
        .downcast_ref::<SafetensorsHeaderPolicyError>()
        .unwrap();
    assert_eq!(cause.maximum_headers(), Some(0));
    drop(policy);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), controls);
    let weak = Arc::downgrade(&pool.0);
    drop(pool);
    assert!(weak.upgrade().is_none());
    drop(first);
}
#[test]
fn invalid_request_and_overflow_refusals_are_typed_and_keep_policy_custody() {
    if !qualified() {
        return;
    }
    let controls = MemoryLedger::safetensors_header_policy_required_bytes(2).unwrap();
    let pool = crate::working_memory::memory_fixture::host_ledger(controls, 0).unwrap();
    let policy = pool.safetensors_header_admission(2, METADATA).unwrap();
    let invalid = policy
        .reserve(SafetensorsHeaderRequest {
            json_bytes: 1,
            buffer_bytes: 8,
            path_bytes: 0,
        })
        .unwrap_err();
    assert!(matches!(
        memory(invalid.as_ref()),
        WorkingMemoryError::IdentityMismatch
    ));
    let overflow = policy
        .reserve(SafetensorsHeaderRequest {
            json_bytes: usize::MAX,
            buffer_bytes: 0,
            path_bytes: 0,
        })
        .unwrap_err();
    assert!(matches!(
        memory(overflow.as_ref()),
        WorkingMemoryError::Overflow
    ));
    drop(policy);
    drop(invalid);
    assert_eq!(pool.payload_used_bytes().unwrap(), controls);
    drop(overflow);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
