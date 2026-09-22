use super::*;
use eredu_core::{AttentionPolicy, LayerSchedule, SharedStorageAccountingId};
use eredu_runtime::{HostMetadataIdentity, HostSlotAttachmentError, HostSlotMetadata};
use safemlx::{Device, DeviceType};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

fn layout(count: usize) -> StateLayout {
    let mut policies = vec![LayerCachePolicy::NoState; count];
    policies[0] = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 4).unwrap();
    StateLayout::new(LayerSchedule::new(count, policies).unwrap()).unwrap()
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn append(state: &mut MlxKeyValueState, values: [f32; 4], stream: &Stream) {
    let keys = Array::from_slice(&values, &[1, 1, 1, 4]);
    let values = Array::from_slice(&values.map(|value| value + 10.), &[1, 1, 1, 4]);
    KeyValueCache::update_for_attention(state.layer(0).unwrap(), keys, values, stream).unwrap();
}

fn keys(state: &MlxKeyValueState, stream: &Stream) -> Vec<f32> {
    state.retained_arrays()[0]
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

struct Charge {
    bytes: u64,
    used: Arc<AtomicU64>,
    _identity: HostMetadataIdentity,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

fn attach(metadata: &HostSlotMetadata, domain: &SharedStorageAccountingId, used: &Arc<AtomicU64>) {
    let bytes = metadata.capacity_bytes().unwrap();
    let identity = metadata.identity().clone();
    assert!(metadata
        .try_attach(domain, || {
            used.fetch_add(bytes, Ordering::SeqCst);
            Ok::<Box<dyn Send + Sync>, Infallible>(Box::new(Charge {
                bytes,
                used: used.clone(),
                _identity: identity,
            }))
        })
        .unwrap());
}

fn assert_retired(metadata: &HostSlotMetadata, domain: &SharedStorageAccountingId) {
    assert!(matches!(
        metadata.try_attach::<Infallible>(domain, || panic!("retired table must not acquire")),
        Err(HostSlotAttachmentError::Retired)
    ));
}

#[test]
fn model_clone_and_clone_from_replace_table_identity_with_preserved_state() {
    let stream = stream();
    let mut source = MlxKeyValueState::device_with_global_layer_start(layout(2), 17).unwrap();
    append(&mut source, [1., 3., 5., 7.], &stream);
    source.paged_transaction_branch = true;
    let source_token = source.layer_slot_metadata().clone();
    let expected_bytes = std::mem::size_of_val(source.as_ref()) as u64;
    assert_eq!(source_token.len(), 2);
    assert_eq!(source_token.capacity_bytes(), Some(expected_bytes));
    assert!(matches!(
        source.as_ref()[1],
        MlxKeyValueLayerState::Stateless
    ));

    let clone = source.clone();
    assert!(!source_token.same_storage(clone.layer_slot_metadata()));
    assert_eq!(
        clone.layer_slot_metadata().capacity_bytes(),
        Some(expected_bytes)
    );
    assert_eq!(clone.global_layer_start, 17);
    assert!(clone.paged_transaction_branch);
    assert!(clone.layout.same_storage(&source.layout));
    assert_eq!(clone.offset(), 1);
    assert_eq!(keys(&clone, &stream), [1., 3., 5., 7.]);
    drop(clone);

    let domain = SharedStorageAccountingId::default();
    let used = Arc::new(AtomicU64::new(0));
    let mut destination = MlxKeyValueState::device(layout(2)).unwrap();
    append(&mut destination, [11., 13., 17., 19.], &stream);
    let old_token = destination.layer_slot_metadata().clone();
    attach(&old_token, &domain, &used);
    destination.clone_from(&source);
    // clone_from preserves independently owned state. Equal
    // extents still replace the actual allocation and retire its old identity.
    assert_retired(&old_token, &domain);
    assert!(!old_token.same_storage(destination.layer_slot_metadata()));
    assert!(!source_token.same_storage(destination.layer_slot_metadata()));
    assert_eq!(destination.global_layer_start, 17);
    assert!(destination.paged_transaction_branch);
    assert_eq!(destination.offset(), 1);
    assert_eq!(keys(&destination, &stream), [1., 3., 5., 7.]);
    assert_eq!(used.load(Ordering::SeqCst), expected_bytes);
    drop(old_token);
    assert_eq!(used.load(Ordering::SeqCst), 0);

    let replaced = destination.layer_slot_metadata().clone();
    let larger = MlxKeyValueState::device_with_global_layer_start(layout(3), 29).unwrap();
    destination.clone_from(&larger);
    assert_retired(&replaced, &domain);
    assert_eq!(destination.as_ref().len(), 3);
    assert_eq!(
        destination.layer_slot_metadata().capacity_bytes(),
        Some(3 * std::mem::size_of::<MlxKeyValueLayerState>() as u64)
    );
    assert_eq!(destination.global_layer_start, 29);
    assert!(!destination.paged_transaction_branch);
    assert_eq!(destination.offset(), 0);
    assert_eq!(source.offset(), 1);
    assert_eq!(keys(&source, &stream), [1., 3., 5., 7.]);
}

#[test]
fn transaction_commit_moves_branch_table_and_checkpoint_restore_reuses_current_table() {
    let stream = stream();
    let mut canonical = MlxKeyValueState::device_with_global_layer_start(layout(2), 23).unwrap();
    append(&mut canonical, [2., 4., 6., 8.], &stream);
    let original_token = canonical.layer_slot_metadata().clone();
    let domain = SharedStorageAccountingId::default();
    let used = Arc::new(AtomicU64::new(0));
    attach(&original_token, &domain, &used);
    let mut branch = canonical.branch().unwrap();
    let branch_token = branch.layer_slot_metadata().clone();
    assert!(!original_token.same_storage(&branch_token));
    assert_eq!(branch.global_layer_start, 23);
    append(&mut branch, [12., 14., 16., 18.], &stream);
    assert_eq!(canonical.offset(), 1);
    assert_eq!(branch.offset(), 2);
    canonical.commit_branch(branch).unwrap();
    assert!(branch_token.same_storage(canonical.layer_slot_metadata()));
    assert_retired(&original_token, &domain);
    assert_eq!(canonical.global_layer_start, 23);
    assert_eq!(
        keys(&canonical, &stream),
        [2., 4., 6., 8., 12., 14., 16., 18.]
    );
    assert!(used.load(Ordering::SeqCst) > 0);
    drop(original_token);
    assert_eq!(used.load(Ordering::SeqCst), 0);

    let checkpoint = canonical.deep_clone_state().unwrap();
    assert!(!branch_token.same_storage(checkpoint.layer_slot_metadata()));
    canonical.clear().unwrap();
    assert_eq!(canonical.offset(), 0);
    canonical.restore_checkpoint(&checkpoint, &stream).unwrap();
    assert!(branch_token.same_storage(canonical.layer_slot_metadata()));
    assert_eq!(canonical.offset(), 2);
    assert_eq!(
        keys(&canonical, &stream),
        [2., 4., 6., 8., 12., 14., 16., 18.]
    );
}

#[test]
fn escaped_layer_metadata_keeps_charge_without_retaining_native_payload() {
    struct NativeRetired(Arc<AtomicUsize>);
    impl Drop for NativeRetired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let stream = stream();
    let mut state = MlxKeyValueState::device(layout(2)).unwrap();
    append(&mut state, [11., 13., 17., 19.], &stream);
    let native_retired = Arc::new(AtomicUsize::new(0));
    {
        let arrays = state.retained_arrays();
        arrays[0].evaluated().unwrap();
        arrays[0]
            .retain_allocation_owner(NativeRetired(native_retired.clone()))
            .unwrap();
    }
    let token = state.layer_slot_metadata().clone();
    let earlier = token.clone();
    let identity = token.identity().clone();
    let domain = SharedStorageAccountingId::default();
    let used = Arc::new(AtomicU64::new(0));
    let capacity = token.capacity_bytes().unwrap();
    attach(&token, &domain, &used);
    assert!(!earlier
        .try_attach::<Infallible>(&domain, || panic!("shared token already attached"))
        .unwrap());
    drop(state);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        native_retired.load(Ordering::SeqCst) == 1
    });
    assert_eq!(used.load(Ordering::SeqCst), capacity);
    assert_retired(&token, &domain);
    assert_retired(&token, &SharedStorageAccountingId::default());
    drop(token);
    assert_eq!(used.load(Ordering::SeqCst), capacity);
    drop(earlier);
    assert_eq!(used.load(Ordering::SeqCst), 0);
    // A separately retained registry key owns neither slots nor charges.
    drop(identity);
    assert_eq!(used.load(Ordering::SeqCst), 0);
}

#[test]
fn borrowed_key_value_whole_visit_preserves_actual_slots_and_empty_invocations() {
    use std::cell::Cell;
    thread_local! { static CALLS:Cell<usize>=const {Cell::new(0)}; }
    fn housekeeping() {
        CALLS.set(CALLS.get() + 1);
    }
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            safemlx::unregister_thread_runtime_housekeeping(housekeeping);
        }
    }
    let stream = stream();
    let mut state = MlxKeyValueState::device_with_global_layer_start(layout(2), 17).unwrap();
    append(&mut state, [1., 3., 5., 7.], &stream);
    let expected = state.retained_arrays();
    assert_eq!(expected.len(), 2);
    let offset = state.offset();
    let manager = CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut paged =
        MlxKeyValueLayerState::Paged(PagedKeyValueCache::new(manager, 0, None).unwrap());
    paged
        .update_and_fetch(
            Array::from_slice(&[2_f32, 4., 6., 8.], &[1, 1, 1, 4]),
            Array::from_slice(&[12_f32, 14., 16., 18.], &[1, 1, 1, 4]),
            &stream,
        )
        .unwrap();
    let paged_expected = paged.retained_arrays();
    assert_eq!(paged_expected.len(), 2);
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = Guard;
    CALLS.set(0);
    let mut count = 0;
    state
        .visit_all_retained_values(&mut |value| {
            assert!(std::ptr::eq(value.as_array(), expected[count]));
            count += 1;
        })
        .unwrap();
    RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(&state.as_ref()[1], &mut |_| {
        panic!("stateless invocation")
    });
    assert_eq!(count, 2);
    let mut tail_count = 0;
    RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(&paged, &mut |value| {
        assert!(std::ptr::eq(value.as_array(), paged_expected[tail_count]));
        tail_count += 1;
    });
    assert_eq!(tail_count, 2);
    assert_eq!(CALLS.get(), 0);
    drop(guard);
    assert_eq!(state.offset(), offset);
}
