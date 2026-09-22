use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use eredu_runtime::working_memory::MemoryLedger;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

#[test]
fn empty_parameter_storage_is_complete_even_with_descriptive_metadata() {
    let mut state = NativeParameterState::default();
    assert_eq!(
        state.retained_storage().unwrap().byte_bound().unwrap(),
        Some(0)
    );
    state.active = Some("overlay-without-local-numerical-slots".into());
    state.floating_state_dtype_bytes = std::num::NonZeroU8::new(4);
    state
        .baseline_transforms
        .push(("projection".into(), ProjectionInputTransform::Identity));
    state.reset_estimate = Some(eredu_core::execution_control::SnapshotEstimate {
        retained_bytes: 128,
        copy_bytes: 256,
    });
    state.epoch = 7;
    state.usage.set(CaptureUsage {
        captures: 3,
        retained_bytes: 512,
        host_bytes: 64,
        encoded_bytes: 128,
    });
    assert_eq!(
        state.retained_storage().unwrap().byte_bound().unwrap(),
        Some(0)
    );
    let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let counted = state.parameter_sources().count(&mut guard).unwrap();
    assert!(std::ptr::eq(counted.source().state(), &state));
    assert_eq!(counted.counts(), ParameterOwnerCounts::default());
    for role in [
        ParameterOwnerRole::Static,
        ParameterOwnerRole::PredictionInner,
        ParameterOwnerRole::PredictionPlaceholder,
        ParameterOwnerRole::PredictionReplacement,
        ParameterOwnerRole::DisplacedOriginal,
        ParameterOwnerRole::PublishedOverlay,
    ] {
        let empty = counted.counts().role(role);
        assert_eq!(empty.parameters.named_slots, 0);
        assert_eq!(empty.parameters.auxiliary_slots, 0);
        assert_eq!(empty.map_key_bytes, 0);
    }
    assert_eq!(
        state.active.as_deref(),
        Some("overlay-without-local-numerical-slots")
    );
    assert_eq!(state.epoch, 7);
    assert_eq!(state.usage.get().captures, 3);
}

struct RetirementProbe(Arc<AtomicUsize>);

impl Drop for RetirementProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn original_and_published_aliases_retain_unique_full_backing_capacity() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let _memory = NativeMemoryOwner::acquire(&pool).unwrap();
    let original = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6., 7., 8.], &[8]);
    let published = Array::from_slice(&[3_f32; 32], &[32]);
    let view = original.as_strided(&[2][..], &[2][..], 1, &stream).unwrap();
    assert_eq!(
        view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        [2., 4.]
    );
    let original_info = original.allocation_info().unwrap().unwrap();
    let published_info = published.allocation_info().unwrap().unwrap();
    assert_ne!(original_info.identity(), published_info.identity());
    assert_eq!(view.allocation_info().unwrap(), Some(original_info));
    let expected = original_info.bytes() as u64 + published_info.bytes() as u64;
    let retired = Arc::new(AtomicUsize::new(0));
    original
        .retain_allocation_owner(RetirementProbe(Arc::clone(&retired)))
        .unwrap();
    published
        .retain_allocation_owner(RetirementProbe(Arc::clone(&retired)))
        .unwrap();

    let mut state = NativeParameterState::default();
    state.originals = fixture_rows([
        ("original", MlxTensor::from_array(original.clone())),
        ("original-alias", MlxTensor::from_array(view.clone())),
    ]);
    state.published = fixture_rows([
        ("published", MlxTensor::from_array(published.clone())),
        ("cross-map-alias", MlxTensor::from_array(view.clone())),
    ]);
    let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let counted = state.parameter_sources().count(&mut guard).unwrap();
    assert!(std::ptr::eq(counted.source().state(), &state));
    for role in [
        ParameterOwnerRole::DisplacedOriginal,
        ParameterOwnerRole::PublishedOverlay,
    ] {
        assert_eq!(counted.counts().role(role).parameters.auxiliary_slots, 2);
        assert_eq!(counted.counts().role(role).parameters.unknown_backings, 0);
    }
    assert_eq!(
        counted
            .counts()
            .role(ParameterOwnerRole::DisplacedOriginal)
            .map_key_bytes,
        0
    );
    assert_eq!(
        counted
            .counts()
            .role(ParameterOwnerRole::PublishedOverlay)
            .map_key_bytes,
        0
    );
    drop(counted);
    drop(guard);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    let storage = state.retained_storage().unwrap();
    assert_eq!(storage.byte_bound().unwrap(), Some(expected));

    // Published model slots contribute these same allocations to the enclosing
    // model inventory. Merging them must not charge either allocation twice.
    let mut enclosing = RetainedStorage::default();
    enclosing.include_array(&published).unwrap();
    enclosing.include_array(&view).unwrap();
    enclosing.merge(storage).unwrap();
    assert_eq!(enclosing.byte_bound().unwrap(), Some(expected));
    drop((state, original, published, view));
    stream.synchronize().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(enclosing.byte_bound().unwrap(), Some(expected));
    drop(enclosing);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::reclaim_allocation_owners();
        retired.load(Ordering::SeqCst) == 2
    });
}

#[test]
fn lazy_parameter_inventory_stays_unknown_without_materializing() {
    let stream = stream();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let _memory = NativeMemoryOwner::acquire(&pool).unwrap();
    let original = Array::from_slice(&[2_f32, -3.], &[2]);
    let lazy = original.square(&stream).unwrap();
    let mut state = NativeParameterState::default();
    state.originals = fixture_rows([("original", MlxTensor::from_array(original))]);
    state.published = fixture_rows([("published", MlxTensor::from_array(lazy.clone()))]);
    assert_eq!(lazy.allocation_info().unwrap(), None);
    let cold = state.retained_storage().unwrap();
    assert_eq!(cold.byte_bound().unwrap(), None);
    assert_eq!(lazy.allocation_info().unwrap(), None);
    assert_eq!(
        lazy.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
        [4., 9.]
    );
    assert_eq!(cold.byte_bound().unwrap(), None);
    assert!(
        state
            .retained_storage()
            .unwrap()
            .byte_bound()
            .unwrap()
            .unwrap()
            > 0
    );
}
