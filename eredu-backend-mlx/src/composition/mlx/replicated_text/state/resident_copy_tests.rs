use super::*;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::cell::Cell;

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct HousekeepingGuard;
impl Drop for HousekeepingGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn layout() -> eredu_runtime::StateLayout {
    eredu_runtime::StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                eredu_core::cache::LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2)
                    .unwrap(),
                eredu_core::cache::LayerCachePolicy::NoState,
            ],
        )
        .unwrap(),
    )
    .unwrap()
}
fn unknown(error: &Error) -> bool {
    match error {
        Error::Other(source) => {
            source.downcast_ref::<WorkingMemoryError>() == Some(&WorkingMemoryError::UnknownBound)
        }
        _ => false,
    }
}

#[test]
fn typed_resident_source_keeps_lazy_operands_and_preexisting_layout_without_housekeeping() {
    use crate::backend::runtime::cache::kv::KeyValueCache;
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut state = MlxKeyValueState::device_with_global_layer_start(layout(), 17).unwrap();
    let base = Array::from_slice(&[1_f32, 3., 5., 7.], &[1, 1, 2, 2]);
    let lazy = base.transpose_axes(&[0, 1, 3, 2], &stream).unwrap();
    state.as_mut()[0]
        .update_and_fetch(lazy.clone(), lazy, &stream)
        .unwrap();
    let layout = state.shared_layout().unwrap().clone();
    let before = state.retained_arrays();
    assert!(
        before
            .iter()
            .all(|a| a.try_metadata_snapshot().unwrap().allocation().is_none())
    );
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.with(|calls| calls.set(0));
    let plan = state.prepare_resident_decoder_copy().unwrap();
    let mut operands = Vec::new();
    plan.dense_key_value()
        .expect("resident KV fixture")
        .visit_operands(&mut |array| operands.push(array));
    assert_eq!(operands.len(), 2);
    assert!(
        operands
            .iter()
            .zip(&before)
            .all(|(a, b)| std::ptr::eq(*a, *b))
    );
    assert!(plan.shared_layout().unwrap().same_storage(&layout));
    assert_eq!(plan.global_layer_start(), Some(17));
    assert!(
        operands
            .iter()
            .all(|a| a.try_metadata_snapshot().unwrap().allocation().is_none())
    );
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(guard);
}

#[test]
fn native_representations_select_exact_supported_plans_without_housekeeping() {
    let hybrid = MlxHybridState::device(layout()).unwrap();
    let pooling_layout = eredu_runtime::StateLayout::new(
        LayerSchedule::new(
            1,
            vec![
                eredu_core::cache::LayerCachePolicy::key_only(
                    AttentionPolicy::sliding(7).unwrap(),
                    1,
                    2,
                )
                .unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let pooling = MlxPoolingAttentionStateFactory::device(pooling_layout).unwrap();
    let stateless = MlxPoolingAttentionState::stateless();
    let manager = CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 4096, 64 << 10, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let paged_hybrid = MlxHybridState::paged(layout(), manager.clone(), None).unwrap();
    let paged = MlxKeyValueState::paged(layout(), manager, None).unwrap();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.with(|calls| calls.set(0));
    // The NoState layer selects the existing grouped Hybrid copy. Preserve
    // that exact source layout instead of retagging it as an ordinary KV copy.
    let hybrid_plan = hybrid.prepare_resident_decoder_copy().unwrap();
    assert_eq!(hybrid_plan.global_layer_start(), Some(0));
    assert!(
        hybrid_plan
            .shared_layout()
            .unwrap()
            .same_storage(hybrid.shared_layout().unwrap())
    );
    let mut hybrid_operands = 0;
    hybrid_plan.visit_operands(&mut |_| hybrid_operands += 1);
    hybrid_plan.visit_retained_arrays(&mut |_| hybrid_operands += 1);
    assert_eq!(hybrid_operands, 0);
    assert!(hybrid_plan.into_dense_key_value().is_none());
    let pooling_plan = pooling.prepare_resident_decoder_copy().unwrap();
    assert_eq!(pooling_plan.global_layer_start(), None);
    assert!(
        pooling_plan
            .shared_layout()
            .unwrap()
            .same_storage(pooling.shared_layout().unwrap())
    );
    let mut operands = 0;
    pooling_plan.visit_operands(&mut |_| operands += 1);
    assert_eq!(operands, 0);
    // Present but empty layers retain their actual table/layout. The actual
    // stateless source remains absent instead of registering an empty table.
    assert!(pooling.layer_slot_metadata().is_some());
    assert!(stateless.layer_slot_metadata().is_none());
    let stateless_plan = stateless.prepare_resident_decoder_copy().unwrap();
    assert!(stateless_plan.shared_layout().is_none());
    assert_eq!(stateless_plan.global_layer_start(), None);
    stateless_plan.visit_operands(&mut |_| operands += 1);
    stateless_plan.visit_retained_arrays(&mut |_| operands += 1);
    assert_eq!(operands, 0);
    assert!(unknown(&paged.prepare_resident_decoder_copy().unwrap_err()));
    assert!(unknown(
        &paged_hybrid.prepare_resident_decoder_copy().unwrap_err()
    ));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(guard);
}
