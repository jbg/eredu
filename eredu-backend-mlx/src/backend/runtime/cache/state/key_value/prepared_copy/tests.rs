use super::*;
use crate::backend::runtime::cache::{
    kv::{ConcatKeyValueCache, KeyValueCache},
    residency::CacheResidencyManager,
};
use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_runtime::{HostSlotTable, StateLayout};
use safemlx::{Device, DeviceType};
use std::cell::Cell;

thread_local! {
    pub(super) static FAIL_AFTER_LAYER: Cell<Option<usize>> = const { Cell::new(None) };
}
pub(super) fn after_layer(index: usize) -> Result<(), Exception> {
    if FAIL_AFTER_LAYER.with(|fail| fail.get() == Some(index)) {
        FAIL_AFTER_LAYER.with(|fail| fail.set(None));
        return Err(Exception::custom(
            "injected resident decoder layer-copy failure",
        ));
    }
    Ok(())
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn layout() -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            3,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
                LayerCachePolicy::NoState,
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}
fn array(seed: f32) -> Array {
    Array::from_slice(&[seed, seed + 1., seed + 2., seed + 3.], &[1, 1, 2, 2])
}
pub(super) fn state(stream: &Stream) -> MlxKeyValueState {
    let mut source = MlxKeyValueState::device_with_global_layer_start(layout(), 13).unwrap();
    let mut padded = ConcatKeyValueCache::new_with_max_size_and_step(8, 4);
    padded
        .update_and_fetch(array(1.), array(11.), stream)
        .unwrap();
    let mut key_only = ConcatKeyValueCache::new_key_only_for_sliding_attention(4);
    key_only
        .update_and_fetch(array(21.), array(31.), stream)
        .unwrap();
    source.layers = HostSlotTable::new(Box::new([
        MlxKeyValueLayerState::Device(padded),
        MlxKeyValueLayerState::Stateless,
        MlxKeyValueLayerState::Device(key_only),
    ]));
    source
}
pub(super) fn operands<'a>(plan: &PreparedResidentKvCopy<'a>) -> Vec<&'a Array> {
    let mut arrays = Vec::new();
    plan.visit_operands(&mut |array| arrays.push(array));
    arrays
}
pub(super) fn values(plan: &PreparedResidentKvCopy<'_>) -> Vec<Vec<f32>> {
    operands(plan)
        .into_iter()
        .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        .collect()
}
pub(super) fn controls(
    plan: &PreparedResidentKvCopy<'_>,
) -> Vec<Option<(i32, Option<i32>, Option<u64>, usize)>> {
    (0..plan.len())
        .map(|index| match plan.layer(index).unwrap() {
            MlxKeyValueLayerState::Stateless => None,
            MlxKeyValueLayerState::Device(cache) => Some((
                cache.offset(),
                cache.max_size(),
                cache.continuation_capacity_bound(1),
                cache.arrays().count(),
            )),
            MlxKeyValueLayerState::Paged(_) => panic!("resident fixture"),
        })
        .collect()
}

#[test]
fn resident_plan_borrows_exact_slots_and_preserves_lazy_alias_operands() {
    let stream = stream();
    let mut source = MlxKeyValueState::device_with_global_layer_start(layout(), 29).unwrap();
    let base = array(3.);
    let lazy = base.transpose_axes(&[0, 1, 3, 2], &stream).unwrap();
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let mut cache = ConcatKeyValueCache::new();
    cache.restore_resident(lazy.clone(), lazy, 2).unwrap();
    source.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache);
    let plan = source.prepare_resident_copy().unwrap();
    assert_eq!(plan.global_layer_start(), 29);
    assert!(plan.shared_layout().same_storage(&source.layout));
    assert!(std::ptr::eq(
        plan.layer(0).unwrap(),
        &source.layers.slots()[0]
    ));
    let arrays = operands(&plan);
    assert_eq!(arrays.len(), 2);
    assert!(arrays
        .iter()
        .all(|a| a.try_metadata_snapshot().unwrap().allocation().is_none()));
    assert_eq!(
        plan.slot_initialization()
            .unwrap()
            .source_metadata()
            .identity(),
        source.layer_slot_metadata().identity()
    );
    assert!(matches!(
        plan.layer(1),
        Some(MlxKeyValueLayerState::Stateless)
    ));
    // Querying this plan has not certified any source backing or copied a slot.
    assert!(source.retained_arrays().iter().all(|a| a
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none()));
}

#[test]
fn paged_source_rejects_before_host_association_or_numerical_work() {
    let manager = CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 4096, 64 << 10, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let source = MlxKeyValueState::paged(layout(), manager.clone(), None).unwrap();
    assert!(source.retained_arrays().is_empty());
    assert!(matches!(
        source.prepare_resident_copy(),
        Err(ResidentKvCopyError::PagedStorage)
    ));
    assert!(source.retained_arrays().is_empty());
    assert_eq!(source.offset(), 0);
    // An empty original snapshot has a real pager even before any arrays exist.
    assert!(source.snapshot_paged_source().unwrap().is_some());
    // The lifecycle keeps a zero-byte end marker after a sealed/cleared tail.
    // It is still a canonical row and must survive exact source-copy census.
    manager.set_tail_state(0, 0, 0).unwrap();
    assert!(source.snapshot_paged_source().unwrap().is_some());
    assert!(source.retained_arrays().is_empty());
    // Presence alone is insufficient: the shared source still checks the exact
    // physical frontier, and the outer census rejects an unrelated layer row.
    manager.set_tail_state(0, 0, 1).unwrap();
    assert!(source.snapshot_paged_source().is_err());
    manager.set_tail_state(0, 0, 0).unwrap();
    manager.set_tail_state(99, 0, 0).unwrap();
    assert!(matches!(source.snapshot_paged_source(),
        Err(crate::backend::runtime::cache::state::SnapshotProjectionCause::ChangedAt("catalog and physical tail population"))));
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
pub(super) mod funded;
