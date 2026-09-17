use super::*;
use crate::backend::runtime::cache::kv::KeyValueCache;
use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::cache::{kv::PoolingCache, residency::CacheResidencyManager},
    },
    MlxTensor,
};
use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_nn::PoolingAttentionCache as _;
use eredu_runtime::{DeviceState, StateLayout};
use safemlx::{Device, DeviceType};
use std::cell::Cell;

thread_local! {
    static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) };
    pub(super) static FAIL_AFTER_LAYER: Cell<Option<usize>> = const { Cell::new(None) };
}
#[derive(Debug, thiserror::Error)]
#[error("injected pooling decoder failure after a completed layer")]
pub(super) struct InjectedPoolingCopyFailure;
pub(super) fn after_layer(index: usize) -> Result<(), Exception> {
    if FAIL_AFTER_LAYER.with(|fail| fail.get() == Some(index)) {
        FAIL_AFTER_LAYER.with(|fail| fail.set(None));
        return Err(Exception::from_source(InjectedPoolingCopyFailure));
    }
    Ok(())
}
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn layout() -> StateLayout {
    let policies = (0..3)
        .map(|streams| {
            let mut fixed = Vec::new();
            for (index, ratio) in [4, 6].into_iter().take(streams).enumerate() {
                fixed.extend(super::super::pooling_layout_tests::pooling_stream(
                    index as u32,
                    ratio,
                    streams == 2,
                ));
            }
            if fixed.is_empty() {
                LayerCachePolicy::key_only(AttentionPolicy::sliding(7).unwrap(), 1, 8).unwrap()
            } else {
                LayerCachePolicy::key_only_with_fixed_state(
                    AttentionPolicy::sliding(7).unwrap(),
                    1,
                    8,
                    fixed,
                )
                .unwrap()
            }
        })
        .collect();
    StateLayout::new(LayerSchedule::new(3, policies).unwrap()).unwrap()
}
fn payload(positions: i32, seed: f32) -> Array {
    Array::from_slice(
        &(0..2 * positions * 8)
            .map(|i| seed + i as f32 * 0.25)
            .collect::<Vec<_>>(),
        &[2, positions, 8],
    )
}
fn fill_pool(pool: &mut PoolingCache, seed: f32, overlapping: bool, stream: &Stream) {
    let windows = pool
        .accumulate_windows(payload(9, seed), payload(9, seed + 100.), 0, stream)
        .unwrap();
    pool.update_and_fetch(payload(9 / pool.ratio(), seed + 200.), stream)
        .unwrap();
    if overlapping {
        pool.replace_overlap(windows.values, windows.gates);
    }
}
pub(super) fn state(stream: &Stream, populated: bool) -> MlxPoolingAttentionState {
    DeviceState::create(layout(), |index, policy| {
        let mut cache = MlxPoolingAttentionCache::resident_from_policy(index, policy)?;
        if populated {
            cache
                .append_local(MlxTensor::from_array(payload(9, index as f32 + 1.)), stream)
                .unwrap();
            match &mut cache {
                MlxPoolingAttentionCache::Local(_) => {}
                MlxPoolingAttentionCache::Compressed { pool, .. } => {
                    fill_pool(pool, 11., false, stream)
                }
                MlxPoolingAttentionCache::Sparse {
                    pool, index_pool, ..
                } => {
                    fill_pool(pool, 31., true, stream);
                    fill_pool(index_pool, 61., true, stream);
                    let alias = payload(4, 101.);
                    pool.replace_overlap(alias.clone(), alias);
                }
            }
        }
        Ok::<_, Exception>(cache)
    })
    .unwrap()
}
pub(super) fn operands<'a>(plan: &PreparedResidentPoolingCopy<'a>) -> Vec<&'a Array> {
    let mut result = Vec::new();
    plan.visit_operands(&mut |array| result.push(array));
    result
}
pub(super) fn values(plan: &PreparedResidentPoolingCopy<'_>) -> Vec<Vec<f32>> {
    operands(plan)
        .into_iter()
        .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        .collect()
}
pub(super) fn controls(
    plan: &PreparedResidentPoolingCopy<'_>,
) -> Vec<(u8, i32, Option<i32>, Vec<(i32, i32, [bool; 5])>)> {
    (0..plan.len())
        .map(|index| {
            let cache = plan.layer(index).unwrap();
            let local = match cache.local() {
                LiveKeyValueCache::Resident(local) => local,
                _ => unreachable!(),
            };
            let pool = |p: &PoolingCache| {
                (
                    p.ratio(),
                    p.processed_tokens(),
                    p.state_arrays().map(|a| a.is_some()),
                )
            };
            let (kind, pools) = match cache {
                MlxPoolingAttentionCache::Local(_) => (0, vec![]),
                MlxPoolingAttentionCache::Compressed { pool: p, .. } => (1, vec![pool(p)]),
                MlxPoolingAttentionCache::Sparse {
                    pool: p,
                    index_pool,
                    ..
                } => (2, vec![pool(p), pool(index_pool)]),
            };
            (kind, cache.offset(), local.max_size(), pools)
        })
        .collect()
}

#[test]
fn cold_plan_borrows_real_layer_owner_and_keeps_unknown_backing_unknown() {
    let source = state(&stream(), true);
    let metadata = source.layer_slot_metadata().unwrap();
    let before = source
        .as_ref()
        .iter()
        .flat_map(MlxPoolingAttentionCache::retained_arrays)
        .collect::<Vec<_>>();
    assert!(before
        .iter()
        .any(|a| a.try_metadata_snapshot().unwrap().allocation().is_none()));
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = Guard;
    HOUSEKEEPING.with(|calls| calls.set(0));
    let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
    let arrays = operands(&plan);
    let mut retained = Vec::new();
    plan.visit_retained_arrays(&mut |a| retained.push(a));
    assert_eq!(arrays.len(), 19);
    assert_eq!(retained.len(), before.len());
    assert!(retained
        .iter()
        .zip(&before)
        .all(|(a, b)| std::ptr::eq(*a, *b)));
    assert!(plan
        .slot_initialization()
        .unwrap()
        .source_metadata()
        .same_storage(metadata));
    assert!(plan
        .shared_layout()
        .same_storage(source.shared_layout().unwrap()));
    assert!(std::ptr::eq(plan.layer(1).unwrap(), &source.as_ref()[1]));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(guard);
    assert!(arrays
        .iter()
        .any(|a| a.try_metadata_snapshot().unwrap().allocation().is_none()));
}

#[test]
fn paged_and_absent_tables_reject_before_work_while_present_empty_layers_prepare() {
    let present = state(&stream(), false);
    let plan = PreparedResidentPoolingCopy::prepare(&present).unwrap();
    assert_eq!(plan.len(), 3);
    assert!(operands(&plan).is_empty());
    let stateless = DeviceState::<MlxNeuralBackend, MlxPoolingAttentionCache>::stateless();
    assert!(stateless.prepare_layer_copy_slots().unwrap().is_none());
    assert!(stateless.shared_layout().is_none());
    let manager = CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap(),
    )
    .unwrap();
    let paged = DeviceState::create(layout(), |index, policy| {
        MlxPoolingAttentionCache::paged_from_policy(index, policy, manager.clone(), index, 0, None)
    })
    .unwrap();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = Guard;
    HOUSEKEEPING.with(|calls| calls.set(0));
    assert!(matches!(
        PreparedResidentPoolingCopy::prepare(&stateless),
        Err(ResidentPoolingCopyError::AbsentTable)
    ));
    assert!(matches!(
        PreparedResidentPoolingCopy::prepare(&paged),
        Err(ResidentPoolingCopyError::PagedStorage)
    ));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(guard);
    assert!(present.as_ref().iter().all(|layer| layer.offset() == 0));
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
pub(super) mod funded;

#[test]
fn borrowed_pooling_whole_visit_includes_both_streams_pending_and_overlap_slots() {
    use eredu_runtime::{RuntimeLayerState, RuntimeState};
    let stream = stream();
    let source = state(&stream, true);
    let expected = source
        .as_ref()
        .iter()
        .flat_map(MlxPoolingAttentionCache::retained_arrays)
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 19);
    let MlxPoolingAttentionCache::Sparse {
        pool, index_pool, ..
    } = &source.as_ref()[2]
    else {
        panic!("sparse fixture")
    };
    assert_eq!(pool.state_arrays().map(|v| v.is_some()), [true; 5]);
    assert_eq!(index_pool.state_arrays().map(|v| v.is_some()), [true; 5]);
    let before = [pool.processed_tokens(), index_pool.processed_tokens()];
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = Guard;
    HOUSEKEEPING.set(0);
    let mut total = 0;
    source
        .visit_all_retained_values(&mut |value| {
            assert!(std::ptr::eq(value.as_array(), expected[total]));
            total += 1;
        })
        .unwrap();
    let mut per_layer = [0; 3];
    for (index, layer) in source.as_ref().iter().enumerate() {
        RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(layer, &mut |_| {
            per_layer[index] += 1
        });
    }
    assert_eq!(total, 19);
    assert_eq!(per_layer, [2, 5, 12]);
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(guard);
    assert_eq!(
        [pool.processed_tokens(), index_pool.processed_tokens()],
        before
    );
    let empty = state(&stream, false);
    empty
        .visit_all_retained_values(&mut |_| panic!("empty streams retain no tensors"))
        .unwrap();
}
