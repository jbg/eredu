use super::*;
use crate::backend::runtime::cache::kv::KeyValueCache;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
    WorkspaceOperationKind,
};
use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType};

#[derive(Debug)]
struct Unpriced;
impl WorkspaceMechanisms for Unpriced {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn array(positions: usize, seed: f32) -> Array {
    Array::from_slice(
        &(0..positions * 4)
            .map(|n| seed + n as f32 * 0.25)
            .collect::<Vec<_>>(),
        &[2, 1, positions as i32, 2],
    )
}

fn values(cache: &ConcatKeyValueCache) -> Vec<Vec<f32>> {
    cache
        .arrays()
        .map(|array| array.evaluated().unwrap().try_to_vec::<f32>().unwrap())
        .collect()
}

fn fields(cache: &ConcatKeyValueCache) -> (bool, i32, i32, i32, i32, Option<i32>, Option<i32>) {
    (
        cache.key_only,
        cache.offset,
        cache.length,
        cache.capacity,
        cache.step,
        cache.max_size,
        cache.attention_window,
    )
}

#[test]
fn empty_and_lazy_prepared_copies_do_not_invent_completed_source_facts() {
    let context = WorkspaceContext::new(Unpriced);
    let empty = ConcatKeyValueCache::new();
    let (complete, report) = empty
        .prepare_isolated_copy()
        .component_report(&context)
        .unwrap();
    assert!(complete);
    assert_eq!(report.total_bytes, Some(0));
    assert_eq!(report.state.unwrap().retained_bytes, Some(0));
    assert!(report.operations.is_empty());
    assert!(empty
        .isolated_snapshot(&stream())
        .unwrap()
        .arrays()
        .next()
        .is_none());

    let stream = stream();
    let keys = array(3, 1.).transpose_axes(&[0, 1, 3, 2], &stream).unwrap();
    let value = keys.clone();
    let mut cache = ConcatKeyValueCache::new();
    cache.restore_resident(keys, value, 2).unwrap();
    let context = WorkspaceContext::new(Unpriced);
    let (complete, report) = cache
        .prepare_isolated_copy()
        .component_report(&context)
        .unwrap();
    assert!(!complete);
    assert!(report.state.unwrap().retained_bytes.is_none());
    assert_eq!(report.operations.len(), 4);
    assert!(cache.arrays().all(|array| array
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none()));
}

#[test]
fn actual_prepared_copy_preserves_key_only_capacity_frontier_and_window_behavior() {
    let stream = stream();
    for key_only in [false, true] {
        for sliding in [false, true] {
            let mut source = if sliding {
                ConcatKeyValueCache::new_for_sliding_attention(4)
            } else {
                ConcatKeyValueCache::new_with_max_size_and_step(16, 8)
            };
            source.key_only = key_only;
            source
                .update_and_fetch(array(3, 1.), array(3, 101.), &stream)
                .unwrap();
            source
                .update_and_fetch(array(2, 11.), array(2, 111.), &stream)
                .unwrap();
            let expected = values(&source);
            let before_fields = fields(&source);
            let source_backing = source
                .arrays()
                .map(|array| array.allocation_info().unwrap().unwrap().identity())
                .collect::<Vec<_>>();
            let context = WorkspaceContext::new(Unpriced);
            let (complete, report) = source
                .prepare_isolated_copy()
                .component_report(&context)
                .unwrap();
            assert!(complete);
            assert_eq!(report.operations.len(), if key_only { 2 } else { 4 });
            for pair in report.operations.chunks_exact(2) {
                assert!(matches!(pair[0].kind, WorkspaceOperationKind::Contiguous));
                assert!(matches!(pair[1].kind, WorkspaceOperationKind::DeepCopy));
            }
            let mut copied = source.isolated_snapshot(&stream).unwrap();
            assert_eq!(fields(&copied), before_fields);
            assert_eq!(values(&copied), expected);
            for (index, array) in copied.arrays().enumerate() {
                assert_ne!(
                    array.allocation_info().unwrap().unwrap().identity(),
                    source_backing[index]
                );
            }
            copied
                .update_and_fetch(array(1, 51.), array(1, 151.), &stream)
                .unwrap();
            assert_eq!(copied.offset, 6);
            assert_eq!(copied.length, if sliding { 3 } else { 6 });
            assert_eq!(copied.capacity, if sliding { 3 } else { 8 });
            assert_eq!(fields(&source), before_fields);
            assert_eq!(values(&source), expected);
            assert_ne!(values(&copied), expected);
        }
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn aliased_keys_and_values_have_one_opening_root_and_two_priced_destinations() {
    use crate::backend::nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms};

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let root = array(16, 1.);
    let keys = root.try_index_device((.., .., 3..6, ..), &stream).unwrap();
    let values = root.try_index_device((.., .., 7..10, ..), &stream).unwrap();
    let mut cache = ConcatKeyValueCache::new();
    cache.restore_resident(keys, values, 3).unwrap();
    let expected = self::values(&cache);
    let source_info = cache
        .keys
        .as_ref()
        .unwrap()
        .allocation_info()
        .unwrap()
        .unwrap();
    assert_eq!(
        cache
            .values
            .as_ref()
            .unwrap()
            .allocation_info()
            .unwrap()
            .unwrap(),
        source_info
    );
    assert!(source_info.bytes() > cache.keys.as_ref().unwrap().nbytes());
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let plan = cache.prepare_isolated_copy();
    let (complete, report) = plan.component_report(&context).unwrap();
    assert!(complete);
    assert_eq!(report.operations.len(), 4);
    assert_eq!(report.host_workspace_bytes, Some(0));
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
    let mut projection = ExistingArrayProjection::new(&context);
    for source in cache.arrays() {
        projection.project(source).unwrap();
    }
    let inventory = projection.into_storage();
    assert_eq!(inventory.iter().len(), 1);
    assert_eq!(
        inventory.iter().next().unwrap().1,
        source_info.bytes() as u64
    );
    let copied = plan.copy(&stream).unwrap();
    assert_eq!(self::values(&copied), expected);
    let destinations = copied
        .arrays()
        .map(|array| array.allocation_info().unwrap().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(destinations.len(), 2);
    assert_ne!(destinations[0].identity(), destinations[1].identity());
    assert!(destinations
        .iter()
        .all(|info| info.identity() != source_info.identity()));
    let destination_bytes = destinations
        .iter()
        .map(|info| info.bytes() as u64)
        .sum::<u64>();
    assert!(report.retained_bytes.unwrap() >= destination_bytes);
    assert!(
        report.state.unwrap().retained_bytes.unwrap()
            >= source_info.bytes() as u64 + destination_bytes
    );
    assert!(report.transient_bytes.unwrap() > 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
fn priced_primitives_do_not_make_a_lazy_source_component_complete() {
    use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;

    let stream = stream();
    let keys = array(3, 1.).transpose_axes(&[0, 1, 3, 2], &stream).unwrap();
    let mut cache = ConcatKeyValueCache::new_key_only();
    cache.restore_resident(keys.clone(), keys, 2).unwrap();
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let (complete, report) = cache
        .prepare_isolated_copy()
        .component_report(&context)
        .unwrap();
    assert!(!complete);
    assert!(report.total_bytes.is_some());
    assert!(report.state.unwrap().retained_bytes.is_none());
    assert!(cache.arrays().all(|array| array
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none()));
}

#[test]
fn retained_copy_uses_padded_slots_and_keeps_independent_outputs_until_collector_retires() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let stream = stream();
    for key_only in [false, true] {
        let mut source = ConcatKeyValueCache::new_with_max_size_and_step(16, 8);
        source.key_only = key_only;
        source
            .update_and_fetch(array(3, 1.), array(3, 101.), &stream)
            .unwrap();
        if !key_only {
            source.values = source.keys.clone();
        }
        let expected = values(&source);
        let before = fields(&source);
        let plan = source.prepare_isolated_copy();
        let mut operands = Vec::new();
        plan.visit_operands(&mut |array| operands.push(array));
        let count = if key_only { 1 } else { 2 };
        assert_eq!(operands.len(), count);
        assert!(std::ptr::eq(operands[0], source.keys.as_ref().unwrap()));
        if !key_only {
            assert!(std::ptr::eq(operands[1], source.values.as_ref().unwrap()));
            assert_eq!(
                operands[0].allocation_info().unwrap(),
                operands[1].allocation_info().unwrap()
            );
        }
        assert!(operands.iter().all(|array| array.dim(2) == source.capacity));
        assert!(source.capacity > source.length);
        let roots = RefCell::new(Vec::new());
        let copied = plan.copy_retained(&stream, &roots).unwrap();
        assert_eq!(fields(&copied), before);
        assert_eq!(values(&copied), expected);
        assert_eq!(roots.borrow().len(), 2 * count);
        let retired = Arc::new(AtomicUsize::new(0));
        let mut allocations = Vec::new();
        for array in copied.arrays() {
            let info = array.allocation_info().unwrap().unwrap();
            assert!(!allocations.contains(&info.identity()));
            allocations.push(info.identity());
            array
                .retain_allocation_owner(Retired(retired.clone()))
                .unwrap();
        }
        drop(operands);
        drop((copied, source));
        stream.synchronize().unwrap();
        safemlx::reclaim_allocation_owners();
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(roots);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            safemlx::reclaim_allocation_owners();
            retired.load(Ordering::SeqCst) == count
        });
    }
}
