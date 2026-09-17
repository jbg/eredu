use super::*;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{Index, Tensor, workspace::*};
use safemlx::{Device, DeviceType};
use std::collections::BTreeMap;

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, ComputeError> {
        panic!("cold projection must not replay native state")
    }
}

pub(super) fn layout(streams: usize, window: u32) -> StateLayout {
    let tensors = (0..streams)
        .flat_map(|id| {
            super::super::pooling_layout_tests::pooling_stream(
                id as u32,
                if id == 0 { 4 } else { 6 },
                streams == 2,
            )
        })
        .collect::<Vec<_>>();
    let policy = if tensors.is_empty() {
        LayerCachePolicy::key_only(AttentionPolicy::sliding(window).unwrap(), 1, 8).unwrap()
    } else {
        LayerCachePolicy::key_only_with_fixed_state(
            AttentionPolicy::sliding(window).unwrap(),
            1,
            8,
            tensors,
        )
        .unwrap()
    };
    StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap()
}

pub(super) fn step(
    cache: &mut MlxPoolingAttentionCache,
    streams: usize,
    count: i32,
    stream: &Stream,
) {
    let offset = cache.offset();
    let values = MlxTensor::from_f32_slice(
        &(0..2 * count * 8)
            .map(|n| 0.25 + n as f32 / 32.)
            .collect::<Vec<_>>(),
        &[2, count, 8],
        stream,
    )
    .unwrap();
    let gates = values.multiply_scalar(-0.5, stream).unwrap();
    cache.append_local(values.clone(), stream).unwrap();
    for id in 0..streams as u32 {
        let ratio = cache.pooling_ratio(id).unwrap();
        let ready = cache
            .accumulate_pooling_windows(id, values.clone(), gates.clone(), offset, stream)
            .unwrap();
        let complete = ready.values.shape()[1];
        if streams == 2 && complete > 0 {
            let tail = |value: MlxTensor| {
                value
                    .index(
                        &[
                            Index::Full,
                            Index::Range(complete - ratio, complete),
                            Index::Full,
                        ],
                        stream,
                    )
                    .unwrap()
            };
            cache
                .replace_pooling_overlap(id, tail(ready.values), tail(ready.gates))
                .unwrap();
        }
        let pooled = MlxTensor::full_f32(1.25, &[2, complete / ratio, 8], stream).unwrap();
        cache.append_pooled(id, pooled, stream).unwrap();
        if let Some(mask) = cache.pooling_mask(id, count, offset, stream).unwrap() {
            let columns = (offset + count) / ratio;
            let expected = (0..count)
                .flat_map(|q| (0..columns).map(move |p| (p + 1) * ratio <= offset + q + 1))
                .collect::<Vec<_>>();
            assert_eq!(
                mask.as_array().evaluated().unwrap().as_slice::<bool>(),
                expected
            );
        }
    }
}

fn arrays(state: &MlxPoolingAttentionState) -> Vec<&Array> {
    state
        .as_ref()
        .iter()
        .flat_map(MlxPoolingAttentionCache::retained_arrays)
        .collect()
}

fn check(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    for streams in 0..=2 {
        for window in [1, 5] {
            let mut native =
                MlxPoolingAttentionStateFactory::device(layout(streams, window)).unwrap();
            for count in [3, 4, 1, 5, 1] {
                step(native.layer(0).unwrap(), streams, count, &stream);
                for array in arrays(&native) {
                    array.evaluated().unwrap();
                }
                let allocations = arrays(&native)
                    .into_iter()
                    .map(|array| {
                        let info = array.allocation_info().unwrap().unwrap();
                        (info.identity(), info.bytes() as u64)
                    })
                    .collect::<BTreeMap<_, _>>();
                let context = WorkspaceContext::new(NoEquations);
                let projected =
                    MlxPoolingAttentionStateFactory::project_resident_workspace_with_storage(
                        &native,
                        NonZeroU32::new(2).unwrap(),
                        &context,
                    )
                    .unwrap();
                assert!(projected.storage.is_complete());
                assert_eq!(
                    projected
                        .storage
                        .iter()
                        .map(|(id, bytes, root)| {
                            assert_eq!(root.capacity_bytes(), Some(bytes));
                            (id, bytes)
                        })
                        .collect::<BTreeMap<_, _>>(),
                    allocations
                );
                let metadata = projected.state;
                let roots = metadata
                    .as_ref()
                    .iter()
                    .flat_map(RuntimeLayerState::retained_values)
                    .cloned()
                    .collect::<Vec<_>>();
                context.begin_state_span(&roots).unwrap();
                let report = context.report(&roots).unwrap();
                assert!(report.operations.is_empty());
                assert_eq!(
                    report.state.unwrap().retained_bytes,
                    Some(allocations.values().sum())
                );
                assert_eq!(metadata.as_ref()[0].position(), native.as_ref()[0].offset());
                assert_eq!(roots.len(), arrays(&native).len());
                if !roots.is_empty() {
                    assert!(
                        MlxPoolingAttentionStateFactory::project_resident_workspace(
                            &native,
                            NonZeroU32::new(1).unwrap(),
                            &context
                        )
                        .is_err()
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires native CPU execution"]
fn pooling_projection_tracks_native_partial_windows_overlap_and_aliases_cpu() {
    check(DeviceType::Cpu);
}

#[test]
#[ignore = "requires native Metal execution"]
fn pooling_projection_tracks_native_partial_windows_overlap_and_aliases_metal() {
    check(DeviceType::Gpu);
}

#[test]
#[ignore = "requires native CPU execution"]
fn pooling_projection_preserves_unknown_backing_without_evaluating_or_replaying() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut native = MlxPoolingAttentionStateFactory::device(layout(2, 5)).unwrap();
    step(native.layer(0).unwrap(), 2, 3, &stream);
    assert!(
        arrays(&native)
            .iter()
            .any(|array| array.allocation_info().unwrap().is_none())
    );
    let context = WorkspaceContext::new(NoEquations);
    let projected = MlxPoolingAttentionStateFactory::project_resident_workspace_with_storage(
        &native,
        NonZeroU32::new(2).unwrap(),
        &context,
    )
    .unwrap();
    assert!(!projected.storage.is_complete());
    let metadata = projected.state;
    let roots = metadata
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::retained_values)
        .cloned()
        .collect::<Vec<_>>();
    context.begin_state_span(&roots).unwrap();
    assert_eq!(
        context
            .report(&roots)
            .unwrap()
            .state
            .unwrap()
            .retained_bytes,
        None
    );
    assert!(
        arrays(&native)
            .iter()
            .any(|array| array.allocation_info().unwrap().is_none())
    );
}
