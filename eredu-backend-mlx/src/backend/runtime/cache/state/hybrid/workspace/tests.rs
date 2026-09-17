use super::*;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::{
    workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound},
    Tensor,
};
use safemlx::{Device, DeviceType};
use std::collections::BTreeMap;

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        panic!("projection must not replay equations")
    }
}
fn context() -> WorkspaceContext {
    WorkspaceContext::new(NoEquations)
}
fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}
fn retained(
    state: &eredu_runtime::DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
) -> Vec<eredu_nn::workspace::WorkspaceTensor> {
    state
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
        .cloned()
        .collect()
}

#[test]
#[ignore = "requires native CPU execution"]
fn resident_state_projection_preserves_cross_layer_aliases_and_lazy_fixed_slots() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let role = StateTensorRole::Convolution { slot: 3 };
    let fixed = StateTensorPolicy::new(
        role,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(3).unwrap(),
            StateTensorDimension::fixed(8).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(
            3,
            vec![
                LayerCachePolicy::FixedState {
                    tensors: vec![fixed.clone()],
                },
                LayerCachePolicy::FixedState {
                    tensors: vec![fixed],
                },
                LayerCachePolicy::key_value(AttentionPolicy::sliding(4).unwrap(), 2, 8).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut native = MlxHybridState::device(layout.clone()).unwrap();
    let shared = Array::from_slice(
        &(0..48).map(|n| n as f32 + 0.25).collect::<Vec<_>>(),
        &[2, 3, 8],
    );
    for layer in &mut native.layers.slots_mut()[..2] {
        *layer.fixed_component(role).unwrap() = Some(MlxTensor::from(shared.clone()));
        layer.advance_fixed(7).unwrap();
    }
    let history = Array::from_slice(
        &(0..224).map(|n| n as f32 + 1.0).collect::<Vec<_>>(),
        &[2, 2, 7, 8],
    );
    let input = MlxTensor::from(history.clone());
    AttentionCache::update_for_attention(
        &mut native.layers.slots_mut()[2],
        input.clone(),
        input,
        &stream,
    )
    .unwrap();
    for array in native.retained_arrays() {
        array.evaluated().unwrap();
    }
    let allocations: BTreeMap<_, _> = native
        .retained_arrays()
        .into_iter()
        .map(|array| {
            let info = array.allocation_info().unwrap().unwrap();
            (info.identity(), info.bytes() as u64)
        })
        .collect();
    assert_eq!(allocations.len(), 2);
    let context = context();
    let projected = native
        .project_resident_workspace_with_storage(nz(2), &context)
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
        native.retained_storage().unwrap().array_allocation_facts()
    );
    let metadata = projected.state;
    assert_eq!(metadata.layout(), &layout);
    let roots = retained(&metadata);
    context.begin_state_span(&roots).unwrap();
    let report = context.report(&roots).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(
        report.state.unwrap().retained_bytes,
        Some(allocations.values().sum())
    );
    assert!(metadata.as_ref().iter().all(|layer| layer.position() == 7));
    assert_eq!(roots[2].shape(), [2, 2, 3, 8]);
    assert!(native.project_resident_workspace(nz(1), &context).is_err());

    // A lazy replacement has known geometry but no observed backing. Import
    // must preserve that gap without evaluating it or inventing an allocation.
    let lazy = shared.square(&stream).unwrap();
    *native.layers.slots_mut()[0].fixed_component(role).unwrap() =
        Some(MlxTensor::from(lazy.clone()));
    let unknown = native
        .project_resident_workspace_with_storage(nz(2), &context)
        .unwrap();
    assert!(!unknown.storage.is_complete());
    let roots = retained(&unknown.state);
    context.begin_state_span(&roots).unwrap();
    assert_eq!(
        context.report(&roots).unwrap().inference_transient_bytes(),
        None
    );
    assert_eq!(lazy.allocation_info().unwrap(), None);
    let unchanged = shared.evaluated().unwrap();
    assert_eq!(unchanged.as_slice::<f32>()[0], 0.25);
    let wrong_dtype = shared.as_dtype(safemlx::Dtype::Bfloat16, &stream).unwrap();
    *native.layers.slots_mut()[0].fixed_component(role).unwrap() =
        Some(MlxTensor::from(wrong_dtype.clone()));
    assert!(native.project_resident_workspace(nz(2), &context).is_err());
    assert_eq!(wrong_dtype.allocation_info().unwrap(), None);
}

#[test]
#[ignore = "requires native CPU execution"]
fn resident_state_projection_preserves_key_only_and_empty_sliding_frontiers() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for attention in [
        AttentionPolicy::Full,
        AttentionPolicy::sliding(4).unwrap(),
        AttentionPolicy::sliding(1).unwrap(),
    ] {
        for key_only in [false, true] {
            let policy = if key_only {
                LayerCachePolicy::key_only(attention, 2, 8).unwrap()
            } else {
                LayerCachePolicy::key_value(attention, 2, 8).unwrap()
            };
            let layout =
                StateLayout::new(LayerSchedule::new(2, vec![policy.clone(), policy]).unwrap())
                    .unwrap();
            let mut ordinary =
                (!key_only).then(|| MlxKeyValueState::device(layout.clone()).unwrap());
            let mut hybrid = key_only.then(|| MlxHybridState::device(layout).unwrap());
            let array = Array::from_slice(
                &(0..224).map(|n| n as f32 + 0.5).collect::<Vec<_>>(),
                &[2, 2, 7, 8],
            );
            let context = context();
            let metadata = if let Some(native) = &mut ordinary {
                for layer in native.layers.slots_mut() {
                    KeyValueCache::update_for_attention(
                        layer,
                        array.clone(),
                        array.clone(),
                        &stream,
                    )
                    .unwrap();
                }
                for value in native.retained_arrays() {
                    value.evaluated().unwrap();
                }
                let projected = native
                    .project_resident_workspace_with_storage(nz(2), &context)
                    .unwrap();
                assert!(projected.storage.is_complete());
                assert_eq!(
                    projected
                        .storage
                        .iter()
                        .filter(|(_, bytes, _)| *bytes != 0)
                        .map(|(id, bytes, _)| (id, bytes))
                        .collect::<BTreeMap<_, _>>(),
                    native.retained_storage().unwrap().array_allocation_facts()
                );
                projected.state
            } else {
                let native = hybrid.as_mut().unwrap();
                for layer in native.layers.slots_mut() {
                    let input = MlxTensor::from(array.clone());
                    AttentionCache::update_for_attention(layer, input.clone(), input, &stream)
                        .unwrap();
                }
                for value in native.retained_arrays() {
                    value.evaluated().unwrap();
                }
                let projected = native
                    .project_resident_workspace_with_storage(nz(2), &context)
                    .unwrap();
                assert!(projected.storage.is_complete());
                assert_eq!(
                    projected
                        .storage
                        .iter()
                        .filter(|(_, bytes, _)| *bytes != 0)
                        .map(|(id, bytes, _)| (id, bytes))
                        .collect::<BTreeMap<_, _>>(),
                    native.retained_storage().unwrap().array_allocation_facts()
                );
                projected.state
            };
            let roots = retained(&metadata);
            context.begin_state_span(&roots).unwrap();
            let report = context.report(&roots).unwrap();
            assert!(report.operations.is_empty());
            assert!(metadata.as_ref().iter().all(|layer| layer.position() == 7));
            let expected = if attention.sliding_window_i32().unwrap() == Some(1) {
                0
            } else {
                array.allocation_info().unwrap().unwrap().bytes() as u64
            };
            assert_eq!(report.state.unwrap().retained_bytes, Some(expected));
        }
    }
}
