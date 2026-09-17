use super::*;
use crate::backend::{
    nn::workspace::MlxMetalWorkspaceMechanisms, runtime::cache::kv::KeyValueCache,
};
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::workspace::WorkspaceTensor;
use eredu_runtime::RuntimeLayerState;
use safemlx::{Device, DeviceType};
use std::cell::Cell;

fn layout() -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
                LayerCachePolicy::key_value(AttentionPolicy::sliding(8).unwrap(), 1, 2).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}
fn source() -> MlxHybridState {
    MlxHybridState::device_with_global_layer_start(layout(), 17).unwrap()
}
fn unknown<T>(result: Result<T, Error>) {
    match result {
        Err(Error::Other(e)) => assert_eq!(
            e.downcast_ref::<WorkingMemoryError>(),
            Some(&WorkingMemoryError::UnknownBound)
        ),
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("unsupported payload was accepted"),
    }
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct Cold;
impl Drop for Cold {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn actual_outer_geometry_and_empty_children_prepare_without_native_work() {
    let source = source();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let _cold = Cold;
    HOUSEKEEPING.set(0);
    let plan = PreparedHybridKvCopy::prepare(&source).unwrap();
    let saved = plan.slot_initialization().unwrap();
    let dense = plan.dense_initialization().unwrap();
    assert!(saved
        .source_metadata()
        .same_storage(source.layers.metadata()));
    assert!(dense
        .source_metadata()
        .same_storage(source.layers.metadata()));
    assert!(std::ptr::eq(
        saved.source_at(0).unwrap(),
        &source.layers.slots()[0]
    ));
    assert_eq!(
        dense.retained_bytes(),
        (2 * std::mem::size_of::<MlxHybridLayerState>()) as u64
    );
    assert_eq!(
        saved.retained_bytes(),
        (2 * std::mem::size_of::<Option<MlxHybridLayerState>>()) as u64
    );
    assert_eq!(plan.global_layer_start(), 17);
    assert!(plan.shared_layout().same_storage(&source.layout));
    assert!(source
        .layers
        .slots()
        .iter()
        .all(|s| s.fixed.is_empty() && s.fixed.payload_bytes() == Some(0)));
    assert_eq!(HOUSEKEEPING.get(), 0);
}

#[test]
fn absent_fixed_values_still_reject_real_role_payload_and_missing_attention() {
    let mut source = source();
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Convolution { slot: 0 },
        vec![StateTensorDimension::fixed(1).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    source.layers.slots_mut()[0].fixed =
        FixedStateSlots::from_policy(&LayerCachePolicy::FixedState {
            tensors: vec![fixed],
        })
        .unwrap();
    assert!(source.layers.slots()[0].fixed.values().all(Option::is_none));
    assert!(source.layers.slots()[0].fixed.payload_bytes().unwrap() > 0);
    unknown(PreparedHybridKvCopy::prepare(&source));
    source.layers.slots_mut()[0].fixed =
        FixedStateSlots::from_policy(&LayerCachePolicy::NoState).unwrap();
    source.layers.slots_mut()[0].attention = None;
    unknown(PreparedHybridKvCopy::prepare(&source));
    source.layers.slots_mut()[0].attention = Some(MlxHybridAttentionState::Compressed(
        CompressedLatentCache::new(),
    ));
    unknown(PreparedHybridKvCopy::prepare(&source));
}

fn populated(stream: &Stream) -> MlxHybridState {
    let mut source = source();
    let array = Array::from_slice(&[1_f32, 3., 5., 7., 9., 11.], &[1, 1, 3, 2]);
    for layer in source.layers.slots_mut() {
        let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) =
            &mut layer.attention
        else {
            unreachable!()
        };
        cache
            .update_and_fetch(array.clone(), array.clone(), stream)
            .unwrap();
        layer.fixed_offset = 4;
    }
    for array in source.retained_arrays() {
        array.evaluated().unwrap();
    }
    source
}
fn arrays(layer: &MlxHybridLayerState) -> Vec<&Array> {
    let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) =
        &layer.attention
    else {
        unreachable!()
    };
    let mut values = Vec::new();
    cache
        .prepare_isolated_copy()
        .visit_operands(&mut |a| values.push(a));
    values
}
#[test]
fn numerical_leaf_copies_aliases_and_only_recreates_empty_child_metadata() {
    // This leaf test owns unquoted values. End-to-end family tests exercise
    // the actual joined account, saved publication and fresh prompt admission.
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = populated(&stream);
    let plan = PreparedHybridKvCopy::prepare(&source).unwrap();
    let roots = RefCell::new(Vec::new());
    let copied = plan.copy_layer(0, &stream, &roots, None).unwrap();
    assert_eq!(copied.fixed_offset, 4);
    assert!(copied.fixed.is_empty());
    assert_eq!(copied.fixed.payload_bytes(), Some(0));
    assert!(!copied
        .fixed
        .metadata()
        .same_storage(source.layers.slots()[0].fixed.metadata()));
    let old = arrays(&source.layers.slots()[0]);
    let new = arrays(&copied);
    for (a, b) in old.iter().zip(&new) {
        assert_eq!(
            a.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            b.evaluated().unwrap().try_to_vec::<f32>().unwrap()
        );
        assert_ne!(
            a.allocation_info().unwrap().unwrap().identity(),
            b.allocation_info().unwrap().unwrap().identity()
        );
    }
    assert_ne!(
        new[0].allocation_info().unwrap().unwrap().identity(),
        new[1].allocation_info().unwrap().unwrap().identity()
    );
    assert_eq!(roots.borrow().len(), 4);
    assert_eq!(
        arrays(&source.layers.slots()[0])[0]
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![1., 3., 5., 7., 9., 11.]
    );
}

#[test]
fn copied_workspace_uses_independent_roots_and_preserves_full_sliding_controls() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = populated(&stream);
    let plan = PreparedHybridKvCopy::prepare(&source).unwrap();
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let copied = plan
        .project_dense_workspace(NonZeroU32::new(1).unwrap(), &context)
        .unwrap();
    assert!(copied.source_storage.is_complete());
    assert_eq!(copied.source_storage.iter().len(), 1);
    let roots: Vec<WorkspaceTensor> = copied
        .state
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
        .cloned()
        .collect();
    assert_eq!(roots.len(), 4);
    context.begin_state_span(&roots).unwrap();
    let retained = context
        .report(&roots)
        .unwrap()
        .state
        .unwrap()
        .retained_bytes
        .unwrap();
    let individual: u64 = roots
        .iter()
        .map(|root| {
            context
                .report(std::slice::from_ref(root))
                .unwrap()
                .state
                .unwrap()
                .retained_bytes
                .unwrap()
        })
        .sum();
    assert!(retained > 0);
    assert_eq!(
        retained, individual,
        "every copied role has an independent bounded root"
    );
    assert!(copied.copy.state.as_ref().unwrap().retained_bytes.is_some());
    assert!(copied
        .copy
        .state
        .as_ref()
        .unwrap()
        .transient_bytes
        .is_some());
    assert_eq!(
        eredu_nn::AttentionCache::offset(&source.layers.slots()[1]),
        3
    );
    assert_eq!(copied.state.layout(), source.layout.layout());
}
