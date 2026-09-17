use super::super::{MlxHybridLayerState, MlxHybridState};
use super::*;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy},
    LayerSchedule,
};
use eredu_runtime::{RuntimeStateComponents, StateError, StateLayout};
use safemlx::{Array, Device, DeviceType, Stream};

const CONV: StateTensorRole = StateTensorRole::Convolution { slot: 3 };
const RECURRENT: StateTensorRole = StateTensorRole::Recurrent;
const PREFIX: StateTensorRole = StateTensorRole::PrefixEmbedding;

fn declaration(role: StateTensorRole) -> StateTensorPolicy {
    StateTensorPolicy::new(
        role,
        vec![
            StateTensorDimension::fixed(2).unwrap(),
            StateTensorDimension::fixed(2).unwrap(),
        ],
        StateTensorDtype::Float32,
        if role == RECURRENT {
            MutableStateResidency::LayerScopedOffloadable
        } else {
            MutableStateResidency::AlwaysDeviceMutable
        },
    )
    .unwrap()
}

fn policy(roles: &[StateTensorRole]) -> LayerCachePolicy {
    LayerCachePolicy::fixed_only(roles.iter().copied().map(declaration).collect()).unwrap()
}

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn tensor(values: [f32; 4]) -> MlxTensor {
    MlxTensor::from_array(Array::from_slice(&values, &[2, 2]))
}

fn values(tensor: &MlxTensor, stream: &Stream) -> Vec<f32> {
    tensor
        .as_array()
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn roles(slots: &FixedStateSlots) -> Vec<StateTensorRole> {
    slots.iter().map(|(role, _)| *role).collect()
}

#[test]
fn declared_roles_are_sorted_and_absent_slots_have_exact_retained_payload() {
    let mut slots = FixedStateSlots::from_policy(&policy(&[PREFIX, CONV, RECURRENT])).unwrap();
    assert_eq!(roles(&slots), [CONV, RECURRENT, PREFIX]);
    assert!(slots.values().all(Option::is_none));
    assert_eq!(
        slots.payload_bytes(),
        Some(3 * std::mem::size_of::<Slot>() as u64)
    );
    assert_eq!(
        slots.payload_bytes(),
        Some(std::mem::size_of_val(slots.slots.slots()) as u64)
    );
    let before = slots.payload_bytes();
    assert!(slots.get_mut(&StateTensorRole::PositionDelta).is_none());
    assert_eq!(roles(&slots), [CONV, RECURRENT, PREFIX]);
    assert_eq!(slots.payload_bytes(), before);
    let empty = FixedStateSlots::from_policy(&LayerCachePolicy::NoState).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.payload_bytes(), Some(0));
    assert_eq!(empty.clone().payload_bytes(), Some(0));
}

#[test]
fn ordinary_clone_preserves_tensor_aliases_but_option_slots_are_independent() {
    let stream = stream();
    let mut source = MlxHybridLayerState::device(0, &policy(&[PREFIX, RECURRENT, CONV])).unwrap();
    *source.fixed_component(CONV).unwrap() = Some(tensor([1., 2., 3., 4.]));
    *source.fixed_component(PREFIX).unwrap() = Some(tensor([5., 6., 7., 8.]));
    source.advance_fixed(7).unwrap();
    let original_payload = source.fixed_slot_payload_bytes();
    let mut copy = source.deep_clone_state().unwrap();
    assert_ne!(
        source.fixed.slots.slots().as_ptr(),
        copy.fixed.slots.slots().as_ptr()
    );
    assert_eq!(copy.fixed_slot_payload_bytes(), original_payload);
    assert!(!source
        .fixed_slot_metadata()
        .same_storage(copy.fixed_slot_metadata()));
    assert_eq!(copy.position(), 7);
    assert!(copy.fixed_component(RECURRENT).unwrap().is_none());
    for ((source_role, source_value), (copy_role, copy_value)) in
        source.fixed.iter().zip(&copy.fixed)
    {
        assert_eq!(source_role, copy_role);
        if let (Some(source), Some(copy)) = (source_value, copy_value) {
            source.as_array().evaluated().unwrap();
            copy.as_array().evaluated().unwrap();
            assert_eq!(
                source
                    .as_array()
                    .allocation_info()
                    .unwrap()
                    .unwrap()
                    .identity(),
                copy.as_array()
                    .allocation_info()
                    .unwrap()
                    .unwrap()
                    .identity()
            );
        }
    }
    *copy.fixed_component(CONV).unwrap() = Some(tensor([11., 12., 13., 14.]));
    *copy.fixed_component(PREFIX).unwrap() = None;
    *copy.fixed_component(RECURRENT).unwrap() = Some(tensor([21., 22., 23., 24.]));
    assert_eq!(
        values(
            source.fixed_component(CONV).unwrap().as_ref().unwrap(),
            &stream
        ),
        [1., 2., 3., 4.]
    );
    assert!(source.fixed_component(PREFIX).unwrap().is_some());
    assert!(source.fixed_component(RECURRENT).unwrap().is_none());
    assert!(matches!(
        copy.fixed_component(StateTensorRole::PositionDelta),
        Err(StateError::UnknownComponent { .. })
    ));
    copy.clear().unwrap();
    assert_eq!(copy.position(), 0);
    assert!(copy.fixed.values().all(Option::is_none));
    assert_eq!(copy.fixed_slot_payload_bytes(), original_payload);
    assert_eq!(roles(&copy.fixed), roles(&source.fixed));
    assert_eq!(source.position(), 7);
}

#[test]
fn clone_from_replaces_all_roles_and_presence_without_spare_slots() {
    let mut source = FixedStateSlots::from_policy(&policy(&[PREFIX, CONV])).unwrap();
    *source.get_mut(&CONV).unwrap() = Some(tensor([2., 4., 6., 8.]));
    let mut destination = FixedStateSlots::from_policy(&policy(&[RECURRENT, CONV])).unwrap();
    *destination.get_mut(&RECURRENT).unwrap() = Some(tensor([3., 6., 9., 12.]));
    let original_allocation = destination.slots.slots().as_ptr();
    let original_metadata = destination.metadata().clone();
    destination.clone_from(&source);
    assert_eq!(destination.slots.slots().as_ptr(), original_allocation);
    assert!(destination.metadata().same_storage(&original_metadata));
    assert_eq!(roles(&destination), [CONV, PREFIX]);
    assert!(destination.get_mut(&CONV).unwrap().is_some());
    assert!(destination.get_mut(&PREFIX).unwrap().is_none());
    assert!(destination.get_mut(&RECURRENT).is_none());
    *destination.get_mut(&CONV).unwrap() = None;
    assert!(source.get_mut(&CONV).unwrap().is_some());
    let larger = FixedStateSlots::from_policy(&policy(&[PREFIX, RECURRENT, CONV])).unwrap();
    destination.clone_from(&larger);
    assert!(!destination.metadata().same_storage(&original_metadata));
    assert!(!destination.metadata().same_storage(larger.metadata()));
    assert_eq!(roles(&destination), [CONV, RECURRENT, PREFIX]);
    assert!(destination.values().all(Option::is_none));
    assert_eq!(
        destination.payload_bytes(),
        Some(std::mem::size_of_val(destination.slots.slots()) as u64)
    );
    assert_eq!(destination.payload_bytes(), larger.payload_bytes());
}

#[test]
fn isolated_fixed_copy_preserves_absent_roles_frontier_and_nonzero_strided_values() {
    let stream = stream();
    let layout =
        StateLayout::new(LayerSchedule::new(1, vec![policy(&[PREFIX, RECURRENT, CONV])]).unwrap())
            .unwrap();
    let mut source = MlxHybridState::device(layout).unwrap();
    for (role, data) in [(CONV, [1., 2., 3., 4.]), (PREFIX, [5., 6., 7., 8.])] {
        let view = Array::from_slice(&data, &[2, 2])
            .transpose(&stream)
            .unwrap();
        *source.layers.slots_mut()[0].fixed_component(role).unwrap() =
            Some(MlxTensor::from_array(view));
    }
    source.layers.slots_mut()[0].advance_fixed(9).unwrap();
    let mut copied = source.isolated_snapshot(&stream).unwrap();
    assert!(!source.layers.slots()[0]
        .fixed_slot_metadata()
        .same_storage(copied.layers.slots()[0].fixed_slot_metadata()));
    assert_eq!(copied.layers.slots_mut()[0].position(), 9);
    assert_eq!(
        roles(&copied.layers.slots_mut()[0].fixed),
        [CONV, RECURRENT, PREFIX]
    );
    assert!(copied.layers.slots_mut()[0]
        .fixed_component(RECURRENT)
        .unwrap()
        .is_none());
    assert_eq!(
        copied.layers.slots_mut()[0].fixed_slot_payload_bytes(),
        source.layers.slots_mut()[0].fixed_slot_payload_bytes()
    );
    for role in [CONV, PREFIX] {
        let original = source.layers.slots_mut()[0]
            .fixed_component(role)
            .unwrap()
            .as_ref()
            .unwrap();
        let copy = copied.layers.slots_mut()[0]
            .fixed_component(role)
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(values(copy, &stream), values(original, &stream));
        original.as_array().evaluated().unwrap();
        copy.as_array().evaluated().unwrap();
        assert_ne!(
            original
                .as_array()
                .allocation_info()
                .unwrap()
                .unwrap()
                .identity(),
            copy.as_array()
                .allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        );
    }
    source.layers.slots_mut()[0].clear().unwrap();
    drop(source);
    assert_eq!(
        values(
            copied.layers.slots_mut()[0]
                .fixed_component(CONV)
                .unwrap()
                .as_ref()
                .unwrap(),
            &stream
        ),
        [1., 3., 2., 4.]
    );
    assert_eq!(
        values(
            copied.layers.slots_mut()[0]
                .fixed_component(PREFIX)
                .unwrap()
                .as_ref()
                .unwrap(),
            &stream
        ),
        [5., 7., 6., 8.]
    );
    assert_eq!(copied.layers.slots_mut()[0].position(), 9);
}

#[test]
fn fallible_value_copy_skips_absent_roles_and_preserves_source_on_failure() {
    let stream = stream();
    let mut source = FixedStateSlots::from_policy(&policy(&[PREFIX, RECURRENT, CONV])).unwrap();
    *source.get_mut(&CONV).unwrap() = Some(tensor([1., 2., 3., 4.]));
    *source.get_mut(&PREFIX).unwrap() = Some(tensor([5., 6., 7., 8.]));
    let source_payload = source.payload_bytes();
    let mut calls = 0;
    let result = source.try_map_values(|value| {
        calls += 1;
        if calls == 2 {
            Err("second present tensor failed")
        } else {
            Ok(value.clone())
        }
    });
    assert_eq!(result.unwrap_err(), "second present tensor failed");
    assert_eq!(calls, 2);
    assert_eq!(source.payload_bytes(), source_payload);
    assert_eq!(roles(&source), [CONV, RECURRENT, PREFIX]);
    assert!(source.get_mut(&RECURRENT).unwrap().is_none());
    assert_eq!(
        values(source.get_mut(&CONV).unwrap().as_ref().unwrap(), &stream),
        [1., 2., 3., 4.]
    );
    assert_eq!(
        values(source.get_mut(&PREFIX).unwrap().as_ref().unwrap(), &stream),
        [5., 6., 7., 8.]
    );
}

#[test]
fn duplicate_private_initialization_preserves_typed_structural_error() {
    let malformed = LayerCachePolicy::FixedState {
        tensors: vec![declaration(CONV), declaration(CONV)],
    };
    assert!(
        matches!(FixedStateSlots::from_policy(&malformed), Err(CachePolicyError::Invalid(message)) if message.contains("duplicate fixed-state tensor role"))
    );
    let error = MlxHybridLayerState::device(0, &malformed).unwrap_err();
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<CachePolicyError>()
        .is_some());
    assert!(malformed.validate().is_err());
}

mod metadata;
