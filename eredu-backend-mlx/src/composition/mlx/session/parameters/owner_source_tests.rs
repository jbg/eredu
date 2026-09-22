use super::*;
use std::time::Duration;

#[test]
fn displaced_source_borrows_real_maps_and_preserves_empty_and_unknown_states() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let value = Array::from_slice(&[2_i32, -3], &[2]);
    let lazy = value.square(&stream).unwrap();
    let empty = Array::from_slice::<i32>(&[], &[0]);
    empty.evaluated().unwrap();
    let mut state = NativeParameterState::default();
    state.active = Some("descriptive-active-overlay".into());
    state.originals = fixture_rows([
        ("old", MlxTensor::from_array(value.clone())),
        ("old-alias", MlxTensor::from_array(value)),
    ]);
    state.published = fixture_rows([
        ("empty", MlxTensor::from_array(empty)),
        ("lazy", MlxTensor::from_array(lazy.clone())),
    ]);
    let mut guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let counted = state.parameter_sources().count(&mut guard).unwrap();
    assert!(std::ptr::eq(counted.source().state(), &state));
    let old = counted.counts();
    assert_eq!(
        old.role(ParameterOwnerRole::DisplacedOriginal)
            .parameters
            .auxiliary_slots,
        2
    );
    assert_eq!(
        old.role(ParameterOwnerRole::DisplacedOriginal)
            .map_key_bytes,
        0
    );
    assert_eq!(
        old.role(ParameterOwnerRole::PublishedOverlay)
            .parameters
            .auxiliary_slots,
        2
    );
    assert_eq!(
        old.role(ParameterOwnerRole::PublishedOverlay)
            .parameters
            .unknown_backings,
        1
    );
    assert_eq!(
        old.role(ParameterOwnerRole::PublishedOverlay)
            .parameters
            .shape_elements,
        2
    );
    assert_eq!(
        old.role(ParameterOwnerRole::Static).parameters.named_slots,
        0
    );
    drop(counted);
    drop(guard);
    assert_eq!(lazy.allocation_info().unwrap(), None);
    assert_eq!(
        lazy.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
        [4, 9]
    );
    let mut guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let new = state
        .parameter_sources()
        .count(&mut guard)
        .unwrap()
        .counts();
    assert_eq!(
        old.role(ParameterOwnerRole::PublishedOverlay)
            .parameters
            .unknown_backings,
        1
    );
    assert_eq!(
        new.role(ParameterOwnerRole::PublishedOverlay)
            .parameters
            .unknown_backings,
        0
    );
}

#[test]
fn displaced_count_controls_borrow_without_owning_or_drop_work() {
    macro_rules! layout {
        ($ty:ty) => {
            println!(
                "{} size={} align={}",
                stringify!($ty),
                std::mem::size_of::<$ty>(),
                std::mem::align_of::<$ty>()
            );
        };
    }
    layout!(DisplacedParameterSource<'_>);
    layout!(CountedDisplacedParameterSource<'_>);
    layout!(Result<CountedDisplacedParameterSource<'_>, ParameterOwnerSourceError>);
    assert_eq!(
        std::mem::size_of::<DisplacedParameterSource<'_>>(),
        std::mem::size_of::<&NativeParameterState>()
    );
    assert!(!std::mem::needs_drop::<CountedDisplacedParameterSource<'_>>());
    assert!(!std::mem::needs_drop::<ParameterOwnerCounts>());
}
