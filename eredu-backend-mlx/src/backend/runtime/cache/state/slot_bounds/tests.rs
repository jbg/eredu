use super::super::*;
use super::*;
use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use safemlx::{Device, DeviceType};

fn count_values<S: RuntimeState<MlxNeuralBackend>>(source: &S) -> usize {
    let mut count = 0;
    source
        .visit_all_retained_values(&mut |_| count += 1)
        .unwrap();
    count
}
#[test]
fn absent_kv_fields_populate_without_changing_fixed_slot_ceiling() {
    let layout = StateLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap(),
                LayerCachePolicy::NoState,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxKeyValueState::device(layout).unwrap();
    let expected = NativeStateSlotCounts {
        arrays: 2,
        layouts: 1,
        slot_tables: 1,
        manager_roles: 0,
    };
    assert_eq!(state.retained_owner_slot_counts(), Some(expected));
    assert_eq!(count_values(&state), 0);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    KeyValueCache::update_for_attention(
        state.layer(0).unwrap(),
        Array::from_slice(&[3_f32, 5.], &[1, 1, 1, 2]),
        Array::from_slice(&[7_f32, 11.], &[1, 1, 1, 2]),
        &stream,
    )
    .unwrap();
    assert_eq!(count_values(&state), 2);
    assert_eq!(state.retained_owner_slot_counts(), Some(expected));
}

#[test]
fn fixed_role_tables_count_absent_optional_values_and_keep_exact_host_extent() {
    let role = StateTensorRole::Recurrent;
    let policy = StateTensorPolicy::new(
        role,
        vec![StateTensorDimension::fixed(2).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap()
    .optional();
    let layout = StateLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::fixed_only(vec![policy]).unwrap()]).unwrap(),
    )
    .unwrap();
    let mut state = MlxHybridState::device(layout).unwrap();
    let expected = NativeStateSlotCounts {
        arrays: 1,
        layouts: 1,
        slot_tables: 2,
        manager_roles: 0,
    };
    assert_eq!(state.retained_owner_slot_counts(), Some(expected));
    assert_eq!(count_values(&state), 0);
    let identity = state
        .fixed_slot_metadata()
        .next()
        .unwrap()
        .identity()
        .clone();
    *state.layers_mut()[0].fixed_component(role).unwrap() = Some(MlxTensor::from_array(
        Array::from_slice(&[13_f32, 17.], &[2]),
    ));
    assert_eq!(count_values(&state), 1);
    assert_eq!(state.retained_owner_slot_counts(), Some(expected));
    assert_eq!(
        *state.fixed_slot_metadata().next().unwrap().identity(),
        identity
    );
}

#[test]
fn concrete_pooling_variants_reserve_all_future_fields_and_checked_sum_rejects_overflow() {
    let local = || LiveKeyValueCache::resident(ConcatKeyValueCache::new());
    let a = MlxPoolingAttentionCache::Local(local());
    let b = MlxPoolingAttentionCache::Compressed {
        local: local(),
        pool: PoolingCache::new(2).unwrap(),
    };
    let c = MlxPoolingAttentionCache::Sparse {
        local: local(),
        pool: PoolingCache::new(2).unwrap(),
        index_pool: PoolingCache::new(2).unwrap(),
    };
    for (value, count) in [(a, 2), (b, 7), (c, 12)] {
        let mut seen = 0;
        RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(&value, &mut |_| seen += 1);
        assert_eq!(seen, 0);
        assert_eq!(value.retained_owner_slot_counts().arrays, count);
    }
    assert_eq!(
        NativeStateSlotCounts::arrays(usize::MAX, 0)
            .checked_add(NativeStateSlotCounts::arrays(1, 0)),
        None
    );
}
