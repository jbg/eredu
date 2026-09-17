use super::*;
use crate::backend::runtime::cache::state::MlxPoolingAttentionCache;
use eredu_architectures::prediction_extension::{
    MaterializedDeepSeekV4Prediction, MaterializedInklingPrediction,
};
use eredu_nn::{AttentionCache, PoolingAttentionCache};
use eredu_runtime::RuntimeStateComponents;

#[test]
fn retained_prediction_storage_keeps_nonzero_state_prototypes_and_their_aliases() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let memory = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = crate::backend::managed_memory::NativeMemoryOwner::acquire(&memory).unwrap();
    let policy = eredu_core::cache::LayerCachePolicy::key_only(
        eredu_core::AttentionPolicy::sliding(8).unwrap(),
        1,
        2,
    )
    .unwrap();
    let mut pool = MlxPoolingAttentionCache::resident_from_policy(0, &policy).unwrap();
    let keys = MlxTensor::from_array(Array::from_slice(&[0.5f32, -1.5, 3.0, 7.0], &[1, 2, 2]));
    PoolingAttentionCache::append_local(&mut pool, keys, &stream).unwrap();
    let mut expected = RetainedStorage::default();
    for value in pool.retained_arrays() {
        value.evaluated().unwrap();
        expected.include_array(value).unwrap();
    }
    let bytes = expected.byte_bound().unwrap().unwrap();
    assert!(bytes > 0);
    let pool = super::super::OwnedPredictionCache::new(pool, Default::default());
    let extension = MaterializedDeepSeekV4Prediction::<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    >::Sequential {
        units: Vec::new(),
        state: vec![pool.clone(), pool],
    };
    let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut counts = ParameterOwnerCounts::default();
    count_parameter_owners::<eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>, _>(
        &extension,
        &mut counts,
        &mut guard,
    )
    .unwrap();
    assert_eq!(
        (
            counts.prediction_modules,
            counts.pooling_prototypes,
            counts.model_prototypes
        ),
        (0, 2, 0)
    );
    drop(guard);
    let mut inventory = retained_storage::<
        eredu_architectures::deepseek::v4::Model<MlxNeuralBackend>,
        _,
    >(&extension)
    .unwrap();
    assert_eq!(
        inventory.byte_bound().unwrap(),
        Some(bytes),
        "shared prototype backing counts once"
    );
    inventory.merge(expected).unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), Some(bytes));
    drop(extension);
    assert_eq!(inventory.byte_bound().unwrap(), Some(bytes));
    drop(inventory);

    use eredu_core::cache::{
        MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
        StateTensorRole,
    };
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![StateTensorDimension::fixed(2).unwrap()],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let layout = eredu_runtime::StateLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![
                eredu_core::cache::LayerCachePolicy::key_value_with_fixed_state(
                    eredu_core::AttentionPolicy::Full,
                    1,
                    2,
                    vec![fixed],
                )
                .unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxHybridState::device(layout).unwrap();
    let value = MlxTensor::from_array(Array::from_slice(&[2.0f32, 5.0], &[1, 1, 1, 2]));
    state.layers_mut()[0]
        .update_for_attention(value.clone(), value, &stream)
        .unwrap();
    *state.layers_mut()[0]
        .fixed_component(StateTensorRole::Recurrent)
        .unwrap() = Some(MlxTensor::from_array(Array::from_slice(
        &[7_f32, 11.],
        &[2],
    )));
    let mut expected = RetainedStorage::default();
    for value in state.retained_arrays() {
        value.evaluated().unwrap();
        expected.include_array(value).unwrap();
    }
    let numerical_bytes = expected.byte_bound().unwrap().unwrap();
    assert!(numerical_bytes > 0);
    let layout = eredu_runtime::RuntimeState::<MlxNeuralBackend>::shared_layout(&state)
        .unwrap()
        .clone();
    let layout_bytes = layout.capacity_bytes().unwrap();
    let tokens = std::iter::once(state.layer_slot_metadata())
        .chain(state.fixed_slot_metadata())
        .cloned()
        .collect::<Vec<_>>();
    let slot_facts = tokens
        .iter()
        .map(|token| (token.identity().clone(), token.capacity_bytes().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        tokens.len(),
        2,
        "outer layer table and its fixed-role table"
    );
    assert!(slot_facts.values().all(|bytes| *bytes > 0));
    let slot_bytes: u64 = slot_facts.values().copied().sum();
    assert!(layout_bytes > 0);
    assert!(slot_bytes > 0);
    expected
        .merge(MlxStateMechanisms::retained_host_storage(&state).unwrap())
        .unwrap();
    let bytes = numerical_bytes + layout_bytes + slot_bytes;
    assert_eq!(expected.byte_bound().unwrap(), Some(bytes));
    let extension =
        MaterializedInklingPrediction::<MlxNeuralBackend, MlxEmbeddedPredictionMaterializer> {
            units: Vec::new(),
            shared: None,
            state,
        };
    let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut counts = ParameterOwnerCounts::default();
    count_parameter_owners::<eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>, _>(
        &extension,
        &mut counts,
        &mut guard,
    )
    .unwrap();
    assert_eq!(
        (
            counts.prediction_modules,
            counts.pooling_prototypes,
            counts.model_prototypes
        ),
        (0, 0, 1)
    );
    drop(guard);
    let mut inventory = retained_storage::<
        eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>,
        _,
    >(&extension)
    .unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), Some(bytes));
    let found_layouts = inventory
        .metadata_sources()
        .map(|source| (source.identity().clone(), source.capacity_bytes().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        found_layouts,
        BTreeMap::from([(layout.identity().clone(), layout_bytes)])
    );
    let found_slots = inventory
        .slot_metadata_sources()
        .map(|token| (token.identity().clone(), token.capacity_bytes().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(found_slots, slot_facts);
    inventory.merge(expected).unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), Some(bytes));
    let publication = inventory.publish_unquoted(&loading).unwrap();
    assert_eq!(memory.used_bytes().unwrap(), bytes);
    drop((extension, layout, publication, loading));
    // Neither the publication nor metadata-only keys pin prototype payloads.
    // Escaped table tokens conservatively retain their exact inline charges.
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        memory.used_bytes().unwrap() == slot_bytes && memory.unquoted_owner_count().unwrap() == 0
    });
    for token in &tokens {
        let error = token
            .try_attach(
                memory.shared_storage_domain(),
                || -> Result<Box<dyn Send + Sync>, std::convert::Infallible> {
                    panic!("retired prototype tables cannot acquire new custody")
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            eredu_runtime::HostSlotAttachmentError::Retired
        ));
    }
    drop(tokens);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        memory.used_bytes().unwrap() == 0
    });
    // These surviving identity maps own neither source payload nor custody.
    assert_eq!(found_slots, slot_facts);
}

#[test]
fn prediction_placeholders_preserve_geometry_with_bounded_completed_backing() {
    for device in [safemlx::DeviceType::Cpu, safemlx::DeviceType::Gpu] {
        if device == safemlx::DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let stream = Stream::new_with_device(&safemlx::Device::new(device, 0));
        for dtype in [
            safemlx::Dtype::Float32,
            safemlx::Dtype::Bfloat16,
            safemlx::Dtype::Uint32,
        ] {
            let lazy = MlxTensor::from_array(
                safemlx::ops::zeros_dtype(&[128, 256], dtype, &stream).unwrap(),
            );
            let unloaded = placeholder(&lazy, &stream).unwrap();
            assert_eq!(unloaded.as_array().shape(), [128, 256]);
            assert_eq!(unloaded.as_array().dtype(), dtype);
            let info = unloaded
                .as_array()
                .allocation_info()
                .unwrap()
                .expect("completed scalar view");
            assert!(
                info.bytes() < 128 * 256 * 2,
                "placeholder must not allocate the full parameter shape: {info:?}"
            );
            let mut guard = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
                .unwrap()
                .enter()
                .unwrap();
            let mut counts = ParameterOwnerCounts::default();
            counts
                .observe_map(
                    ParameterOwnerRole::PredictionPlaceholder,
                    Some(0),
                    &BTreeMap::from([("actual-shape".into(), unloaded.clone())]),
                    &mut guard,
                )
                .unwrap();
            assert_eq!(
                counts
                    .role(ParameterOwnerRole::PredictionPlaceholder)
                    .parameters
                    .auxiliary_slots,
                1
            );
            assert_eq!(
                counts
                    .role(ParameterOwnerRole::PredictionPlaceholder)
                    .parameters
                    .shape_elements,
                2
            );
            assert_eq!(
                counts
                    .role(ParameterOwnerRole::PredictionPlaceholder)
                    .parameters
                    .unknown_backings,
                0
            );
            drop(guard);
            assert_eq!(
                lazy.as_array().allocation_info().unwrap(),
                None,
                "constructor graph is never evaluated"
            );
        }
    }
}
