//! Real module registration/transfer mechanics; this is not a fabricated
//! architecture PreparedPredictionUnit or a whole-extension admission fixture.
use super::*;
use eredu_checkpoint::store::{MemoryWeightStore, TensorSelection};
use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy};
use eredu_nn::{Parameter, ParameterSpec};
use std::time::Duration;

#[derive(Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "MlxTensor")]
struct Unit {
    weight: Parameter<MlxTensor>,
}

fn module(stream: &Stream) -> MlxPredictionModule<Unit> {
    let source: RetainedCheckpointSource = (Arc::new(
        MemoryWeightStore::from_safetensors([(
            "w".to_owned(),
            safetensors::Dtype::I32,
            vec![2],
            [3_i32, 7].into_iter().flat_map(i32::to_le_bytes).collect(),
        )])
        .unwrap(),
    ))
    .into();
    let mut inner = Unit {
        weight: Parameter::unloaded_i32(ParameterSpec::trainable("weight").unwrap(), &[2], stream)
            .unwrap(),
    };
    let placeholders = placeholders(&mut inner, stream).unwrap();
    MlxPredictionModule {
        inner,
        parameters: Vec::new(),
        tasks: Arc::new(Vec::new()),
        layout: None,
        materialization: Default::default(),
        source,
        bindings: vec![
            eredu_runtime::WeightBinding::new("weight", "w", TensorSelection::Full, 8).unwrap(),
        ],
        residency: eredu_runtime::LayerWeightResidency::LayerwiseHost(Default::default()),
        shared: false,
        manager: Arc::new(OnceLock::new()),
        id: None,
        original: None,
        placeholders,
        replacements: BTreeMap::new(),
        stream: stream.clone(),
    }
}
fn count(module: &MlxPredictionModule<Unit>) -> ParameterOwnerCounts {
    let mut guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut counts = ParameterOwnerCounts::default();
    module
        .count_parameter_owners(0, &mut counts, &mut guard)
        .unwrap();
    counts
}

#[test]
fn prediction_count_tracks_actual_registration_shared_manager_and_transfer_restoration() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let memory = eredu_runtime::working_memory::WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let _loading = crate::backend::managed_memory::NativeMemoryOwner::acquire(&memory).unwrap();
    let mut module = module(&stream);
    let early_alias = module.clone();
    let initial = count(&module);
    assert_eq!(
        (
            initial.prediction_modules,
            initial.prediction_ids_present,
            initial.prediction_managers_installed
        ),
        (1, 0, 0)
    );
    assert_eq!(
        initial
            .role(ParameterOwnerRole::PredictionInner)
            .parameters
            .named_slots,
        1
    );
    assert_eq!(
        initial
            .role(ParameterOwnerRole::PredictionPlaceholder)
            .parameters
            .auxiliary_slots,
        1
    );
    assert_eq!(
        initial
            .role(ParameterOwnerRole::PredictionPlaceholder)
            .map_key_bytes,
        "weight".len()
    );
    let placeholder_identity = module
        .inner
        .weight
        .as_ref()
        .as_array()
        .allocation_info()
        .unwrap();
    assert!(placeholder_identity.is_some());
    assert_eq!(
        module.placeholders["weight"]
            .as_array()
            .allocation_info()
            .unwrap(),
        placeholder_identity
    );

    let mut registry = PredictionResidency::default();
    module.register_residency(0, &mut registry).unwrap();
    assert_eq!(
        (
            count(&module).prediction_ids_present,
            count(&module).prediction_managers_installed
        ),
        (1, 0)
    );
    assert!(module.register_residency(0, &mut registry).is_err());
    assert_eq!(registry.units.len(), 1);
    let id = module.id.as_ref().unwrap().clone();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(65_536), Some(65_536), 1).unwrap(),
        [OffloadUnitSpec::new(id, 8, ResidencyPolicy::Cacheable, MemoryTier::Disk).unwrap()],
    )
    .unwrap();
    let manager = ResidencyManager::new_shared(
        module.source.clone(),
        plan,
        registry
            .units
            .iter()
            .map(|unit| unit.definition.clone())
            .collect::<Vec<_>>(),
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    registry.install(&manager).unwrap();
    let reads = manager.report().unwrap().weight_store().physical_reads;
    let installed = count(&module);
    assert_eq!(
        (
            installed.prediction_ids_present,
            installed.prediction_managers_installed
        ),
        (1, 1)
    );
    let aliased = count(&early_alias);
    assert_eq!(
        (
            aliased.prediction_ids_present,
            aliased.prediction_managers_installed
        ),
        (0, 1),
        "real clone shares installation but predates identity assignment"
    );
    assert_eq!(
        manager.report().unwrap().weight_store().physical_reads,
        reads
    );

    let loaded = module
        .invoke(&stream, |inner| {
            let value = inner.weight.as_ref().clone();
            let result = value
                .as_array()
                .evaluated()
                .map(|a| a.try_to_vec::<i32>().unwrap())
                .map_err(Error::from);
            (result, vec![value])
        })
        .unwrap();
    assert_eq!(loaded, [3, 7]);
    assert_eq!(
        module
            .inner
            .weight
            .as_ref()
            .as_array()
            .allocation_info()
            .unwrap(),
        placeholder_identity
    );
    let replacement = MlxTensor::from_array(Array::from_slice(&[11_i32, 13], &[2]));
    let replacement_identity = replacement.as_array().allocation_info().unwrap();
    module
        .replacements
        .insert("weight".into(), replacement.clone());
    let before_failure = count(&module);
    assert_eq!(
        before_failure
            .role(ParameterOwnerRole::PredictionReplacement)
            .parameters
            .auxiliary_slots,
        1
    );
    let failure = module
        .invoke::<()>(&stream, |inner| {
            let value = inner.weight.as_ref().clone();
            assert_eq!(
                value
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<i32>()
                    .unwrap(),
                [11, 13]
            );
            (
                Err(Error::ArchitectureModel(
                    "actual prediction operation sentinel".into(),
                )),
                vec![value],
            )
        })
        .unwrap_err();
    assert!(
        matches!(failure, Error::ArchitectureModel(ref text) if text == "actual prediction operation sentinel")
    );
    assert_eq!(
        module
            .inner
            .weight
            .as_ref()
            .as_array()
            .allocation_info()
            .unwrap(),
        placeholder_identity
    );
    assert_eq!(
        module.replacements["weight"]
            .as_array()
            .allocation_info()
            .unwrap(),
        replacement_identity
    );
    assert_eq!(count(&module), before_failure);
    let successful = module
        .invoke(&stream, |inner| {
            let value = inner.weight.as_ref().clone();
            let result = value
                .as_array()
                .evaluated()
                .map(|a| a.try_to_vec::<i32>().unwrap())
                .map_err(Error::from);
            (result, vec![value])
        })
        .unwrap();
    assert_eq!(successful, [11, 13]);
    assert_eq!(count(&module), before_failure);
    assert_eq!(
        module
            .inner
            .weight
            .as_ref()
            .as_array()
            .allocation_info()
            .unwrap(),
        placeholder_identity
    );
}
