use super::*;
use eredu_checkpoint::store::{MemoryWeightStore, TensorSelection};
use eredu_core::residency::OffloadConfig;
use eredu_nn::{
    Parameter, ParameterSourceVisitor, ParameterSpec, ParameterVisitor, ParameterVisitorMut,
};
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};
use safemlx::{Array, Device, DeviceType};
use std::{cell::Cell, collections::BTreeMap, time::Duration};

#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "MlxTensor")]
struct Unit {
    weight: Parameter<MlxTensor>,
    #[parameter(skip, retained_optional_value)]
    auxiliary: Option<MlxTensor>,
}
fn fixture() -> (MlxResidentPolicy<Unit>, Stream) {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source: RetainedCheckpointSource = (Arc::new(
        MemoryWeightStore::from_safetensors((0..3).map(|i| {
            (
                format!("w{i}"),
                safetensors::Dtype::I32,
                vec![2],
                [3_i32 + i, 7 + i]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            )
        }))
        .unwrap(),
    ))
    .into();
    let ids: Vec<_> = (0..3)
        .map(|i| OffloadUnitId::new(format!("unit{i}")).unwrap())
        .collect();
    let definitions: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            OffloadUnit::new(
                id.clone(),
                [
                    WeightBinding::new("weight", format!("w{i}"), TensorSelection::Full, 8)
                        .unwrap(),
                ],
            )
            .unwrap()
        })
        .collect();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(65_536), Some(65_536), 1).unwrap(),
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                .unwrap()
        }),
    )
    .unwrap();
    let manager = ResidencyManager::new_shared(
        source.clone(),
        plan,
        definitions,
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("vision"),
            ExecutionGroupSpec::with_dependencies("text", ["vision"]),
        ],
        "text",
    )
    .unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [1, 2]).unwrap();
    let policy = super::super::super::MlxLayerwisePolicy::new(
        manager,
        source,
        ids,
        layout,
        1,
        (),
        Vec::new(),
        None,
        false,
        false,
    )
    .unwrap();
    let units = (0..3)
        .map(|_| Unit {
            weight: Parameter::unloaded_i32(
                ParameterSpec::trainable("weight").unwrap(),
                &[2],
                &stream,
            )
            .unwrap(),
            auxiliary: None,
        })
        .collect();
    (policy.into_resident_units(units, &stream).unwrap(), stream)
}
thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(hook);
    }
}

#[test]
fn loaded_resident_loan_preserves_selection_source_and_recovery() {
    let (mut policy, stream) = fixture();
    for (i, unit) in policy.units.iter().enumerate() {
        assert_eq!(
            unit.as_ref()
                .unwrap()
                .inner
                .weight
                .as_ref()
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<i32>()
                .unwrap(),
            [3 + i as i32, 7 + i as i32]
        );
    }
    let mut runtime = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    safemlx::register_thread_runtime_housekeeping(hook);
    let hook_owner = Hook;
    HOOKS.with(|count| count.set(0));
    let source = policy.parameter_sources().unwrap();
    let address = source.layout().address(0).unwrap();
    assert!(source.unit(0, source.layout().address(1).unwrap()).is_err());
    let counted = source.count(&mut runtime).unwrap();
    assert_eq!(
        counted.counts(),
        ResidentParameterCounts {
            named_slots: 3,
            metadata_name_bytes: 18,
            shape_elements: 3,
            ..Default::default()
        }
    );
    assert!(std::ptr::eq(
        counted.source().unit(0, address).unwrap(),
        &policy.units[0].as_ref().unwrap().inner
    ));
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(counted);
    drop(hook_owner);
    drop(runtime);
    let lease = policy
        .acquire::<(), _>(
            0,
            address,
            |_| -> Result<Unit, ()> { panic!("resident source must not construct a replacement") },
            &stream,
        )
        .unwrap();
    assert!(matches!(
        policy.parameter_sources(),
        Err(ResidentSourceError::NotIdle { unit: 0 })
    ));
    policy.abort(Some((0, address, lease)), &stream);
    assert!(policy.parameter_sources().is_ok());
    let original = policy.unit_ids[0].clone();
    policy.unit_ids[0] = OffloadUnitId::new("foreign").unwrap();
    assert!(matches!(
        policy.parameter_sources(),
        Err(ResidentSourceError::Selection { unit: 0 })
    ));
    policy.unit_ids[0] = original;
    let replacement = MlxTensor::from_array(Array::from_slice(&[41_i32, 43], &[2]));
    replacement.as_array().evaluated().unwrap();
    assert!(policy
        .publish_parameter_replacements(&BTreeMap::from([("weight".into(), replacement)]), false)
        .unwrap());
    let source = policy.parameter_sources().unwrap();
    assert_eq!(
        source
            .unit(0, address)
            .unwrap()
            .weight
            .as_ref()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<i32>()
            .unwrap(),
        [41, 43]
    );
}
#[test]
fn actual_descriptor_count_keeps_alias_replicas_known_empty_and_lazy_distinct() {
    let (mut policy, stream) = fixture();
    let first = policy.units[0]
        .as_ref()
        .unwrap()
        .inner
        .weight
        .as_ref()
        .clone();
    policy.units[0].as_mut().unwrap().inner.auxiliary = Some(first.clone());
    let lazy = MlxTensor::from_array(first.as_array().square(&stream).unwrap());
    policy.units[1].as_mut().unwrap().inner.auxiliary = Some(lazy);
    let empty = Array::from_slice::<i32>(&[], &[0]);
    empty.evaluated().unwrap();
    policy.units[2].as_mut().unwrap().inner.auxiliary = Some(MlxTensor::from_array(empty));
    let replica = Array::from_slice(&[3_i32, 7], &[2]);
    replica.evaluated().unwrap();
    let mut runtime = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let original = runtime
        .descriptor(first.as_array())
        .unwrap()
        .facts()
        .allocation()
        .unwrap();
    let alias = runtime
        .descriptor(
            policy.units[0]
                .as_ref()
                .unwrap()
                .inner
                .auxiliary
                .as_ref()
                .unwrap()
                .as_array(),
        )
        .unwrap()
        .facts()
        .allocation()
        .unwrap();
    let independent = runtime
        .descriptor(&replica)
        .unwrap()
        .facts()
        .allocation()
        .unwrap();
    assert_eq!(original, alias);
    assert_ne!(original, independent);
    let empty = runtime
        .descriptor(
            policy.units[2]
                .as_ref()
                .unwrap()
                .inner
                .auxiliary
                .as_ref()
                .unwrap()
                .as_array(),
        )
        .unwrap()
        .facts();
    assert!(empty.allocation().is_some());
    assert_eq!(empty.logical_bytes(), 0);
    let counts = policy
        .parameter_sources()
        .unwrap()
        .count(&mut runtime)
        .unwrap();
    assert_eq!(
        counts.counts(),
        ResidentParameterCounts {
            named_slots: 3,
            auxiliary_slots: 3,
            metadata_name_bytes: 18,
            shape_elements: 6,
            unknown_backings: 1
        }
    );
    drop(counts);
    assert!(runtime
        .descriptor(
            policy.units[1]
                .as_ref()
                .unwrap()
                .inner
                .auxiliary
                .as_ref()
                .unwrap()
                .as_array()
        )
        .unwrap()
        .facts()
        .allocation()
        .is_none());
}
struct Partial {
    value: Parameter<MlxTensor>,
}
impl Parameterized<MlxTensor> for Partial {
    fn visit_parameter_sources<'a, V: ParameterSourceVisitor<'a, MlxTensor>>(
        &'a self,
        visitor: &mut V,
    ) -> Result<(), ParameterSourceError> {
        self.value.visit_parameter_sources(visitor)?;
        Err(ParameterSourceError::Unavailable)
    }
    fn visit_parameters<'a, V: ParameterVisitor<'a, MlxTensor>>(&'a self, visitor: &mut V) {
        self.value.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, MlxTensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.value.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, value: bool) {
        self.value.set_trainable(value);
    }
}
#[test]
fn fixed_observer_error_precedes_independent_coverage_failure_without_success() {
    let source = Partial {
        value: Parameter::new(
            ParameterSpec::trainable("w").unwrap(),
            MlxTensor::from_array(Array::from_slice(&[2_i32], &[1])),
        ),
    };
    let mut runtime = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let mut counter = Counter {
        guard: &mut runtime,
        counts: ResidentParameterCounts {
            named_slots: usize::MAX,
            ..Default::default()
        },
        slot: 0,
        failure: None,
    };
    assert_eq!(
        counter
            .observe_source(&source)
            .map_err(|error| resident_error(2, error)),
        Err(ResidentSourceError::CountOverflow)
    );
    assert_eq!(counter.counts.named_slots, usize::MAX);
    counter.failure = None;
    counter.counts = Default::default();
    assert_eq!(
        counter
            .observe_source(&source)
            .map_err(|error| resident_error(2, error)),
        Err(ResidentSourceError::Traversal {
            unit: 2,
            source: ParameterSourceError::Unavailable
        })
    );
    let error = ResidentSourceError::Descriptor {
        unit: 2,
        slot: 3,
        source: ArrayDescriptorError::SourceChanged,
    };
    let actual = std::error::Error::source(&error).unwrap();
    assert_eq!(
        actual.downcast_ref::<ArrayDescriptorError>(),
        Some(&ArrayDescriptorError::SourceChanged)
    );
}

#[test]
fn resident_source_control_layouts_are_concrete_and_reported_without_authority() {
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
    layout!(ResidentSourceError);
    layout!(ResidentParameterCounts);
    layout!(Counter<'_>);
    layout!(ResidentParameterSource<'_, Unit>);
    layout!(CountedResidentParameterSource<'_, Unit>);
    layout!(Result<ResidentParameterSource<'_, Unit>, ResidentSourceError>);
    layout!(Result<CountedResidentParameterSource<'_, Unit>, ResidentSourceError>);
    layout!(Result<(),ResidentSourceError>);
    layout!(Option<ResidentSourceError>);
    layout!(eredu_nn::ParameterMetadataView<'_>);
    layout!(ParameterSourceError);
    layout!(Result<(),ParameterSourceError>);
    layout!(safemlx::ArrayDescriptorFacts);
    layout!(safemlx::ArrayDescriptorLoan<'_>);
    layout!(Result<safemlx::ArrayDescriptorLoan<'_>, ArrayDescriptorError>);
    assert_eq!(
        std::mem::size_of::<ResidentParameterSource<'_, Unit>>(),
        std::mem::size_of::<&MlxResidentPolicy<Unit>>()
    );
    assert!(!std::mem::needs_drop::<ResidentParameterCounts>());
    assert!(!std::mem::needs_drop::<ResidentSourceError>());
}
