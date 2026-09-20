use super::*;
use crate::backend::runtime::{
    execution::generic::MlxLayerwisePolicy, residency::manager::ResidencyManager,
};
use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitId, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout, OffloadUnit, WeightBinding,
};
use safemlx::{Device, DeviceType, Stream};
use safetensors::tensor::{TensorView, serialize_to_file};
use std::{collections::BTreeMap, sync::Arc};

fn source(
    device: DeviceType,
    context: &WorkspaceContext,
    allocation: NativeAllocationFacts,
) -> (tempfile::TempDir, LayerwiseWorkspace) {
    source_with_constructors(device, context, allocation, false, false)
}

fn source_with_constructors(
    device: DeviceType,
    context: &WorkspaceContext,
    allocation: NativeAllocationFacts,
    independent: bool,
    complete: bool,
) -> (tempfile::TempDir, LayerwiseWorkspace) {
    use crate::backend::runtime::execution::generic::MlxSelectiveUnitPopulator;
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let dir = tempfile::tempdir().unwrap();
    let data = [1.25_f32, -3.5, 2.0, 0.125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "a",
                TensorView::new(safetensors::Dtype::F32, vec![2, 2], &data).unwrap(),
            ),
            (
                "b",
                TensorView::new(safetensors::Dtype::F32, vec![4], &data).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let ids = [
        OffloadUnitId::new("owner").unwrap(),
        OffloadUnitId::new("alias").unwrap(),
    ];
    let units = [
        OffloadUnit::new(
            ids[0].clone(),
            [WeightBinding::new("weight", "a", TensorSelection::Full, 16)
                .unwrap()
                .with_logical_target("shared.weight")
                .unwrap()],
        )
        .unwrap(),
        OffloadUnit::new(
            ids[1].clone(),
            [
                WeightBinding::alias("shared", "shared.weight", 16).unwrap(),
                WeightBinding::new("local", "b", TensorSelection::Full, 16).unwrap(),
            ],
        )
        .unwrap(),
    ];
    let plan = OffloadPlan::new(
        OffloadConfig::new(None, None, 1).unwrap(),
        // The second unit owns only b; its alias still constructs a separate
        // unloaded slot, but the immutable a backing belongs to the first unit.
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 16, ResidencyPolicy::Cacheable, MemoryTier::Host)
                .unwrap()
        }),
    )
    .unwrap();
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("layers")], "layers").unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [2]).unwrap();
    let host = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let destination = Stream::new_with_device(&Device::new(device, 0));
    let exclusions = if independent {
        std::collections::BTreeSet::from(["independent.bank".to_owned()])
    } else {
        std::collections::BTreeSet::new()
    };
    let mut owner = BTreeMap::from([(
        "shared.weight".to_owned(), context.layout(&[2, 2], WorkspaceDtype::Float32).unwrap(),
    )]);
    if independent {
        owner.insert("independent.bank".to_owned(),
            context.layout(&[2, 2, 2], WorkspaceDtype::Float32).unwrap());
    }
    let alias = BTreeMap::from([
        ("shared".to_owned(), context.layout(&[2, 2], WorkspaceDtype::Float32).unwrap()),
        ("local".to_owned(), context.layout(&[4], WorkspaceDtype::Float32).unwrap()),
    ]);
    let constructors = [
        ParameterConstructors::from_layouts(&owner).unwrap(),
        ParameterConstructors::from_layouts(&alias).unwrap(),
    ];
    let manager = ResidencyManager::prepare_original_host(
        store.clone().into(),
        BTreeMap::new(),
        &plan,
        &units,
        &["layers".into()],
        &ids,
        &layout,
        1,
        &exclusions,
        complete.then_some(constructors.as_slice()),
        &host,
        &destination,
        &pool,
    )
    .unwrap()
    .expect("genuine prepared Host source on the selected destination");
    let populator = MlxSelectiveUnitPopulator::from_prepared(
        manager.original_parameter_exclusions().unwrap().for_selection(&exclusions).unwrap());
    let policy = MlxLayerwisePolicy::<(), _>::new(
        manager,
        store,
        ids.to_vec(),
        layout,
        1,
        populator,
        vec![],
        None,
        false,
        false,
    )
    .unwrap();
    let source = policy
        .layerwise_workspace_with_metadata(allocation, context)
        .unwrap();
    assert_eq!(source.destination_device_type(), device);
    (dir, source)
}

fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::Sequence,
    }
}
fn selected_mechanism(device: DeviceType) -> ResidentExecutionMechanisms {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    match device {
        DeviceType::Gpu => ResidentExecutionMechanisms::Metal(ordinary),
        DeviceType::Cpu => ResidentExecutionMechanisms::Cpu {
            ordinary,
            cpu: MlxCpuWorkspaceMechanisms::new(
                ordinary.allocation(),
                MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles)
                    .unwrap(),
            ),
        },
    }
}
fn context(mechanism: ResidentExecutionMechanisms) -> WorkspaceContext {
    let context = match mechanism {
        ResidentExecutionMechanisms::Metal(facts) => WorkspaceContext::new(facts),
        ResidentExecutionMechanisms::Cpu { cpu, .. } => WorkspaceContext::new(cpu),
    };
    context.set_borrowed_storage(
        eredu_nn::workspace::WorkspaceBorrowedStorage::new(&context, []).unwrap(),
    ).unwrap();
    context
}
fn trace(context: &WorkspaceContext, slots: usize) -> WorkspaceTraceReport {
    trace_shapes(context, std::iter::repeat_n(&[2, 2][..], slots))
}
fn trace_shapes<'a>(context: &WorkspaceContext, shapes: impl IntoIterator<Item = &'a [i32]>) -> WorkspaceTraceReport {
    context.begin_state_span([]).unwrap();
    for shape in shapes {
        drop(WorkspaceTensor::unloaded_f32(shape, context).unwrap());
    }
    let input = WorkspaceTensor::existing(
        context
            .layout(&[2, 2], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            ))),
        context,
    )
    .unwrap();
    let output = input.tanh(context).unwrap();
    context.report(&[output]).unwrap()
}

#[test]
fn layerwise_constructor_sources_include_independent_slots_and_refuse_missing_declarations() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let mechanism = selected_mechanism(device);
        let context = context(mechanism);
        let (_dir, incomplete) = source_with_constructors(device, &context, mechanism.allocation(), true, false);
        assert!(constructor_source(&incomplete, None).is_err(),
            "excluded lease rows cannot certify the complete unloaded module");
        let (_dir, complete) = source_with_constructors(device, &context, mechanism.allocation(), true, true);
        let counted = complete.parameter_source().count().unwrap();
        assert_eq!(counted.counts().rows, 3, "independent slot is not a lease payload");
        drop(counted);
        let (constructors, query) = constructor_source(&complete, None).unwrap();
        assert_eq!(constructors.slots, 4);
        assert_eq!(constructors.scalar_bytes, 16);
        assert_eq!(constructors.rank, 3, "bank placeholder shape still needs descriptor storage");
        let (native, _) = constructors.native_source(query).unwrap();
        assert_eq!(native.primitives(), 12);
        assert_eq!(native.seeds(), 4);
        assert_eq!(native.maximum_rank(), 3);
        let report = trace_shapes(&context, [&[2, 2][..], &[2, 2, 2], &[2, 2], &[4]]);
        let mut recorder = mechanism.recorder(geometry(), &context).unwrap();
        recorder.bind_layerwise_constructor_source(&complete).unwrap();
        record(&mut recorder, &report).unwrap();
        assert_eq!(recorder.records[0].first_missing_operation, None);
        let missing = trace(&context, 3);
        let mut recorder = mechanism.recorder(geometry(), &context).unwrap();
        recorder.bind_layerwise_constructor_source(&complete).unwrap();
        assert!(record(&mut recorder, &missing).is_err());
    }
}
fn record(
    recorder: &mut ResidentRecipeRecorder,
    report: &WorkspaceTraceReport,
) -> Result<(), Error> {
    recorder.record_equation(
        &InferenceWorkspaceSpan::Decode {
            index: 0,
            position: 1,
            output: eredu_core::OutputDemand::Sequence,
        },
        report,
        0,
        1,
        None,
        None,
        false,
    )
}

#[test]
fn layerwise_constructor_sources_preserve_alias_slots_and_selected_device() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let mechanism = selected_mechanism(device);
        let context = context(mechanism);
        let (_dir, source) = source(device, &context, mechanism.allocation());
        let (constructors, query) = constructor_source(&source, None).unwrap();
        assert_eq!(
            constructors.slots, 3,
            "logical aliases still construct separate unloaded slots"
        );
        assert_eq!(constructors.scalar_bytes, 12);
        assert_eq!(constructors.rank, 2);
        assert_eq!(constructors.dtypes, [3, 0, 0, 0, 0]);
        let (native, controls) = constructors.native_source(query).unwrap();
        assert_eq!(native.primitives(), 9);
        assert_eq!(native.seeds(), 3);
        assert_eq!(native.maximum_rank(), 2);
        assert!(controls > query);

        let report = trace(&context, 3);
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        assert!(report.tensor_buffers.total_bytes.unwrap() > 0);
        assert_eq!(report.host_workspace_bytes, Some(0));
        assert!(report.total_bytes.unwrap() > 0);
        assert!(report.inference_transient_bytes().unwrap() > 0);
        assert!(report.residual.as_ref().unwrap().total_bytes.unwrap() > 0);
        if device == DeviceType::Cpu {
            let uncertified = mechanism.recorder(geometry(), &context).unwrap()
                .reduce_trace(&report, None, 0, 1).unwrap();
            assert_eq!(uncertified.first_missing_operation, Some(0));
            assert!(uncertified.graph.is_none(),
                "descriptive completeness cannot substitute for retained constructor source");
        }
        let mut recorder = mechanism.recorder(geometry(), &context).unwrap();
        recorder.bind_layerwise_constructor_source(&source).unwrap();
        assert!(recorder.bind_layerwise_constructor_source(&source).is_err());
        record(&mut recorder, &report).unwrap();
        let row = &recorder.records[0];
        assert_eq!(row.first_missing_operation, None);
        if device == DeviceType::Cpu {
            assert_eq!(
                row.dispatch.unwrap().cpu_model.unwrap().primitives,
                1,
                "replaced constructors never enter the CPU evaluation tape"
            );
        }
        SpeculativeNumericalRecipe::inspect_owned_child_with_sources(
            &report,
            1,
            mechanism,
            &context,
            Default::default(),
            Some(&source),
        )
        .unwrap();

        // This same recorder subsequently quotes sampling. Its trace has no
        // model construction and must not be compared with the source inventory.
        let sampler = eredu_runtime::ConfiguredTextSampler::Standard(
            eredu_runtime::GenerationSampler::default(),
        );
        let logits = context
            .layout(&[1, 4], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )));
        eredu_runtime::working_memory::quote_sampling_workspace_with_observer(
            &sampler,
            0.0,
            None,
            &logits,
            &eredu_core::TokenFilter::All,
            1,
            &context,
            Some(&mut recorder),
        )
        .unwrap();
        assert_eq!(recorder.sampling.len(), 2);
        assert!(
            recorder
                .sampling
                .iter()
                .all(|row| row.first_missing_operation.is_none() && row.mutable_storage.is_some())
        );

        let opposite = selected_mechanism(if device == DeviceType::Cpu {
            DeviceType::Gpu
        } else {
            DeviceType::Cpu
        });
        let mut foreign = opposite.recorder(geometry(), &context).unwrap();
        assert!(foreign.bind_layerwise_constructor_source(&source).is_err());
        assert!(foreign.layerwise_constructors.is_none());
        let mut late = mechanism.recorder(geometry(), &context).unwrap();
        record(&mut late, &trace(&context, 0)).unwrap();
        assert!(late.bind_layerwise_constructor_source(&source).is_err());
    }
}

#[test]
fn layerwise_constructor_inventory_refuses_changed_normal_and_owned_child_traces() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let mechanism = selected_mechanism(device);
        let context = context(mechanism);
        let (_dir, source) = source(device, &context, mechanism.allocation());
        let report = trace(&context, 3);
        for change in 0..5 {
            let mut invalid = report.clone();
            match change {
                0 => {
                    invalid.operations.remove(0);
                }
                1 => {
                    invalid.operations.push(invalid.operations[0].clone());
                }
                2 => {
                    invalid.operations[0].outputs[0] =
                        context.layout(&[], WorkspaceDtype::Int32).unwrap();
                }
                3 => {
                    invalid.operations[0].outputs[0] =
                        context.layout(&[1], WorkspaceDtype::Float32).unwrap();
                }
                4 => {
                    let input = invalid.operations[0].outputs[0].clone();
                    invalid.operations[0].inputs.push(input);
                }
                _ => unreachable!(),
            }
            let mut recorder = mechanism.recorder(geometry(), &context).unwrap();
            recorder.bind_layerwise_constructor_source(&source).unwrap();
            assert!(
                record(&mut recorder, &invalid).is_err(),
                "device {device:?}, change {change}"
            );
            assert!(recorder.records.is_empty());
            assert!(
                SpeculativeNumericalRecipe::inspect_owned_child_with_sources(
                    &invalid,
                    1,
                    mechanism,
                    &context,
                    Default::default(),
                    Some(&source)
                )
                .is_err(),
                "owned child must authenticate the same inventory: device {device:?}, change {change}"
            );
        }
    }
}
