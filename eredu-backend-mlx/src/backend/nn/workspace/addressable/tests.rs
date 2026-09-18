use super::*;
use crate::backend::runtime::residency::parameter_bank::{AddressableParameterBank,
    IndexedBankSource, ParameterBankEntry, ParameterBankKey, SelectedAddressableEntries,
    SharedAddressableParameterBank};
use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::OffloadConfig;
use eredu_nn::{GroupedLinearActivation, GroupedLinearSpec, GroupedProjectionSpec,
    LinearFormatSpec, ParameterSpec};
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFundingError, WorkspaceExpertRegionView};
use eredu_runtime::{AddressableBankDistribution, AddressableBankMemberPlacement, ExecutionGroupId,
    OffloadUnit, ParameterBankLoadOptions, WeightBinding};
use safemlx::{Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, TensorView};
use std::{collections::BTreeMap, sync::{Arc, atomic::{AtomicUsize, Ordering}}};

#[derive(Clone, Debug, Default)]
struct Account(Arc<AtomicUsize>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(bytes))
            .map(|_| ()).map_err(|_| HostMetadataFundingError::Overflow)
    }
}

#[test]
fn independent_local_quote_specializes_actual_rows_from_one_physical_bank() {
    check_local_quote(false, 38, 5, DeviceType::Cpu);
}

#[test]
fn independent_gated_quote_covers_outer_tail_and_inner_kernel_transitions() {
    check_local_quote(true, 230, 96, DeviceType::Cpu);
}

#[test]
fn independent_metal_local_quote_keeps_separate_source_copy_producers() {
    check_local_quote(false, 10, 5, DeviceType::Gpu);
}

fn check_local_quote(gated: bool, maximum: usize, chunk_rows: usize, device_type: DeviceType) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let runtime = crate::backend::managed_memory::input_allocator::borrow_admitted(&pool).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let fields = if gated { vec![("read", vec![1, 8, 4]), ("write", vec![1, 4, 4])] }
        else { vec![("weight", vec![1, 4, 4])] };
    let values = fields.iter().map(|(_, shape)| (0..shape.iter().product::<usize>())
        .flat_map(|n| ((n as f32 + 1.0) / 17.0).to_le_bytes()).collect::<Vec<_>>()).collect::<Vec<_>>();
    serialize_to_file((0..3).flat_map(|member| fields.iter().zip(&values).map(move |((name, shape), bytes)|
        (format!("member.{member}.{name}"), TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap()))),
        None, &directory.path().join("model.safetensors")).unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let member_bytes = values.iter().map(|v| v.len() as u64).sum::<u64>();
    let mut selected = SelectedAddressableEntries { parameter_targets: BTreeMap::new(),
        entries: Vec::new(), transformations: BTreeMap::new(), expected_bytes: BTreeMap::new(), placements: BTreeMap::new() };
    for (ordinal, member) in [0, 2, 5].into_iter().enumerate() {
        let key = ParameterBankKey::new(0, 0, member);
        let bindings = fields.iter().zip(&values).map(|((name, _), bytes)| {
            selected.parameter_targets.insert((key, (*name).into()), (*name).into());
            WeightBinding::new(*name, format!("member.{ordinal}.{name}"), TensorSelection::Full, bytes.len() as u64).unwrap()
        }).collect::<Vec<_>>();
        selected.entries.push(ParameterBankEntry::new(key, OffloadUnit::new(key.unit_id(), bindings).unwrap(), member_bytes).unwrap());
        selected.expected_bytes.insert(key, member_bytes);
        selected.placements.insert(key, AddressableBankMemberPlacement::new(ExecutionGroupId::new("layers").unwrap(),
            0, "layers.0", AddressableBankDistribution::ExpertParallel).unwrap());
    }
    let bulk_target = member_bytes * chunk_rows as u64;
    let scratch = bulk_target.max(member_bytes * 3);
    let options = ParameterBankLoadOptions::new(OffloadConfig::new(Some(scratch), Some(scratch), 1).unwrap(), scratch, bulk_target).unwrap();
    let host = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let device = Stream::new_with_device(&Device::new(device_type, 0));
    let source: eredu_checkpoint::store::RetainedCheckpointSource = store.clone().into();
    let prepared = AddressableParameterBank::prepare_selected_manager(source.clone(), selected.clone(), options,
        &host, &device, &pool).unwrap().expect("actual prepared indexed manager");
    let bank = AddressableParameterBank::new_selected_shared_with_manager(source, selected, options, host, device, Some(prepared)).unwrap();
    let bank = SharedAddressableParameterBank::new(bank).scoped(0).unwrap();
    let source = IndexedBankSource::new(bank, options);
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let mechanism = match device_type {
        DeviceType::Cpu => ResidentExecutionMechanisms::Cpu { ordinary,
            cpu: MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),
                MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap()) },
        DeviceType::Gpu => ResidentExecutionMechanisms::Metal(ordinary),
    };
    let cold_account = Account::default();
    let funding = HostMetadataFunding::new(cold_account.clone()).unwrap();
    let sources = AddressableSources::new([(0, &source)].into_iter(), mechanism, &runtime, Some(&pool), &funding).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone()).unwrap();
    let projection = |name| GroupedProjectionSpec::new(ParameterSpec::trainable(name).unwrap(), None,
        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()).unwrap();
    let kernel = if gated {
        WorkspaceGroupedBank::GatedProduct(eredu_nn::GroupedGatedProductSpec::new(3, 4, 4, 4,
            eredu_nn::GatedProductPolicy::ordinary_silu(), eredu_nn::GatedProductGroupLayout::Packed {
                gate_up: projection("read"), down: projection("write") }).unwrap())
    } else { WorkspaceGroupedBank::Linear(GroupedLinearSpec::new(3, 4, 4, GroupedLinearActivation::Identity,
        projection("weight")).unwrap()) };
    let kernel_view = match &kernel {
        WorkspaceGroupedBank::GatedProduct(value) => eredu_nn::workspace::WorkspaceExpertKernel::Gated(value),
        WorkspaceGroupedBank::Linear(value) => eredu_nn::workspace::WorkspaceExpertKernel::Linear(value),
        _ => unreachable!(),
    };
    let chunks = eredu_runtime::expert::AddressableChunkPlan::new(maximum, 1, 3,
        eredu_runtime::ParameterBankAccess::Bulk, Some(member_bytes), bulk_target).unwrap().workspace_source();
    let local = WorkspaceAddressableRegionView { owner_group: "layers", bank: 0, unit: 0, prefill: true,
        chunks, local_members: Some(&[0, 2, 5]), kernel: kernel_view,
        tensor_partitions: None, compact_scratch_bytes: scratch, bulk_target_bytes: bulk_target, callback_control_bytes: 0 };
    let region = WorkspaceExpertRegionView { bank: 0, unit: 0, prefill: true,
        group: eredu_core::CollectiveGroupId::new(0), rank: 0, peers: 2, source_rows: maximum / 2, routes_per_row: 1,
        owners: &[0, 1, 0, 1, 1, 0], owner_local: &[0, 0, 1, 1, 2, 2], kernel: local.kernel,
        tensor_partitions: None, provider_tensor_group: None, provider_wave_group: None,
        movement: eredu_nn::workspace::WorkspaceExpertMovementPopulation { row_gathers: 1, scalar_gathers: 2, zeros: 1, indexed_adds: 1 }, transfers: eredu_nn::workspace::WorkspaceExpertTransfers([None; 9]),
        addressable: Some(local) }.retain(&context).unwrap();
    let representation = Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true));
    let floating = |shape: &[i32]| context.layout(shape, WorkspaceDtype::Float32).unwrap().with_representation(representation);
    let operation = WorkspaceOperation { kind: WorkspaceOperationKind::ExpertRegion(Box::new(region)),
        inputs: vec![floating(&[(maximum / 2) as i32, 4]), context.layout(&[(maximum / 2) as i32, 1], WorkspaceDtype::Int32).unwrap(),
            floating(&[(maximum / 2) as i32, 1]), floating(&[(maximum / 2) as i32, 1])], outputs: vec![floating(&[(maximum / 2) as i32, 4])] };
    let quote = sources.local_quote(operation.as_view()).unwrap_or_else(|cause|
        panic!("device={device_type:?}, gated={gated}, maximum={maximum}, chunk_rows={chunk_rows}: {cause:?}"));
    assert!(quote.is_local());
    assert_eq!(quote.constructor_facts.maximum_partitions(), maximum.div_ceil(chunk_rows));
    assert!(quote.select_rows(0, &funding).is_err());
    assert!(quote.select_rows(maximum + 1, &funding).is_err());
    let envelope = quote.local_numerical().unwrap();
    let execution_account = Account::default();
    let execution_funding = HostMetadataFunding::new(execution_account.clone()).unwrap();
    for rows in 1..=maximum {
        let cold_before = cold_account.0.load(Ordering::SeqCst);
        let execution_before = execution_account.0.load(Ordering::SeqCst);
        let specialized = quote.select_rows(rows, &execution_funding).unwrap();
        assert_eq!(cold_account.0.load(Ordering::SeqCst), cold_before, "runtime specialization must not debit the retained cold source");
        let execution_debit = execution_account.0.load(Ordering::SeqCst) - execution_before;
        assert!(execution_debit > 0 && execution_debit <= quote.specialization_control_bytes().unwrap(), "the accepted caller pays every specialization owner within its quoted census");
        assert_eq!(specialized.declaration.as_view().chunks.rows, rows);
        assert_eq!(specialized.inputs[0].shape(), [rows as i32, 4]);
        assert_eq!(specialized.residency.census().total_rows(), rows);
        assert!(specialized.identity().source().same_binding(&source));
        assert_eq!(specialized.residency.census().plan().workspace_source().chunk_rows, chunk_rows);
        let declaration = specialized.declaration.as_view().retain(&context).unwrap();
        let actual = sources.quote(WorkspaceOperationView { kind: WorkspaceOperationKindView::AddressableRegion(&declaration),
            inputs: WorkspaceLayoutList::Owned(&specialized.inputs), outputs: WorkspaceLayoutList::Owned(&specialized.outputs) }).unwrap();
        let equation = actual.equation;
        let completed = actual.numerical;
        assert!(completed.completion.graph.primitives() > equation.completion.graph.primitives(),
            "rows={rows}: actual copies add constructors to the shared graph");
        assert!(completed.graph_capacity > equation.graph_capacity, "rows={rows}: copy/aggregate graph producers survive");
        assert!(completed.record_capacity > equation.record_capacity, "rows={rows}: copy/aggregate records survive");
        assert!(completed.completion.nested_completions > equation.completion.nested_completions,
            "rows={rows}: independent source completion frontiers survive");
        assert!(completed.storage.maximum_births() > equation.storage.maximum_births(), "rows={rows}: copy destinations stay charged");
        assert!(completed.storage.mutable_bytes() > equation.storage.mutable_bytes(), "rows={rows}: copied physical storage stays charged");
        if device_type == DeviceType::Cpu {
            let tape = completed.completion.traversal.limits().tape_entries;
            assert_eq!(tape, equation.completion.traversal.limits().tape_entries,
                "rows={rows}: separate copy tasks never become equation tape entries");
        } else {
            assert!(completed.kernels > equation.kernels, "rows={rows}: source copy and aggregate kernels survive");
        }
        assert!(actual.capacity.graph <= quote.native_capacity().graph, "rows={rows}");
        assert!(actual.capacity.records <= quote.native_capacity().records, "rows={rows}");
        assert!(actual.capacity.backing <= quote.native_capacity().backing, "rows={rows}");
        assert!(actual.numerical.controls <= envelope.controls, "rows={rows}");
        assert!(actual.numerical.kernels <= envelope.kernels, "rows={rows}");
        assert!(actual.numerical.completion.nested_completions <= envelope.completion.nested_completions, "rows={rows}");
        assert!(actual.numerical.completion.traversal.limits().captures <= envelope.completion.traversal.limits().captures, "rows={rows}");
        assert!(actual.numerical.storage.maximum_births() <= envelope.storage.maximum_births(), "rows={rows}");
        assert!(actual.host_bytes <= quote.host_capacity(), "rows={rows}");
    }
    assert!(sources.local_quote(operation.as_view()).unwrap().same_quote(&quote));
    let mut wrong = local;
    wrong.bulk_target_bytes += 1;
    assert!(source.first_census(wrong, &funding).is_err(), "selected options stay authenticated");
}
