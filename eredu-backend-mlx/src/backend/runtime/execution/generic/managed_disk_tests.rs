use super::*;
use crate::backend::nn::workspace::{NativeAllocationFacts, MlxMetalWorkspaceMechanisms};
use eredu_checkpoint::store::{
    CheckpointSource, MemoryWeightStore, SafetensorsWeightStore, TensorSelection,
};
use eredu_core::residency::{OffloadConfig, TransferDirection};
use eredu_nn::{Parameter, ParameterSpec, Tensor};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, ExecutionGraph, ExecutionGroupSpec, ResidencyWindowManager,
};
use safemlx::{Array, Device, DeviceType};
use safetensors::tensor::{serialize_to_file, TensorView};
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

struct AllocationRetirement(Arc<AtomicUsize>);
impl Drop for AllocationRetirement {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

type Unit = Vec<Parameter<MlxTensor>>;
type Policy = MlxLayerwisePolicy<Unit, MlxSelectiveUnitPopulator>;

fn id(index: usize) -> OffloadUnitId {
    OffloadUnitId::new(format!("unit{index}")).unwrap()
}

struct Fixture {
    _dir: tempfile::TempDir,
    source: Arc<dyn CheckpointSource>,
    manager: ResidencyManager,
    stream: Stream,
    policy: Policy,
    definitions: Vec<OffloadUnit>,
}

fn fixture(aliases: bool, depth: usize) -> Fixture {
    fixture_with_groups(aliases, depth, &[3, 2], true)
}

fn fixture_with_groups(aliases: bool, depth: usize, counts: &[usize], direct: bool) -> Fixture {
    fixture_with_host_capacity(aliases, depth, counts, direct, 0)
}

fn fixture_with_host_capacity(
    aliases: bool,
    depth: usize,
    counts: &[usize],
    direct: bool,
    host_capacity: u64,
) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let sizes = [2_usize, 3, 5, 7, 11];
    let payloads: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| {
            vec![index as i32 + 1; *size]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        })
        .collect();
    serialize_to_file(
        payloads.iter().enumerate().map(|(index, bytes)| {
            (
                format!("w{index}"),
                TensorView::new(safetensors::Dtype::I32, vec![sizes[index]], bytes).unwrap(),
            )
        }),
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source: Arc<dyn CheckpointSource> = if direct {
        Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap())
    } else {
        Arc::new(
            MemoryWeightStore::from_safetensors(payloads.iter().enumerate().map(
                |(index, bytes)| {
                    (
                        format!("w{index}"),
                        safetensors::Dtype::I32,
                        vec![sizes[index]],
                        bytes.clone(),
                    )
                },
            ))
            .unwrap(),
        )
    };
    let definitions: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| {
            let mut bindings = vec![WeightBinding::new(
                "weight",
                format!("w{index}"),
                TensorSelection::Full,
                (*size * 4) as u64,
            )
            .unwrap()
            .with_logical_target(format!("owner{index}"))
            .unwrap()];
            if aliases && index == 0 {
                // A persistent canonical unit owns a second independent output;
                // closure pricing must include the complete unit, not just aliases.
                bindings.push(
                    WeightBinding::new("companion", "w1", TensorSelection::Full, 12).unwrap(),
                );
                bindings.push(WeightBinding::alias("same", "owner0", 8).unwrap());
            }
            if aliases && index == 4 {
                bindings.push(WeightBinding::alias("shared", "owner0", 8).unwrap());
            }
            OffloadUnit::new(id(index), bindings).unwrap()
        })
        .collect();
    let specs = definitions.iter().map(|unit| {
        OffloadUnitSpec::new(
            unit.id().clone(),
            unit.bindings()
                .iter()
                // Logical residency counts materialized canonical bindings.
                // Alias names reuse their owner's output rather than adding it.
                .filter(|binding| !binding.is_alias())
                .map(WeightBinding::expected_bytes)
                .sum(),
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )
        .unwrap()
    });
    // Deliberately roomy: automatic budget eviction cannot hide accumulation.
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(1_000_000), Some(host_capacity), 1).unwrap(),
        specs,
    )
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let manager = ResidencyManager::new_shared(
        source.clone(),
        plan,
        definitions.clone(),
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let names: Vec<_> = (0..counts.len())
        .map(|index| format!("group{index}"))
        .collect();
    let groups: Vec<_> = names
        .iter()
        .enumerate()
        .map(|(index, name)| {
            if index == 0 {
                ExecutionGroupSpec::root(name.clone())
            } else {
                ExecutionGroupSpec::with_dependencies(name.clone(), [names[index - 1].clone()])
            }
        })
        .collect();
    let graph = ExecutionGraph::new(groups, names.last().unwrap()).unwrap();
    let layout = ExecutionUnitLayout::new(&graph, counts.iter().copied()).unwrap();
    let controller = Arc::new(
        DenseStreamController::new(
            &manager,
            DenseDiskStreamLoadOptions::new(1_000_000, 0, 0, 0).unwrap(),
            5,
            112,
            44,
            0,
            names.iter().enumerate().map(|(group, name)| {
                (
                    name.clone(),
                    layout.group_range(group).unwrap().map(id).collect(),
                )
            }),
        )
        .unwrap(),
    );
    let policy = MlxLayerwisePolicy::new(
        manager.clone(),
        source.clone(),
        (0..5).map(id).collect(),
        layout,
        depth,
        MlxSelectiveUnitPopulator::new(BTreeSet::new()),
        Vec::new(),
        Some(controller),
        false,
        false,
    )
    .unwrap();
    Fixture {
        _dir: dir,
        source,
        manager,
        stream,
        policy,
        definitions,
    }
}

fn build(unit: &OffloadUnit, stream: &Stream) -> Result<Unit, eredu_nn::Error> {
    unit.bindings()
        .iter()
        .map(|binding| {
            Parameter::unloaded_i32(
                ParameterSpec::trainable(binding.name())
                    .map_err(eredu_nn::Error::backend_retained_source)?,
                &[(binding.expected_bytes() / 4) as i32],
                stream,
            )
        })
        .collect()
}

fn parameter<'a>(unit: &'a Unit, name: &str) -> &'a MlxTensor {
    let topology = eredu_nn::validate_parameter_topology::<MlxTensor, _>(unit).unwrap();
    let index = topology
        .iter()
        .position(|metadata| metadata.id.as_str() == name)
        .expect("fixture declares the requested parameter identity");
    unit[index].as_ref()
}

fn resident(manager: &ResidencyManager) -> BTreeSet<OffloadUnitId> {
    manager
        .unit_reports()
        .unwrap()
        .into_iter()
        .filter(|unit| unit.device_resident())
        .map(|unit| unit.id().clone())
        .collect()
}

fn forward(f: &mut Fixture, managed: bool, aliases: bool) {
    forward_with_retirement_probe(f, managed, aliases, None);
}

fn forward_with_retirement_probe(
    f: &mut Fixture,
    managed: bool,
    aliases: bool,
    retirement: Option<&Arc<AtomicUsize>>,
) {
    let initial = MlxTensor::from_array(Array::from_slice(&[1_i32], &[1, 1]));
    f.policy.begin(&initial, &f.stream).unwrap();
    let mut last = None;
    for index in 0..5 {
        let address = f.policy.layout.address(index).unwrap();
        let unit = &f.definitions[index];
        if let Some(retirement) = retirement {
            // Do not retain an escaped output graph in this witness. The
            // manager still owns unit zero until the next window evicts it.
            drop(last.take());
            if index == 1 {
                safemlx::reclaim_allocation_owners();
                assert_eq!(retirement.load(Ordering::SeqCst), 0);
            }
        }
        let lease =
            f.policy
                .acquire(
                    index,
                    address,
                    |stream| {
                        if let Some(retirement) = retirement.filter(|_| index == 1) {
                            // Acquisition has drained and evicted, but has not built
                            // this unit or refilled the window with unit two yet.
                            safemlx::reclaim_allocation_owners();
                            assert_eq!(retirement.load(Ordering::SeqCst), 1,
                        "the old native backing must retire before the next constructor/refill");
                        }
                        build(unit, stream)
                    },
                    &f.stream,
                )
                .unwrap();
        let weight = parameter(&lease, "weight");
        if let Some(retirement) = retirement.filter(|_| index == 0) {
            weight.as_array().evaluated().unwrap();
            weight
                .as_array()
                .retain_allocation_owner(AllocationRetirement(Arc::clone(retirement)))
                .unwrap();
            assert_eq!(retirement.load(Ordering::SeqCst), 0);
        }
        let output = weight.add(weight, &f.stream).unwrap();
        let numerical = output.as_array().evaluated().unwrap();
        assert!(numerical
            .as_slice::<i32>()
            .iter()
            .all(|value| *value == 2 * (index as i32 + 1)));
        if aliases && index == 4 {
            let shared = parameter(&lease, "shared").as_array();
            assert_eq!(shared.evaluated().unwrap().as_slice::<i32>(), &[1, 1]);
        }
        f.policy
            .complete(
                index,
                address,
                lease,
                &output,
                std::iter::empty(),
                std::iter::empty(),
                &f.stream,
            )
            .unwrap();
        if managed {
            let range = f
                .policy
                .layout
                .window_range(index, std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
            let mut expected: BTreeSet<_> = range.map(id).collect();
            if aliases {
                expected.insert(id(0));
            }
            assert_eq!(
                resident(&f.manager),
                expected,
                "window {index} retained extra device copies"
            );
        }
        last = Some(output);
    }
    f.policy.finish(&last.unwrap(), &f.stream).unwrap();
}

#[test]
fn admitted_roomy_windows_evict_across_group_tails_and_warm_second_forward() {
    let mut f = fixture(false, 2);
    let before = f.source.source_diagnostics().unwrap();
    let workspace = f
        .policy
        .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
        .unwrap();
    let identity = workspace.identity();
    let receipt = workspace.disk_receipt().unwrap();
    assert!(workspace.pin_sources().unwrap().is_none());
    receipt.validate().unwrap();
    assert_eq!(
        f.source.source_diagnostics().unwrap(),
        before,
        "quote must reuse loading metadata"
    );
    let facts = NativeAllocationFacts::current_host().unwrap();
    assert_eq!(
        workspace.materialization().bytes(),
        Some(facts.buffer_capacity(28).unwrap() + facts.buffer_capacity(44).unwrap())
    );
    for _ in 0..2 {
        receipt.validate().unwrap();
        let guard = receipt.activate().unwrap();
        forward(&mut f, true, false);
        drop(guard);
        assert!(!f.manager.admitted_disk_route_active());
        let warm = f.policy.layerwise_workspace(facts).unwrap();
        assert_eq!(
            warm.identity(),
            identity,
            "warm allocation IDs are not policy identity"
        );
    }
}

#[test]
fn canonical_owner_full_unit_is_priced_persistent_and_aliases_share_projection() {
    let mut f = fixture(true, 2);
    let facts = NativeAllocationFacts::current_host().unwrap();
    let workspace = f.policy.layerwise_workspace(facts).unwrap();
    let persistent = facts.buffer_capacity(8).unwrap() + facts.buffer_capacity(12).unwrap();
    let moving = facts.buffer_capacity(28).unwrap() + facts.buffer_capacity(44).unwrap();
    assert_eq!(
        workspace.materialization().bytes(),
        Some(persistent + moving)
    );
    let context = eredu_nn::workspace::WorkspaceContext::new(
        MlxMetalWorkspaceMechanisms::current_host().unwrap(),
    );
    let owner = workspace
        .parameters(0, f.policy.layout.address(0).unwrap(), &context)
        .unwrap();
    let alias = workspace
        .parameters(4, f.policy.layout.address(4).unwrap(), &context)
        .unwrap();
    let values: Vec<_> = owner.into_values().chain(alias.into_values()).collect();
    context.begin_state_span(&values).unwrap();
    assert_eq!(
        context
            .report(&values)
            .unwrap()
            .state
            .unwrap()
            .retained_bytes,
        Some(persistent + facts.buffer_capacity(44).unwrap()),
        "same-unit and cross-unit aliases share the canonical root"
    );
    let receipt = workspace.disk_receipt().unwrap();
    for _ in 0..2 {
        let guard = receipt.activate().unwrap();
        forward(&mut f, true, true);
        drop(guard);
    }
}

#[test]
fn ordinary_roomy_windows_evict_across_groups_and_repeated_forwards() {
    let mut f = fixture(false, 2);
    let units = u64::try_from(f.definitions.len()).unwrap();
    let bytes: u64 = f.definitions.iter().flat_map(OffloadUnit::bindings)
        .map(WeightBinding::expected_bytes).sum();
    for _ in 0..2 {
        let before = f.manager.report().unwrap().offload().transfer(TransferDirection::DiskToDevice);
        // No admitted receipt is installed: ordinary execution must enforce
        // the same selected depth even though all five units fit the budget.
        assert!(!f.manager.admitted_disk_route_active());
        forward(&mut f, true, false);
        let report = f.manager.report().unwrap();
        assert_eq!(resident(&f.manager), BTreeSet::from([id(4)]));
        assert!(report.active_window().is_empty());
        assert_eq!(report.offload().peak_resident_units().get(MemoryTier::Device), 2);
        // Direct window acquisition records each actual copy publication;
        // it does not call the separate prefetch hit/miss producer.
        let after = report.offload().transfer(TransferDirection::DiskToDevice);
        assert_eq!(after.count() - before.count(), units);
        assert_eq!(after.bytes() - before.bytes(), bytes);
    }
}

#[test]
fn ordinary_dense_group_switch_preserves_unconsumed_transfers_on_rejection() {
    let mut f = fixture(false, 2);
    let initial = MlxTensor::from_array(Array::from_slice(&[1_i32], &[1, 1]));
    f.policy.begin(&initial, &f.stream).unwrap();
    let mut last = None;
    for index in 0..5 {
        let address = f.policy.layout.address(index).unwrap();
        let lease = f.policy.acquire(index, address,
            |stream| build(&f.definitions[index], stream), &f.stream).unwrap();
        let weight = parameter(&lease, "weight");
        let output = weight.add(weight, &f.stream).unwrap();
        assert!(output.as_array().evaluated().unwrap().as_slice::<i32>()
            .iter().all(|value| *value == 2 * (index as i32 + 1)));
        f.policy.complete(index, address, lease, &output,
            std::iter::empty(), std::iter::empty(), &f.stream).unwrap();
        if index == 0 {
            let before = f.manager.report().unwrap();
            let wrong_group = f.policy.layout.address(3).unwrap();
            let result = f.policy.acquire(3, wrong_group,
                |stream| build(&f.definitions[3], stream), &f.stream);
            assert!(matches!(result, Err(LayerwiseAcquireError::Policy(Error::Parallel(_)))));
            let after = f.manager.report().unwrap();
            assert_eq!(resident(&f.manager), BTreeSet::from([id(0), id(1)]));
            assert_eq!(after.active_window(), before.active_window());
            assert_eq!(after.offload().transfer(TransferDirection::DiskToDevice),
                before.offload().transfer(TransferDirection::DiskToDevice));
            assert!(after.units().iter().find(|unit| unit.id() == &id(1)).unwrap().device_pins() > 0,
                "the unconsumed transfer must retain its actual lease");
        }
        last = Some(output);
    }
    f.policy.finish(&last.unwrap(), &f.stream).unwrap();
    assert_eq!(resident(&f.manager), BTreeSet::from([id(4)]));
    assert!(f.manager.report().unwrap().active_window().is_empty());
}

#[test]
fn managed_inspection_and_wrong_address_reject_before_parameter_construction() {
    let mut f = fixture(false, 2);
    let workspace = f
        .policy
        .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
        .unwrap();
    let receipt = workspace.disk_receipt().unwrap();
    let _guard = receipt.activate().unwrap();
    let called = Cell::new(false);
    let address = f.policy.layout.address(0).unwrap();
    let result = f.policy.inspect_unit(
        0,
        address,
        |stream| {
            called.set(true);
            build(&f.definitions[0], stream)
        },
        |_| Ok(()),
        &f.stream,
    );
    assert!(result.is_err());
    assert!(!called.get());
    let wrong = f.policy.layout.address(1).unwrap();
    let result = f.policy.acquire(
        0,
        wrong,
        |stream| {
            called.set(true);
            build(&f.definitions[0], stream)
        },
        &f.stream,
    );
    assert!(result.is_err());
    assert!(!called.get());
    assert!(resident(&f.manager).is_empty());
}

#[test]
fn mismatched_dense_depth_does_not_claim_a_two_entry_transfer_quote() {
    let f = fixture(false, 1);
    assert!(f
        .policy
        .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
        .is_err());
}

#[test]
fn singleton_groups_accept_equivalent_selected_and_actual_window_depths() {
    let mut f = fixture_with_groups(false, 1, &[1, 1, 1, 1, 1], true);
    let workspace = f
        .policy
        .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
        .unwrap();
    let receipt = workspace.disk_receipt().unwrap();
    let _guard = receipt.activate().unwrap();
    forward(&mut f, true, false);
}

#[test]
fn ordinary_source_loading_keeps_unquoted_execution_and_typed_unknown_quote() {
    let mut f = fixture_with_groups(false, 2, &[3, 2], false);
    // Ordinary exact read geometry is diagnostic. It does not create the
    // original source identity required by a paid construction context.
    let allocation = NativeAllocationFacts::current_host().unwrap();
    f.policy.layerwise_workspace(allocation).unwrap();
    let context = eredu_nn::workspace::WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let error = f.policy.layerwise_workspace_with_metadata(allocation, &context).unwrap_err();
    let mut current: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound) =
            current.downcast_ref()
        {
            break;
        }
        current = current
            .source()
            .expect("original unknown workspace cause must survive");
    }
    forward(&mut f, false, false);
}

#[test]
fn admitted_window_physically_retires_old_weight_before_next_constructor_and_refill() {
    let mut f = fixture(false, 2);
    let workspace = f
        .policy
        .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
        .unwrap();
    let receipt = workspace.disk_receipt().unwrap();
    // Match SessionOperation: this outer scope remains open while each unit
    // enters and retires its nested scope. The next-constructor retirement
    // assertion therefore runs before the outer operation is sealed.
    let mut operation = safemlx::SubmissionScope::begin().unwrap();
    let _guard = receipt.activate().unwrap();
    let retirement = Arc::new(AtomicUsize::new(0));
    forward_with_retirement_probe(&mut f, true, false, Some(&retirement));
    assert_eq!(retirement.load(Ordering::SeqCst), 1);
    operation.seal();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let status = operation.progress();
        assert!(!status.failed() && !status.blocked(), "{status:?}");
        if status.is_settled() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "outer operation did not settle: {status:?}"
        );
        std::thread::yield_now();
    }
}

mod slot_bounds;

mod parameter_source_tests {
    use super::*;
    use eredu_runtime::working_memory::*;

    struct Destinations {
        units: Vec<WorkspaceParameterUnitRecord>,
        requested: Vec<WorkspaceParameterRequestRecord>,
        rows: Vec<WorkspaceParameterRecord>,
        names: Vec<u8>,
        shapes: Vec<i32>,
        roots: Vec<WorkspaceParameterRootRecord>,
        members: Vec<usize>,
    }
    impl Destinations {
        fn new(c: WorkspaceParameterCounts) -> Self {
            Self {
                units: vec![Default::default(); c.units],
                requested: vec![Default::default(); c.requested],
                rows: vec![Default::default(); c.rows],
                names: vec![0xaa; c.name_bytes],
                shapes: vec![-1; c.shape_elements],
                roots: vec![Default::default(); c.roots],
                members: vec![usize::MAX; c.window_members],
            }
        }
        fn lend(&mut self) -> WorkspaceParameterDestinations<'_> {
            WorkspaceParameterDestinations {
                units: &mut self.units,
                requested: &mut self.requested,
                rows: &mut self.rows,
                names: &mut self.names,
                shapes: &mut self.shapes,
                roots: &mut self.roots,
                window_members: &mut self.members,
            }
        }
    }
    fn retained(
        context: &eredu_nn::workspace::WorkspaceContext,
        values: &[eredu_nn::workspace::WorkspaceTensor],
    ) -> u64 {
        context.begin_state_span(values).unwrap();
        context
            .report(values)
            .unwrap()
            .state
            .unwrap()
            .retained_bytes
            .unwrap()
    }

    #[test]
    fn actual_host_source_retains_selected_independent_parameter_exclusions() {
        use eredu_nn::workspace::{WorkspaceContext, WorkspaceDtype, WorkspaceTensor};
        let f = fixture_with_host_capacity(true, 2, &[3, 2], true, 1_000_000);
        for definition in &f.definitions {
            drop(f.manager.acquire(definition.id(), MemoryTier::Host).unwrap());
        }
        let selected = MlxSelectiveUnitPopulator::new(
            ["independent.bank".to_owned()].into());
        let host = Policy::new(f.manager.clone(), f.source.clone(),
            (0..5).map(id).collect(), f.policy.layout.clone(), 2,
            selected, Vec::new(), None, false, false).unwrap();
        let workspace = host.layerwise_workspace(NativeAllocationFacts::current_host().unwrap()).unwrap();
        assert!(workspace.excludes_parameter("independent.bank"));
        assert!(!workspace.excludes_parameter("missing.unit.weight"));
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let address = host.layout.address(0).unwrap();
        let rows = workspace.parameters(0, address, &context).unwrap();
        let mut module = rows.into_iter().map(|(id, value)|
            Parameter::new(ParameterSpec::trainable(id.as_str()).unwrap(), value)
        ).collect::<Vec<_>>();
        let bank = WorkspaceTensor::existing(
            context.layout(&[3], WorkspaceDtype::Float32).unwrap(), &context).unwrap();
        module.push(Parameter::new(ParameterSpec::trainable("independent.bank").unwrap(), bank.clone()));
        // Drop the native policy first: the source must retain its exact
        // immutable selection, not borrow a temporary filter or infer absence.
        drop(host);
        assert!(workspace.excludes_parameter("independent.bank"));
        assert!(workspace.known_retained_control_bytes(true).unwrap() > 0,
            "ordinary source retains its exact unpriced name payload in the quote");
        let mut projection = workspace.parameter_source().prepare_projection(&context).unwrap();
        projection.bind(&mut module, 0, address, &context).unwrap();
        assert_eq!(retained(&context, &[bank, module.last().unwrap().as_ref().clone()]), 12);
        drop(projection);
        drop(workspace);

    }

    #[test]
    fn actual_host_rows_preserve_alias_dispatch_order_and_independent_roots() {
        let f = fixture_with_host_capacity(true, 2, &[3, 2], true, 1_000_000);
        for definition in &f.definitions {
            drop(
                f.manager
                    .acquire(definition.id(), MemoryTier::Host)
                    .unwrap(),
            );
        }
        let host = Policy::new(
            f.manager.clone(),
            f.source.clone(),
            (0..5).map(id).collect(),
            f.policy.layout.clone(),
            2,
            MlxSelectiveUnitPopulator::new(BTreeSet::new()),
            Vec::new(),
            None,
            false,
            false,
        )
        .unwrap();
        let facts = NativeAllocationFacts::current_host().unwrap();
        let workspace = host.layerwise_workspace(facts).unwrap();
        assert!(workspace
            .parameter_source()
            .same_source(&workspace.parameter_source()));
        let before = f.source.source_diagnostics().unwrap();
        let counted = workspace.parameter_source().count().unwrap();
        assert_eq!(counted.counts().rows, counted.counts().roots);
        let mut destinations = Destinations::new(counted.counts());
        let table = counted.fill(destinations.lend()).unwrap();
        assert_eq!(
            f.source.source_diagnostics().unwrap(),
            before,
            "count/fill must not read checkpoint headers or payloads"
        );
        assert!(!f.manager.admitted_disk_route_active());
        let context = eredu_nn::workspace::WorkspaceContext::new(
            MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        );
        let mut expected = 0;
        let mut values = Vec::new();
        for ordinal in 0..5 {
            let unit = table
                .requested_unit(ordinal, host.layout.address(ordinal).unwrap())
                .unwrap();
            let ordinary = workspace
                .parameters(ordinal, host.layout.address(ordinal).unwrap(), &context)
                .unwrap();
            for row in table.unit_rows(unit).unwrap() {
                let name = table.row_name(row).unwrap();
                let key = eredu_nn::ParameterId::new(name).unwrap();
                assert_eq!(
                    ordinary[&key].layout().shape(),
                    table.row_layout(row).unwrap().shape()
                );
                expected += table
                    .root_capacity_bytes(table.row_root(row).unwrap())
                    .unwrap();
            }
            values.extend(ordinary.into_values());
        }
        assert_eq!(
            retained(&context, &values),
            expected,
            "shared physical host sources must not collapse independent future copy roots"
        );
        assert_eq!(table.window(2), Some(&[2][..]));
        assert_eq!(table.window(3), Some(&[3, 4][..]));
    }

    #[test]
    fn actual_disk_rows_preserve_persistent_companions_and_invocation_lifetimes() {
        let f = fixture(true, 2);
        let facts = NativeAllocationFacts::current_host().unwrap();
        let workspace = f.policy.layerwise_workspace(facts).unwrap();
        let before = f.source.source_diagnostics().unwrap();
        let counted = workspace.parameter_source().count().unwrap();
        assert_eq!((counted.counts().rows, counted.counts().roots), (8, 6));
        let mut destinations = Destinations::new(counted.counts());
        let table = counted.fill(destinations.lend()).unwrap();
        assert_eq!(f.source.source_diagnostics().unwrap(), before);
        assert!(!f.manager.admitted_disk_route_active());
        let locate = |unit: usize, name: &str| {
            table
                .unit_rows(unit)
                .unwrap()
                .find(|r| table.row_name(*r) == Some(name))
                .unwrap()
        };
        let owner = locate(0, "weight");
        let local_alias = locate(0, "same");
        let external_alias = locate(4, "shared");
        assert_eq!(table.row_root(owner), table.row_root(local_alias));
        assert_eq!(table.row_root(owner), table.row_root(external_alias));
        assert_ne!(
            table.row_root(owner),
            table.row_root(locate(0, "companion"))
        );
        assert_eq!(
            table
                .root_owner(table.row_root(owner).unwrap())
                .unwrap()
                .lifetime,
            WorkspaceParameterLifetime::Trace
        );
        assert_eq!(
            table
                .root_owner(table.row_root(locate(2, "weight")).unwrap())
                .unwrap()
                .lifetime,
            WorkspaceParameterLifetime::Invocation
        );
        let context = eredu_nn::workspace::WorkspaceContext::new(
            MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        );
        let persistent: Vec<_> = workspace
            .parameters(0, f.policy.layout.address(0).unwrap(), &context)
            .unwrap()
            .into_values()
            .chain(
                workspace
                    .parameters(0, f.policy.layout.address(0).unwrap(), &context)
                    .unwrap()
                    .into_values(),
            )
            .collect();
        assert_eq!(
            retained(&context, &persistent),
            facts.buffer_capacity(8).unwrap() + facts.buffer_capacity(12).unwrap()
        );
        let repeated: Vec<_> = workspace
            .parameters(2, f.policy.layout.address(2).unwrap(), &context)
            .unwrap()
            .into_values()
            .chain(
                workspace
                    .parameters(2, f.policy.layout.address(2).unwrap(), &context)
                    .unwrap()
                    .into_values(),
            )
            .collect();
        assert_eq!(
            retained(&context, &repeated),
            2 * facts.buffer_capacity(20).unwrap()
        );
        let second = eredu_nn::workspace::WorkspaceContext::new(
            MlxMetalWorkspaceMechanisms::current_host().unwrap(),
        );
        let fresh: Vec<_> = workspace
            .parameters(0, f.policy.layout.address(0).unwrap(), &second)
            .unwrap()
            .into_values()
            .collect();
        assert!(second.validate_values(&persistent).is_err());
        assert_eq!(
            retained(&second, &fresh),
            facts.buffer_capacity(8).unwrap() + facts.buffer_capacity(12).unwrap()
        );
    }

    #[test]
    fn actual_snapshot_loan_does_not_activate_route_or_adopt_warm_identity() {
        let mut f = fixture(true, 2);
        let workspace = f
            .policy
            .layerwise_workspace(NativeAllocationFacts::current_host().unwrap())
            .unwrap();
        let initial = workspace.parameter_source().count().unwrap().counts();
        let receipt = workspace.disk_receipt().unwrap();
        forward(&mut f, false, true);
        let before = f.source.source_diagnostics().unwrap();
        let counted = workspace.parameter_source().count().unwrap();
        assert_eq!(counted.counts(), initial);
        let mut destinations = Destinations::new(initial);
        let table = counted.fill(destinations.lend()).unwrap();
        assert_eq!(table.counts(), initial);
        assert_eq!(f.source.source_diagnostics().unwrap(), before);
        assert!(
            !f.manager.admitted_disk_route_active(),
            "a metadata loan cannot install an operation receipt"
        );
        receipt.validate().unwrap();
        let guard = receipt.activate().unwrap();
        assert!(f.manager.admitted_disk_route_active());
        drop(guard);
        assert!(!f.manager.admitted_disk_route_active());
    }
}
