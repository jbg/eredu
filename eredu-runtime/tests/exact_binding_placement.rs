//! Local-slot validation must consume the selected placement before native work.

use std::{
    collections::BTreeSet,
    convert::Infallible,
    sync::atomic::{AtomicUsize, Ordering},
};

use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{
        CheckpointLease, CheckpointSource, MemoryWeightStore, StoreError, TensorMetadata,
        TensorReadRequest, TensorSelection, TensorSourceProvenance, WeightStoreDiagnostics,
    },
    LinearFormat, SourceTensorEncoding, StoredDtype,
};
use eredu_nn::{
    ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut, Parameterized,
};
use eredu_runtime::{
    build_exact_replicated_text_bindings, BindingPlanError, LocalModelLayout, LocalTensorLayout,
    ModuleBindingPlanError, ParameterBindingTarget, ParameterRole,
    ReplicatedTextMaterializationTask, ReplicatedTextParameterOwner, ReplicatedTextParameterRole,
    ReplicatedTextPhysicalSource, TensorPlacement, WeightLoweringDescriptor, WeightLoweringKind,
};

struct Module(ParameterBindingTarget);

impl Parameterized<ParameterBindingTarget> for Module {
    fn visit_parameters<'a, V: ParameterVisitor<'a, ParameterBindingTarget>>(
        &'a self,
        visitor: &mut V,
    ) {
        visitor.visit(
            ParameterMetadata::from_spec(&ParameterSpec::trainable("weight").unwrap(), true),
            &self.0,
        );
    }

    fn visit_parameters_mut<'a, V: ParameterVisitorMut<'a, ParameterBindingTarget>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        visitor.visit_mut(
            ParameterMetadata::from_spec(&ParameterSpec::trainable("weight").unwrap(), true),
            &mut self.0,
        );
    }

    fn set_trainable(&mut self, _: bool) {}
}

struct Source {
    store: MemoryWeightStore,
    overlay: bool,
    leases: AtomicUsize,
}

impl CheckpointSource for Source {
    fn source_keys(&self) -> Vec<String> {
        self.store.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.store.source_metadata(key)
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        let mut provenance = self.store.source_provenance(key)?;
        provenance.backing_shard = Some("/exact/model.safetensors".into());
        Ok(provenance)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.leases.fetch_add(1, Ordering::SeqCst);
        self.store.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.store.source_diagnostics()
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.overlay && key == "weight"
    }
}

fn task(
    shape: &[usize],
    lowering: WeightLoweringKind,
    executable: LinearFormat,
) -> ReplicatedTextMaterializationTask {
    owned_task(
        "weight",
        ReplicatedTextParameterOwner::StaticRole("head".into()),
        shape,
        lowering,
        executable,
    )
}

fn owned_task(
    name: &str,
    owner: ReplicatedTextParameterOwner,
    shape: &[usize],
    lowering: WeightLoweringKind,
    executable: LinearFormat,
) -> ReplicatedTextMaterializationTask {
    let encoding = SourceTensorEncoding::Safetensors(StoredDtype::F32);
    ReplicatedTextMaterializationTask::from_exact_source(
        name,
        ReplicatedTextPhysicalSource::new(
            "checkpoint.weight",
            "checkpoint.weight",
            "/exact/model.safetensors",
            "checkpoint.weight",
            encoding.clone(),
            (shape.iter().product::<usize>() * 4) as u64,
        )
        .unwrap(),
        Vec::new(),
        shape.to_vec(),
        shape.to_vec(),
        ReplicatedTextParameterRole::LinearWeight,
        owner,
        executable,
        lowering,
        WeightLoweringDescriptor::new(
            encoding,
            executable,
            shape.to_vec(),
            shape.to_vec(),
            Some(shape.len() - 1),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn acquisition_preserves_independent_owners_packed_bytes_and_sparse_execution_order() {
    use eredu_runtime::{
        local_parameter_acquisition_units, order_parameter_acquisition_units,
        parameter_acquisition_units, partition_parameter_acquisition_bindings,
        ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
        MemberSharding, OwnedParameterGroupSpec, ParameterAcquisitionError, ParameterGroupSpec,
        ParameterMemberSpec,
    };
    let owner = |unit| ReplicatedTextParameterOwner::ExecutionUnit {
        group: "decoder".into(),
        unit,
    };
    let encoded = owned_task(
        "encoded",
        owner(1),
        &[8, 64],
        WeightLoweringKind::Transform,
        LinearFormat::Affine(Default::default()),
    );
    let outputs = encoded
        .transform_output_companions(
            encoded.logical_shape(),
            "encoded.scales",
            Some("encoded.biases"),
        )
        .unwrap();
    let encoded = encoded.with_output_companions(outputs).unwrap();
    let tasks = vec![
        owned_task(
            "first",
            owner(4),
            &[8, 64],
            WeightLoweringKind::Direct,
            LinearFormat::Dense,
        ),
        encoded,
        owned_task(
            "third",
            owner(4),
            &[8, 64],
            WeightLoweringKind::Direct,
            LinearFormat::Dense,
        ),
    ];
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let complete = ExecutionUnitLayout::new(&graph, [5]).unwrap();
    let selected = complete
        .select([1, 4].map(|index| complete.address(index).unwrap()))
        .unwrap();
    let units =
        order_parameter_acquisition_units(parameter_acquisition_units(&tasks).unwrap(), &selected)
            .unwrap();
    assert_eq!(
        units
            .iter()
            .map(|unit| unit.owner().clone())
            .collect::<Vec<_>>(),
        [owner(1), owner(4)]
    );
    assert_eq!(
        units
            .iter()
            .map(|unit| unit.parameter_bytes())
            .collect::<Vec<_>>(),
        [320, 4096]
    );
    let groups = tasks
        .iter()
        .map(|task| {
            let members = std::iter::once(ParameterMemberSpec::new(
                task.name(),
                task.logical_shape().to_vec(),
                MemberSharding::Balanced { axis: 0 },
            ))
            .chain(task.output_companions().iter().map(|companion| {
                ParameterMemberSpec::new(
                    companion.name(),
                    companion.logical_shape().to_vec(),
                    MemberSharding::Balanced { axis: 0 },
                )
            }));
            OwnedParameterGroupSpec::new(
                task.owner().parameter_group_owner().unwrap(),
                ParameterGroupSpec::new(task.name(), ParameterRole::ColumnProjection, members)
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let description = ArchitectureParameterDescription::new(
        &graph,
        &complete,
        groups.iter().map(|group| group.group().clone()),
        groups.clone(),
    )
    .unwrap();
    // Parameter lookup uses canonical aliases and companions independently of
    // rank-local task order or checkpoint source sharing.
    let aliased = |aliases| {
        let first = &tasks[0];
        ReplicatedTextMaterializationTask::from_exact_source(
            first.name(),
            first.physical_sources()[0].clone(),
            aliases,
            first.logical_shape().to_vec(),
            first.physical_shape().to_vec(),
            first.role(),
            first.owner().clone(),
            first.executable(),
            first.lowering(),
            first.lowering_descriptor().clone(),
        )
        .unwrap()
    };
    let mut lookup_tasks = tasks.clone();
    lookup_tasks[0] = aliased(vec!["first-consumer".into()]);
    let materialized = eredu_runtime::MaterializationTaskIndex::new(&lookup_tasks).unwrap();
    assert!(materialized.get("first-consumer").is_none());
    assert!(materialized.get("encoded.scales").is_none());
    assert!(materialized.get("encoded.biases").is_none());
    let index = eredu_runtime::ParameterMemberIndex::new(
        &lookup_tasks,
        std::sync::Arc::new(description.clone()),
    );
    for (query, canonical) in [
        ("first-consumer", "first"),
        ("first", "first"),
        ("encoded.scales", "encoded.scales"),
        ("encoded.biases", "encoded.biases"),
        ("third", "third"),
    ] {
        let retained = index.resolve(query).unwrap();
        let borrowed =
            eredu_runtime::resolve_parameter_member(&lookup_tasks, &description, query).unwrap();
        assert_eq!(retained.member().target(), canonical);
        assert_eq!(retained.group().owner(), borrowed.group().owner());
        assert_eq!(retained.member(), borrowed.member());
    }
    assert!(matches!(
        index.resolve("missing"),
        Err(eredu_core::parameters::ParameterError::Missing(_))
    ));
    lookup_tasks.clear();
    assert_eq!(
        index.resolve("first-consumer").unwrap().member().target(),
        "first"
    );
    let duplicate = vec![tasks[0].clone(), tasks[0].clone()];
    let ambiguous = eredu_runtime::ParameterMemberIndex::new(
        &duplicate,
        std::sync::Arc::new(description.clone()),
    );
    assert!(matches!(
        ambiguous.resolve("first"),
        Err(eredu_core::parameters::ParameterError::Invalid(_))
    ));
    let ambiguous_alias = vec![aliased(vec!["third".into()])];
    let ambiguous = eredu_runtime::ParameterMemberIndex::new(
        &ambiguous_alias,
        std::sync::Arc::new(description.clone()),
    );
    assert!(matches!(
        ambiguous.resolve("first"),
        Err(eredu_core::parameters::ParameterError::Invalid(_))
    ));
    let rank = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        1,
    )
    .unwrap();
    let layout = eredu_runtime::derive_partitioned_local_layout(&description, rank).unwrap();
    struct MetadataBudget {
        used: u64,
        limit: u64,
    }
    impl eredu_core::capture::CaptureReservation for MetadataBudget {
        fn reserve(
            &mut self,
            usage: eredu_core::capture::CaptureUsage,
        ) -> Result<Option<eredu_core::capture::CaptureSkipReason>, eredu_core::capture::CaptureError>
        {
            let next = self.used.checked_add(usage.host_bytes).unwrap();
            if next > self.limit {
                return Err(eredu_core::capture::CaptureError::Invalid(
                    "metadata limit".into(),
                ));
            }
            self.used = next;
            Ok(None)
        }
    }
    let mut effective = layout.clone();
    effective.insert(
        "encoded".into(),
        LocalTensorLayout::new(
            "encoded",
            ParameterRole::ColumnProjection,
            vec![8, 8],
            vec![4, 8],
            TensorPlacement::Shard {
                axis: 0,
                index: 1,
                parts: 2,
            },
            None,
            None,
            false,
        ),
    );
    let mut budget = MetadataBudget {
        used: 0,
        limit: u64::MAX,
    };
    for (query, expected_shape) in [
        ("first-consumer", vec![4, 64]),
        ("encoded", vec![4, 64]),
        ("encoded.scales", vec![4, 1]),
    ] {
        let local = eredu_runtime::derive_parameter_member_layout(
            index.resolve(query).unwrap(),
            rank,
            rank,
            true,
            Some(&effective),
            &mut budget,
        )
        .unwrap();
        assert_eq!(local.coordinates().unwrap().local_shape(), expected_shape);
        let remote = eredu_runtime::derive_parameter_member_layout(
            index.resolve(query).unwrap(),
            rank,
            rank,
            false,
            Some(&effective),
            &mut budget,
        )
        .unwrap();
        assert!(remote.coordinates().is_none());
        assert_eq!(remote.target(), local.target());
        assert_eq!(remote.global_shape(), local.global_shape());
    }
    let mut denied = MetadataBudget {
        used: 0,
        limit: 320,
    };
    assert!(matches!(
        eredu_runtime::derive_parameter_member_layout(
            index.resolve("encoded").unwrap(),
            rank,
            rank,
            true,
            Some(&effective),
            &mut denied
        ),
        Err(eredu_core::parameters::ParameterError::Budget(_))
    ));
    assert!(denied.used > 0);
    let consumed = denied.used;
    denied.limit = u64::MAX;
    eredu_runtime::derive_parameter_member_layout(
        index.resolve("encoded").unwrap(),
        rank,
        rank,
        true,
        Some(&effective),
        &mut denied,
    )
    .unwrap();
    assert!(denied.used > consumed);
    let source = source(vec![8, 64], false);
    let local = order_parameter_acquisition_units(
        local_parameter_acquisition_units(&tasks, &layout, &source).unwrap(),
        &selected,
    )
    .unwrap();
    assert_eq!(
        local
            .iter()
            .map(|unit| unit.parameter_bytes())
            .collect::<Vec<_>>(),
        [160, 2048]
    );
    assert_eq!(
        local[0]
            .outputs()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["encoded", "encoded.biases", "encoded.scales"]
    );
    assert_eq!(local[1].task_indices(), [0, 2]);
    // Module paths differ from canonical destinations, and source sharing does
    // not merge independent acquisition owners or change packed-companion costs.
    let binding = |target: &str, bytes| {
        eredu_runtime::WeightBinding::new(
            format!("native.{target}"),
            "checkpoint.weight",
            TensorSelection::Full,
            bytes,
        )
        .unwrap()
        .with_logical_target(target)
        .unwrap()
    };
    let bindings = vec![
        binding("first", 1024),
        binding("encoded.scales", 16),
        binding("third", 1024),
        binding("encoded", 128),
        binding("encoded.biases", 16),
    ];
    let projected = partition_parameter_acquisition_bindings(&local, bindings.clone()).unwrap();
    assert_eq!(
        projected
            .iter()
            .map(|unit| unit
                .iter()
                .map(|binding| binding.logical_target().unwrap())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        [
            vec!["encoded.scales", "encoded", "encoded.biases"],
            vec!["first", "third"]
        ]
    );
    let mut missing = bindings.clone();
    missing.pop();
    assert!(matches!(
        partition_parameter_acquisition_bindings(&local, missing),
        Err(ParameterAcquisitionError::Selection { missing, .. })
            if missing == ["encoded.biases"]
    ));
    let mut duplicate = bindings.clone();
    duplicate.push(bindings[0].clone());
    assert!(matches!(
        partition_parameter_acquisition_bindings(&local, duplicate),
        Err(ParameterAcquisitionError::Output { target, .. }) if target == "first"
    ));
    let mut foreign = bindings.clone();
    foreign.push(binding("neighbor", 1024));
    assert!(matches!(
        partition_parameter_acquisition_bindings(&local, foreign),
        Err(ParameterAcquisitionError::Output { target, .. }) if target == "neighbor"
    ));
    let mut wrong_extent = bindings;
    wrong_extent[0] = binding("first", 2048);
    assert!(matches!(
        partition_parameter_acquisition_bindings(&local, wrong_extent),
        Err(ParameterAcquisitionError::Extent { owner: actual_owner, expected: 2048, actual: 3072 })
            if actual_owner == owner(4)
    ));
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
    assert!(matches!(
        order_parameter_acquisition_units(units, &complete),
        Err(ParameterAcquisitionError::ExecutionOwnership { .. })
    ));
    let duplicate = vec![tasks[0].clone(), tasks[0].clone()];
    assert!(matches!(
        eredu_runtime::MaterializationTaskIndex::new(&duplicate),
        Err(eredu_runtime::MaterializationTaskIndexError { target, .. }) if target == "first"
    ));
    let error = parameter_acquisition_units(&duplicate).unwrap_err();
    assert!(
        matches!(&error, ParameterAcquisitionError::Output { target, .. } if target == "first")
    );
    let error: eredu_core::artifact::ArtifactError = error.into();
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<ParameterAcquisitionError>()
        .is_some());
}

fn source(shape: Vec<usize>, overlay: bool) -> Source {
    Source {
        store: MemoryWeightStore::from_safetensors([(
            if overlay {
                "weight"
            } else {
                "checkpoint.weight"
            }
            .into(),
            if overlay {
                safetensors::Dtype::U32
            } else {
                safetensors::Dtype::F32
            },
            shape.clone(),
            vec![0; shape.iter().product::<usize>() * 4],
        )])
        .unwrap(),
        overlay,
        leases: AtomicUsize::new(0),
    }
}

fn layout(global: Vec<usize>, local: Vec<usize>, placement: TensorPlacement) -> LocalModelLayout {
    let mut layout = LocalModelLayout::default();
    layout.insert(
        "weight".into(),
        LocalTensorLayout::new(
            "head",
            ParameterRole::ColumnProjection,
            global,
            local,
            placement,
            None,
            None,
            false,
        ),
    );
    layout
}

fn bind(
    source: &Source,
    task: &ReplicatedTextMaterializationTask,
    shape: Vec<usize>,
    layout: Option<&LocalModelLayout>,
) -> Result<Vec<eredu_runtime::WeightBinding>, ModuleBindingPlanError> {
    let module = Module(ParameterBindingTarget {
        shape,
        dtype: if source.overlay {
            RecipeDtype::U32
        } else {
            RecipeDtype::F32
        },
        permitted_source_dtypes: Vec::new(),
    });
    build_exact_replicated_text_bindings(
        &module,
        source,
        &[task],
        &BTreeSet::new(),
        layout,
        |value| Some(value.clone()),
        |_, recipe, _| Ok::<_, Infallible>(recipe),
    )
}

#[test]
fn exact_binding_places_each_rank_before_validating_its_local_slot() {
    let source = source(vec![4, 2], false);
    let task = task(&[4, 2], WeightLoweringKind::Direct, LinearFormat::Dense);
    assert!(matches!(
        bind(&source, &task, vec![2, 2], None),
        Err(ModuleBindingPlanError::Plan(
            BindingPlanError::ShapeMismatch { .. }
        ))
    ));
    for rank in 0..2 {
        let layout = layout(
            vec![4, 2],
            vec![2, 2],
            TensorPlacement::Shard {
                axis: 0,
                index: rank,
                parts: 2,
            },
        );
        let bindings = bind(&source, &task, vec![2, 2], Some(&layout)).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].logical_target(), Some("weight"));
        assert_eq!(bindings[0].checkpoint_key(), "checkpoint.weight");
        assert_eq!(
            bindings[0].selection(),
            &TensorSelection::Range {
                axis: 0,
                start: rank * 2,
                end: rank * 2 + 2
            }
        );
        assert_eq!(
            bindings[0]
                .source_recipe()
                .infer(&source as &dyn CheckpointSource)
                .unwrap()
                .shape(),
            [2, 2]
        );
        assert_eq!(bindings[0].expected_bytes(), 16);
        assert!(matches!(
            bind(&source, &task, vec![4, 2], Some(&layout)),
            Err(ModuleBindingPlanError::Plan(
                BindingPlanError::ShapeMismatch { .. }
            ))
        ));
    }
    assert!(bind(
        &source,
        &task,
        vec![2, 2],
        Some(&LocalModelLayout::default())
    )
    .is_err());
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_binding_applies_compound_member_and_tensor_placements_once() {
    let source = source(vec![4, 8, 2], false);
    let task = task(&[4, 8, 2], WeightLoweringKind::Direct, LinearFormat::Dense);
    let mut layout = LocalModelLayout::default();
    layout.insert(
        "weight".into(),
        LocalTensorLayout::new(
            "experts",
            ParameterRole::ExpertIntermediate,
            vec![4, 8, 2],
            vec![2, 4, 2],
            TensorPlacement::Shard {
                axis: 1,
                index: 1,
                parts: 2,
            },
            None,
            None,
            false,
        )
        .with_additional_placement(TensorPlacement::Range {
            axis: 0,
            start: 2,
            end: 4,
        }),
    );
    let bindings = bind(&source, &task, vec![2, 4, 2], Some(&layout)).unwrap();
    assert_eq!(
        bindings[0]
            .source_recipe()
            .infer(&source as &dyn CheckpointSource)
            .unwrap()
            .shape(),
        [2, 4, 2]
    );
    assert_eq!(bindings[0].expected_bytes(), 64);
    // Independently selected members consume their bank axis before the same
    // tensor placement, retaining separate payload charges and global identity.
    let bank_tasks = [owned_task(
        "weight",
        ReplicatedTextParameterOwner::ExecutionUnit {
            group: "decoder".into(),
            unit: 7,
        },
        &[4, 8, 2],
        WeightLoweringKind::Direct,
        LinearFormat::Dense,
    )];
    let index = eredu_runtime::MaterializationTaskIndex::new(&bank_tasks).unwrap();
    let mut cache = eredu_runtime::AddressableBankTaskCache::new(&index);
    let shared = cache
        .resolve::<Box<dyn std::error::Error>>("weight", &[1, 8, 2], None, |_, _| {
            panic!("direct task has no generated outputs")
        })
        .unwrap();
    let members = [3, 2].map(|member| {
        let recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::source(
            "checkpoint.weight",
            TensorSelection::Range {
                axis: 0,
                start: member,
                end: member + 1,
            },
        );
        let metadata = recipe.infer(&source as &dyn CheckpointSource).unwrap();
        let parameter = eredu_runtime::AddressableBankParameter::from_shared_task(
            "weight",
            shared.clone(),
            recipe,
            metadata.clone(),
            metadata.byte_len(),
            None,
        )
        .unwrap();
        eredu_runtime::AddressableBankMember::new(
            eredu_runtime::ParameterBankKey::new(1, 7, member),
            eredu_runtime::AddressableBankMemberPlacement::new(
                eredu_runtime::ExecutionGroupId::new("decoder").unwrap(),
                7,
                "decoder.7",
                eredu_runtime::AddressableBankDistribution::ExpertParallel,
            )
            .unwrap()
            .with_owner_rank(11),
            [parameter],
        )
        .unwrap()
    });
    let local = eredu_runtime::place_addressable_bank_members(
        &members,
        &source as &dyn CheckpointSource,
        &layout,
    )
    .unwrap();
    assert_eq!(local.iter().map(|m| m.selected_bytes()).sum::<u64>(), 64);
    assert_eq!(local.iter().map(|m| m.source_bytes()).sum::<u64>(), 64);
    for (global, local) in members.iter().zip(&local) {
        assert_eq!(local.key(), global.key());
        assert_eq!(local.placement(), global.placement());
        assert_eq!(local.parameters()[0].source_output().shape(), [1, 4, 2]);
        assert!(std::ptr::eq(
            local.parameters()[0].task(),
            global.parameters()[0].task()
        ));
    }
    let plans = eredu_runtime::plan_addressable_bank_bindings(
        &local,
        &source,
        |_, _, _| -> Result<_, String> {
            panic!("dense member requires no checkpoint-format lowering")
        },
    )
    .unwrap();
    assert_eq!(plans.iter().map(|p| p.source_bytes()).sum::<u64>(), 64);
    assert_eq!(plans.iter().map(|p| p.selected_bytes()).sum::<u64>(), 64);
    for plan in &plans {
        assert_eq!(
            plan.parameter_targets().get("weight").map(String::as_str),
            Some("weight")
        );
    }
    let catalog = eredu_runtime::AddressableBankCatalog::from_binding_plans(plans).unwrap();
    use eredu_core::residency::{MemoryTier, OffloadConfig};
    let (entries, _transformations, preparation) =
        eredu_runtime::AddressableBankPreparation::admit(
            catalog,
            Vec::new(),
            OffloadConfig::new(Some(32), Some(32), 1).unwrap(),
            &source as &dyn CheckpointSource,
        )
        .unwrap();
    let (residency, prepared, _placements) = preparation
        .bind(entries, &source as &dyn CheckpointSource)
        .unwrap()
        .into_parts();
    for member in prepared.members() {
        let address = prepared.binding(member.key, &member.binding).unwrap();
        assert_eq!(address.row(), member.key.member() - 2);
    }
    let declaration =
        ParameterMetadata::from_spec(&ParameterSpec::trainable("weight").unwrap(), true);
    let parameters = eredu_runtime::parameter_operations::prepare_bank_parameter_slots(
        &[declaration],
        prepared.into_members(),
    )
    .unwrap();
    assert_eq!(parameters.len(), 1);
    assert_eq!(
        parameters[0].slot.materialized,
        bindings[0]
            .source_recipe()
            .infer(&source as &dyn CheckpointSource)
            .unwrap()
    );
    assert_eq!(
        parameters[0]
            .members
            .iter()
            .map(|member| member.key.member())
            .collect::<Vec<_>>(),
        [2, 3]
    );

    // Full reads follow canonical member order through the shared slot driver.
    struct ReadSlots(Vec<(String, Vec<usize>)>);
    impl eredu_nn::ParameterSlotVisitor<Vec<usize>> for ReadSlots {
        fn visit_slot(&mut self, metadata: ParameterMetadata, value: &mut Vec<usize>) {
            self.0
                .push((metadata.id.as_str().to_owned(), value.clone()));
        }
    }
    let read = eredu_runtime::parameter_operations::BankParameterReadPlan::new(
        1,
        7,
        &parameters,
        &BTreeSet::from(["weight".to_owned()]),
    )
    .unwrap();
    let mut slots = ReadSlots(Vec::new());
    read.with_slots(
        |_| None,
        |member| Ok::<_, Infallible>(member.key.member()),
        |copies| Ok(copies),
        &mut |traversal| {
            traversal(&mut slots);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(slots.0, [("weight".to_owned(), vec![2, 3])]);
    // Final independently leased inputs enter the ordinary controller with their
    // sparse global identities and shared checkpoint authority intact.
    let (members, definitions, rows, plan) = residency.into_parts();
    // One canonical replacement changes every independently retained member's
    // effective charge; removing the overlay restores all original charges.
    use eredu_runtime::parameter_operations::{
        prepare_bank_parameter_replacements, BankParameterReplacementFacts,
    };
    let replacement =
        prepare_bank_parameter_replacements(&parameters[0].members, &members, true, |parameter| {
            (parameter == "weight").then_some(BankParameterReplacementFacts {
                leading_dimension: Some(2),
                element_type: Some(eredu_nn::TensorElementType::F16),
                byte_len: 32,
            })
        })
        .unwrap();
    let (selected, effective_bytes) = replacement.into_parts();
    assert_eq!(selected, BTreeSet::from(["weight".to_owned()]));
    assert_eq!(effective_bytes.values().copied().sum::<u64>(), 32);
    assert!(effective_bytes.values().all(|bytes| *bytes == 16));
    let (_, restored) =
        prepare_bank_parameter_replacements(&parameters[0].members, &members, false, |_| {
            panic!("inactive overlay must not inspect replacements")
        })
        .unwrap()
        .into_parts();
    assert_eq!(restored, members);
    // Compact admission charges every coexistent member at its current extent,
    // once per physical owner regardless of its logical use count.
    let requests: Vec<_> = members
        .keys()
        .rev()
        .map(|key| (*key, key.member() as u64))
        .collect();
    let (admitted, uses, compact_bytes) =
        eredu_runtime::AddressableBankDemand::new(&requests, &members, &effective_bytes, 32)
            .unwrap()
            .into_parts();
    assert_eq!(
        admitted.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        members.keys().copied().collect::<Vec<_>>()
    );
    assert_eq!(uses, 5);
    assert_eq!(compact_bytes, 32);
    assert!(matches!(
        eredu_runtime::AddressableBankDemand::new(&requests, &members, &restored, 32),
        Err(
            eredu_runtime::AddressableBankDemandError::ScratchLimitExceeded {
                required_bytes: 64,
                distinct_entries: 2,
                ..
            }
        )
    ));
    let ids: Vec<_> = members.keys().map(|key| key.unit_id()).collect();
    let mut controller = eredu_runtime::ResidencyController::new_with_row_ranges(
        |_| &source as &dyn CheckpointSource,
        plan,
        definitions,
        rows,
    )
    .unwrap();
    controller.ledger_mut().mark_initialized();
    controller
        .ledger_mut()
        .reserve_copy(&ids[0], MemoryTier::Device, 32, &BTreeSet::new())
        .unwrap();
    assert!(controller
        .ledger_mut()
        .reserve_copy(&ids[1], MemoryTier::Device, 32, &BTreeSet::new())
        .is_err());
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn exact_binding_does_not_reshard_selected_local_transform_output() {
    let source = source(vec![2, 4], true);
    let task = task(&[4, 32], WeightLoweringKind::Transform, LinearFormat::MxFp4);
    let layout = layout(
        vec![4, 32],
        vec![2, 32],
        TensorPlacement::Shard {
            axis: 0,
            index: 1,
            parts: 2,
        },
    );
    let bindings = bind(&source, &task, vec![2, 4], Some(&layout)).unwrap();
    assert_eq!(bindings[0].checkpoint_key(), "weight");
    assert_eq!(bindings[0].selection(), &TensorSelection::Full);
    assert_eq!(
        bindings[0]
            .source_recipe()
            .infer(&source as &dyn CheckpointSource)
            .unwrap()
            .shape(),
        [2, 4]
    );
    assert_eq!(source.leases.load(Ordering::SeqCst), 0);
}

mod component_preparation {
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{MemoryWeightStore, SharedCheckpointSource, TensorSelection},
    };
    use eredu_runtime::{
        ArchitectureParameterDescription, ComponentPreparationError, ExecutionGraph,
        ExecutionGroupSpec, ExecutionUnitLayout, MemberSharding, OwnedParameterGroupSpec,
        ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        PreparedComponent,
    };
    use std::sync::Arc;

    #[test]
    fn component_declarations_preserve_authoritative_parameter_geometry() {
        use eredu_runtime::{ArchitectureParameterError, PartitionUnitAssignment};
        let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let member = |shape, axis| {
            ParameterMemberSpec::new("weight", shape, MemberSharding::Partitioned { axis })
        };
        let group = |name, role, units, member| {
            ParameterGroupSpec::partitioned(name, role, units, [member]).unwrap()
        };
        let expected = group(
            "projection",
            ParameterRole::ColumnProjection,
            4,
            member(vec![4, 4], 0),
        );
        let owner = ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), 0);
        for changed in [
            group(
                "projection",
                ParameterRole::ColumnProjection,
                4,
                member(vec![8, 4], 0),
            ),
            group(
                "projection",
                ParameterRole::ColumnProjection,
                4,
                member(vec![4, 4], 1),
            ),
            group(
                "projection",
                ParameterRole::RowProjection,
                4,
                member(vec![4, 4], 0),
            ),
            group(
                "other",
                ParameterRole::ColumnProjection,
                4,
                member(vec![4, 4], 0),
            ),
            group(
                "projection",
                ParameterRole::ColumnProjection,
                2,
                member(vec![4, 4], 0),
            ),
            expected
                .clone()
                .with_partition_unit_assignment(PartitionUnitAssignment::Repeat {
                    semantic_units: 8,
                })
                .unwrap(),
        ] {
            assert!(matches!(
                ArchitectureParameterDescription::new(&graph, &layout, [expected.clone()], [OwnedParameterGroupSpec::new(owner.clone(), changed)]),
                Err(ArchitectureParameterError::InconsistentDeclaration { target, .. }) if target == "weight"
            ));
        }
        let companion =
            ParameterMemberSpec::new("scale", vec![4], MemberSharding::Partitioned { axis: 0 });
        let expected = ParameterGroupSpec::partitioned(
            "projection",
            ParameterRole::ColumnProjection,
            4,
            [expected.members()[0].clone(), companion],
        )
        .unwrap();
        let reordered = ParameterGroupSpec::partitioned(
            "projection",
            ParameterRole::ColumnProjection,
            4,
            expected.members().iter().rev().cloned(),
        )
        .unwrap();
        let description = ArchitectureParameterDescription::new(
            &graph,
            &layout,
            [expected],
            [OwnedParameterGroupSpec::new(owner, reordered)],
        )
        .unwrap();
        let selected = eredu_runtime::derive_partitioned_local_layout(
            &description,
            eredu_core::ParallelRankTopology::new(
                eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
                1,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(selected.tensor("weight").unwrap().local_shape(), [2, 4]);
        assert_eq!(selected.tensor("scale").unwrap().local_shape(), [2]);
    }

    #[test]
    fn bank_recipe_members_keep_global_ids_companions_and_local_geometry() {
        use super::{AtomicUsize, Ordering, Source};
        use eredu_checkpoint::store::CheckpointSource;
        use eredu_runtime::{ParameterBankRecipes, PreparedParameterRecipes, RecipeBindingError};
        use std::collections::BTreeMap;
        let original = Arc::new(Source {
            store: MemoryWeightStore::from_safetensors((0..3).flat_map(|member| {
                [
                    (
                        format!("weight.{member}"),
                        safetensors::Dtype::U32,
                        vec![2, 4],
                        vec![0; 32],
                    ),
                    (
                        format!("scale.{member}"),
                        safetensors::Dtype::F32,
                        vec![2, 1],
                        vec![0; 8],
                    ),
                ]
            }))
            .unwrap(),
            overlay: false,
            leases: AtomicUsize::new(0),
        });
        let stack = |prefix: &str, count| DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: (0..count)
                .map(|member| {
                    DerivedWeightRecipe::source(format!("{prefix}.{member}"), TensorSelection::Full)
                })
                .collect(),
        };
        let recipes = BTreeMap::from([
            ("weight".into(), stack("weight", 3)),
            ("scales".into(), stack("scale", 3)),
        ]);
        let catalog = original.as_ref() as &dyn CheckpointSource;
        let bank = ParameterBankRecipes::new(catalog, recipes.clone(), 3).unwrap();
        assert_eq!(bank.member_count(), 3);
        assert!(bank.member(3).is_none());
        let rank =
            eredu_runtime::select_parameter_bank_recipes(catalog, 3, 0, &[2, 0], recipes.clone())
                .unwrap();
        let rank = PreparedParameterRecipes::new("ordered rank", original.clone(), rank).unwrap();
        assert_eq!(
            rank.source().source_keys(),
            ["scale.0", "scale.2", "weight.0", "weight.2"]
        );
        assert_eq!(
            rank.recipes()["weight"]
                .infer(rank.source().as_ref())
                .unwrap()
                .shape(),
            [2, 2, 4]
        );
        assert!(matches!(
            eredu_runtime::select_parameter_bank_recipes(catalog, 3, 0, &[], recipes.clone()),
            Err(RecipeBindingError::EmptyBankSelection)
        ));
        assert!(matches!(
            eredu_runtime::select_parameter_bank_recipes(catalog, 3, 0, &[2, 2], recipes.clone()),
            Err(RecipeBindingError::DuplicateBankMember { member: 2 })
        ));
        assert!(matches!(
            eredu_runtime::select_parameter_bank_recipes(catalog, 3, 0, &[3], recipes.clone()),
            Err(RecipeBindingError::BankMemberOutOfBounds {
                member: 3,
                global_members: 3
            })
        ));
        let duplicate_destination = eredu_runtime::select_parameter_bank_recipes(
            catalog,
            3,
            0,
            &[2, 0],
            vec![
                ("weight".into(), recipes["weight"].clone()),
                ("weight".into(), recipes["weight"].clone()),
            ],
        );
        assert!(matches!(
            duplicate_destination,
            Err(RecipeBindingError::Declaration(
                eredu_runtime::ResidencyDeclarationError::DuplicateLogicalBinding { name }
            )) if name == "weight"
        ));
        let local = bank
            .map_recipes(catalog, |target, recipe| {
                eredu_runtime::select_parameter_recipe(
                    catalog,
                    target,
                    recipe,
                    Some(&[TensorSelection::Range {
                        axis: 1,
                        start: 1,
                        end: 2,
                    }]),
                )
            })
            .unwrap();
        let member = PreparedParameterRecipes::new(
            "bank-global member 2",
            original.clone(),
            local.member(2).unwrap().clone(),
        )
        .unwrap();
        assert_eq!(member.source().source_keys(), ["scale.2", "weight.2"]);
        assert_eq!(
            member.recipes()["weight"]
                .infer(member.source().as_ref())
                .unwrap()
                .shape(),
            [1, 1, 4]
        );
        assert_eq!(
            member.recipes()["scales"]
                .infer(member.source().as_ref())
                .unwrap()
                .shape(),
            [1, 1, 1]
        );
        let segmented = bank
            .map_recipes(catalog, |target, recipe| {
                eredu_runtime::select_parameter_recipe(
                    catalog,
                    target,
                    recipe,
                    Some(&[
                        TensorSelection::Range {
                            axis: 1,
                            start: 1,
                            end: 2,
                        },
                        TensorSelection::Range {
                            axis: 1,
                            start: 0,
                            end: 1,
                        },
                    ]),
                )
            })
            .unwrap();
        let segmented = PreparedParameterRecipes::new(
            "ordered segments of member 2",
            original.clone(),
            segmented.member(2).unwrap().clone(),
        )
        .unwrap();
        assert_eq!(segmented.source().source_keys(), ["scale.2", "weight.2"]);
        assert_eq!(
            segmented.recipes()["weight"]
                .infer(segmented.source().as_ref())
                .unwrap()
                .shape(),
            [1, 2, 4]
        );
        for invalid in [
            vec![],
            vec![
                TensorSelection::Range {
                    axis: 1,
                    start: 0,
                    end: 1,
                },
                TensorSelection::Range {
                    axis: 2,
                    start: 0,
                    end: 1,
                },
            ],
            vec![TensorSelection::Full, TensorSelection::Full],
        ] {
            assert!(matches!(
                eredu_runtime::select_parameter_recipe(catalog, "weight", &recipes["weight"], Some(&invalid)),
                Err(RecipeBindingError::InvalidParameterSelection { target }) if target == "weight"
            ));
        }
        assert_eq!(bank.recipes(), &recipes);
        assert!(matches!(ParameterBankRecipes::new(catalog, BTreeMap::from([
            ("weight".into(), stack("weight", 3)),
            ("scales".into(), stack("scale", 2)),
        ]), 3), Err(RecipeBindingError::BankMemberCount { target, expected: 3, actual: 2 }) if target == "scales"));
        assert!(matches!(
            bank.map_recipes(catalog, |_, recipe| {
                Ok::<_, RecipeBindingError>(recipe.select_bounded(
                    catalog,
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 2,
                    },
                )?)
            }),
            Err(RecipeBindingError::BankMemberCount {
                expected: 3,
                actual: 2,
                ..
            })
        ));
        assert_eq!(original.leases.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn resource_preparation_keeps_independent_materializations_and_scopes_aliases() {
        use super::{AtomicUsize, Ordering, Source};
        use eredu_checkpoint::{recipe::RecipeAlias, store::CheckpointSource};
        use eredu_runtime::{
            prepare_resource_bindings, ParameterResourceRecipes, PreparedParameterRecipes,
            WeightBindingPlan,
        };
        use std::collections::BTreeMap;
        let original = Arc::new(Source {
            store: MemoryWeightStore::from_safetensors(
                ["shared", "neighbor"]
                    .map(|key| (key.into(), safetensors::Dtype::F32, vec![4, 2], vec![0; 32])),
            )
            .unwrap(),
            overlay: false,
            leases: AtomicUsize::new(0),
        });
        let recipe = || DerivedWeightRecipe::source("shared", TensorSelection::Full);
        let owner = PreparedParameterRecipes::new(
            "ordinary owner",
            original.clone(),
            BTreeMap::from([("owner".into(), recipe())]),
        )
        .unwrap();
        assert_eq!(owner.source().source_keys(), ["shared"]);
        assert!(owner.source().source_metadata("neighbor").is_err());
        let targets = ["owner".to_owned()].into_iter().collect();
        let projected = owner.clone().project("same owner", &targets).unwrap();
        assert_eq!(projected.recipes(), owner.recipes());
        assert_eq!(projected.source().source_keys(), ["shared"]);
        assert!(matches!(
            owner.clone().project("foreign", &["peer".into()].into_iter().collect()),
            Err(eredu_runtime::RecipeBindingError::UnknownParameter { target }) if target == "peer"
        ));
        assert!(owner
            .clone()
            .map_recipes("foreign input", |_, _, _| {
                Ok::<_, eredu_runtime::RecipeBindingError>(DerivedWeightRecipe::source(
                    "neighbor",
                    TensorSelection::Full,
                ))
            })
            .is_err());
        let both = PreparedParameterRecipes::new(
            "two sources",
            original.clone(),
            BTreeMap::from([
                ("owner".into(), recipe()),
                (
                    "peer".into(),
                    DerivedWeightRecipe::source("neighbor", TensorSelection::Full),
                ),
            ]),
        )
        .unwrap();
        let selected = both
            .map_recipes("local recipes", |_, recipe, source| {
                Ok::<_, eredu_runtime::RecipeBindingError>(recipe.select_bounded(
                    source,
                    TensorSelection::Range {
                        axis: 0,
                        start: 2,
                        end: 4,
                    },
                )?)
            })
            .unwrap()
            .project("peer only", &["peer".into()].into_iter().collect())
            .unwrap();
        assert_eq!(selected.source().source_keys(), ["neighbor"]);
        assert_eq!(
            selected.source().source_provenance("neighbor").unwrap(),
            original.source_provenance("neighbor").unwrap()
        );
        assert_eq!(
            selected.recipes()["peer"]
                .infer(selected.source().as_ref())
                .unwrap()
                .shape(),
            [2, 2]
        );
        let recipes = selected.recipes().clone();
        let selected = selected
            .project(
                "same local selection",
                &["peer".into()].into_iter().collect(),
            )
            .unwrap();
        assert_eq!(selected.recipes(), &recipes);
        let empty_projection = selected
            .project("empty projection", &Default::default())
            .unwrap();
        assert!(empty_projection.source().source_keys().is_empty());
        assert!(empty_projection.recipes().is_empty());
        let empty = PreparedParameterRecipes::new("empty owner", original.clone(), BTreeMap::new())
            .unwrap();
        assert!(empty.source().source_keys().is_empty());
        let resources = BTreeMap::from([
            (
                "first",
                ParameterResourceRecipes {
                    outputs: owner.recipes().clone(),
                    aliases: vec![RecipeAlias::new("consumer", "owner")],
                },
            ),
            (
                "second",
                ParameterResourceRecipes {
                    outputs: BTreeMap::from([("replica".into(), recipe())]),
                    aliases: vec![],
                },
            ),
        ]);
        let (source, resources) =
            prepare_resource_bindings(original.clone(), "resources", resources).unwrap();
        assert_eq!(source.source_keys(), ["shared"]);
        assert!(source.source_metadata("neighbor").is_err());
        let plans = resources
            .values()
            .map(|bindings| WeightBindingPlan::new(bindings).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            plans
                .iter()
                .flat_map(|plan| plan.owners())
                .map(|binding| binding.expected_bytes())
                .sum::<u64>(),
            64
        );
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.aliases().count())
                .sum::<usize>(),
            1
        );
        assert_eq!(original.leases.load(Ordering::SeqCst), 0);

        let crossing = BTreeMap::from([
            (
                "first",
                ParameterResourceRecipes {
                    outputs: BTreeMap::from([("owner".into(), recipe())]),
                    aliases: vec![],
                },
            ),
            (
                "second",
                ParameterResourceRecipes {
                    outputs: BTreeMap::new(),
                    aliases: vec![RecipeAlias::new("consumer", "owner")],
                },
            ),
        ]);
        assert!(prepare_resource_bindings(original.clone(), "invalid", crossing).is_err());
        assert_eq!(original.leases.load(Ordering::SeqCst), 0);
    }

    fn description(selected: &[usize]) -> ArchitectureParameterDescription {
        let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [3]).unwrap();
        let groups = selected
            .iter()
            .map(|index| {
                ParameterGroupSpec::partitioned(
                    format!("parameter.{index}"),
                    ParameterRole::ColumnProjection,
                    4,
                    [ParameterMemberSpec::new(
                        format!("weight.{index}"),
                        vec![4, 2],
                        MemberSharding::Partitioned { axis: 0 },
                    )],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let owned = selected
            .iter()
            .zip(&groups)
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        layout.group_id(0).unwrap().clone(),
                        *index,
                    ),
                    group.clone(),
                )
            })
            .collect::<Vec<_>>();
        ArchitectureParameterDescription::new(&graph, &layout, groups, owned).unwrap()
    }

    fn state_layout(heads: i32) -> eredu_runtime::StateLayout {
        eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(
                3,
                vec![
                    eredu_core::cache::LayerCachePolicy::key_value(
                        eredu_core::AttentionPolicy::Full,
                        heads,
                        4,
                    )
                    .unwrap();
                    3
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn prepared() -> PreparedComponent<eredu_runtime::ComponentDefinition> {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors((0..4).map(|index| {
                (
                    format!("physical.{index}"),
                    safetensors::Dtype::F32,
                    vec![4, 2],
                    vec![0; 32],
                )
            }))
            .unwrap(),
        );
        let declaration = eredu_runtime::ComponentDefinition::new(
            "shared-contract",
            description(&[0, 1, 2]),
            (0..3)
                .map(|index| eredu_runtime::ComponentUnit {
                    path: format!("units.{index}"),
                    state: index..index + 1,
                })
                .collect(),
            vec![eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Projector,
                first_owner_static_roles: vec![],
                last_owner_static_roles: vec![],
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            "units",
            vec![],
            state_layout(2),
            eredu_runtime::ArchitectureStatePartitionPlan::new([
                eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..3),
            ]),
        )
        .unwrap();
        PreparedComponent::prepare(
            "shared-contract",
            declaration,
            source,
            |member| {
                let index = member.target().strip_prefix("weight.").unwrap();
                Ok::<_, ComponentPreparationError>(DerivedWeightRecipe::source(
                    format!("physical.{index}"),
                    TensorSelection::Full,
                ))
            },
        )
        .unwrap()
    }

    #[test]
    fn sparse_component_selection_retains_global_address_and_only_owned_sources() {
        let prepared = prepared();
        let address = prepared.description().unit_layout().address(2).unwrap();
        let unselected = prepared.description().unit_layout().address(0).unwrap();
        let selected = prepared.select_units([address]).unwrap();
        assert_eq!(selected.source().source_keys(), ["physical.2"]);
        assert_eq!(
            selected.unit_source(address).unwrap().source_keys(),
            ["physical.2"]
        );
        assert!(selected.unit_source(unselected).is_err());
        assert!(selected.pinned_source().unwrap().source_keys().is_empty());
        let (definition, _, layout, source, units) = selected.into_declared_unit_parts().unwrap();
        assert_eq!(definition.id(), "shared-contract");
        assert_eq!(definition.unit_count(0).unwrap(), 3);
        assert_eq!(layout.storage_layout().address(0), Some(address));
        assert_eq!(layout.storage_layout().ordinal(0, 2), Some(0));
        assert!(layout.storage_layout().ordinal(0, 0).is_none());
        assert_eq!(
            units[0][0]
                .source_recipe()
                .infer(source.as_ref())
                .unwrap()
                .shape(),
            [4, 2]
        );
        let prototype = self::prepared();
        let graph = prototype.description().graph();
        let complete = ExecutionUnitLayout::new(graph, [8]).unwrap();
        let addresses = [complete.address(3).unwrap(), complete.address(7).unwrap()];
        let sparse = complete.select(addresses).unwrap();
        let groups = [0, 2]
            .into_iter()
            .zip(addresses)
            .map(|(index, address)| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        sparse.group_id(0).unwrap().clone(),
                        address.index(),
                    ),
                    prototype.description().groups()[index].group().clone(),
                )
            })
            .collect::<Vec<_>>();
        let declare = |layout: &ExecutionUnitLayout, groups: Vec<OwnedParameterGroupSpec>| {
            ArchitectureParameterDescription::new(
                graph,
                layout,
                groups.iter().map(|group| group.group().clone()),
                groups.clone(),
            )
        };
        for missing in [0, 4] {
            let invalid = OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(sparse.group_id(0).unwrap().clone(), missing),
                groups[0].group().clone(),
            );
            assert!(matches!(declare(&sparse, vec![invalid]),
                Err(eredu_runtime::ArchitectureParameterError::UnitOutOfRange { global_unit, .. }) if global_unit == missing));
        }
        let parameters = declare(&sparse, groups.clone()).unwrap();
        let prepared = PreparedComponent::prepare(
            "sparse-parameters",
            parameters,
            prototype.source().clone(),
            |member| {
                Ok::<_, ComponentPreparationError>(DerivedWeightRecipe::source(
                    format!(
                        "physical.{}",
                        member.target().strip_prefix("weight.").unwrap()
                    ),
                    TensorSelection::Full,
                ))
            },
        )
        .unwrap();
        assert_eq!(
            prepared.unit_source(addresses[0]).unwrap().source_keys(),
            ["physical.0"]
        );
        assert_eq!(
            prepared.unit_source(addresses[1]).unwrap().source_keys(),
            ["physical.2"]
        );
        let selected_layout = sparse.select([addresses[1]]).unwrap();
        let selected = prepared
            .project(
                "sparse-owner",
                declare(&selected_layout, vec![groups[1].clone()]).unwrap(),
            )
            .unwrap();
        assert_eq!(selected.unit_layout().address(0), Some(addresses[1]));
        assert_eq!(selected.source().source_keys(), ["physical.2"]);
        let (_, _, retained, source, units) = selected.into_addressed_unit_parts().unwrap();
        assert_eq!(retained.storage_layout(), &selected_layout);
        assert_eq!(units[0].0, addresses[1]);
        assert_eq!(
            units[0].1[0]
                .source_recipe()
                .infer(source.as_ref())
                .unwrap()
                .shape(),
            [4, 2]
        );
    }

    #[test]
    fn logical_layout_admission_preserves_exact_selection_and_resource_authority() {
        use eredu_runtime::{
            validate_parameter_layout, LocalModelLayout, LocalTensorLayout, ParallelPlanError,
            TensorPlacement,
        };
        let prepared = prepared();
        let description = prepared.description();
        let groups = || description.groups().iter().map(|owned| owned.group());
        // A caller selects an explicit interval; it need not be a balanced rank cut.
        let selected = 1..4;
        let mut layout = LocalModelLayout::default();
        for group in groups() {
            for member in group.members() {
                layout.insert(
                    member.target().into(),
                    LocalTensorLayout::new(
                        group.logical_name(),
                        group.role(),
                        member.global_shape().to_vec(),
                        vec![3, 2],
                        TensorPlacement::Range {
                            axis: 0,
                            start: 1,
                            end: 4,
                        },
                        Some(4),
                        Some(selected.clone()),
                        false,
                    ),
                );
            }
        }
        let validate = |layout: &LocalModelLayout, range: Option<std::ops::Range<usize>>| {
            validate_parameter_layout(groups(), layout, |group| {
                let retained = eredu_runtime::logical_parameter_group_range(group, layout)?;
                if Some(retained) != range {
                    return Err(ParallelPlanError::InvalidGroup(
                        "selection differs from retained semantic geometry".into(),
                    ));
                }
                Ok(eredu_runtime::ParameterGroupSelection::Logical(
                    range.clone(),
                ))
            })
        };
        validate(&layout, Some(selected.clone())).unwrap();
        assert!(validate(&layout, Some(3..5)).is_err());
        assert!(validate(&layout, None).is_err());
        let mut foreign = layout.clone();
        foreign.insert(
            "neighbor".into(),
            layout.tensor("weight.0").unwrap().clone(),
        );
        assert!(validate(&foreign, Some(selected.clone())).is_err());
        let mut incomplete = LocalModelLayout::default();
        incomplete.insert(
            "weight.0".into(),
            layout.tensor("weight.0").unwrap().clone(),
        );
        assert!(validate(&incomplete, Some(selected)).is_err());

        let selected = prepared.select(&layout).unwrap();
        assert_eq!(
            selected.source().source_keys(),
            ["physical.0", "physical.1", "physical.2"]
        );
        let bindings = selected
            .unit_bindings()
            .iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.expected_bytes())
                .sum::<u64>(),
            72
        );
        for binding in bindings {
            assert_eq!(
                binding
                    .source_recipe()
                    .infer(selected.source().as_ref())
                    .unwrap()
                    .shape(),
                [3, 2]
            );
        }
    }

    #[test]
    fn axis_layout_admission_keeps_companions_aligned_and_replicas_whole() {
        use super::{AtomicUsize, Ordering, Source};
        use eredu_runtime::{
            derive_partitioned_local_layout, validate_parameter_layout, LocalModelLayout,
            LocalTensorLayout, ParallelPlanError, ParameterGroupSelection, TensorPlacement,
        };
        let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [2]).unwrap();
        let projection = ParameterGroupSpec::new(
            "projection",
            ParameterRole::Vocabulary,
            [ParameterMemberSpec::new(
                "weight",
                [11, 64],
                MemberSharding::Balanced { axis: 0 },
            )],
        )
        .unwrap();
        let projection =
            eredu_runtime::expand_linear_format_parameter_groups(vec![projection], |_| {
                Ok(Some(
                    eredu_nn::LinearFormatSpec::scaled(
                        eredu_checkpoint::LinearFormat::MxFp4,
                        eredu_nn::ParameterSpec::trainable("scale").unwrap(),
                    )
                    .unwrap(),
                ))
            })
            .unwrap()
            .remove(0);
        let groups = [
            projection,
            ParameterGroupSpec::new(
                "replica",
                ParameterRole::Replicated,
                [ParameterMemberSpec::new(
                    "recurrent",
                    [2],
                    MemberSharding::Replicated,
                )],
            )
            .unwrap(),
        ];
        let owned = groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(units.group_id(0).unwrap().clone(), index),
                    group.clone(),
                )
            })
            .collect::<Vec<_>>();
        let description =
            ArchitectureParameterDescription::new(&graph, &units, groups, owned).unwrap();
        let original = Arc::new(Source {
            store: MemoryWeightStore::from_safetensors([
                (
                    "physical.weight".into(),
                    safetensors::Dtype::U32,
                    vec![11, 8],
                    vec![0; 352],
                ),
                (
                    "physical.scale".into(),
                    safetensors::Dtype::U8,
                    vec![11, 2],
                    vec![0; 22],
                ),
                (
                    "physical.recurrent".into(),
                    safetensors::Dtype::F32,
                    vec![2],
                    vec![0; 8],
                ),
                (
                    "neighbor".into(),
                    safetensors::Dtype::F32,
                    vec![2],
                    vec![0; 8],
                ),
            ])
            .unwrap(),
            overlay: false,
            leases: AtomicUsize::new(0),
        });
        let prepared =
            PreparedComponent::prepare("axis-contract", description, original.clone(), |member| {
                Ok::<_, ComponentPreparationError>(DerivedWeightRecipe::source(
                    format!("physical.{}", member.target()),
                    TensorSelection::Full,
                ))
            })
            .unwrap();
        let description = prepared.description();
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
        )
        .unwrap();
        let layout = derive_partitioned_local_layout(description, topology).unwrap();
        eredu_runtime::validate_tensor_parallel_layout("axis-contract", description, &layout, 2, 1)
            .unwrap();
        assert!(matches!(
            eredu_runtime::validate_tensor_parallel_layout(
                "axis-contract",
                description,
                &layout,
                2,
                0,
            ),
            Err(eredu_runtime::TensorParallelLayoutError::Mismatch { .. })
        ));
        let validate = |layout: &LocalModelLayout, range: std::ops::Range<usize>| {
            validate_parameter_layout(
                description.groups().iter().map(|owned| owned.group()),
                layout,
                |group| {
                    Ok::<_, ParallelPlanError>(if group.role() == ParameterRole::Vocabulary {
                        ParameterGroupSelection::BalancedAxis {
                            axis: 0,
                            global_extent: 11,
                            range: range.clone(),
                        }
                    } else {
                        ParameterGroupSelection::Logical(None)
                    })
                },
            )
        };
        validate(&layout, 6..11).unwrap();
        assert!(validate(&layout, 11..11).is_err());
        assert!(validate(&layout, 6..12).is_err());
        for (target, shape, range) in [("scale", vec![5, 2], 0..5), ("recurrent", vec![1], 0..1)] {
            let mut invalid = layout.clone();
            let tensor = layout.tensor(target).unwrap();
            invalid.insert(
                target.into(),
                LocalTensorLayout::new(
                    tensor.logical_name(),
                    tensor.role(),
                    tensor.global_shape().to_vec(),
                    shape,
                    TensorPlacement::Range {
                        axis: 0,
                        start: range.start,
                        end: range.end,
                    },
                    None,
                    None,
                    false,
                ),
            );
            assert!(validate(&invalid, 6..11).is_err());
        }
        let prepared = prepared.select(&layout).unwrap();
        assert_eq!(
            prepared
                .unit_bindings()
                .iter()
                .flatten()
                .map(|binding| binding.expected_bytes())
                .sum::<u64>(),
            178
        );
        assert_eq!(
            prepared.source().source_keys(),
            ["physical.recurrent", "physical.scale", "physical.weight"]
        );
        assert_eq!(original.leases.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn component_projection_preserves_placement_without_transforming_twice() {
        let prepared = prepared();
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
        )
        .unwrap();
        let layout = prepared.parallel_layout(topology).unwrap();
        let prepared = prepared.select(&layout).unwrap();
        let before = prepared.unit_bindings()[2][0].source_recipe();
        let prepared = prepared.with_state_layout(state_layout(1)).unwrap();
        let declaration = prepared.declaration();
        let selected = declaration
            .state_partition_plan()
            .resolve_partition(
                declaration.state_layout(),
                declaration.parameters().unit_layout(),
                |_| Some(2..3),
                false,
            )
            .unwrap()
            .unwrap();
        assert_eq!(selected.global_layers(), 2..3);
        assert_eq!(selected.layout(), &state_layout(1).slice(2..3).unwrap());
        let projected = prepared.project("projected", description(&[2])).unwrap();
        assert_eq!(projected.source().source_keys(), ["physical.2"]);
        assert_eq!(projected.local_layout().unwrap().len(), 1);
        let retained = projected.parallel_layout(topology).unwrap();
        let other_rank = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            0,
        )
        .unwrap();
        assert!(matches!(
            projected.parallel_layout(other_rank),
            Err(ComponentPreparationError::SelectionMismatch(_))
        ));
        let projected = projected.select(&retained).unwrap();
        let after = projected.unit_bindings()[2][0].source_recipe();
        assert_eq!(before, after);
        assert_eq!(
            after.infer(projected.source().as_ref()).unwrap().shape(),
            [2, 2]
        );
        assert!(projected.select(&layout).is_err());
    }

    #[test]
    fn component_projection_rebases_groups_without_losing_sparse_addresses_or_placement() {
        let projected_description = description(&[0, 1, 2]);
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("ingress"),
                ExecutionGroupSpec::with_dependencies("units", ["ingress"]),
            ],
            "units",
        )
        .unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [1, 3]).unwrap();
        let complete = ArchitectureParameterDescription::new(
            &graph,
            &layout,
            projected_description
                .groups()
                .iter()
                .map(|owned| owned.group().clone()),
            projected_description.groups().iter().cloned(),
        )
        .unwrap();
        let source = prepared().source().clone();
        let original = PreparedComponent::prepare("complete", complete, source, |member| {
            let index = member.target().strip_prefix("weight.").unwrap();
            Ok::<_, ComponentPreparationError>(DerivedWeightRecipe::source(
                format!("physical.{index}"),
                TensorSelection::Full,
            ))
        })
        .unwrap();
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
        )
        .unwrap();
        let local =
            eredu_runtime::derive_partitioned_local_layout(original.description(), topology)
                .unwrap();
        let original = original.select(&local).unwrap();
        let recipes = original
            .unit_bindings()
            .iter()
            .flatten()
            .map(|binding| (binding.name().to_owned(), binding.source_recipe()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let selected =
            [1, 2].map(|index| projected_description.unit_layout().address(index).unwrap());
        let projected = original
            .project("projected", projected_description)
            .unwrap()
            .select_units(selected)
            .unwrap();
        let retained = projected.parallel_layout(topology).unwrap();
        let projected = projected.select(&retained).unwrap();
        assert_eq!(
            projected.source().source_keys(),
            ["physical.1", "physical.2"]
        );
        assert_eq!(projected.unit_layout().ordinal(0, 1), Some(0));
        assert_eq!(projected.unit_layout().ordinal(0, 2), Some(1));
        assert_eq!(
            projected
                .addressed_unit_bindings()
                .map(|(address, _)| address)
                .collect::<Vec<_>>(),
            selected
        );
        for (_, bindings) in projected.addressed_unit_bindings() {
            for binding in bindings {
                assert_eq!(binding.source_recipe(), recipes[binding.name()]);
                assert_eq!(
                    binding
                        .source_recipe()
                        .infer(projected.source().as_ref())
                        .unwrap()
                        .shape(),
                    [2, 2]
                );
            }
        }
        let (_, placement, layout, source, units) = projected.into_addressed_unit_parts().unwrap();
        assert_eq!(placement, Some(retained));
        assert_eq!(layout.storage_layout().address(0), Some(selected[0]));
        let local_graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
        let local = ExecutionUnitLayout::new(&local_graph, [2]).unwrap();
        let addresses = eredu_runtime::ExecutionUnitAddressMap::new(
            &local,
            selected
                .into_iter()
                .enumerate()
                .map(|(ordinal, address)| (address, local.address(ordinal).unwrap())),
        )
        .unwrap();
        for (ordinal, address) in selected.into_iter().enumerate() {
            let lease = addresses.resolve(ordinal, address).unwrap();
            lease.validate_return(ordinal, address).unwrap();
            let storage = lease.storage_location();
            assert_eq!(storage.ordinal(), ordinal);
            assert_eq!(storage.address(), local.address(ordinal).unwrap());
            assert_eq!(addresses.local_location(ordinal), Some(storage));
            assert!(matches!(
                lease.validate_return(ordinal + 1, address),
                Err(eredu_runtime::ExecutionUnitAddressMapError::ReturnOrdinalMismatch { .. })
            ));
            assert!(matches!(
                lease.validate_return(ordinal, selected[1 - ordinal]),
                Err(eredu_runtime::ExecutionUnitAddressMapError::AddressMismatch { .. })
            ));
        }
        assert!(matches!(
            addresses.resolve(0, selected[1]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::AddressMismatch { .. })
        ));
        assert!(matches!(
            addresses.resolve(2, selected[1]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::UnknownOrdinal { .. })
        ));
        assert_eq!(addresses.local_location(2), None);
        assert!(matches!(
            eredu_runtime::ExecutionUnitAddressMap::new(
                &local,
                [(selected[0], local.address(0).unwrap().with_index(99))],
            ),
            Err(eredu_runtime::ExecutionUnitAddressMapError::UnknownStorage { .. })
        ));
        assert!(matches!(
            eredu_runtime::ExecutionUnitAddressMap::new(
                &local,
                [(selected[0], local.address(0).unwrap())]
            ),
            Err(eredu_runtime::ExecutionUnitAddressMapError::UnreferencedStorage { .. })
        ));
        assert!(matches!(
            eredu_runtime::ExecutionUnitAddressMap::new(
                &local,
                [(selected[0], local.address(0).unwrap()); 2]
            ),
            Err(eredu_runtime::ExecutionUnitAddressMapError::DuplicateAddress { .. })
        ));
        assert_eq!(
            units
                .into_iter()
                .map(|(address, _)| address)
                .collect::<Vec<_>>(),
            selected
        );
        assert_eq!(source.source_keys(), ["physical.1", "physical.2"]);
    }

    #[test]
    fn component_projection_rejects_unretained_storage() {
        let prepared = prepared().project("subset", description(&[2])).unwrap();
        assert!(matches!(prepared.project("invalid", description(&[0])),
            Err(ComponentPreparationError::InvalidProjection { target, .. }) if target == "weight.0"));
    }
    #[test]
    fn shared_component_storage_keeps_aliases_coherent_through_placement_and_precision() {
        use eredu_checkpoint::recipe::RecipeDtype;
        use eredu_runtime::{
            ComponentRecipe, ComponentRecipeConversion, RecipeBindingError, WeightBindingPlan,
        };
        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("shared")], "shared").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        let parameters = ParameterGroupSpec::partitioned(
            "shared",
            ParameterRole::ColumnProjection,
            4,
            ["owner", "consumer"].map(|name| {
                ParameterMemberSpec::new(name, vec![4, 2], MemberSharding::Partitioned { axis: 0 })
            }),
        )
        .unwrap();
        let description = ArchitectureParameterDescription::new(
            &graph,
            &units,
            [parameters.clone()],
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(units.group_id(0).unwrap().clone(), 0),
                parameters,
            )],
        )
        .unwrap();
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "physical".into(),
                safetensors::Dtype::BF16,
                vec![4, 2],
                vec![0; 16],
            )])
            .unwrap(),
        );
        let prepared = PreparedComponent::prepare("shared", description, source, |member| {
            Ok::<_, ComponentPreparationError>(if member.target() == "owner" {
                ComponentRecipe::Physical {
                    recipe: DerivedWeightRecipe::source("physical", TensorSelection::Full),
                    conversion: Some(ComponentRecipeConversion {
                        dtype: RecipeDtype::F32,
                        from: &[RecipeDtype::F16, RecipeDtype::BF16],
                    }),
                }
            } else {
                ComponentRecipe::Alias("owner".into())
            })
        })
        .unwrap();
        let plan = WeightBindingPlan::new(&prepared.unit_bindings()[0]).unwrap();
        assert_eq!(
            plan.owners()
                .map(|binding| binding.expected_bytes())
                .sum::<u64>(),
            32
        );
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            0,
        )
        .unwrap();
        let layout =
            eredu_runtime::derive_partitioned_local_layout(prepared.description(), topology)
                .unwrap();
        let prepared = prepared
            .select(&layout)
            .unwrap()
            .transform_units(|source, units| {
                for binding in units
                    .iter_mut()
                    .flatten()
                    .filter(|binding| !binding.is_alias())
                {
                    let recipe = DerivedWeightRecipe::Cast {
                        input: Box::new(binding.source_recipe()),
                        dtype: RecipeDtype::F16,
                    };
                    let bytes = recipe
                        .infer(source.as_ref())
                        .map_err(ComponentPreparationError::from)?
                        .byte_len();
                    *binding = binding
                        .clone()
                        .with_source_recipe(recipe, bytes)
                        .map_err(RecipeBindingError::from)?;
                }
                Ok::<_, ComponentPreparationError>(())
            })
            .unwrap();
        let plan = WeightBindingPlan::new(&prepared.unit_bindings()[0]).unwrap();
        assert_eq!(plan.owners().count(), 1);
        assert_eq!(plan.aliases().count(), 1);
        for (alias, owner) in plan.aliases() {
            assert_eq!(alias.expected_bytes(), 8);
            assert_eq!(owner.expected_bytes(), 8);
        }
        assert_eq!(prepared.source().source_keys(), ["physical"]);
        let description = prepared.description().clone();
        let projected = prepared.project("retained", description);
        assert!(projected.is_ok());
    }
    #[test]
    fn shared_pinned_storage_preserves_consumer_state_and_pipeline_replicas() {
        use eredu_runtime::{
            ArchitecturePartition, ComponentDefinition, ComponentRecipe, NoAuxiliaryBoundarySchema,
            PartitionOwnership, WeightBindingPlan,
        };
        let prototype = prepared().declaration().clone();
        let graph = prototype.parameters().graph();
        let complete = ExecutionUnitLayout::new(graph, [12]).unwrap();
        let indices = [3, 7, 11];
        let units = complete
            .select(indices.map(|index| complete.address(index).unwrap()))
            .unwrap();
        let group = units.group_id(0).unwrap().clone();
        let groups = ["owner", "consumer"]
            .into_iter()
            .zip([3, 11])
            .map(|(name, index)| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_unit_consumers("shared", [(group.clone(), index)]),
                    ParameterGroupSpec::new(
                        name,
                        ParameterRole::Replicated,
                        [ParameterMemberSpec::new(
                            name,
                            [4, 2],
                            MemberSharding::Replicated,
                        )],
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let parameters = ArchitectureParameterDescription::new(
            graph,
            &units,
            groups.iter().map(|g| g.group().clone()).collect::<Vec<_>>(),
            groups,
        )
        .unwrap();
        let invalid = OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_unit_consumers("shared", [(group.clone(), 0)]),
            parameters.groups()[0].group().clone(),
        );
        assert!(matches!(
            ArchitectureParameterDescription::new(
                graph,
                &units,
                [invalid.group().clone()],
                [invalid],
            ),
            Err(eredu_runtime::ArchitectureParameterError::UnitOutOfRange { global_unit: 0, .. })
        ));
        let definition = ComponentDefinition::new(
            "shared-pinned",
            parameters,
            (0..3)
                .map(|i| prototype.unit(0, i).unwrap().clone())
                .collect(),
            vec![prototype.group_transport(0).unwrap().clone()],
            "units",
            vec![],
            prototype.state_layout().clone(),
            prototype.state_partition_plan().clone(),
        )
        .unwrap();
        assert_eq!(
            definition.parameter_storage(0, 3).unwrap(),
            definition.parameter_storage(0, 11).unwrap()
        );
        assert_ne!(
            definition.state_slots(0, 3).unwrap().collect::<Vec<_>>(),
            definition.state_slots(0, 11).unwrap().collect::<Vec<_>>()
        );
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "physical".into(),
                safetensors::Dtype::F32,
                vec![4, 2],
                vec![0; 32],
            )])
            .unwrap(),
        );
        let prepare = || {
            PreparedComponent::prepare(
                "shared-pinned",
                definition.clone(),
                source.clone(),
                |member| {
                    Ok::<_, ComponentPreparationError>(if member.target() == "owner" {
                        ComponentRecipe::from(DerivedWeightRecipe::source(
                            "physical",
                            TensorSelection::Full,
                        ))
                    } else {
                        ComponentRecipe::Alias("owner".into())
                    })
                },
            )
            .unwrap()
        };
        let mut replica_bytes = 0;
        for (position, replica) in [(0, false), (1, false), (1, true), (2, false)] {
            let index = indices[position];
            let partition = ArchitecturePartition::from_description(
                definition.parameters(),
                [("units", index..index + 1)],
                PartitionOwnership::new(position == 0, position == 2, std::iter::empty::<String>())
                    .unwrap()
                    .with_replicated_static_roles(if replica { vec!["shared"] } else { vec![] })
                    .unwrap(),
                definition.state_layout(),
                definition.state_partition_plan(),
                (),
                NoAuxiliaryBoundarySchema::new(2),
            )
            .unwrap();
            let owns_unit = |group: &eredu_runtime::ExecutionGroupId, unit| {
                partition
                    .groups()
                    .iter()
                    .any(|owned| owned.group() == group && owned.contains(unit))
            };
            let storage_groups = definition
                .parameters()
                .stored_groups(partition.ownership(), owns_unit)
                .collect::<Vec<_>>();
            let stores_owner = definition.parameters().stores_owner(
                definition.parameters().groups()[0].owner(),
                partition.ownership(),
                owns_unit,
            );
            if position == 1 && !replica {
                assert!(storage_groups.is_empty());
                assert!(!stores_owner);
                assert!(partition.parameter_bindings().is_empty());
                continue;
            }
            assert_eq!(storage_groups.len(), 2);
            assert!(stores_owner);
            if replica {
                assert!(!definition.parameters().groups()[0]
                    .owner()
                    .is_owned_by(partition.ownership(), owns_unit));
            }
            let selected = definition.select_partition(&partition).unwrap();
            assert_eq!(
                selected.state().unwrap().global_layers(),
                position..position + 1
            );
            let groups = definition.parameters().select_owned(&partition);
            assert_eq!(groups.len(), 2);
            let parameters = ArchitectureParameterDescription::new(
                graph,
                &units,
                groups.iter().map(|g| g.group().clone()).collect::<Vec<_>>(),
                groups,
            )
            .unwrap();
            let prepared = prepare()
                .project("stage", parameters)
                .unwrap()
                .select_units(partition.units())
                .unwrap();
            assert_eq!(prepared.pinned_bindings().len(), 1);
            assert!(prepared.unit_bindings().iter().all(Vec::is_empty));
            let bindings = prepared.pinned_bindings().values().next().unwrap();
            let plan = WeightBindingPlan::new(bindings).unwrap();
            assert_eq!(plan.owners().count(), 1);
            assert_eq!(plan.aliases().count(), 1);
            let pinned = eredu_runtime::StaticUnitBindings::new("shared", bindings.clone())
                .unwrap()
                .prepare::<Box<dyn std::error::Error>>()
                .unwrap();
            replica_bytes += pinned.bytes;
            let (definition, spec) = pinned.resource.unwrap();
            let insufficient =
                eredu_core::residency::OffloadConfig::new(Some(31), Some(0), 1).unwrap();
            assert!(matches!(
                eredu_core::residency::OffloadPlan::new(insufficient, [spec.clone()]),
                Err(eredu_core::residency::OffloadError::BudgetExceeded { .. })
            ));
            let exact = eredu_core::residency::OffloadConfig::new(Some(32), Some(0), 1).unwrap();
            let plan = eredu_core::residency::OffloadPlan::new(exact, [spec]).unwrap();
            eredu_runtime::ResidencyController::new(prepared.source().as_ref(), plan, [definition])
                .unwrap();
            assert_eq!(prepared.source().source_keys(), ["physical"]);
        }
        assert_eq!(replica_bytes, 96);

        let leased_groups = definition
            .parameters()
            .groups()
            .iter()
            .map(|owned| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit_consumers(group.clone(), 3, indices),
                    owned.group().clone(),
                )
            })
            .collect::<Vec<_>>();
        let parameters = ArchitectureParameterDescription::new(
            graph,
            &units,
            leased_groups.iter().map(|owned| owned.group().clone()),
            leased_groups.clone(),
        )
        .unwrap();
        let leased = ComponentDefinition::new(
            "shared-leased",
            parameters,
            (0..3)
                .map(|i| prototype.unit(0, i).unwrap().clone())
                .collect(),
            vec![prototype.group_transport(0).unwrap().clone()],
            "units",
            vec![],
            prototype.state_layout().clone(),
            prototype.state_partition_plan().clone(),
        )
        .unwrap();
        assert_eq!(leased.parameters().invocations().storage_layout().len(), 1);
        assert_eq!(
            leased.parameter_storage(0, 3).unwrap(),
            leased.parameter_storage(0, 11).unwrap()
        );
        assert_ne!(
            leased.state_slots(0, 3).unwrap().collect::<Vec<_>>(),
            leased.state_slots(0, 11).unwrap().collect::<Vec<_>>()
        );
        let prepare_leased = || {
            PreparedComponent::prepare("shared-leased", leased.clone(), source.clone(), |member| {
                Ok::<_, ComponentPreparationError>(if member.target() == "owner" {
                    ComponentRecipe::from(DerivedWeightRecipe::source(
                        "physical",
                        TensorSelection::Full,
                    ))
                } else {
                    ComponentRecipe::Alias("owner".into())
                })
            })
            .unwrap()
        };
        let full = prepare_leased();
        assert_eq!(full.unit_bindings().len(), 1);
        assert_eq!(full.unit_bindings()[0].len(), 2);
        let mut leased_replica_bytes = 0;
        for (position, index) in indices.into_iter().enumerate() {
            let partition = ArchitecturePartition::from_description(
                leased.parameters(),
                [("units", index..index + 1)],
                PartitionOwnership::new(position == 0, position == 2, std::iter::empty::<String>())
                    .unwrap(),
                leased.state_layout(),
                leased.state_partition_plan(),
                (),
                NoAuxiliaryBoundarySchema::new(2),
            )
            .unwrap();
            assert_eq!(partition.parameter_bindings().len(), 2);
            assert_eq!(
                partition.state().unwrap().global_layers(),
                position..position + 1
            );
            let address = complete.address(index).unwrap();
            let selected = prepare_leased()
                .select_units([address])
                .unwrap()
                .select_units([address])
                .unwrap();
            assert_eq!(
                selected
                    .invocations()
                    .logical_addresses()
                    .collect::<Vec<_>>(),
                [address]
            );
            assert_eq!(
                selected
                    .invocations()
                    .storage_layout()
                    .address(0)
                    .unwrap()
                    .index(),
                3
            );
            assert_eq!(
                selected.unit_source(address).unwrap().source_keys(),
                ["physical"]
            );
            if index != 3 {
                assert!(selected.unit_source(complete.address(3).unwrap()).is_err());
            }
            let (_, _, invocations, source, bindings) =
                selected.into_declared_unit_parts().unwrap();
            let mut resources = eredu_runtime::ExecutionUnitResourcePreparation::new(
                "shared",
                &invocations,
                eredu_runtime::LayerWeightResidency::FullyResident,
            );
            for binding in bindings {
                resources = resources
                    .push::<Box<dyn std::error::Error>>(binding, |_| Ok(0))
                    .unwrap();
            }
            let resources = resources.finish().unwrap();
            leased_replica_bytes += resources.parameter_bytes;
            let empty = eredu_runtime::StaticUnitBindings::new("empty", vec![])
                .unwrap()
                .prepare::<Box<dyn std::error::Error>>()
                .unwrap();
            let resources = eredu_runtime::ParameterResourcePreparation::new(empty, resources)
                .finish()
                .unwrap();
            let restricted = resources.restrict_primary_source(source).unwrap();
            eredu_runtime::ResidencyController::new(
                restricted.as_ref(),
                resources.plan,
                resources.definitions,
            )
            .unwrap();
        }
        assert_eq!(leased_replica_bytes, 96);
        assert!(matches!(
            prepare_leased()
                .select_units([complete.address(7).unwrap()])
                .unwrap()
                .project("restored", leased.clone()),
            Err(ComponentPreparationError::InvalidProjection { .. })
        ));

        for consumers in [vec![], vec![3, 3], vec![0]] {
            let invalid = OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit_consumers(group.clone(), 3, consumers),
                leased_groups[0].group().clone(),
            );
            assert!(ArchitectureParameterDescription::new(
                graph,
                &units,
                [invalid.group().clone()],
                [invalid]
            )
            .is_err());
        }
        let mut inconsistent = leased_groups.clone();
        inconsistent[1] = OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit_consumers(group.clone(), 3, [3, 11]),
            inconsistent[1].group().clone(),
        );
        assert!(matches!(
            ArchitectureParameterDescription::new(
                graph,
                &units,
                inconsistent.iter().map(|owned| owned.group().clone()),
                inconsistent.clone()
            ),
            Err(eredu_runtime::ArchitectureParameterError::InconsistentUnitConsumers { .. })
        ));
        let mut conflicting = leased_groups.clone();
        conflicting[1] = OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit_consumers(group.clone(), 11, [11]),
            conflicting[1].group().clone(),
        );
        assert!(matches!(
            ArchitectureParameterDescription::new(
                graph,
                &units,
                conflicting.iter().map(|owned| owned.group().clone()),
                conflicting.clone()
            ),
            Err(eredu_runtime::ArchitectureParameterError::ConflictingUnitOwners { .. })
        ));
        let empty = eredu_runtime::StaticUnitBindings::new("empty", vec![])
            .unwrap()
            .prepare::<Box<dyn std::error::Error>>()
            .unwrap();
        let zero = eredu_core::residency::OffloadConfig::new(Some(0), Some(0), 1).unwrap();
        eredu_core::residency::OffloadPlan::new(
            zero,
            empty.resource.into_iter().map(|(_, spec)| spec),
        )
        .unwrap();
    }

    fn assert_mapped_resource_preparation() {
        use eredu_core::residency::OffloadConfig;
        use eredu_runtime::{
            DenseDiskStreamLoadOptions, ExecutionUnitResourcePreparation, LayerWeightResidency,
            LayerwiseLoadOptions, ParameterResourcePreparation, ParameterResourcePreparationError,
            StaticUnitBindings, WeightBinding,
        };
        type Failure = Box<dyn std::error::Error>;
        let graph = ExecutionGraph::chain(["units", "readout"]).unwrap();
        let storage = ExecutionUnitLayout::new(&graph, [3, 1]).unwrap();
        let logical = ExecutionUnitLayout::new(&graph, [8, 4]).unwrap();
        let invocations = eredu_runtime::ExecutionUnitAddressMap::new(
            &storage,
            (3..8)
                .chain([11])
                .map(|ordinal| logical.address(ordinal).unwrap())
                .zip([0, 0, 2, 1, 2, 3].map(|ordinal| storage.address(ordinal).unwrap())),
        )
        .unwrap();
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors((0..5).map(|index| {
                (
                    format!("physical.{index}"),
                    safetensors::Dtype::F32,
                    if index == 3 { vec![10] } else { vec![4, 2] },
                    vec![0; if index == 3 { 40 } else { 32 }],
                )
            }))
            .unwrap(),
        );
        let prepare_units =
            |mode| -> Result<eredu_runtime::PreparedExecutionUnitResources, Failure> {
                let mut units = ExecutionUnitResourcePreparation::new("mapped", &invocations, mode);
                for (index, bytes, selection) in [
                    (0, 32, TensorSelection::Full),
                    (
                        1,
                        8,
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                    ),
                    (
                        2,
                        16,
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 2,
                        },
                    ),
                    (3, 40, TensorSelection::Full),
                ] {
                    let binding = WeightBinding::new(
                        format!("mapped.weight.{index}"),
                        format!("physical.{index}"),
                        selection,
                        bytes,
                    )?;
                    units = units.push(vec![binding], |_| Ok::<_, Failure>(bytes * 2))?;
                }
                Ok(units.finish()?)
            };
        let finish = |units| -> Result<eredu_runtime::PreparedParameterResources, Failure> {
            let pinned = StaticUnitBindings::new("empty", vec![])?.prepare::<Failure>()?;
            Ok(ParameterResourcePreparation::new(pinned, units).finish()?)
        };
        let assemble = |mode| -> Result<eredu_runtime::PreparedParameterResources, Failure> {
            finish(prepare_units(mode)?)
        };
        let host = |device, host| {
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(device), Some(host), 2).unwrap(),
            ))
        };
        let stream = |device, host| {
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(
                    device,
                    host,
                    if host == 0 { 0 } else { 2 },
                    if host == 0 { 0 } else { 1 },
                )
                .unwrap(),
            )
        };
        for (mode, window) in [
            (LayerWeightResidency::FullyResident, 96),
            (host(48, 192), 48),
            (stream(48, 96), 48),
            (stream(48, 0), 48),
        ] {
            let resources = assemble(mode).unwrap();
            assert_eq!(resources.invocations, invocations);
            assert_eq!(resources.metadata.maximum_device_layer_bytes(), window);
            assert_eq!(resources.metadata.layer_parameter_bytes(), 96);
            let restricted = resources.restrict_primary_source(source.clone()).unwrap();
            assert_eq!(
                restricted.source_keys(),
                ["physical.0", "physical.1", "physical.2", "physical.3"]
            );
            assert!(restricted.source_metadata("physical.4").is_err());
            eredu_runtime::ResidencyController::new(
                restricted.as_ref(),
                resources.plan,
                resources.definitions,
            )
            .unwrap();
        }
        let address = |index| logical.address(index).unwrap();
        let consumers = [address(3), address(4)];
        let shared = prepare_units(LayerWeightResidency::FullyResident)
            .unwrap()
            .select_invocations(consumers)
            .unwrap();
        let shared = finish(shared).unwrap();
        assert_eq!(shared.metadata.layer_parameter_bytes(), 32);
        assert_eq!(
            shared
                .invocations
                .resolve(1, consumers[1])
                .unwrap()
                .storage_location()
                .ordinal(),
            0
        );
        let owner = shared.invocations.storage_layout().address(0).unwrap();
        assert!(!consumers.contains(&owner));
        let selected = prepare_units(host(48, 192))
            .unwrap()
            .select_invocations([address(4), address(7)])
            .unwrap()
            .select_invocations([address(4)])
            .unwrap();
        let selected = finish(selected).unwrap();
        assert_eq!(selected.metadata.layer_parameter_bytes(), 32);
        assert!(matches!(
            selected.invocations.select_invocations([address(3)]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::UnknownInvocation { .. })
        ));
        let restricted = selected.restrict_primary_source(source.clone()).unwrap();
        assert_eq!(restricted.source_keys(), ["physical.0"]);
        assert!(restricted.source_metadata("physical.2").is_err());
        eredu_runtime::ResidencyController::new(
            restricted.as_ref(),
            selected.plan,
            selected.definitions,
        )
        .unwrap();
        for invalid in [[address(7), address(4)], [address(4), address(4)]] {
            assert!(matches!(
                invocations.select_invocations(invalid),
                Err(eredu_runtime::ExecutionUnitAddressMapError::NonCanonicalSelection)
            ));
        }
        let linked_units = || {
            let mut units = prepare_units(LayerWeightResidency::FullyResident).unwrap();
            let mut bindings = units.definitions[0].bindings().to_vec();
            bindings.push(WeightBinding::alias("mapped.alias", "mapped.weight.2", 16).unwrap());
            units.definitions[0] =
                eredu_runtime::OffloadUnit::new(units.unit_ids[0].clone(), bindings).unwrap();
            units
        };
        assert!(matches!(linked_units().select_invocations([address(4)]),
            Err(eredu_runtime::ExecutionUnitPreparationError::DiscardedAliasOwner { owner, .. }) if owner == "mapped.weight.2"));
        let linked = finish(
            linked_units()
                .select_invocations([address(4), address(5)])
                .unwrap(),
        )
        .unwrap();
        let restricted = linked.restrict_primary_source(source.clone()).unwrap();
        eredu_runtime::ResidencyController::new(
            restricted.as_ref(),
            linked.plan,
            linked.definitions,
        )
        .unwrap();
        let mut replicas = 0;
        for consumer in consumers {
            let units = prepare_units(LayerWeightResidency::FullyResident)
                .unwrap()
                .select_invocations([consumer])
                .unwrap();
            let resources = finish(units).unwrap();
            replicas += resources.metadata.layer_parameter_bytes();
            let restricted = resources.restrict_primary_source(source.clone()).unwrap();
            assert_eq!(restricted.source_keys(), ["physical.0"]);
            eredu_runtime::ResidencyController::new(
                restricted.as_ref(),
                resources.plan,
                resources.definitions,
            )
            .unwrap();
        }
        assert_eq!(replicas, 64);
        for (mode, host_failure) in [(stream(47, 96), false), (stream(48, 95), true)] {
            let error = assemble(mode).err().unwrap();
            match error
                .downcast_ref::<ParameterResourcePreparationError>()
                .unwrap()
            {
                ParameterResourcePreparationError::HostBudget {
                    required: 96,
                    budget: 95,
                } if host_failure => {}
                ParameterResourcePreparationError::DeviceBudget {
                    window_bytes: 48,
                    required: 48,
                    budget: 47,
                    ..
                } if !host_failure => {}
                other => panic!("unexpected mapped capacity rejection: {other}"),
            }
        }
        assert_eq!(
            eredu_runtime::execution_unit_parameter_window_bytes(
                &invocations,
                &[u64::MAX, 1, 0, 0],
                host(48, 192)
            )
            .unwrap(),
            u64::MAX
        );
        assert!(matches!(
            eredu_runtime::execution_unit_parameter_window_bytes(
                &invocations,
                &[u64::MAX, 1, 0, 0],
                LayerWeightResidency::FullyResident
            ),
            Err(eredu_runtime::BoundedResidencySizingError::ArithmeticOverflow(_))
        ));
    }

    #[test]
    fn execution_unit_admission_keeps_sparse_addresses_and_independent_capacity() {
        use eredu_core::residency::MemoryTier;
        use eredu_runtime::{
            ExecutionUnitPreparationError, ExecutionUnitResourcePreparation, LayerWeightResidency,
            WeightBinding,
        };
        type Failure = Box<dyn std::error::Error>;
        let prepared = prepared();
        let selected = prepared
            .select_units([0, 2].map(|i| description(&[0, 1, 2]).unit_layout().address(i).unwrap()))
            .unwrap();
        let layout = selected.unit_layout();
        let admission_map = eredu_runtime::ExecutionUnitAddressMap::new(
            layout,
            (0..layout.len()).map(|ordinal| {
                let address = layout.address(ordinal).unwrap();
                (address, address)
            }),
        )
        .unwrap();
        let options =
            LayerWeightResidency::LayerwiseHost(eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(256), Some(512), 1).unwrap(),
            ));
        let mut bindings = selected.unit_bindings().to_vec();
        let owner = bindings[0][0].name().to_owned();
        bindings[0].push(WeightBinding::alias("alias", owner, 32).unwrap());
        assert!(matches!(
            ExecutionUnitResourcePreparation::new("model", &admission_map, options).finish(),
            Err(ExecutionUnitPreparationError::UnitCount {
                expected: 2,
                actual: 0
            })
        ));
        let resources = ExecutionUnitResourcePreparation::new("model", &admission_map, options)
            .push(bindings[0].clone(), |_| Ok::<_, Failure>(64))
            .unwrap()
            .push(bindings[1].clone(), |_| Ok::<_, Failure>(128))
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(resources.definitions.len(), 2);
        assert_eq!(resources.unit_bytes, [32, 32]);
        assert_eq!(resources.parameter_bytes, 64);
        assert_eq!(resources.total_host_bytes, 192);
        assert_eq!(resources.maximum_host_bytes, 128);
        assert_eq!(
            resources.source_keys,
            std::collections::BTreeSet::from(["physical.0".into(), "physical.2".into()])
        );
        assert_eq!(
            resources
                .unit_ids
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>(),
            ["model.units.00000", "model.units.00002"]
        );
        // Repeated invocations share copies and pins while transfer occurrences
        // remain ordered. Two ready uses of one resource do not charge two copies.
        let plan = eredu_core::residency::OffloadPlan::new(
            eredu_core::residency::OffloadConfig::new(Some(64), Some(512), 2).unwrap(),
            resources.specs,
        )
        .unwrap();
        let mut controller = eredu_runtime::ResidencyController::new(
            selected.source().as_ref(),
            plan,
            resources.definitions,
        )
        .unwrap();
        let ledger = controller.ledger_mut();
        ledger.mark_initialized();
        let logical = ExecutionUnitLayout::new(selected.description().graph(), [8]).unwrap();
        let invocations = (3..8)
            .map(|ordinal| logical.address(ordinal).unwrap())
            .collect::<Vec<_>>();
        let mapping = eredu_runtime::ExecutionUnitAddressMap::new(
            layout,
            invocations
                .iter()
                .copied()
                .zip([0, 0, 1, 0, 1].map(|ordinal| layout.address(ordinal).unwrap())),
        )
        .unwrap();
        let mut schedule = eredu_runtime::DenseTransferSchedule::new(
            mapping.group_storage_ordinals(0).unwrap(),
            2,
        )
        .unwrap();
        let mut completed = Vec::new();
        let mut first = None;
        let mut rejected_replay = false;
        while !schedule.is_exhausted() {
            let protected = schedule
                .storage_window(2)
                .into_iter()
                .map(|index| resources.unit_ids[index].clone())
                .collect::<Vec<_>>();
            ledger
                .set_group_window("dense", &protected, MemoryTier::Device)
                .unwrap();
            while schedule.can_admit() {
                let step = schedule.next_pending().unwrap();
                let invocation = mapping
                    .resolve(step.ordinal(), invocations[step.ordinal()])
                    .unwrap();
                let id = &resources.unit_ids[invocation.storage_location().ordinal()];
                if !ledger.is_resident(id, MemoryTier::Device).unwrap() {
                    ledger
                        .reserve_copy(id, MemoryTier::Device, 32, &Default::default())
                        .unwrap();
                    ledger
                        .publish_reserved(id, MemoryTier::Device, 32, None)
                        .unwrap();
                }
                ledger.pin(id, MemoryTier::Device, 1).unwrap();
                schedule.admit(step, id.clone()).unwrap();
            }
            let (step, id) = schedule.pop_ready().unwrap();
            first.get_or_insert(step);
            if let Some(next) = schedule.next_pending() {
                let old = first.unwrap();
                if next.storage_index() == old.storage_index() {
                    assert!(matches!(
                        schedule.admit(old, id.clone()),
                        Err(eredu_runtime::DenseTransferScheduleError::OutOfOrder { expected, actual })
                            if expected == next && actual == old
                    ));
                    rejected_replay = true;
                }
            }
            let reports = ledger.unit_reports();
            assert!(
                reports
                    .iter()
                    .map(|report| report.device_allocated_bytes())
                    .sum::<u64>()
                    <= 64
            );
            if completed.is_empty() {
                assert_eq!(
                    reports
                        .iter()
                        .map(|report| report.device_pins())
                        .sum::<u64>(),
                    2
                );
                assert_eq!(
                    reports
                        .iter()
                        .map(|report| report.device_allocated_bytes())
                        .sum::<u64>(),
                    32
                );
            }
            completed.push((step.ordinal(), step.storage_index()));
            ledger.unpin(&id, MemoryTier::Device);
        }
        assert!(rejected_replay);
        assert_eq!(completed, [(0, 0), (1, 0), (2, 1), (3, 0), (4, 1)]);
        assert!(!mapping.is_group_end(2));
        assert!(mapping.is_group_end(4));
        assert_eq!(
            mapping
                .storage_window(2, 2)
                .unwrap()
                .iter()
                .map(|storage| storage.ordinal())
                .collect::<Vec<_>>(),
            [1, 0]
        );
        assert!(ledger
            .unit_reports()
            .iter()
            .all(|report| report.device_pins() == 0));
        let assemble = |mode| -> Result<eredu_runtime::PreparedParameterResources, Failure> {
            let units = ExecutionUnitResourcePreparation::new("model", &admission_map, mode)
                .push(bindings[0].clone(), |_| Ok::<_, Failure>(64))?
                .push(bindings[1].clone(), |_| Ok::<_, Failure>(128))?
                .finish()?;
            let pinned = eredu_runtime::StaticUnitBindings::new(
                "pinned",
                vec![WeightBinding::new(
                    "pinned",
                    "physical.0",
                    TensorSelection::Full,
                    32,
                )?],
            )?
            .prepare::<Failure>()?;
            let mut preparation = eredu_runtime::ParameterResourcePreparation::new(pinned, units);
            for (index, bytes, capacity, shared) in
                [(0, 16, 64, true), (1, 48, 128, false), (2, 8, 32, false)]
            {
                // The same key names different geometry in each retained source.
                let source: SharedCheckpointSource =
                    Arc::new(MemoryWeightStore::from_safetensors([
                        (
                            "physical.0".into(),
                            safetensors::Dtype::F32,
                            vec![bytes as usize / 4],
                            vec![0; bytes as usize],
                        ),
                        (
                            "neighbor".into(),
                            safetensors::Dtype::F32,
                            vec![1],
                            vec![0; 4],
                        ),
                    ])?);
                let mut bindings = vec![WeightBinding::new(
                    format!("aux.weight.{index}"),
                    "physical.0",
                    TensorSelection::Full,
                    bytes,
                )?];
                if shared {
                    bindings.push(WeightBinding::alias(
                        format!("aux.alias.{index}"),
                        format!("aux.weight.{index}"),
                        bytes,
                    )?);
                }
                let definition = eredu_runtime::OffloadUnit::new(
                    eredu_core::residency::OffloadUnitId::new(format!("aux.{index}"))?,
                    bindings,
                )?;
                preparation = preparation
                    .push_auxiliary(definition, source, shared, |_| Ok::<_, Failure>(capacity))?;
            }
            assert_eq!(preparation.source_keys(), &resources.source_keys);
            Ok(preparation.finish()?)
        };
        let streamed = LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(256, 512, 2, 1).unwrap(),
        );
        let direct = LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(256, 0, 0, 0).unwrap(),
        );
        for (mode, window) in [
            (LayerWeightResidency::FullyResident, 64),
            (options, 32),
            (streamed, 64),
            (direct, 64),
        ] {
            let prepared = assemble(mode).unwrap();
            assert_eq!(prepared.invocations, admission_map);
            assert_eq!(prepared.metadata.maximum_device_layer_bytes(), window);
            assert_eq!(prepared.metadata.static_device_bytes(), 32);
            for source in prepared.sources.values() {
                assert_eq!(source.source_keys(), ["physical.0"]);
                assert!(source.source_metadata("neighbor").is_err());
            }
            eredu_runtime::ResidencyController::new_with_catalogs(
                |id| {
                    prepared
                        .sources
                        .get(id)
                        .map(|source| source.as_ref())
                        .unwrap_or(selected.source().as_ref())
                },
                prepared.plan,
                prepared.definitions,
            )
            .unwrap();
        }
        for (device, host, host_failure) in [(256, 415, true), (95, 512, false)] {
            let mode =
                LayerWeightResidency::LayerwiseHost(eredu_runtime::LayerwiseLoadOptions::new(
                    eredu_core::residency::OffloadConfig::new(Some(device), Some(host), 1).unwrap(),
                ));
            let error = assemble(mode).err().unwrap();
            match error
                .downcast_ref::<eredu_runtime::ParameterResourcePreparationError>()
                .unwrap()
            {
                eredu_runtime::ParameterResourcePreparationError::HostBudget {
                    required: 416,
                    budget: 415,
                } if host_failure => {}
                eredu_runtime::ParameterResourcePreparationError::DeviceBudget {
                    required: 96,
                    budget: 95,
                    ..
                } if !host_failure => {}
                other => panic!("unexpected capacity rejection: {other}"),
            }
        }
        assert_mapped_resource_preparation();
        let overflow = ExecutionUnitResourcePreparation::new("model", &admission_map, options)
            .push(bindings[0].clone(), |_| Ok::<_, Failure>(u64::MAX))
            .unwrap()
            .push(bindings[1].clone(), |_| Ok::<_, Failure>(1))
            .err()
            .unwrap();
        assert!(
            matches!(overflow.downcast_ref::<ExecutionUnitPreparationError>(),
            Some(ExecutionUnitPreparationError::ArithmeticOverflow { address, .. }) if address.index() == 2)
        );
        let failure = ExecutionUnitResourcePreparation::new("model", &admission_map, options)
            .push(bindings[0].clone(), |_| {
                Err::<u64, Failure>(std::io::Error::other("host capacity unavailable").into())
            })
            .err()
            .unwrap();
        assert!(failure.downcast_ref::<std::io::Error>().is_some());
    }
}
