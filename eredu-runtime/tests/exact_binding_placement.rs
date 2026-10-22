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
        ParameterMemberSpec, ReplicatedTextOutputCompanion,
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
    )
    .with_output_companions(
        [
            ("encoded.scales", eredu_nn::LinearCompanionRole::Scale),
            ("encoded.biases", eredu_nn::LinearCompanionRole::AffineBias),
        ]
        .map(|(name, role)| {
            ReplicatedTextOutputCompanion::new(
                name,
                role,
                vec![8, 1],
                owner(1).parameter_group_owner().unwrap(),
            )
            .unwrap()
        })
        .to_vec(),
    )
    .unwrap();
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
    let rank = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        1,
    )
    .unwrap();
    let layout = eredu_runtime::derive_partitioned_local_layout(&description, rank).unwrap();
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
    fn resource_preparation_keeps_independent_materializations_and_scopes_aliases() {
        use super::{AtomicUsize, Ordering, Source};
        use eredu_checkpoint::recipe::RecipeAlias;
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

    fn prepared() -> PreparedComponent {
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
        PreparedComponent::prepare(
            "shared-contract",
            description(&[0, 1, 2]),
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
        let (_, layout, source, units) = selected.into_unit_parts().unwrap();
        assert_eq!(layout.address(0), Some(address));
        assert_eq!(layout.ordinal(0, 2), Some(0));
        assert!(layout.ordinal(0, 0).is_none());
        assert_eq!(
            units[0][0]
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
        let (placement, layout, source, units) = projected.into_addressed_unit_parts().unwrap();
        assert_eq!(placement, Some(retained));
        assert_eq!(layout.address(0), Some(selected[0]));
        let local_graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("units")], "units").unwrap();
        let local = ExecutionUnitLayout::new(&local_graph, [2]).unwrap();
        let addresses = eredu_runtime::ExecutionUnitAddressMap::new(&local, &selected).unwrap();
        for (ordinal, address) in selected.into_iter().enumerate() {
            assert_eq!(
                addresses.resolve(ordinal, address).unwrap(),
                local.address(ordinal).unwrap()
            );
            assert_eq!(addresses.local_address(ordinal), local.address(ordinal));
        }
        assert!(matches!(
            addresses.resolve(0, selected[1]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::AddressMismatch { .. })
        ));
        assert!(matches!(
            addresses.resolve(2, selected[1]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::UnknownOrdinal { .. })
        ));
        assert_eq!(addresses.local_address(2), None);
        assert!(matches!(
            eredu_runtime::ExecutionUnitAddressMap::new(&local, &selected[..1]),
            Err(eredu_runtime::ExecutionUnitAddressMapError::CountMismatch { .. })
        ));
        assert!(matches!(
            eredu_runtime::ExecutionUnitAddressMap::new(&local, &[selected[0]; 2]),
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
}
