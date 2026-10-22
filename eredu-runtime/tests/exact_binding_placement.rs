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
    let encoding = SourceTensorEncoding::Safetensors(StoredDtype::F32);
    ReplicatedTextMaterializationTask::from_exact_source(
        "weight",
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
        ReplicatedTextParameterOwner::StaticRole("head".into()),
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
    fn resource_preparation_keeps_independent_materializations_and_scopes_aliases() {
        use super::{AtomicUsize, Ordering, Source};
        use eredu_checkpoint::recipe::RecipeAlias;
        use eredu_runtime::{
            prepare_resource_bindings, ParameterResourceRecipes, WeightBindingPlan,
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
        let resources = BTreeMap::from([
            (
                "first",
                ParameterResourceRecipes {
                    outputs: BTreeMap::from([("owner".into(), recipe())]),
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
    fn component_projection_preserves_placement_without_transforming_twice() {
        let prepared = prepared();
        let topology = eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
        )
        .unwrap();
        let layout =
            eredu_runtime::derive_partitioned_local_layout(prepared.description(), topology)
                .unwrap();
        let prepared = prepared.select(&layout).unwrap();
        let before = prepared.unit_bindings()[2][0].source_recipe();
        let projected = prepared.project("projected", description(&[2])).unwrap();
        assert_eq!(projected.source().source_keys(), ["physical.2"]);
        assert_eq!(projected.local_layout().unwrap().len(), 1);
        let retained = projected.local_layout().unwrap().clone();
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
    fn component_projection_rejects_unretained_storage() {
        let prepared = prepared().project("subset", description(&[2])).unwrap();
        assert!(matches!(prepared.project("invalid", description(&[0])),
            Err(ComponentPreparationError::InvalidProjection { target, .. }) if target == "weight.0"));
    }
    #[test]
    fn shared_component_storage_keeps_aliases_coherent_through_placement_and_precision() {
        use eredu_checkpoint::recipe::RecipeDtype;
        use eredu_runtime::{ComponentRecipe, RecipeBindingError, WeightBindingPlan};
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
                safetensors::Dtype::F32,
                vec![4, 2],
                vec![0; 32],
            )])
            .unwrap(),
        );
        let prepared = PreparedComponent::prepare("shared", description, source, |member| {
            Ok::<_, ComponentPreparationError>(if member.target() == "owner" {
                ComponentRecipe::Physical(DerivedWeightRecipe::source(
                    "physical",
                    TensorSelection::Full,
                ))
            } else {
                ComponentRecipe::Alias("owner".into())
            })
        })
        .unwrap();
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
