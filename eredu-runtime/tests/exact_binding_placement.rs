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
    let catalog = |shape, role: &str| {
        let graph = eredu_runtime::ExecutionGraph::chain(["decoder"]).unwrap();
        let units = eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap();
        eredu_runtime::ArchitectureParameterDescription::from_owned(
            &graph,
            &units,
            [eredu_runtime::OwnedParameterGroupSpec::new(
                eredu_runtime::ParameterGroupOwner::static_role(role),
                eredu_runtime::ParameterGroupSpec::new(
                    "head",
                    ParameterRole::ColumnProjection,
                    [eredu_runtime::ParameterMemberSpec::new(
                        "weight",
                        shape,
                        eredu_runtime::MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            )],
        )
        .unwrap()
    };
    let global = catalog(vec![4, 2], "readout");
    let local = catalog(vec![2, 2], "readout");
    let selection = |ranges| std::collections::BTreeMap::from([("weight".to_owned(), ranges)]);
    for rank in 0..2 {
        let retained = selection(vec![TensorSelection::Range {
            axis: 0,
            start: rank * 2,
            end: rank * 2 + 2,
        }]);
        let layout =
            eredu_runtime::project_parameter_selection_layout(&retained, &global, Some(&local))
                .unwrap();
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
    // Fused physical ranges preserve their order through the actual binding plan.
    let fused = selection(vec![
        TensorSelection::Range {
            axis: 0,
            start: 2,
            end: 3,
        },
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 1,
        },
    ]);
    let layout =
        eredu_runtime::project_parameter_selection_layout(&fused, &global, Some(&local)).unwrap();
    let bindings = bind(&source, &task, vec![2, 2], Some(&layout)).unwrap();
    let selected = layout.tensor("weight").unwrap();
    assert_eq!(
        eredu_runtime::placement_selection(selected, selected.placement(), &[4, 2]).unwrap(),
        TensorSelection::Indices {
            axis: 0,
            indices: vec![2, 0]
        },
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
    let repeated = selection(vec![
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 2,
        },
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 3,
        },
    ]);
    assert!(matches!(
        eredu_runtime::project_parameter_selection_layout(&repeated, &global, None),
        Err(eredu_runtime::ParameterSelectionLayoutError::RepeatedIndices(_))
    ));
    assert!(matches!(
        eredu_runtime::project_parameter_selection_layout(
            &fused,
            &global,
            Some(&catalog(vec![3, 2], "readout"))
        ),
        Err(eredu_runtime::ParameterSelectionLayoutError::LocalShape { .. })
    ));
    assert!(matches!(
        eredu_runtime::project_parameter_selection_layout(
            &fused,
            &global,
            Some(&catalog(vec![2, 2], "other"))
        ),
        Err(eredu_runtime::ParameterSelectionLayoutError::ChangedOwner(
            _
        ))
    ));
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
            vec![4, 4, 2],
            TensorPlacement::Shard {
                axis: 1,
                index: 1,
                parts: 2,
            },
            None,
            None,
            false,
        ),
    );
    let ranges =
        |axis, range| std::collections::BTreeMap::from([("weight".to_owned(), (axis, range))]);
    let layout =
        eredu_runtime::project_additional_parameter_ranges(&layout, &ranges(0, 2..4), None)
            .unwrap();
    assert!(matches!(
        eredu_runtime::project_additional_parameter_ranges(&layout, &ranges(0, 0..2), None),
        Err(eredu_runtime::ParameterSelectionLayoutError::OverlappingAxis(_))
    ));
    assert!(matches!(
        eredu_runtime::project_additional_parameter_ranges(&layout, &ranges(1, 0..4), None),
        Err(eredu_runtime::ParameterSelectionLayoutError::OverlappingAxis(_))
    ));
    assert!(matches!(
        eredu_runtime::project_additional_parameter_ranges(&layout, &ranges(2, 1..3), None),
        Err(eredu_runtime::ParameterSelectionLayoutError::InvalidRange(
            _
        ))
    ));
    let composed =
        eredu_runtime::project_additional_parameter_ranges(&layout, &ranges(2, 1..2), None)
            .unwrap();
    let composed_bindings = bind(&source, &task, vec![2, 4, 1], Some(&composed)).unwrap();
    assert_eq!(
        composed_bindings[0]
            .source_recipe()
            .infer(&source as &dyn CheckpointSource)
            .unwrap()
            .shape(),
        [2, 4, 1]
    );
    assert_eq!(composed_bindings[0].expected_bytes(), 32);
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
    use eredu_runtime::{
        derive_partitioned_transform_source_layout, prepare_transform_source_layout,
        ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
        MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
        ParameterMemberSpec,
    };
    let description = |units, width| {
        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let units = ExecutionUnitLayout::new(&graph, [units]).unwrap();
        ArchitectureParameterDescription::from_owned(
            &graph,
            &units,
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(units.group_id(0).unwrap().clone(), 0),
                ParameterGroupSpec::new(
                    "head",
                    ParameterRole::ColumnProjection,
                    [ParameterMemberSpec::new(
                        "weight",
                        vec![4, width],
                        MemberSharding::Balanced { axis: 0 },
                    )],
                )
                .unwrap(),
            )],
        )
        .unwrap()
    };
    let rank = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        1,
    )
    .unwrap();
    let source_parameters = description(1, 32);
    let target_parameters = description(1, 4);
    let layout = prepare_transform_source_layout([task.lowering()], || {
        derive_partitioned_transform_source_layout(&source_parameters, &target_parameters, rank)
    })
    .unwrap()
    .unwrap();
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
