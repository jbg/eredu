//! Exact architecture-owned acquisition partitions for prepared assistants.

use std::collections::BTreeSet;

use eredu_runtime::{ReplicatedTextMaterializationTask, ReplicatedTextParameterOwner};

/// One independently acquired assistant parameter unit. Task indices address the
/// retained exact materialization tasks; output names include packed companions.
/// This partition describes immutable storage, not invocation or mutable state.
#[derive(Debug, Clone)]
pub struct ExternalAssistantParameterUnit {
    owner: ReplicatedTextParameterOwner,
    task_indices: Vec<usize>,
    outputs: BTreeSet<String>,
    parameter_bytes: u64,
}

impl ExternalAssistantParameterUnit {
    /// Architecture-declared physical owner, independent of native placement.
    pub const fn owner(&self) -> &ReplicatedTextParameterOwner {
        &self.owner
    }

    /// Indices into the retained exact task sequence, in its original order.
    pub fn task_indices(&self) -> &[usize] {
        &self.task_indices
    }

    /// Complete canonical destination identities, including scales and biases.
    pub fn outputs(&self) -> &BTreeSet<String> {
        &self.outputs
    }

    /// Exact selected parameter payload, including quantization companions.
    /// This excludes loading scratch, activations, state and source caches.
    pub const fn parameter_bytes(&self) -> u64 {
        self.parameter_bytes
    }
}

pub(super) fn parameter_units(
    tasks: &[ReplicatedTextMaterializationTask],
) -> Result<Vec<ExternalAssistantParameterUnit>, String> {
    let mut units = Vec::<ExternalAssistantParameterUnit>::new();
    let mut declared = BTreeSet::new();
    for (index, task) in tasks.iter().enumerate() {
        let owner = task.owner();
        let companion_owner = owner.parameter_group_owner().map_err(|e| e.to_string())?;
        let unit_index = units
            .iter()
            .position(|unit| unit.owner == *owner)
            .unwrap_or_else(|| {
                units.push(ExternalAssistantParameterUnit {
                    owner: owner.clone(),
                    task_indices: Vec::new(),
                    outputs: BTreeSet::new(),
                    parameter_bytes: 0,
                });
                units.len() - 1
            });
        let unit = &mut units[unit_index];
        unit.task_indices.push(index);
        let mut bytes =
            eredu_runtime::selected_materialization_task_bytes(task).map_err(|e| e.to_string())?;
        if !matches!(
            task.lowering(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        ) {
            for companion in task.output_companions() {
                let companion_bytes = companion
                    .derived_output()
                    .map(|m| m.byte_len())
                    .or_else(|| companion.catalog_source().map(|s| s.encoded_byte_len()))
                    .ok_or_else(|| {
                        format!(
                            "assistant companion {:?} has no exact output extent",
                            companion.name()
                        )
                    })?;
                bytes = bytes
                    .checked_add(companion_bytes)
                    .ok_or("assistant parameter bytes overflow")?;
            }
        }
        unit.parameter_bytes = unit
            .parameter_bytes
            .checked_add(bytes)
            .ok_or("assistant unit bytes overflow")?;
        let mut insert = |name: &str| {
            if !declared.insert(name.to_owned()) {
                return Err(format!(
                    "assistant output {name:?} has multiple acquisition owners"
                ));
            }
            unit.outputs.insert(name.to_owned());
            Ok(())
        };
        insert(task.name())?;
        for companion in task.output_companions() {
            if companion.owner() != &companion_owner {
                return Err(format!(
                    "assistant companion {:?} differs from its primary acquisition owner",
                    companion.name()
                ));
            }
            insert(companion.name())?;
        }
    }
    if units.is_empty() {
        return Err("assistant has no parameter acquisition units".into());
    }
    Ok(units)
}

pub(super) fn canonical_unit_order(
    units: Vec<ExternalAssistantParameterUnit>,
    layout: &eredu_runtime::ExecutionUnitLayout,
) -> Result<Vec<ExternalAssistantParameterUnit>, String> {
    let mut indexed = std::collections::BTreeMap::new();
    for source in units {
        let ReplicatedTextParameterOwner::ExecutionUnit { group, unit } = source.owner() else {
            return Err("unit-policy assistant has an unplaced static owner".into());
        };
        if indexed.insert((group.clone(), *unit), source).is_some() {
            return Err("assistant repeats an execution owner".into());
        }
    }
    let result = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("canonical assistant layout");
            let group = layout
                .group_id(address.group())
                .expect("canonical assistant group")
                .as_str()
                .to_owned();
            indexed.remove(&(group, address.index())).ok_or_else(|| {
                "assistant execution owner is absent from prepared sources".to_owned()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !indexed.is_empty() {
        return Err("assistant sources contain an unused execution owner".into());
    }
    Ok(result)
}

/// Sizes the same bounded recipes consumed by the exact native task binder.
/// Source tasks and provenance remain global; placement is a retained input.
pub(super) fn local_parameter_units(
    tasks: &[ReplicatedTextMaterializationTask],
    layout: &eredu_runtime::LocalModelLayout,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExternalAssistantParameterUnit>, String> {
    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeMetadata},
        store::TensorSelection,
    };
    fn metadata(
        name: &str,
        recipe: DerivedWeightRecipe,
        layout: &eredu_runtime::LocalModelLayout,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<RecipeMetadata, String> {
        let bytes = recipe.infer(source).map_err(|e| e.to_string())?.byte_len();
        let binding = eredu_runtime::WeightBinding::from_recipe(name, recipe, bytes)
            .map_err(|e| e.to_string())?
            .with_logical_target(name)
            .map_err(|e| e.to_string())?;
        let mut placed = eredu_runtime::place_weight_bindings(vec![binding], source, layout)
            .map_err(|e| e.to_string())?;
        placed
            .pop()
            .expect("one binding placement preserves cardinality")
            .source_recipe()
            .infer(source)
            .map_err(|e| e.to_string())
    }
    let mut units = parameter_units(tasks)?;
    let names = units
        .iter()
        .flat_map(|unit| unit.outputs.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let selected = layout
        .tensors()
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>();
    if names != selected {
        return Err("assistant placement and admitted task outputs differ".into());
    }
    for unit in &mut units {
        unit.parameter_bytes = 0;
        for index in &unit.task_indices {
            let task = &tasks[*index];
            let primary = metadata(
                task.name(),
                task.source_recipe().map_err(|e| e.to_string())?,
                layout,
                source,
            )?;
            let mut bytes = eredu_runtime::selected_addressable_parameter_bytes(task, &primary)
                .map_err(|e| e.to_string())?;
            if !matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
                    | eredu_runtime::WeightLoweringKind::DerivedTransform
            ) {
                for companion in task.output_companions() {
                    let recipe = if let Some(task) = companion.materialization_task() {
                        task.source_recipe().map_err(|e| e.to_string())?
                    } else if let Some(recipe) = companion.derived_recipe() {
                        recipe.clone()
                    } else if let Some(source) = companion.catalog_source() {
                        DerivedWeightRecipe::source(source.catalog_key(), TensorSelection::Full)
                    } else {
                        return Err(format!(
                            "assistant companion {:?} has no admitted source recipe",
                            companion.name()
                        ));
                    };
                    bytes = bytes
                        .checked_add(metadata(companion.name(), recipe, layout, source)?.byte_len())
                        .ok_or("assistant local companion bytes overflow")?;
                }
            }
            unit.parameter_bytes = unit
                .parameter_bytes
                .checked_add(bytes)
                .ok_or("assistant local unit bytes overflow")?;
        }
    }
    Ok(units)
}
