//! Exact parameter acquisition partitions over retained materialization tasks.

use std::collections::BTreeSet;

use crate::{
    AddressableBankMemberError, BindingPlacementError, ReplicatedTextContractError,
    ReplicatedTextMaterializationTask, ReplicatedTextParameterOwner, ResidencyDeclarationError,
};
use eredu_checkpoint::recipe::RecipeError;

/// Invalid resource ownership, selected geometry or parameter accounting.
#[derive(Debug, thiserror::Error)]
pub enum ParameterAcquisitionError {
    /// An exact materialization task is inconsistent.
    #[error(transparent)]
    Contract(#[from] ReplicatedTextContractError),
    /// A source or placement recipe cannot be inferred.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A binding declaration is invalid.
    #[error(transparent)]
    Binding(#[from] ResidencyDeclarationError),
    /// Exact rank-local binding placement failed.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// Selected executable geometry is invalid.
    #[error(transparent)]
    Storage(#[from] AddressableBankMemberError),
    /// No independently acquired resource was declared.
    #[error("no parameter acquisition units")]
    Empty,
    /// A primary or companion output has inconsistent ownership or source facts.
    #[error("parameter {target:?}: {reason}")]
    Output {
        /// Canonical parameter identity.
        target: String,
        /// Exact resource contract failure.
        reason: &'static str,
    },
    /// Parameter or resource payload accounting overflowed.
    #[error("parameter {target:?} owned by {owner:?}: {reason}")]
    Accounting {
        /// Canonical parameter at the accounting boundary.
        target: String,
        /// Independent acquisition owner.
        owner: ReplicatedTextParameterOwner,
        /// Exact accounting failure.
        reason: &'static str,
    },
    /// The requested invocation order disagrees with retained resource ownership.
    #[error("execution owner {owner:?}: {reason}")]
    ExecutionOwnership {
        /// Missing, repeated or incompatible resource owner.
        owner: ReplicatedTextParameterOwner,
        /// Exact ownership failure.
        reason: &'static str,
    },
    /// Materialized bindings disagree with the selected resource footprint.
    #[error("parameter owner {owner:?} binds {actual} bytes; selected {expected}")]
    Extent {
        /// Independent acquisition owner.
        owner: ReplicatedTextParameterOwner,
        /// Retained cold parameter payload.
        expected: u64,
        /// Exact payload in the destination bindings.
        actual: u64,
    },
    /// Selected physical tensors differ from the admitted task outputs.
    #[error("parameter placement omits {missing:?} and adds {unexpected:?}")]
    Selection {
        /// Declared outputs absent from the physical selection.
        missing: Vec<String>,
        /// Physical tensors outside the declared outputs.
        unexpected: Vec<String>,
    },
}

impl From<ParameterAcquisitionError> for eredu_core::artifact::ArtifactError {
    fn from(error: ParameterAcquisitionError) -> Self {
        Self::Preparation {
            source: Box::new(error),
        }
    }
}

/// One independently acquired parameter unit. Task indices address the
/// retained exact materialization tasks; output names include packed companions.
/// This partition describes immutable storage, not invocation or mutable state.
#[derive(Debug, Clone)]
pub struct ParameterAcquisitionUnit {
    owner: ReplicatedTextParameterOwner,
    task_indices: Vec<usize>,
    outputs: BTreeSet<String>,
    parameter_bytes: u64,
}

impl ParameterAcquisitionUnit {
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

/// Groups tasks by their declared independent resource owner, retaining first-owner
/// order, canonical outputs and exact executable bytes. No payload is acquired.
pub fn parameter_acquisition_units(
    tasks: &[ReplicatedTextMaterializationTask],
) -> Result<Vec<ParameterAcquisitionUnit>, ParameterAcquisitionError> {
    let mut units = Vec::<ParameterAcquisitionUnit>::new();
    let mut declared = BTreeSet::new();
    for (index, task) in tasks.iter().enumerate() {
        let owner = task.owner();
        let companion_owner = owner.parameter_group_owner()?;
        let unit_index = units
            .iter()
            .position(|unit| unit.owner == *owner)
            .unwrap_or_else(|| {
                units.push(ParameterAcquisitionUnit {
                    owner: owner.clone(),
                    task_indices: Vec::new(),
                    outputs: BTreeSet::new(),
                    parameter_bytes: 0,
                });
                units.len() - 1
            });
        let unit = &mut units[unit_index];
        unit.task_indices.push(index);
        let mut bytes = crate::selected_materialization_task_bytes(task)?;
        if !matches!(
            task.lowering(),
            crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
        ) {
            for companion in task.output_companions() {
                let companion_bytes = companion
                    .derived_output()
                    .map(|m| m.byte_len())
                    .or_else(|| companion.catalog_source().map(|s| s.encoded_byte_len()))
                    .ok_or_else(|| ParameterAcquisitionError::Output {
                        target: companion.name().into(),
                        reason: "companion has no exact output extent",
                    })?;
                bytes = bytes.checked_add(companion_bytes).ok_or_else(|| {
                    ParameterAcquisitionError::Accounting {
                        target: task.name().into(),
                        owner: owner.clone(),
                        reason: "parameter bytes overflow",
                    }
                })?;
            }
        }
        unit.parameter_bytes = unit.parameter_bytes.checked_add(bytes).ok_or_else(|| {
            ParameterAcquisitionError::Accounting {
                target: task.name().into(),
                owner: owner.clone(),
                reason: "unit bytes overflow",
            }
        })?;
        let mut insert = |name: &str| {
            if !declared.insert(name.to_owned()) {
                return Err(ParameterAcquisitionError::Output {
                    target: name.into(),
                    reason: "multiple acquisition owners",
                });
            }
            unit.outputs.insert(name.to_owned());
            Ok(())
        };
        insert(task.name())?;
        for companion in task.output_companions() {
            if companion.owner() != &companion_owner {
                return Err(ParameterAcquisitionError::Output {
                    target: companion.name().into(),
                    reason: "companion differs from primary acquisition owner",
                });
            }
            insert(companion.name())?;
        }
    }
    if units.is_empty() {
        return Err(ParameterAcquisitionError::Empty);
    }
    Ok(units)
}

/// Orders execution-owned resources against a retained canonical invocation layout.
/// Static, missing, repeated and unselected owners are rejected.
pub fn order_parameter_acquisition_units(
    units: Vec<ParameterAcquisitionUnit>,
    layout: &crate::ExecutionUnitLayout,
) -> Result<Vec<ParameterAcquisitionUnit>, ParameterAcquisitionError> {
    let mut indexed = std::collections::BTreeMap::new();
    for source in units {
        let ReplicatedTextParameterOwner::ExecutionUnit { group, unit } = source.owner() else {
            return Err(ParameterAcquisitionError::ExecutionOwnership {
                owner: source.owner().clone(),
                reason: "unplaced static owner",
            });
        };
        let key = (group.clone(), *unit);
        if let Some(previous) = indexed.insert(key, source) {
            return Err(ParameterAcquisitionError::ExecutionOwnership {
                owner: previous.owner,
                reason: "repeated execution owner",
            });
        }
    }
    let result = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("canonical execution layout");
            let group = layout
                .group_id(address.group())
                .expect("canonical execution group")
                .as_str()
                .to_owned();
            indexed
                .remove(&(group.clone(), address.index()))
                .ok_or_else(|| ParameterAcquisitionError::ExecutionOwnership {
                    owner: ReplicatedTextParameterOwner::ExecutionUnit {
                        group,
                        unit: address.index(),
                    },
                    reason: "owner absent from prepared sources",
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(unused) = indexed.into_values().next() {
        return Err(ParameterAcquisitionError::ExecutionOwnership {
            owner: unused.owner,
            reason: "unused execution owner",
        });
    }
    Ok(result)
}

/// Sizes the same bounded recipes consumed by the exact native task binder.
/// Source tasks and provenance remain global; placement is a retained input.
pub fn local_parameter_acquisition_units(
    tasks: &[ReplicatedTextMaterializationTask],
    layout: &crate::LocalModelLayout,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterAcquisitionUnit>, ParameterAcquisitionError> {
    use eredu_checkpoint::{
        recipe::{DerivedWeightRecipe, RecipeMetadata},
        store::TensorSelection,
    };
    fn metadata(
        name: &str,
        recipe: DerivedWeightRecipe,
        layout: &crate::LocalModelLayout,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<RecipeMetadata, ParameterAcquisitionError> {
        let bytes = recipe.infer(source)?.byte_len();
        let binding =
            crate::WeightBinding::from_recipe(name, recipe, bytes)?.with_logical_target(name)?;
        let mut placed = crate::place_weight_bindings(vec![binding], source, layout)?;
        placed
            .pop()
            .expect("one binding placement preserves cardinality")
            .source_recipe()
            .infer(source)
            .map_err(ParameterAcquisitionError::from)
    }
    let mut units = parameter_acquisition_units(tasks)?;
    let names = units
        .iter()
        .flat_map(|unit| unit.outputs.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let selected = layout
        .tensors()
        .map(|(name, _)| name)
        .collect::<BTreeSet<_>>();
    if names != selected {
        return Err(ParameterAcquisitionError::Selection {
            missing: names
                .difference(&selected)
                .map(|name| (*name).to_owned())
                .collect(),
            unexpected: selected
                .difference(&names)
                .map(|name| (*name).to_owned())
                .collect(),
        });
    }
    for unit in &mut units {
        unit.parameter_bytes = 0;
        for index in &unit.task_indices {
            let task = &tasks[*index];
            let primary = metadata(task.name(), task.source_recipe()?, layout, source)?;
            let mut bytes = crate::selected_addressable_parameter_bytes(task, &primary)?;
            if !matches!(
                task.lowering(),
                crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
            ) {
                for companion in task.output_companions() {
                    let recipe = if let Some(task) = companion.materialization_task() {
                        task.source_recipe()?
                    } else if let Some(recipe) = companion.derived_recipe() {
                        recipe.clone()
                    } else if let Some(source) = companion.catalog_source() {
                        DerivedWeightRecipe::source(source.catalog_key(), TensorSelection::Full)
                    } else {
                        return Err(ParameterAcquisitionError::Output {
                            target: companion.name().into(),
                            reason: "companion has no admitted source recipe",
                        });
                    };
                    bytes = bytes
                        .checked_add(metadata(companion.name(), recipe, layout, source)?.byte_len())
                        .ok_or_else(|| ParameterAcquisitionError::Accounting {
                            target: companion.name().into(),
                            owner: task.owner().clone(),
                            reason: "local companion bytes overflow",
                        })?;
                }
            }
            unit.parameter_bytes = unit.parameter_bytes.checked_add(bytes).ok_or_else(|| {
                ParameterAcquisitionError::Accounting {
                    target: task.name().into(),
                    owner: task.owner().clone(),
                    reason: "local unit bytes overflow",
                }
            })?;
        }
    }
    Ok(units)
}

/// Partitions exact destination bindings by retained canonical resource ownership.
///
/// Binding names may be local module paths; logical targets select the canonical
/// outputs. Every output occurs exactly once, and each resource's non-alias bytes
/// must equal its selected footprint. Input order is preserved within each unit.
/// No source payload is read and independent resources remain separately leased.
pub fn partition_parameter_acquisition_bindings(
    units: &[ParameterAcquisitionUnit],
    bindings: Vec<crate::WeightBinding>,
) -> Result<Vec<Vec<crate::WeightBinding>>, ParameterAcquisitionError> {
    let mut owners = std::collections::BTreeMap::new();
    for (index, unit) in units.iter().enumerate() {
        for target in unit.outputs() {
            if owners.insert(target.as_str(), index).is_some() {
                return Err(ParameterAcquisitionError::Output {
                    target: target.clone(),
                    reason: "multiple acquisition owners",
                });
            }
        }
    }
    let mut partitioned = vec![Vec::new(); units.len()];
    let mut assigned = BTreeSet::new();
    for binding in bindings {
        let target = binding.logical_target().unwrap_or_else(|| binding.name());
        let index = owners
            .get(target)
            .ok_or_else(|| ParameterAcquisitionError::Output {
                target: target.into(),
                reason: "binding has no acquisition owner",
            })?;
        if !assigned.insert(target.to_owned()) {
            return Err(ParameterAcquisitionError::Output {
                target: target.into(),
                reason: "binding has duplicate destinations",
            });
        }
        partitioned[*index].push(binding);
    }
    let missing = owners
        .keys()
        .filter(|target| !assigned.contains(**target))
        .map(|target| (*target).to_owned())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ParameterAcquisitionError::Selection {
            missing,
            unexpected: Vec::new(),
        });
    }
    for (unit, bindings) in units.iter().zip(&partitioned) {
        let actual = bindings
            .iter()
            .filter(|binding| !binding.is_alias())
            .try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    ParameterAcquisitionError::Accounting {
                        target: binding
                            .logical_target()
                            .unwrap_or_else(|| binding.name())
                            .into(),
                        owner: unit.owner().clone(),
                        reason: "binding bytes overflow",
                    }
                })
            })?;
        if actual != unit.parameter_bytes() {
            return Err(ParameterAcquisitionError::Extent {
                owner: unit.owner().clone(),
                expected: unit.parameter_bytes(),
                actual,
            });
        }
    }
    Ok(partitioned)
}
