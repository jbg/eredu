//! Backend-neutral checkpoint materialization and parameter binding.

use std::{collections::BTreeMap, marker::PhantomData};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeCatalog, RecipeDtype, RecipeError},
    store::{
        CheckpointSource, ReadPolicy, SharedCheckpointSource, StoreError, TensorReadRequest,
        TensorSelection,
    },
};
use eredu_nn::{
    ParameterId, ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized,
};

use crate::{
    ParameterBackend, ReplicatedTextMaterializationTask, ResidencyDeclarationError, WeightBinding,
    WeightBindingPlan, WeightLoweringKind,
};

/// One logical parameter target and the exact recipe that produces it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlannedBinding {
    /// Stable parameter identity populated by this binding.
    pub target_name: String,
    /// Exact logical shape required by the destination.
    pub expected_shape: Vec<usize>,
    /// Logical dtype required by the destination.
    pub expected_dtype: RecipeDtype,
    /// Declarative transformation from checkpoint sources to the target.
    pub recipe: DerivedWeightRecipe,
}

impl PlannedBinding {
    /// Creates a direct full-source declaration.
    pub fn direct(
        target_name: impl Into<String>,
        source_key: impl Into<String>,
        expected_shape: impl Into<Vec<usize>>,
        expected_dtype: RecipeDtype,
    ) -> Self {
        Self {
            target_name: target_name.into(),
            expected_shape: expected_shape.into(),
            expected_dtype,
            recipe: DerivedWeightRecipe::source(source_key, TensorSelection::Full),
        }
    }
}

/// Canonical deterministic declaration for one atomic parameter-binding unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BindingPlan {
    bindings: Vec<PlannedBinding>,
    shared_source_keys: std::collections::BTreeSet<String>,
    permitted_dtype_conversions: BTreeMap<String, Vec<RecipeDtype>>,
}

impl BindingPlan {
    /// Builds a plan in which every physical source claim is exclusive.
    pub fn new(bindings: Vec<PlannedBinding>) -> Result<Self, BindingPlanError> {
        Self::with_explicit_exceptions(bindings, std::collections::BTreeSet::new(), BTreeMap::new())
    }

    /// Builds a plan with an explicit allowlist for intentionally shared sources.
    pub fn allowing_shared_sources(
        bindings: Vec<PlannedBinding>,
        shared_source_keys: std::collections::BTreeSet<String>,
    ) -> Result<Self, BindingPlanError> {
        Self::with_explicit_exceptions(bindings, shared_source_keys, BTreeMap::new())
    }

    /// Builds a plan with explicit shared-source and native dtype-conversion declarations.
    pub fn with_explicit_exceptions(
        mut bindings: Vec<PlannedBinding>,
        shared_source_keys: std::collections::BTreeSet<String>,
        permitted_dtype_conversions: BTreeMap<String, Vec<RecipeDtype>>,
    ) -> Result<Self, BindingPlanError> {
        bindings.sort_by(|left, right| left.target_name.cmp(&right.target_name));
        let mut targets = std::collections::BTreeSet::new();
        let mut claims = BTreeMap::<String, String>::new();
        for binding in &bindings {
            if binding.target_name.trim().is_empty() {
                return Err(BindingPlanError::EmptyTarget);
            }
            if binding.expected_shape.contains(&0) {
                return Err(BindingPlanError::InvalidTargetShape {
                    target: binding.target_name.clone(),
                    shape: binding.expected_shape.clone(),
                });
            }
            binding
                .expected_shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| BindingPlanError::TargetShapeOverflow {
                    target: binding.target_name.clone(),
                })?;
            if !targets.insert(binding.target_name.clone()) {
                return Err(BindingPlanError::DuplicateTarget {
                    target: binding.target_name.clone(),
                });
            }
            let source_keys = binding.recipe.source_keys();
            if source_keys.is_empty() {
                return Err(BindingPlanError::EmptyRecipeSources {
                    target: binding.target_name.clone(),
                });
            }
            for source in source_keys {
                if source.trim().is_empty() {
                    return Err(BindingPlanError::InvalidSourceKey {
                        target: binding.target_name.clone(),
                    });
                }
                if shared_source_keys.contains(source) {
                    continue;
                }
                if let Some(first) = claims.insert(source.into(), binding.target_name.clone()) {
                    return Err(BindingPlanError::DuplicateSourceClaim {
                        source_key: source.into(),
                        first,
                        second: binding.target_name.clone(),
                    });
                }
            }
        }
        Ok(Self {
            bindings,
            shared_source_keys,
            permitted_dtype_conversions,
        })
    }

    /// Returns declarations in deterministic target order.
    pub fn bindings(&self) -> &[PlannedBinding] {
        &self.bindings
    }

    /// Returns distinct physical source keys in lexical order.
    pub fn source_keys(&self) -> Vec<&str> {
        let mut keys = std::collections::BTreeSet::new();
        for binding in &self.bindings {
            keys.extend(binding.recipe.source_keys());
        }
        keys.into_iter().collect()
    }

    /// Returns sources explicitly declared as shared by multiple targets.
    pub fn shared_source_keys(&self) -> &std::collections::BTreeSet<String> {
        &self.shared_source_keys
    }

    /// Infers every recipe and lowers this declaration to residency bindings.
    ///
    /// An authoritative materialized overlay wins over its semantic recipe.
    pub fn build_bindings(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, BindingPlanError> {
        let mut output = Vec::with_capacity(self.bindings.len());
        for planned in &self.bindings {
            let recipe = if source.is_authoritative_materialized_key(&planned.target_name) {
                DerivedWeightRecipe::source(&planned.target_name, TensorSelection::Full)
            } else {
                planned.recipe.clone()
            };
            let metadata = recipe.infer(source)?;
            if metadata.shape() != planned.expected_shape {
                return Err(BindingPlanError::ShapeMismatch {
                    target: planned.target_name.clone(),
                    expected: planned.expected_shape.clone(),
                    actual: metadata.shape().to_vec(),
                });
            }
            let explicitly_permitted = self
                .permitted_dtype_conversions
                .get(&planned.target_name)
                .is_some_and(|dtypes| dtypes.contains(metadata.dtype()));
            if planned.expected_dtype != *metadata.dtype() && !explicitly_permitted {
                return Err(BindingPlanError::DtypeMismatch {
                    target: planned.target_name.clone(),
                    expected: planned.expected_dtype.clone(),
                    actual: metadata.dtype().clone(),
                });
            }
            let binding = match recipe {
                DerivedWeightRecipe::Source { key, selection } => {
                    WeightBinding::new(&planned.target_name, key, selection, metadata.byte_len())?
                }
                recipe => {
                    WeightBinding::from_recipe(&planned.target_name, recipe, metadata.byte_len())?
                }
            };
            output.push(binding.with_logical_target(&planned.target_name)?);
        }
        Ok(output)
    }
}

/// Invalid canonical binding declarations or inferred outputs.
#[derive(Debug, thiserror::Error)]
pub enum BindingPlanError {
    /// A declaration has no stable destination identity.
    #[error("binding target must not be empty")]
    EmptyTarget,
    /// A destination contains a zero-sized dimension.
    #[error("binding target {target:?} has invalid shape {shape:?}")]
    InvalidTargetShape {
        /// Invalid target identity.
        target: String,
        /// Invalid logical shape.
        shape: Vec<usize>,
    },
    /// Destination element-count computation overflowed.
    #[error("binding target {target:?} shape element count overflows")]
    TargetShapeOverflow {
        /// Target whose element count overflowed.
        target: String,
    },
    /// Two declarations claim the same destination.
    #[error("binding plan contains duplicate target {target:?}")]
    DuplicateTarget {
        /// Repeated target identity.
        target: String,
    },
    /// A recipe contains no physical source.
    #[error("binding target {target:?} has a recipe with no checkpoint sources")]
    EmptyRecipeSources {
        /// Target with no physical source.
        target: String,
    },
    /// A recipe contains an empty physical source identity.
    #[error("binding target {target:?} has an empty checkpoint source key")]
    InvalidSourceKey {
        /// Target containing the invalid source.
        target: String,
    },
    /// Two destinations claimed an undeclared shared physical source.
    #[error("checkpoint source {source_key:?} is claimed by both {first:?} and {second:?}")]
    DuplicateSourceClaim {
        /// Physical source claimed twice.
        source_key: String,
        /// First target claiming the source.
        first: String,
        /// Second target claiming the source.
        second: String,
    },
    /// Inferred and declared shapes differ.
    #[error("binding target {target:?} expects shape {expected:?}, recipe produces {actual:?}")]
    ShapeMismatch {
        /// Stable target identity.
        target: String,
        /// Declared logical shape.
        expected: Vec<usize>,
        /// Inferred recipe shape.
        actual: Vec<usize>,
    },
    /// Inferred and declared dtypes differ.
    #[error("binding target {target:?} expects dtype {expected:?}, recipe produces {actual:?}")]
    DtypeMismatch {
        /// Stable target identity.
        target: String,
        /// Declared logical dtype.
        expected: RecipeDtype,
        /// Inferred recipe dtype.
        actual: RecipeDtype,
    },
    /// Recipe inference failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Lowered residency declaration is invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
}

/// Backend-described geometry of one statically traversed parameter slot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterBindingTarget {
    /// Logical shape expected by the destination.
    pub shape: Vec<usize>,
    /// Logical dtype expected by the destination.
    pub dtype: RecipeDtype,
    /// Source dtypes the backend explicitly converts while materializing this slot.
    pub permitted_source_dtypes: Vec<RecipeDtype>,
}

/// A canonical binding plan paired with fully qualified logical destinations.
#[derive(Debug, Clone)]
pub struct ModuleBindingPlan {
    plan: BindingPlan,
    logical_targets: BTreeMap<String, String>,
}

impl ModuleBindingPlan {
    /// Returns the canonical declarative plan.
    pub const fn plan(&self) -> &BindingPlan {
        &self.plan
    }

    /// Resolves this declaration against one exact checkpoint source.
    pub fn build_bindings(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<Vec<WeightBinding>, ModuleBindingPlanError> {
        self.plan
            .build_bindings(source)?
            .into_iter()
            .map(|binding| {
                let target = self
                    .logical_targets
                    .get(binding.name())
                    .expect("module binding declaration retains every target");
                binding
                    .with_logical_target(target)
                    .map_err(ModuleBindingPlanError::from)
            })
            .collect()
    }
}

/// Builds the singular declarative plan by traversing any neutral parameterized module.
///
/// `describe` is the backend's only type-binding hook: it exposes native slot
/// geometry but cannot select sources, inspect payloads, or publish values.
pub fn build_module_binding_plan<P, M, F, D>(
    module: &M,
    prefix: &str,
    source: &dyn CheckpointSource,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
    shared_source_keys: std::collections::BTreeSet<String>,
    excluded: F,
    describe: D,
) -> Result<ModuleBindingPlan, ModuleBindingPlanError>
where
    P: 'static,
    M: Parameterized<P>,
    F: Fn(&str) -> bool,
    D: Fn(&P) -> Option<ParameterBindingTarget>,
{
    struct Collector<'a, P> {
        parameters: BTreeMap<String, &'a P>,
        duplicate: Option<String>,
    }

    impl<'a, P: 'a> ParameterVisitor<'a, P> for Collector<'a, P> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'a P) {
            let name = metadata.id.as_str().to_owned();
            if self.parameters.insert(name.clone(), parameter).is_some() {
                self.duplicate = Some(name);
            }
        }
    }

    let mut collector = Collector {
        parameters: BTreeMap::new(),
        duplicate: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(parameter) = collector.duplicate {
        return Err(ModuleBindingPlanError::DuplicateParameter { parameter });
    }

    recipes.retain(|name, _| !excluded(name));
    let source_keys = source
        .source_keys()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let mut declarations = Vec::new();
    let mut logical_targets = BTreeMap::new();
    let mut permitted_dtype_conversions = BTreeMap::new();

    for (local_name, parameter) in collector
        .parameters
        .into_iter()
        .filter(|(name, _)| !excluded(name))
    {
        let destination = qualify_parameter(prefix, &local_name);
        logical_targets.insert(local_name.clone(), destination.clone());
        let description = describe(parameter).ok_or_else(|| {
            ModuleBindingPlanError::InvalidParameterRepresentation {
                parameter: destination.clone(),
            }
        })?;

        let recipe = if source.is_authoritative_materialized_key(&destination) {
            recipes.remove(&local_name);
            DerivedWeightRecipe::source(destination.clone(), TensorSelection::Full)
        } else if let Some(recipe) = recipes.remove(&local_name) {
            recipe
        } else if source_keys.contains(&destination) {
            DerivedWeightRecipe::source(destination.clone(), TensorSelection::Full)
        } else {
            return Err(ModuleBindingPlanError::MissingParameter { destination });
        };
        if !description.permitted_source_dtypes.is_empty() {
            permitted_dtype_conversions
                .insert(local_name.clone(), description.permitted_source_dtypes);
        }
        declarations.push(PlannedBinding {
            target_name: local_name,
            expected_shape: description.shape,
            expected_dtype: description.dtype,
            recipe,
        });
    }

    if !recipes.is_empty() {
        return Err(ModuleBindingPlanError::UnknownRecipeParameters {
            parameters: recipes.into_keys().collect(),
        });
    }
    Ok(ModuleBindingPlan {
        plan: BindingPlan::with_explicit_exceptions(
            declarations,
            shared_source_keys,
            permitted_dtype_conversions,
        )?,
        logical_targets,
    })
}

/// Builds canonical bindings for an exact architecture-selected task partition.
///
/// Module traversal supplies destination geometry only. Task identities (including
/// admitted aliases), physical provenance, recipes, companions, and coverage are
/// validated here before any backend payload or native operation. `lower_mxfp4`
/// is the sole backend hook and may only replace an admitted derived F4 recipe for
/// an MXFP4 executable.
pub fn build_exact_replicated_text_bindings<P, M, D, L, E>(
    module: &M,
    source: &dyn CheckpointSource,
    tasks: &[&ReplicatedTextMaterializationTask],
    addressable_parameters: &std::collections::BTreeSet<String>,
    describe: D,
    mut lower_mxfp4: L,
) -> Result<Vec<WeightBinding>, ModuleBindingPlanError>
where
    P: 'static,
    M: Parameterized<P>,
    D: Fn(&P) -> Option<ParameterBindingTarget>,
    L: FnMut(
        &ReplicatedTextMaterializationTask,
        DerivedWeightRecipe,
        &dyn CheckpointSource,
    ) -> Result<DerivedWeightRecipe, E>,
    E: std::fmt::Display,
{
    struct Collector<'a, P> {
        parameters: BTreeMap<String, (&'a P, ParameterBindingTarget)>,
        duplicate: Option<String>,
    }
    impl<'a, P: 'a> ParameterVisitor<'a, P> for Collector<'a, P> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'a P) {
            let name = metadata.id.as_str().to_owned();
            // The description is filled after traversal because the visitor cannot
            // retain a borrowed callback without complicating this neutral API.
            if self
                .parameters
                .insert(
                    name.clone(),
                    (
                        parameter,
                        ParameterBindingTarget {
                            shape: Vec::new(),
                            dtype: RecipeDtype::Other(String::new()),
                            permitted_source_dtypes: Vec::new(),
                        },
                    ),
                )
                .is_some()
            {
                self.duplicate = Some(name);
            }
        }
    }

    let mut collector = Collector {
        parameters: BTreeMap::new(),
        duplicate: None,
    };
    module.visit_parameters(&mut collector);
    if let Some(parameter) = collector.duplicate {
        return Err(ModuleBindingPlanError::DuplicateParameter { parameter });
    }
    for (name, (parameter, description)) in &mut collector.parameters {
        *description = describe(parameter).ok_or_else(|| {
            ModuleBindingPlanError::InvalidParameterRepresentation {
                parameter: name.clone(),
            }
        })?;
    }

    let parameter_names = collector
        .parameters
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut covered = std::collections::BTreeSet::new();
    let mut declarations = Vec::new();
    let mut conversions = BTreeMap::new();
    let shared_source_keys = tasks
        .iter()
        .flat_map(|task| task.shared_source_keys().iter().cloned())
        .collect();

    for task in tasks {
        let candidates = std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .filter(|candidate| parameter_names.contains(*candidate))
            .collect::<std::collections::BTreeSet<_>>();
        if candidates.len() != 1 {
            return Err(ModuleBindingPlanError::ExactTask {
                details: format!(
                    "task {:?} resolves to {} module destinations through its canonical identity and aliases: {candidates:?}",
                    task.name(),
                    candidates.len()
                ),
            });
        }
        let target = (*candidates.first().expect("one candidate was validated")).to_owned();
        if !covered.insert(target.clone()) {
            return Err(ModuleBindingPlanError::ExactTask {
                details: format!("module destination {target:?} is claimed more than once"),
            });
        }
        validate_exact_task_provenance(task, source)?;
        let recipe = exact_task_recipe(task, source, &mut lower_mxfp4)?;
        push_exact_declaration(
            &collector.parameters,
            &mut declarations,
            &mut conversions,
            target,
            recipe,
            task.permitted_native_source_dtypes(),
        )?;

        for companion in task.output_companions() {
            let target = companion.name().to_owned();
            if !parameter_names.contains(&target) {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!(
                        "task {:?} companion {:?} has no module destination",
                        task.name(),
                        companion.name()
                    ),
                });
            }
            if !covered.insert(target.clone()) {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!("companion destination {target:?} is claimed more than once"),
                });
            }
            let recipe = if let Some(exact) = companion.materialization_task() {
                validate_exact_task_provenance(exact, source)?;
                exact_task_recipe(exact, source, &mut lower_mxfp4)?
            } else if let (Some(recipe), Some(expected)) =
                (companion.derived_recipe(), companion.derived_output())
            {
                let actual = recipe.infer(source)?;
                if &actual != expected {
                    return Err(ModuleBindingPlanError::ExactTask {
                        details: format!(
                            "companion {:?} differs from its admitted derived output",
                            companion.name()
                        ),
                    });
                }
                recipe.clone()
            } else if let Some(physical) = companion.catalog_source() {
                validate_physical_source(physical, source)?;
                DerivedWeightRecipe::source(physical.catalog_key(), TensorSelection::Full)
            } else if matches!(
                task.lowering(),
                WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
            ) || matches!(
                task.source_encoding(),
                eredu_checkpoint::SourceTensorEncoding::Gguf { .. }
            ) {
                if !source.is_authoritative_materialized_key(companion.name()) {
                    return Err(ModuleBindingPlanError::ExactTask {
                        details: format!(
                            "generated companion {:?} has no authoritative materialized output",
                            companion.name()
                        ),
                    });
                }
                DerivedWeightRecipe::source(companion.name(), TensorSelection::Full)
            } else {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!(
                        "companion {:?} has neither an exact task nor causal source",
                        companion.name()
                    ),
                });
            };
            push_exact_declaration(
                &collector.parameters,
                &mut declarations,
                &mut conversions,
                target,
                recipe,
                companion.materialization_task().map_or(
                    &[][..],
                    ReplicatedTextMaterializationTask::permitted_native_source_dtypes,
                ),
            )?;
        }
    }

    let missing = parameter_names
        .difference(&covered)
        .filter(|name| !addressable_parameters.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ModuleBindingPlanError::ExactTask {
            details: format!("module parameters have no exact materialization task: {missing:?}"),
        });
    }

    BindingPlan::with_explicit_exceptions(declarations, shared_source_keys, conversions)?
        .build_bindings(source)
        .map_err(Into::into)
}

fn push_exact_declaration<P>(
    parameters: &BTreeMap<String, (&P, ParameterBindingTarget)>,
    declarations: &mut Vec<PlannedBinding>,
    conversions: &mut BTreeMap<String, Vec<RecipeDtype>>,
    target: String,
    recipe: DerivedWeightRecipe,
    explicitly_permitted_source_dtypes: &[RecipeDtype],
) -> Result<(), ModuleBindingPlanError> {
    let description = &parameters
        .get(&target)
        .expect("validated module destination remains present")
        .1;
    let mut permitted = description.permitted_source_dtypes.clone();
    for dtype in explicitly_permitted_source_dtypes {
        if !permitted.contains(dtype) {
            permitted.push(dtype.clone());
        }
    }
    if !permitted.is_empty() {
        conversions.insert(target.clone(), permitted);
    }
    declarations.push(PlannedBinding {
        target_name: target,
        expected_shape: description.shape.clone(),
        expected_dtype: description.dtype.clone(),
        recipe,
    });
    Ok(())
}

fn validate_exact_task_provenance(
    task: &ReplicatedTextMaterializationTask,
    source: &dyn CheckpointSource,
) -> Result<(), ModuleBindingPlanError> {
    if matches!(
        task.lowering(),
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
    ) {
        return Ok(());
    }
    let declared = task
        .sources()
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let physical = task
        .physical_sources()
        .iter()
        .map(|source| source.catalog_key())
        .collect::<std::collections::BTreeSet<_>>();
    if declared != physical || physical.len() != task.physical_sources().len() {
        return Err(ModuleBindingPlanError::ExactTask {
            details: format!(
                "task {:?} has inconsistent exact physical-source coverage",
                task.name()
            ),
        });
    }
    for physical in task.physical_sources() {
        validate_physical_source(physical, source)?;
    }
    Ok(())
}

fn validate_physical_source(
    physical: &crate::ReplicatedTextPhysicalSource,
    source: &dyn CheckpointSource,
) -> Result<(), ModuleBindingPlanError> {
    let provenance = source.source_provenance(physical.catalog_key())?;
    let metadata = source.source_metadata(physical.catalog_key())?;
    if provenance.catalog_key != physical.catalog_key()
        || provenance.physical_tensor != physical.tensor()
        || provenance.output != physical.output()
        || provenance.backing_shard.as_deref() != Some(physical.shard())
        || provenance.source_encoding != *physical.source_encoding()
        || metadata.encoded_byte_len != physical.encoded_byte_len()
    {
        return Err(ModuleBindingPlanError::ExactTask {
            details: format!(
                "physical source {:?} differs from admitted provenance",
                physical.catalog_key()
            ),
        });
    }
    Ok(())
}

fn exact_task_recipe<L, E>(
    task: &ReplicatedTextMaterializationTask,
    source: &dyn CheckpointSource,
    lower_mxfp4: &mut L,
) -> Result<DerivedWeightRecipe, ModuleBindingPlanError>
where
    L: FnMut(
        &ReplicatedTextMaterializationTask,
        DerivedWeightRecipe,
        &dyn CheckpointSource,
    ) -> Result<DerivedWeightRecipe, E>,
    E: std::fmt::Display,
{
    match task.lowering() {
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform => {
            if !source.is_authoritative_materialized_key(task.name()) {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!(
                        "transformed task {:?} has no authoritative materialized output",
                        task.name()
                    ),
                });
            }
            Ok(DerivedWeightRecipe::source(
                task.name(),
                TensorSelection::Full,
            ))
        }
        WeightLoweringKind::Direct => {
            let [key] = task.sources() else {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!("direct task {:?} must name exactly one source", task.name()),
                });
            };
            Ok(DerivedWeightRecipe::source(key, TensorSelection::Full))
        }
        WeightLoweringKind::Derived => {
            let recipe =
                task.source_recipe()
                    .map_err(|error| ModuleBindingPlanError::ExactTask {
                        details: error.to_string(),
                    })?;
            let admitted = recipe.infer(source)?;
            if task
                .derived_output()
                .is_some_and(|expected| expected != &admitted)
            {
                return Err(ModuleBindingPlanError::ExactTask {
                    details: format!(
                        "derived task {:?} differs from its admitted output",
                        task.name()
                    ),
                });
            }
            if task.executable() == eredu_checkpoint::LinearFormat::MxFp4
                && admitted.dtype() == &RecipeDtype::F4
            {
                lower_mxfp4(task, recipe, source).map_err(|error| {
                    ModuleBindingPlanError::ExactTask {
                        details: format!(
                            "MXFP4 recipe lowering for {:?} failed: {error}",
                            task.name()
                        ),
                    }
                })
            } else {
                Ok(recipe)
            }
        }
    }
}

fn qualify_parameter(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Failure while converting a neutral module traversal into a binding plan.
#[derive(Debug, thiserror::Error)]
pub enum ModuleBindingPlanError {
    /// Immutable traversal repeated a stable parameter identity.
    #[error("module traversal repeats parameter {parameter:?}")]
    DuplicateParameter {
        /// Repeated identity.
        parameter: String,
    },
    /// A native parameter could not expose portable geometry.
    #[error("parameter {parameter:?} has an invalid native representation")]
    InvalidParameterRepresentation {
        /// Invalid destination.
        parameter: String,
    },
    /// No recipe, overlay, or direct checkpoint source exists.
    #[error("checkpoint is missing parameter {destination:?}")]
    MissingParameter {
        /// Fully qualified logical destination.
        destination: String,
    },
    /// Recipe declarations remained after complete module traversal.
    #[error("recipes target unknown or excluded parameters: {parameters:?}")]
    UnknownRecipeParameters {
        /// Unknown local identities.
        parameters: Vec<String>,
    },
    /// An exact architecture-selected task disagreed with its module or source.
    #[error("exact materialization task is invalid: {details}")]
    ExactTask {
        /// Exact causal mismatch.
        details: String,
    },
    /// Checkpoint metadata access failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Recipe metadata inference failed while validating an exact task.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The canonical declaration was invalid.
    #[error(transparent)]
    Plan(#[from] BindingPlanError),
    /// A lowered residency binding was invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
}

/// Consumes one atomic recipe publication into owner bindings and lightweight
/// logical aliases without cloning any derived recipe.
pub fn bindings_from_recipe_set<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    set: AtomicRecipeSet,
) -> Result<Vec<WeightBinding>, RecipeBindingError> {
    let (outputs, aliases) = set.into_parts();
    let mut bytes = BTreeMap::new();
    for (name, recipe) in &outputs {
        bytes.insert(name.clone(), recipe.infer(catalog)?.byte_len());
    }
    let mut bindings = outputs
        .into_iter()
        .map(|(name, recipe)| {
            let expected = bytes[&name];
            WeightBinding::from_recipe(name, recipe, expected)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (alias, owner) in aliases {
        bindings.push(WeightBinding::alias(alias, owner.clone(), bytes[&owner])?);
    }
    WeightBindingPlan::new(&bindings)?;
    Ok(bindings)
}

/// Failure while lowering an atomic neutral recipe set into runtime bindings.
#[derive(Debug, thiserror::Error)]
pub enum RecipeBindingError {
    /// Recipe metadata inference failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A runtime binding declaration was invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
}

/// One fully realized atomic unit keyed by stable parameter identity.
pub struct MaterializedUnit<B: ParameterBackend> {
    weights: BTreeMap<ParameterId, B::MaterializedWeight>,
}

/// An immutable binding unit admitted by one backend against one exact source.
///
/// Construction is intentionally restricted to [`select_bindings`]. Consuming
/// this token materializes the already selected operations without asking the
/// backend to select them again.
pub struct SelectedBindingPlan<B: ParameterBackend> {
    source: SharedCheckpointSource,
    bindings: Vec<WeightBinding>,
    backend: PhantomData<fn() -> B>,
}

impl<B: ParameterBackend> std::fmt::Debug for SelectedBindingPlan<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedBindingPlan")
            .field("bindings", &self.bindings)
            .finish_non_exhaustive()
    }
}

fn validated_binding_plan<'a, B: ParameterBackend>(
    source: &dyn CheckpointSource,
    bindings: &'a [WeightBinding],
) -> Result<WeightBindingPlan<'a>, ParameterOrchestrationError<B::ParameterError>> {
    let plan = WeightBindingPlan::new(bindings)?;
    for binding in plan.owners() {
        let inferred = binding.source_recipe().infer(source)?;
        if inferred.byte_len() != binding.expected_bytes() {
            return Err(ParameterOrchestrationError::ByteMismatch {
                parameter: binding.name().to_owned(),
                expected: binding.expected_bytes(),
                actual: inferred.byte_len(),
            });
        }
    }
    for binding in plan.owners() {
        B::preflight_recipe(&binding.source_recipe(), source)
            .map_err(ParameterOrchestrationError::Backend)?;
    }
    Ok(plan)
}

/// Validates a complete binding unit before any payload read or native work.
pub fn preflight_bindings<B: ParameterBackend>(
    source: &dyn CheckpointSource,
    bindings: &[WeightBinding],
) -> Result<(), ParameterOrchestrationError<B::ParameterError>> {
    validated_binding_plan::<B>(source, bindings).map(|_| ())
}

/// Selects a complete immutable binding unit against one exact source.
///
/// Selection validates every declaration, recipe, byte count, and backend
/// capability without reading payloads, allocating native tensors, or
/// publishing parameters. The returned token owns both the source and tasks so
/// later materialization cannot substitute either or repeat backend selection.
pub fn select_bindings<B: ParameterBackend>(
    source: SharedCheckpointSource,
    bindings: Vec<WeightBinding>,
) -> Result<SelectedBindingPlan<B>, ParameterOrchestrationError<B::ParameterError>> {
    validated_binding_plan::<B>(source.as_ref(), &bindings)?;
    Ok(SelectedBindingPlan {
        source,
        bindings,
        backend: PhantomData,
    })
}

impl<B: ParameterBackend> MaterializedUnit<B> {
    /// Creates a realized unit from already completed backend-native weights.
    ///
    /// This supports bounded backend executors while preserving the singular
    /// neutral traversal and atomic publication path.
    pub fn try_from_weights(
        weights: impl IntoIterator<Item = (ParameterId, B::MaterializedWeight)>,
    ) -> Result<Self, ParameterOrchestrationError<B::ParameterError>> {
        let mut collected = BTreeMap::new();
        for (id, weight) in weights {
            if collected.insert(id.clone(), weight).is_some() {
                return Err(ParameterOrchestrationError::DuplicateBinding { parameter: id });
            }
        }
        Ok(Self { weights: collected })
    }

    /// Returns the number of realized parameter values.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns whether this unit contains no values.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Returns whether a stable parameter identity is present.
    pub fn contains(&self, id: &ParameterId) -> bool {
        self.weights.contains_key(id)
    }
}

/// Materializes one atomic unit without inspecting backend-native values.
///
/// Each encoded source remains retained by its backend materialization guard
/// until exact completion. The returned weights are independently owned native
/// handles ready for binding.
pub fn materialize_bindings<B: ParameterBackend>(
    source: &dyn CheckpointSource,
    bindings: &[WeightBinding],
    context: &B::MaterializationContext,
) -> Result<MaterializedUnit<B>, ParameterOrchestrationError<B::ParameterError>> {
    let plan = validated_binding_plan::<B>(source, bindings)?;
    materialize_validated_plan::<B>(source, plan, context)
}

/// Materializes a previously selected binding unit without repeating backend
/// capability selection.
pub fn materialize_selected_bindings<B: ParameterBackend>(
    selected: SelectedBindingPlan<B>,
    context: &B::MaterializationContext,
) -> Result<MaterializedUnit<B>, ParameterOrchestrationError<B::ParameterError>> {
    let SelectedBindingPlan {
        source,
        bindings,
        backend: _,
    } = selected;
    let plan = WeightBindingPlan::new(&bindings)
        .expect("selected binding declarations remain immutable after admission");
    materialize_validated_plan::<B>(source.as_ref(), plan, context)
}

fn materialize_validated_plan<B: ParameterBackend>(
    source: &dyn CheckpointSource,
    plan: WeightBindingPlan<'_>,
    context: &B::MaterializationContext,
) -> Result<MaterializedUnit<B>, ParameterOrchestrationError<B::ParameterError>> {
    let mut weights = BTreeMap::new();
    for binding in plan.owners() {
        let materialization = match binding.recipe() {
            Some(recipe) => B::materialize_recipe(recipe, source, context),
            None => {
                let lease = source.acquire_lease(TensorReadRequest {
                    key: binding.checkpoint_key().to_owned(),
                    selection: binding.selection().clone(),
                    policy: ReadPolicy::RequireBounded,
                })?;
                B::materialize(lease, context)
            }
        }
        .map_err(ParameterOrchestrationError::Backend)?;
        let weight = B::finish_materialization(materialization)
            .map_err(ParameterOrchestrationError::Backend)?;
        let id = ParameterId::new(binding.name()).map_err(|error| {
            ParameterOrchestrationError::InvalidParameterIdentity(error.to_string())
        })?;
        if weights.insert(id.clone(), weight).is_some() {
            return Err(ParameterOrchestrationError::DuplicateBinding { parameter: id });
        }
    }
    for (alias, owner) in plan.aliases() {
        let owner_id = ParameterId::new(owner.name()).map_err(|error| {
            ParameterOrchestrationError::InvalidParameterIdentity(error.to_string())
        })?;
        let weight = weights
            .get(&owner_id)
            .expect("validated owner was materialized before aliases");
        let weight =
            B::share_materialized_weight(weight).map_err(ParameterOrchestrationError::Backend)?;
        let alias_id = ParameterId::new(alias.name()).map_err(|error| {
            ParameterOrchestrationError::InvalidParameterIdentity(error.to_string())
        })?;
        weights.insert(alias_id, weight);
    }
    Ok(MaterializedUnit { weights })
}

/// Binds a realized unit into a statically traversed native module.
///
/// Binding is keyed exclusively by stable parameter identity. Missing and
/// unexpected values fail closed; native tensor shape or storage remains
/// backend-owned.
pub fn bind_materialized_unit<B, M>(
    module: &mut M,
    unit: MaterializedUnit<B>,
) -> Result<(), ParameterOrchestrationError<B::ParameterError>>
where
    B: ParameterBackend,
    M: Parameterized<B::Parameter>,
{
    bind_materialized_unit_excluding::<B, M, _>(module, unit, |_| false)
}

/// Binds a realized unit while leaving explicitly excluded destinations untouched.
pub fn bind_materialized_unit_excluding<B, M, F>(
    module: &mut M,
    mut unit: MaterializedUnit<B>,
    excluded: F,
) -> Result<(), ParameterOrchestrationError<B::ParameterError>>
where
    B: ParameterBackend,
    M: Parameterized<B::Parameter>,
    F: Fn(&ParameterId) -> bool,
{
    struct Validator<'a, B: ParameterBackend> {
        weights: &'a BTreeMap<ParameterId, B::MaterializedWeight>,
        visited: BTreeMap<ParameterId, ()>,
        error: Option<ParameterOrchestrationError<B::ParameterError>>,
        excluded: &'a dyn Fn(&ParameterId) -> bool,
    }

    impl<'a, 'value, B: ParameterBackend> ParameterVisitor<'value, B::Parameter> for Validator<'a, B> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'value B::Parameter) {
            if self.error.is_some() {
                return;
            }
            if (self.excluded)(&metadata.id) {
                return;
            }
            if self.visited.insert(metadata.id.clone(), ()).is_some() {
                self.error = Some(ParameterOrchestrationError::DuplicateParameter {
                    parameter: metadata.id,
                });
                return;
            }
            let Some(weight) = self.weights.get(&metadata.id) else {
                self.error = Some(ParameterOrchestrationError::MissingBinding {
                    parameter: metadata.id,
                });
                return;
            };
            if let Err(error) = B::validate_bind(parameter, weight) {
                self.error = Some(ParameterOrchestrationError::Backend(error));
            }
        }
    }

    let mut validator = Validator::<B> {
        weights: &unit.weights,
        visited: BTreeMap::new(),
        error: None,
        excluded: &excluded,
    };
    module.visit_parameters(&mut validator);
    if let Some(error) = validator.error {
        return Err(error);
    }
    let unexpected = unit
        .weights
        .keys()
        .filter(|id| !validator.visited.contains_key(*id))
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(ParameterOrchestrationError::UnexpectedBindings {
            parameters: unexpected,
        });
    }

    struct MutableTopologyValidator<'a> {
        expected: &'a BTreeMap<ParameterId, ()>,
        visited: BTreeMap<ParameterId, ()>,
        unexpected: Vec<ParameterId>,
        duplicate: Option<ParameterId>,
        excluded: &'a dyn Fn(&ParameterId) -> bool,
    }

    impl<'a, 'value, P: 'value> ParameterVisitorMut<'value, P> for MutableTopologyValidator<'a> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, _: &'value mut P) {
            if self.duplicate.is_some() {
                return;
            }
            if (self.excluded)(&metadata.id) {
                return;
            }
            if self.visited.insert(metadata.id.clone(), ()).is_some() {
                self.duplicate = Some(metadata.id);
            } else if !self.expected.contains_key(&metadata.id) {
                self.unexpected.push(metadata.id);
            }
        }
    }

    let mut mutable_topology = MutableTopologyValidator {
        expected: &validator.visited,
        visited: BTreeMap::new(),
        unexpected: Vec::new(),
        duplicate: None,
        excluded: &excluded,
    };
    module.visit_parameters_mut(&mut mutable_topology);
    if let Some(parameter) = mutable_topology.duplicate {
        return Err(ParameterOrchestrationError::DuplicateParameter { parameter });
    }
    let mut mismatch = mutable_topology.unexpected;
    mismatch.extend(
        validator
            .visited
            .keys()
            .filter(|id| !mutable_topology.visited.contains_key(*id))
            .cloned(),
    );
    if !mismatch.is_empty() {
        mismatch.sort();
        mismatch.dedup();
        return Err(ParameterOrchestrationError::ParameterTraversalMismatch {
            parameters: mismatch,
        });
    }

    struct Binder<'a, B: ParameterBackend> {
        weights: &'a mut BTreeMap<ParameterId, B::MaterializedWeight>,
        excluded: &'a dyn Fn(&ParameterId) -> bool,
    }

    impl<'a, 'value, B: ParameterBackend> ParameterVisitorMut<'value, B::Parameter> for Binder<'a, B> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, parameter: &'value mut B::Parameter) {
            if (self.excluded)(&metadata.id) {
                return;
            }
            let weight = self
                .weights
                .remove(&metadata.id)
                .expect("prepublication mutable traversal validated every binding identity");
            B::bind(parameter, weight);
        }
    }

    let mut binder = Binder::<B> {
        weights: &mut unit.weights,
        excluded: &excluded,
    };
    module.visit_parameters_mut(&mut binder);
    assert!(
        unit.weights.is_empty(),
        "prepublication mutable traversal validated complete binding consumption"
    );
    Ok(())
}

/// Failure in backend-neutral parameter materialization or binding.
#[derive(Debug, thiserror::Error)]
pub enum ParameterOrchestrationError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Binding aliases were invalid before materialization began.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// Neutral checkpoint access failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Neutral recipe validation failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A stable runtime parameter identity was invalid.
    #[error("invalid runtime parameter identity: {0}")]
    InvalidParameterIdentity(String),
    /// Two bindings targeted the same stable parameter.
    #[error("duplicate materialized binding for parameter {parameter}")]
    DuplicateBinding {
        /// Duplicated identity.
        parameter: ParameterId,
    },
    /// A module traversal repeated one stable destination identity.
    #[error("module parameter traversal repeats identity {parameter}")]
    DuplicateParameter {
        /// Repeated module parameter identity.
        parameter: ParameterId,
    },
    /// Immutable and mutable module traversals declared different identities.
    #[error("immutable and mutable module parameter traversals disagree: {parameters:?}")]
    ParameterTraversalMismatch {
        /// Identities present in only one traversal.
        parameters: Vec<ParameterId>,
    },
    /// Inferred and declared materialized byte sizes disagreed.
    #[error("parameter {parameter:?} declares {expected} bytes but its recipe produces {actual}")]
    ByteMismatch {
        /// Stable binding name.
        parameter: String,
        /// Declared bytes.
        expected: u64,
        /// Inferred bytes.
        actual: u64,
    },
    /// A traversed parameter had no realized value.
    #[error("materialized unit has no value for parameter {parameter}")]
    MissingBinding {
        /// Missing parameter identity.
        parameter: ParameterId,
    },
    /// Realized values remained after complete parameter traversal.
    #[error("materialized unit contains values for unknown parameters: {parameters:?}")]
    UnexpectedBindings {
        /// Unknown parameter identities.
        parameters: Vec<ParameterId>,
    },
    /// Backend-native realization or binding failed.
    #[error("backend parameter operation failed: {0}")]
    Backend(E),
}
