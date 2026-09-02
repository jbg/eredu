//! Canonical checkpoint bindings for unloaded module parameter trees.

//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use eredu_checkpoint::{
    store::{ReadPolicy, TensorReadRequest},
    SourceTensorEncoding, WeightQuantization,
};
use eredu_nn::{validate_parameter_topology, LinearCompanionRole, Parameterized};
use eredu_runtime::{
    ParameterGroupSpec, ParameterRole, ReplicatedTextMaterializationTask, WeightBinding,
    WeightBindingPlan, WeightLoweringKind,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use safemlx::{Array, Stream};

use crate::{
    backend::error::Error,
    backend::nn::shared::{neutral_parameter_refs, neutral_parameter_refs_mut},
    backend::runtime::checkpoint::binding_plan::{BindingPlan, BindingPlanError, PlannedBinding},
    backend::runtime::checkpoint::load::{
        load_array_quantized_strict, QuantizedLoadRecipe, StrictLoadReport,
    },
    backend::runtime::checkpoint::recipe::{
        recipe_dtype_from_mlx, MlxWeightRecipeExt, WeightRecipeError,
    },
    backend::runtime::checkpoint::store::{
        MlxParameterMaterializationContext, WeightMaterialization,
    },
    backend::runtime::residency::manager::ResidentUnitLease,
    module::FlattenedModuleParamRef,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};

const MODEL_LOAD_MATERIALIZATION_BUFFERS: usize = 2;

/// Exact local parameter targets owned by one neutral semantic role.
///
/// Targets come from the architecture parameter contract, so packed
/// companions and alias destinations participate atomically with their base
/// weight rather than being rediscovered from family-specific path syntax.
pub fn parameter_role_targets(
    groups: &[ParameterGroupSpec],
    role: ParameterRole,
) -> BTreeSet<String> {
    groups
        .iter()
        .filter(|group| group.role() == role)
        .flat_map(ParameterGroupSpec::members)
        .map(|member| member.target().to_owned())
        .collect()
}

/// Returns whether a parameter name is one of the declared targets.
pub fn parameter_name_in_targets(name: &str, targets: &BTreeSet<String>) -> bool {
    targets.contains(name)
}

/// Builds exact full-tensor residency bindings for an unloaded module.
///
/// Every module parameter must resolve to exactly one checkpoint key and have
/// the same shape. Binding names are local module parameter names so a lease
/// can later populate a freshly constructed module without architecture-aware
/// rewriting.
pub fn build_module_bindings<M>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    build_module_bindings_excluding(module, prefix, store, |_| false)
}

/// Builds exact bindings for non-excluded local module parameters.
///
/// The predicate receives module-local flattened names and runs before any
/// checkpoint lookup, allowing independently managed parameter groups to use a
/// different checkpoint layout.
pub fn build_module_bindings_excluding<M, F>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    build_module_bindings_with_recipes_excluding(module, prefix, store, BTreeMap::new(), exclude)
}

/// Builds module bindings while replacing selected local parameters with recipes.
///
/// Recipe keys use the module-local flattened parameter names. Every override
/// is shape- and dtype-checked against the unloaded runtime parameter before
/// residency initialization.
pub fn build_module_bindings_with_recipes<M>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    build_module_binding_plan_with_recipes(module, prefix, store, recipes)?.build_bindings(store)
}

/// Replaces global binding sources with architecture-produced rank-local recipes.
///
/// Recipe keys are exact architecture-logical parameter targets. This lowering
/// knows nothing about architecture-specific axes or physical checkpoint layouts; those
/// have already been expressed by the recipes.
pub fn apply_rank_local_parameter_recipes(
    bindings: Vec<WeightBinding>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    let bindings = bindings
        .into_iter()
        .map(|binding| {
            let Some(target) = binding.logical_target() else {
                return Ok(binding);
            };
            let Some(recipe) = recipes.remove(target) else {
                return Ok(binding);
            };
            let bytes = recipe.infer(store)?.byte_len();
            binding
                .with_source_recipe(recipe, bytes)
                .map_err(ModuleBindingError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !recipes.is_empty() {
        return Err(ModuleBindingError::UnknownRecipeParameters {
            parameters: recipes.into_keys().collect(),
        });
    }
    Ok(bindings)
}

/// A declarative module binding plan plus fully qualified logical targets.
pub struct ModuleBindingPlan {
    plan: BindingPlan,
    logical_targets: BTreeMap<String, String>,
}

impl ModuleBindingPlan {
    /// Resolves the plan into runtime bindings against a checkpoint source.
    pub fn build_bindings(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<Vec<WeightBinding>, ModuleBindingError> {
        self.plan
            .build_bindings(store)?
            .into_iter()
            .map(|binding| {
                let logical_target = self
                    .logical_targets
                    .get(binding.name())
                    .expect("module binding plan contains every local target");
                Ok(binding.with_logical_target(logical_target)?)
            })
            .collect()
    }
}

/// Builds a complete module binding plan including derived-weight recipes.
pub fn build_module_binding_plan_with_recipes<M>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<ModuleBindingPlan, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, |_| false)
}

/// Builds exact bindings for one architecture-owned static module, consuming
/// any architecture recipes whose destinations belong to that module.
pub fn build_neutral_module_bindings_with_recipes<M>(
    module: &M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    build_neutral_module_bindings_with_recipes_excluding(module, store, recipes, |_| false)
}

/// Builds neutral bindings while excluding parameter identities managed by a
/// separate architecture-owned alias or residency plan.
pub fn build_neutral_module_bindings_with_recipes_excluding<M, F>(
    module: &M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    let parameters = neutral_parameter_refs(module, false).flatten();
    let names = parameters.keys().cloned().collect::<BTreeSet<_>>();
    let selected = recipes
        .keys()
        .filter(|name| names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let selected = selected
        .into_iter()
        .map(|name| {
            let recipe = recipes
                .remove(&name)
                .expect("recipe key came from the same map");
            (name, recipe)
        })
        .collect();
    build_flattened_module_binding_plan_with_recipes_excluding(
        parameters, "", store, selected, exclude,
    )?
    .build_bindings(store)
}

/// Resolves one exact replicated-text task partition to native module handles.
///
/// The task partition is authoritative for source identity, recipes, lowering,
/// and transformed companion identities. Module traversal only verifies that
/// every selected destination has one native handle, while the checkpoint
/// source supplies byte metadata without opening payloads.
fn binding_from_exact_replicated_text_task(
    task: &ReplicatedTextMaterializationTask,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, ModuleBindingError> {
    let binding = match task.lowering() {
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform => {
            if !store.is_authoritative_materialized_key(task.name()) {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact transformed task {:?} has no authoritative materialized output",
                    task.name()
                )));
            }
            let metadata = store.source_metadata(task.name())?;
            WeightBinding::new(
                task.name(),
                task.name(),
                TensorSelection::Full,
                metadata.encoded_byte_len,
            )?
        }
        WeightLoweringKind::Direct => {
            let [source] = task.sources() else {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact direct task {:?} must name exactly one source",
                    task.name()
                )));
            };
            let metadata = store.source_metadata(source)?;
            WeightBinding::new(
                task.name(),
                source,
                TensorSelection::Full,
                metadata.encoded_byte_len,
            )?
        }
        WeightLoweringKind::Derived => {
            let recipe = task
                .source_recipe()
                .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
            let metadata = recipe.infer(store)?;
            if task
                .derived_output()
                .is_some_and(|expected| expected != &metadata)
            {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact derived task {:?} differs from its admitted output",
                    task.name()
                )));
            }
            WeightBinding::from_recipe(task.name(), recipe, metadata.byte_len())?
        }
        _ => {
            return Err(ModuleBindingError::BindingPlan(format!(
                "exact task {:?} selected an unsupported lowering",
                task.name()
            )))
        }
    };
    binding.with_logical_target(task.name()).map_err(Into::into)
}

/// Resolves one exact replicated-text task partition to native module handles.
///
/// Each primary and companion binding consumes its complete selected task;
/// checkpoint target-name probing is not a source-selection mechanism.
pub fn build_exact_replicated_text_bindings<M>(
    module: &M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    tasks: &[&ReplicatedTextMaterializationTask],
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    let parameter_names = neutral_parameter_refs(module, false)
        .flatten()
        .into_keys()
        .map(|name| name.as_ref().to_owned())
        .collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    let mut bindings = Vec::new();
    for task in tasks {
        if !parameter_names.contains(task.name()) {
            return Err(ModuleBindingError::BindingPlan(format!(
                "exact materialization task {:?} has no native module destination",
                task.name()
            )));
        }
        if !covered.insert(task.name().to_owned()) {
            return Err(ModuleBindingError::BindingPlan(format!(
                "exact materialization task {:?} was consumed more than once",
                task.name()
            )));
        }
        bindings.push(binding_from_exact_replicated_text_task(task, store)?);

        for companion in task.output_companions() {
            if !parameter_names.contains(companion.name()) {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact task {:?} companion {:?} has no native module destination",
                    task.name(),
                    companion.name()
                )));
            }
            if !covered.insert(companion.name().to_owned()) {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact companion {:?} was consumed more than once",
                    companion.name()
                )));
            }
            let binding = if let Some(exact) = companion.materialization_task() {
                binding_from_exact_replicated_text_task(exact, store)?
            } else if let (Some(recipe), Some(expected)) =
                (companion.derived_recipe(), companion.derived_output())
            {
                let metadata = recipe.infer(store)?;
                if &metadata != expected {
                    return Err(ModuleBindingError::BindingPlan(format!(
                        "exact companion {:?} differs from its admitted derived output",
                        companion.name()
                    )));
                }
                WeightBinding::from_recipe(companion.name(), recipe.clone(), metadata.byte_len())?
                    .with_logical_target(companion.name())?
            } else if let Some(source) = companion.catalog_source() {
                let provenance = store.source_provenance(companion.name())?;
                if provenance.physical_tensor != source.tensor()
                    || provenance.output != source.output()
                    || provenance.backing_shard.as_deref() != Some(source.shard())
                    || &provenance.source_encoding != task.source_encoding()
                {
                    return Err(ModuleBindingError::BindingPlan(format!(
                        "translated catalog companion {:?} differs from its admitted provenance",
                        companion.name()
                    )));
                }
                let metadata = store.source_metadata(companion.name())?;
                WeightBinding::new(
                    companion.name(),
                    companion.name(),
                    TensorSelection::Full,
                    metadata.encoded_byte_len,
                )?
                .with_logical_target(companion.name())?
            } else if matches!(
                task.lowering(),
                WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
            ) || matches!(task.source_encoding(), SourceTensorEncoding::Gguf { .. })
            {
                if !store.is_authoritative_materialized_key(companion.name()) {
                    return Err(ModuleBindingError::BindingPlan(format!(
                        "generated companion {:?} has no authoritative materialized output",
                        companion.name()
                    )));
                }
                let metadata = store.source_metadata(companion.name())?;
                WeightBinding::new(
                    companion.name(),
                    companion.name(),
                    TensorSelection::Full,
                    metadata.encoded_byte_len,
                )?
                .with_logical_target(companion.name())?
            } else {
                return Err(ModuleBindingError::BindingPlan(format!(
                    "exact companion {:?} has neither a selected task nor a causal primary output",
                    companion.name()
                )));
            };
            bindings.push(binding);
        }
    }
    let missing = parameter_names
        .difference(&covered)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        let selected = tasks.iter().map(|task| task.name()).collect::<Vec<_>>();
        return Err(ModuleBindingError::BindingPlan(format!(
            "native module parameters have no exact materialization tasks: {missing:?}; selected tasks: {selected:?}"
        )));
    }
    WeightBindingPlan::new(&bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    Ok(bindings)
}

/// Materializes a set of direct or derived bindings without constructing a
/// residency manager.
///
/// This is useful for loaders that keep one parameter class resident while a
/// separate cache owns another class from the same type-erased checkpoint
/// store. Direct and recipe outputs share a fixed two-completion window;
/// source leases remain retained through their exact event, and outputs are
/// copied to the execution stream when the streams differ. Shard-cache-capacity
/// pressure drains the oldest completion before retrying.
pub fn materialize_module_bindings(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<BTreeMap<String, Array>, ModuleBindingError> {
    let plan = WeightBindingPlan::new(bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    for binding in plan.owners() {
        let actual = binding
            .source_recipe()
            .infer(store)
            .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?
            .byte_len();
        if actual != binding.expected_bytes() {
            return Err(ModuleBindingError::BindingPlan(format!(
                "binding {:?} declares {} bytes but materializes {actual}",
                binding.name(),
                binding.expected_bytes()
            )));
        }
    }
    let mut arrays = BTreeMap::new();
    let mut pending = VecDeque::with_capacity(MODEL_LOAD_MATERIALIZATION_BUFFERS);
    let context = MlxParameterMaterializationContext::new(source_stream, execution_stream);
    for binding in plan.owners() {
        let materialization = loop {
            match submit_module_binding(store, binding, source_stream, execution_stream, &context) {
                Ok(materialization) => break materialization,
                Err(error) if !pending.is_empty() && is_shard_cache_capacity_error(&error) => {
                    finish_module_binding(&mut pending, &mut arrays)?;
                }
                Err(error) => return Err(error),
            }
        };
        pending.push_back((
            binding.name().to_string(),
            binding.checkpoint_key().to_string(),
            materialization,
        ));
        if pending.len() == MODEL_LOAD_MATERIALIZATION_BUFFERS {
            finish_module_binding(&mut pending, &mut arrays)?;
        }
    }
    while !pending.is_empty() {
        finish_module_binding(&mut pending, &mut arrays)?;
    }
    for (alias, owner) in plan.aliases() {
        let value = arrays
            .get(owner.name())
            .expect("validated canonical binding was materialized")
            .clone();
        arrays.insert(alias.name().to_owned(), value);
    }
    Ok(arrays)
}

fn submit_module_binding(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    binding: &WeightBinding,
    source_stream: &Stream,
    execution_stream: &Stream,
    context: &MlxParameterMaterializationContext,
) -> Result<WeightMaterialization, ModuleBindingError> {
    if let Some(recipe) = binding.recipe() {
        let pending = recipe.prepare_materialization(store, context)?;
        let (source, sources) = pending.into_parts();
        let output = if source_stream == execution_stream {
            source
        } else {
            source
                .copy(execution_stream)
                .map_err(WeightRecipeError::from)?
        };
        Ok(WeightMaterialization::submit_retained(output, sources)?)
    } else {
        let lease = store.acquire_lease(TensorReadRequest {
            key: binding.checkpoint_key().to_owned(),
            selection: binding.selection().clone(),
            policy: ReadPolicy::RequireBounded,
        })?;
        Ok(context
            .weight_lease(lease)?
            .materialize(source_stream, execution_stream)?)
    }
}

type PendingModuleBinding = (String, String, WeightMaterialization);

fn finish_module_binding(
    pending: &mut VecDeque<PendingModuleBinding>,
    arrays: &mut BTreeMap<String, Array>,
) -> Result<(), ModuleBindingError> {
    let (name, checkpoint_key, materialization) = pending
        .pop_front()
        .expect("non-empty model-loading window has a front");
    let value = materialization.synchronize()?;
    if arrays.insert(name.clone(), value).is_some() {
        return Err(ModuleBindingError::DuplicateCheckpointBinding {
            checkpoint_key,
            first: name.clone(),
            second: name,
        });
    }
    Ok(())
}

fn is_shard_cache_capacity_error(error: &ModuleBindingError) -> bool {
    matches!(
        error,
        ModuleBindingError::CheckpointStore(
            eredu_checkpoint::store::StoreError::CapacityExhausted { .. },
        ) | ModuleBindingError::WeightRecipe(WeightRecipeError::CheckpointStore(
            eredu_checkpoint::store::StoreError::CapacityExhausted { .. },
        ))
    )
}

/// Populates an unloaded module from materialized local-name bindings while
/// permitting an independently managed parameter class to remain unloaded.
pub fn populate_module_from_arrays_excluding<M, F>(
    module: &mut M,
    arrays: &BTreeMap<String, Array>,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    let expected = {
        let params = neutral_parameter_refs(module, false).flatten();
        params
            .iter()
            .filter(|(name, _)| !excluded(name))
            .map(|(name, _)| name.to_string())
            .collect::<BTreeSet<_>>()
    };
    let mut resolved = BTreeMap::<String, &Array>::new();
    let mut unexpected = Vec::new();
    for (name, value) in arrays {
        if expected.contains(name) {
            resolved.insert(name.clone(), value);
        } else {
            unexpected.push(name.clone());
        }
    }
    let missing = expected
        .iter()
        .filter(|name| !resolved.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModuleBindingError::LeaseContents {
            unit: "materialized checkpoint bindings".into(),
            missing,
            unexpected,
        });
    }
    let mut params = neutral_parameter_refs_mut(module).flatten();
    for (name, parameter) in &mut params {
        if !expected.contains(name.as_ref()) {
            continue;
        }
        let value = *resolved
            .get(name.as_ref())
            .expect("validated materialized binding must exist");
        if parameter.shape() != value.shape() {
            return Err(ModuleBindingError::ResidentShapeMismatch {
                unit: "materialized checkpoint bindings".into(),
                parameter: name.to_string(),
                expected: parameter.shape().to_vec(),
                actual: value.shape().to_vec(),
            });
        }
        **parameter = value.clone();
    }
    Ok(())
}

/// Populates a quantized target from dense direct or derived bindings while an
/// independent residency manager may own an excluded parameter class.
pub(crate) fn populate_module_from_dense_arrays_quantized_excluding<M, F>(
    module: &mut M,
    arrays: &BTreeMap<String, Array>,
    quantization: WeightQuantization,
    stream: &Stream,
    excluded: F,
) -> Result<(), Error>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    quantization.validate()?;
    let recipes = quantized_load_recipes(module, quantization)?;
    let mut report = StrictLoadReport::default();
    {
        let mut parameters = neutral_parameter_refs_mut(module).flatten();
        for (name, value) in arrays {
            load_array_quantized_strict(
                &mut parameters,
                name.clone(),
                value.clone(),
                stream,
                quantization,
                recipes.get(name),
                &mut report,
            )?;
        }
    }
    let parameter_names = neutral_parameter_refs(module, false)
        .flatten()
        .into_keys()
        .map(|name| name.to_string());
    report.finish_parameter_names(parameter_names, excluded)
}

fn quantized_load_recipes<M>(
    module: &M,
    quantization: WeightQuantization,
) -> Result<BTreeMap<String, QuantizedLoadRecipe>, Error>
where
    M: Parameterized<crate::MlxTensor>,
{
    #[derive(Default)]
    struct Companions {
        scales: Option<String>,
        biases: Option<String>,
    }

    let topology = validate_parameter_topology(module)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameters = topology
        .iter()
        .map(|parameter| parameter.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut companions = BTreeMap::<String, Companions>::new();
    for parameter in &topology {
        let (Some(role), Some(owner)) = (
            parameter.linear_companion,
            parameter.linear_companion_of.as_ref(),
        ) else {
            continue;
        };
        if !parameters.contains(owner.as_str()) {
            return Err(Error::ArchitectureModel(format!(
                "linear companion {:?} names missing owner {:?}",
                parameter.id.as_str(),
                owner.as_str()
            )));
        }
        let target = companions.entry(owner.as_str().to_owned()).or_default();
        let slot = match role {
            LinearCompanionRole::Scale => &mut target.scales,
            LinearCompanionRole::AffineBias => &mut target.biases,
        };
        if slot.replace(parameter.id.as_str().to_owned()).is_some() {
            return Err(Error::ArchitectureModel(format!(
                "linear weight {:?} declares duplicate {role:?} companions",
                owner.as_str()
            )));
        }
    }

    companions
        .into_iter()
        .map(|(weight, companions)| {
            let scales = companions.scales.ok_or_else(|| {
                Error::ArchitectureModel(format!(
                    "quantized linear weight {weight:?} has no declared scale companion"
                ))
            })?;
            if quantization.has_biases() != companions.biases.is_some() {
                return Err(Error::ArchitectureModel(format!(
                    "quantized linear weight {weight:?} companion declaration does not match {quantization:?}"
                )));
            }
            Ok((
                weight.clone(),
                QuantizedLoadRecipe::new(weight, scales, companions.biases),
            ))
        })
        .collect()
}

/// Builds bindings while excluding parameters managed by another loader.
pub fn build_module_bindings_with_recipes_excluding<M, F>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, exclude)?
        .build_bindings(store)
}

/// Builds a derived binding plan while excluding independently managed parameters.
pub fn build_module_binding_plan_with_recipes_excluding<M, F>(
    module: &M,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<ModuleBindingPlan, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    build_flattened_module_binding_plan_with_recipes_excluding(
        neutral_parameter_refs(module, false).flatten(),
        prefix,
        store,
        recipes,
        exclude,
    )
}

fn build_flattened_module_binding_plan_with_recipes_excluding<F>(
    params: FlattenedModuleParamRef<'_>,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<ModuleBindingPlan, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut local_names = params
        .iter()
        .filter(|(name, _)| !exclude(name))
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    local_names.sort();
    recipes.retain(|name, _| !exclude(name));
    let mut claimed = BTreeMap::<String, String>::new();
    let mut planned = Vec::with_capacity(local_names.len());
    let mut logical_targets = BTreeMap::new();
    let mut source_claim_counts = BTreeMap::<String, usize>::new();

    for local_name in local_names {
        let parameter = params
            .get(local_name.as_str())
            .expect("parameter name came from the same flattened tree");
        let destination = qualify(prefix, &local_name);
        logical_targets.insert(local_name.clone(), destination.clone());
        let authoritative_key = if store.is_authoritative_materialized_key(&destination) {
            Some(destination.clone())
        } else {
            None
        };
        if authoritative_key.is_none() {
            if let Some(recipe) = recipes.remove(&local_name) {
                let metadata = recipe.infer(store)?;
                let expected_shape = parameter
                    .shape()
                    .iter()
                    .map(|&dimension| usize::try_from(dimension))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| ModuleBindingError::InvalidModuleShape {
                        parameter: qualify(prefix, &local_name),
                        shape: parameter.shape().to_vec(),
                    })?;
                if metadata.shape() != expected_shape {
                    return Err(ModuleBindingError::RecipeShapeMismatch {
                        parameter: qualify(prefix, &local_name),
                        expected: expected_shape,
                        actual: metadata.shape().to_vec(),
                    });
                }
                let expected_dtype = recipe_dtype_from_mlx(parameter.dtype());
                if !recipe_dtype_matches(&expected_dtype, metadata.dtype()) {
                    return Err(ModuleBindingError::RecipeDtypeMismatch {
                        parameter: qualify(prefix, &local_name),
                        expected: expected_dtype,
                        actual: metadata.dtype().clone(),
                    });
                }
                for source in recipe.source_keys() {
                    *source_claim_counts.entry(source.into()).or_default() += 1;
                }
                planned.push(PlannedBinding {
                    target_name: local_name,
                    expected_shape,
                    expected_dtype,
                    recipe,
                });
                continue;
            }
        } else {
            recipes.remove(&local_name);
        }
        let checkpoint_key = if let Some(authoritative_key) = authoritative_key {
            authoritative_key
        } else if keys.contains(&destination) {
            destination.clone()
        } else {
            return Err(ModuleBindingError::MissingParameter { destination });
        };

        if let Some(previous) = claimed.insert(checkpoint_key.clone(), destination.clone()) {
            return Err(ModuleBindingError::DuplicateCheckpointBinding {
                checkpoint_key,
                first: previous,
                second: destination,
            });
        }

        let metadata = store.source_metadata(&checkpoint_key)?;
        let expected_shape = parameter
            .shape()
            .iter()
            .map(|&dimension| usize::try_from(dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ModuleBindingError::InvalidModuleShape {
                parameter: destination.clone(),
                shape: parameter.shape().to_vec(),
            })?;
        if metadata.logical_shape != expected_shape {
            return Err(ModuleBindingError::ShapeMismatch {
                checkpoint_key,
                parameter: destination,
                expected: expected_shape,
                actual: metadata.logical_shape,
            });
        }
        // Direct checkpoint tensors retain their stored dtype. Some modules
        // intentionally use an unloaded placeholder with a different dtype
        // and adopt the checkpoint representation when the binding is loaded.
        let expected_dtype = RecipeDtype::from(metadata.stored_dtype.clone());
        let recipe = DerivedWeightRecipe::source(metadata.name, TensorSelection::Full);
        for source in recipe.source_keys() {
            *source_claim_counts.entry(source.into()).or_default() += 1;
        }
        planned.push(PlannedBinding {
            target_name: local_name,
            expected_shape,
            expected_dtype,
            recipe,
        });
    }

    if !recipes.is_empty() {
        return Err(ModuleBindingError::UnknownRecipeParameters {
            parameters: recipes.into_keys().collect(),
        });
    }

    let shared_source_keys = source_claim_counts
        .into_iter()
        .filter_map(|(source, claims)| (claims > 1).then_some(source))
        .collect();
    let plan = BindingPlan::allowing_shared_sources(planned, shared_source_keys)?;
    Ok(ModuleBindingPlan {
        plan,
        logical_targets,
    })
}

/// Assigns every module parameter from a protected resident unit.
///
/// `Array::clone` only clones the MLX handle; it does not copy the resident
/// allocation. The caller must therefore keep `lease` alive through forward
/// execution and synchronize before releasing it.
pub fn populate_module_from_lease<M>(
    module: &mut M,
    lease: &ResidentUnitLease,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    populate_module_from_lease_excluding(module, lease, |_| false)
}

/// Assigns non-excluded module parameters from a protected resident unit.
pub fn populate_module_from_lease_excluding<M, F>(
    module: &mut M,
    lease: &ResidentUnitLease,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    let resident_names = lease.binding_names().collect::<BTreeSet<_>>();
    let expected_names = {
        let params = neutral_parameter_refs(module, false).flatten();
        params
            .iter()
            .filter(|(name, _)| !excluded(name))
            .map(|(name, _)| name.to_string())
            .collect::<BTreeSet<_>>()
    };

    let missing = expected_names
        .iter()
        .filter(|name| !resident_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = resident_names
        .iter()
        .filter(|name| !expected_names.contains(**name))
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModuleBindingError::LeaseContents {
            unit: lease.id().to_string(),
            missing,
            unexpected,
        });
    }

    let mut params = neutral_parameter_refs_mut(module).flatten();
    for (name, parameter) in &mut params {
        if !expected_names.contains(name.as_ref()) {
            continue;
        }
        let value = lease.device_value(name)?;
        if parameter.shape() != value.shape() {
            return Err(ModuleBindingError::ResidentShapeMismatch {
                unit: lease.id().to_string(),
                parameter: name.to_string(),
                expected: parameter.shape().to_vec(),
                actual: value.shape().to_vec(),
            });
        }
        **parameter = value.clone();
    }
    Ok(())
}

/// Returns the checked total byte count of a binding collection.
pub fn binding_bytes(bindings: &[WeightBinding]) -> Result<u64, ModuleBindingError> {
    bindings.iter().try_fold(0u64, |total, binding| {
        if binding.is_alias() {
            return Ok(total);
        }
        total
            .checked_add(binding.expected_bytes())
            .ok_or(ModuleBindingError::ArithmeticOverflow {
                context: "module binding byte total",
            })
    })
}

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

fn recipe_dtype_matches(expected: &RecipeDtype, actual: &RecipeDtype) -> bool {
    expected == actual
        || matches!((expected, actual), (RecipeDtype::U8, RecipeDtype::F8E4M3))
        // Dense module placeholders default to F32, while direct checkpoint
        // bindings replace them with the checkpoint's native floating dtype.
        // Derived bindings (including key rewrites and leading-axis stacking) must
        // follow the same rule or valid BF16 checkpoints are rejected solely
        // because their public tensor names require a recipe.
        || (is_floating_recipe_dtype(expected) && is_floating_recipe_dtype(actual))
}

fn is_floating_recipe_dtype(dtype: &RecipeDtype) -> bool {
    matches!(
        dtype,
        RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
    )
}

/// Structured module-to-checkpoint binding failures.
#[derive(Debug, thiserror::Error)]
pub enum ModuleBindingError {
    /// The declarative binding plan was invalid or disagreed with source metadata.
    #[error("checkpoint binding plan is invalid: {0}")]
    BindingPlan(String),
    /// A recipe override did not name a runtime parameter.
    #[error("derived-weight recipes name unknown local parameters: {parameters:?}")]
    UnknownRecipeParameters {
        /// Unknown local parameter names.
        parameters: Vec<String>,
    },
    /// A recipe output shape differed from its runtime placeholder.
    #[error("derived weight for {parameter:?} has shape {actual:?}, expected {expected:?}")]
    RecipeShapeMismatch {
        /// Fully qualified runtime parameter.
        parameter: String,
        /// Runtime placeholder shape.
        expected: Vec<usize>,
        /// Recipe output shape.
        actual: Vec<usize>,
    },
    /// A recipe output dtype differed from its runtime placeholder.
    #[error("derived weight for {parameter:?} has dtype {actual:?}, expected {expected:?}")]
    RecipeDtypeMismatch {
        /// Fully qualified runtime parameter.
        parameter: String,
        /// Runtime placeholder dtype.
        expected: RecipeDtype,
        /// Recipe output dtype.
        actual: RecipeDtype,
    },
    /// A module parameter had no matching checkpoint tensor.
    #[error("checkpoint is missing module parameter {destination:?}")]
    MissingParameter {
        /// Full module parameter name.
        destination: String,
    },
    /// Two parameters resolved to one checkpoint tensor.
    #[error("checkpoint tensor {checkpoint_key:?} resolves to both {first:?} and {second:?}")]
    DuplicateCheckpointBinding {
        /// Ambiguous checkpoint tensor.
        checkpoint_key: String,
        /// First module parameter.
        first: String,
        /// Second module parameter.
        second: String,
    },
    /// A module placeholder exposed an invalid dimension.
    #[error("module parameter {parameter:?} has invalid shape {shape:?}")]
    InvalidModuleShape {
        /// Full parameter name.
        parameter: String,
        /// Invalid MLX shape.
        shape: Vec<i32>,
    },
    /// Checkpoint and unloaded-module shapes differed.
    #[error("checkpoint tensor {checkpoint_key:?} for {parameter:?} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Source checkpoint key.
        checkpoint_key: String,
        /// Destination module parameter.
        parameter: String,
        /// Unloaded module shape.
        expected: Vec<usize>,
        /// Checkpoint shape.
        actual: Vec<usize>,
    },
    /// A resident unit did not exactly match the module parameter tree.
    #[error("resident unit {unit} cannot populate module: missing {missing:?}, unexpected {unexpected:?}")]
    LeaseContents {
        /// Resident unit identifier.
        unit: String,
        /// Expected module parameters absent from the lease.
        missing: Vec<String>,
        /// Lease bindings absent from the module.
        unexpected: Vec<String>,
    },
    /// A resident array shape differs from its unloaded placeholder.
    #[error(
        "resident unit {unit} parameter {parameter:?} has shape {actual:?}, expected {expected:?}"
    )]
    ResidentShapeMismatch {
        /// Resident unit identifier.
        unit: String,
        /// Local module parameter.
        parameter: String,
        /// Unloaded module shape.
        expected: Vec<i32>,
        /// Resident array shape.
        actual: Vec<i32>,
    },
    /// Checked accounting overflowed.
    #[error("module binding arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Backend-neutral checkpoint inspection or lease acquisition failed.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// Derived-weight metadata validation failed.
    #[error(transparent)]
    WeightRecipe(#[from] crate::backend::runtime::checkpoint::recipe::WeightRecipeError),
    /// Backend-neutral recipe validation failed.
    #[error(transparent)]
    NeutralRecipe(#[from] eredu_checkpoint::recipe::RecipeError),
    /// A backend-neutral residency declaration was invalid.
    #[error(transparent)]
    ResidencyDeclaration(#[from] eredu_runtime::ResidencyDeclarationError),
    /// Residency binding or lookup failed.
    #[error(transparent)]
    Residency(#[from] crate::backend::runtime::residency::manager::ResidencyError),
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "binding tests stay adjacent to the binding planners they exercise"
)]
mod tests {
    use super::*;
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use crate::backend::ExecutionContext;
    use eredu_checkpoint::store::MemoryWeightStore;
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use eredu_checkpoint::{AffineQuantization, LinearFormat};
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use eredu_nn::{
        LinearFormatSpec, LinearSpec, NeuralBackend, ParameterMetadata, ParameterSpec,
        ParameterVisitor, ParameterVisitorMut,
    };
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    use safemlx::{Device, DeviceType};

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn architecture_scale_matching_native_sentinel_shape_is_bound() {
        struct Module {
            weight: crate::MlxTensor,
            scales: crate::MlxTensor,
            weight_spec: ParameterSpec,
            scales_spec: ParameterSpec,
        }

        impl Parameterized<crate::MlxTensor> for Module {
            fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
            where
                V: ParameterVisitor<'a, crate::MlxTensor>,
            {
                visitor.visit(
                    ParameterMetadata::from_spec(&self.weight_spec, true),
                    &self.weight,
                );
                visitor.visit(
                    ParameterMetadata::from_spec(&self.scales_spec, true),
                    &self.scales,
                );
            }

            fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
            where
                V: ParameterVisitorMut<'a, crate::MlxTensor>,
            {
                visitor.visit_mut(
                    ParameterMetadata::from_spec(&self.weight_spec, true),
                    &mut self.weight,
                );
                visitor.visit_mut(
                    ParameterMetadata::from_spec(&self.scales_spec, true),
                    &mut self.scales,
                );
            }

            fn set_trainable(&mut self, _trainable: bool) {}
        }

        let module = Module {
            weight: crate::MlxTensor::from_array(Array::from_slice(&[1u8, 2, 3, 4], &[4])),
            scales: crate::MlxTensor::from_array(Array::from_slice(&[0.5f32], &[1])),
            weight_spec: ParameterSpec::trainable("weight").unwrap(),
            scales_spec: ParameterSpec::trainable("scales").unwrap(),
        };
        let store = MemoryWeightStore::from_safetensors([
            (
                "weight".to_owned(),
                safetensors::Dtype::U8,
                vec![4],
                vec![1, 2, 3, 4],
            ),
            (
                "scales".to_owned(),
                safetensors::Dtype::F32,
                vec![1],
                0.5f32.to_le_bytes().to_vec(),
            ),
        ])
        .unwrap();

        let names = build_module_bindings(&module, "", &store)
            .unwrap()
            .into_iter()
            .map(|binding| binding.name().to_owned())
            .collect::<BTreeSet<_>>();

        assert_eq!(names, BTreeSet::from(["scales".into(), "weight".into()]));
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn quantized_population_uses_declared_companion_identities() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let quantization = AffineQuantization::default();
        let parameter = |name| ParameterSpec::trainable(name).unwrap();
        let linear = <crate::backend::nn::shared::MlxNeuralBackend as NeuralBackend>::linear(
            LinearSpec {
                input: 64,
                output: 8,
                weight: parameter("unconventional.matrix"),
                bias: None,
                format: LinearFormatSpec::affine(
                    LinearFormat::Affine(quantization),
                    parameter("separate.scale.factor"),
                    parameter("another.affine.offset"),
                )
                .unwrap(),
            },
            stream,
        )
        .unwrap();
        let mut module = crate::backend::nn::shared::MlxModule::new(linear);
        let dense = Array::from_slice(&vec![0.25f32; 8 * 64], &[8, 64]);
        populate_module_from_dense_arrays_quantized_excluding(
            &mut module,
            &BTreeMap::from([("unconventional.matrix".into(), dense)]),
            quantization.into(),
            stream,
            |_| false,
        )
        .unwrap();

        let parameters = neutral_parameter_refs(&module, false).flatten();
        assert_eq!(parameters["unconventional.matrix"].shape(), &[8, 8]);
        assert_eq!(parameters["separate.scale.factor"].shape(), &[8, 1]);
        assert_eq!(parameters["another.affine.offset"].shape(), &[8, 1]);
    }

    #[test]
    fn architecture_inner_path_is_an_exact_parameter_identity() {
        let targets =
            BTreeSet::from(["projection.weight".to_owned(), "projection.bias".to_owned()]);
        assert!(!parameter_name_in_targets(
            "projection.inner.weight",
            &targets
        ));
        assert!(!parameter_name_in_targets(
            "projection.inner.bias",
            &targets
        ));
    }

    #[test]
    fn rank_local_recipes_replace_sources_by_exact_logical_target() {
        let store = MemoryWeightStore::from_safetensors([(
            "physical.experts".to_owned(),
            safetensors::Dtype::F32,
            vec![4, 2],
            vec![0; 4 * 2 * size_of::<f32>()],
        )])
        .unwrap();
        let global = DerivedWeightRecipe::source("physical.experts", TensorSelection::Full);
        let binding = WeightBinding::from_recipe("local.experts", global, 32)
            .unwrap()
            .with_logical_target("model.experts")
            .unwrap();
        let local = DerivedWeightRecipe::source(
            "physical.experts",
            TensorSelection::Indices {
                axis: 0,
                indices: vec![3, 1],
            },
        );

        let bindings = apply_rank_local_parameter_recipes(
            vec![binding],
            &store,
            BTreeMap::from([("model.experts".into(), local.clone())]),
        )
        .unwrap();

        assert_eq!(bindings[0].source_recipe(), local);
        assert_eq!(bindings[0].expected_bytes(), 16);
    }
}

impl From<BindingPlanError> for ModuleBindingError {
    fn from(error: BindingPlanError) -> Self {
        Self::BindingPlan(error.to_string())
    }
}
