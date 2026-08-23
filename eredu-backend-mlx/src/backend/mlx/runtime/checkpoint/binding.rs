//! Canonical checkpoint bindings for unloaded module parameter trees.

//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use eredu_checkpoint::{
    store::{ReadPolicy, TensorReadRequest},
    WeightQuantization,
};
use eredu_nn::Parameterized;
use eredu_runtime::{ParameterGroupSpec, ParameterRole, WeightBinding, WeightBindingPlan};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use safemlx::{
    module::{FlattenedModuleParamRef, ModuleParameters},
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::shared::neutral_parameter_refs,
    backend::mlx::runtime::checkpoint::binding_plan::{
        BindingPlan, BindingPlanError, PlannedBinding,
    },
    backend::mlx::runtime::checkpoint::load::{
        load_array_quantized_strict, StrictLoadConfig, StrictLoadReport,
    },
    backend::mlx::runtime::checkpoint::recipe::{
        recipe_dtype_from_mlx, DerivedWeightRecipe, MlxWeightRecipeExt, RecipeDtype,
        WeightRecipeError,
    },
    backend::mlx::runtime::checkpoint::store::{
        MlxParameterMaterializationContext, TensorSelection, WeightMaterialization,
        WeightStoreError,
    },
    backend::mlx::runtime::residency::manager::ResidentUnitLease,
};

const MODEL_LOAD_MATERIALIZATION_BUFFERS: usize = 2;

/// Converts a module parameter name to its canonical checkpoint spelling.
///
/// Quantized modules wrap their packed matrix (and ordinary biased modules
/// wrap their bias) in an `inner` module. MLX-compatible checkpoints omit that
/// implementation detail while retaining companion `.scales` and `.biases`
/// tensors unchanged.
pub fn canonical_checkpoint_name(parameter_name: &str) -> String {
    let canonical = parameter_name
        .replace(".inner.weight", ".weight")
        .replace(".inner.bias", ".bias");
    match canonical.as_str() {
        "inner.weight" => "weight".into(),
        "inner.bias" => "bias".into(),
        _ => canonical,
    }
}

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

pub fn parameter_name_in_targets(name: &str, targets: &BTreeSet<String>) -> bool {
    targets.contains(name) || targets.contains(&canonical_checkpoint_name(name))
}

/// Whether a flattened module parameter has checkpoint-backed storage.
///
/// Native GGML modules expose a one-element `scales` parameter solely to keep
/// the public quantized-module shape compatible with affine quantization. The
/// packed `u8` weight contains its own quantization metadata, so that sentinel
/// must not become a residency or distributed-planning member. Affine packed
/// weights use a non-`u8` storage dtype and retain their real companions.
pub fn is_materialized_module_parameter(
    name: &str,
    parameter: &Array,
    parameters: &FlattenedModuleParamRef<'_>,
) -> bool {
    let weight_names = if name == "scales" {
        Some(["inner.weight".to_string(), "weight".to_string()])
    } else {
        name.strip_suffix(".scales")
            .map(|prefix| [format!("{prefix}.inner.weight"), format!("{prefix}.weight")])
    };
    let weight_dtype = weight_names
        .as_ref()
        .and_then(|names| names.iter().find_map(|name| parameters.get(name.as_str())))
        .map(|weight| weight.dtype());
    !is_native_scale_sentinel(name, parameter.shape(), weight_dtype)
}

fn is_native_scale_sentinel(name: &str, shape: &[i32], weight_dtype: Option<Dtype>) -> bool {
    (name == "scales" || name.ends_with(".scales"))
        && shape == [1]
        && weight_dtype == Some(Dtype::Uint8)
}

/// Returns the checkpoint-backed parameter names exposed by `module` under `prefix`.
pub fn full_parameter_names(module: &impl ModuleParameters, prefix: &str) -> Vec<String> {
    let parameters = module.parameters().flatten();
    let mut names = parameters
        .iter()
        .filter(|(name, parameter)| is_materialized_module_parameter(name, parameter, &parameters))
        .map(|(name, _)| qualify(prefix, name))
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Builds exact full-tensor residency bindings for an unloaded module.
///
/// Every module parameter must resolve to exactly one checkpoint key and have
/// the same shape. Binding names are local module parameter names so a lease
/// can later populate a freshly constructed module without architecture-aware
/// rewriting.
pub fn build_module_bindings(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    build_module_bindings_excluding(module, prefix, store, |_| false)
}

/// Builds exact bindings for non-excluded local module parameters.
///
/// The predicate receives module-local flattened names and runs before any
/// checkpoint lookup, allowing independently managed parameter groups to use a
/// different checkpoint layout.
pub fn build_module_bindings_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    build_module_bindings_with_recipes_excluding(module, prefix, store, BTreeMap::new(), exclude)
}

/// Builds module bindings while replacing selected local parameters with recipes.
///
/// Recipe keys use the module-local flattened parameter names. Every override
/// is shape- and dtype-checked against the unloaded runtime parameter before
/// residency initialization.
pub fn build_module_bindings_with_recipes(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    build_module_binding_plan_with_recipes(module, prefix, store, recipes)?.build_bindings(store)
}

/// Readdresses fully qualified semantic bindings to one module's local
/// parameter names while preserving their architecture-logical targets.
pub fn localize_module_bindings(
    module: &impl ModuleParameters,
    bindings: Vec<WeightBinding>,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    let parameters = module.parameters().flatten();
    let local_names = parameters
        .iter()
        .filter(|(name, parameter)| is_materialized_module_parameter(name, parameter, &parameters))
        .map(|(name, _)| name.to_string())
        .collect::<BTreeSet<_>>();
    bindings
        .into_iter()
        .map(|binding| {
            let candidates = local_names
                .iter()
                .filter(|local| {
                    binding.name() == local.as_str()
                        || binding
                            .name()
                            .strip_suffix(local.as_str())
                            .is_some_and(|prefix| prefix.ends_with('.'))
                })
                .cloned()
                .collect::<Vec<_>>();
            if candidates.len() != 1 {
                return Err(ModuleBindingError::LocalParameterTarget {
                    binding: binding.name().to_owned(),
                    candidates,
                });
            }
            Ok(binding.with_name(&candidates[0])?)
        })
        .collect()
}

/// A declarative module binding plan plus fully qualified logical targets.
pub struct ModuleBindingPlan {
    plan: BindingPlan,
    logical_targets: BTreeMap<String, String>,
}

impl ModuleBindingPlan {
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

pub fn build_module_binding_plan_with_recipes(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<ModuleBindingPlan, ModuleBindingError> {
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, |_| false)
}

/// Builds exact checkpoint bindings from a backend-neutral parameter module.
///
/// Stable parameter identities are already fully qualified, so no
/// family-specific checkpoint root is accepted here.
pub fn build_neutral_module_bindings<M>(
    module: &M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    build_flattened_module_binding_plan_with_recipes_excluding(
        neutral_parameter_refs(module, false).flatten(),
        "",
        store,
        BTreeMap::new(),
        |_| false,
    )?
    .build_bindings(store)
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
    let parameters = neutral_parameter_refs(module, false).flatten();
    let names = parameters
        .iter()
        .map(|(name, _)| name.to_owned())
        .collect::<BTreeSet<_>>();
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
        parameters,
        "",
        store,
        selected,
        |_| false,
    )?
    .build_bindings(store)
}

/// Materializes a set of direct or derived bindings without constructing a
/// residency manager.
///
/// This is useful for loaders that keep one parameter class resident while a
/// separate cache owns another class from the same type-erased checkpoint
/// store. Direct and recipe outputs share a fixed two-completion window;
/// source mappings remain retained through their exact event, and outputs are
/// copied to the execution stream when the streams differ. Mapping-capacity
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
                Err(error) if !pending.is_empty() && is_mapping_capacity_error(&error) => {
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
        let lease = store
            .acquire_lease(TensorReadRequest {
                key: binding.checkpoint_key().to_owned(),
                selection: binding.selection().clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .map_err(crate::backend::mlx::runtime::checkpoint::store::neutral_store_error)?;
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

fn is_mapping_capacity_error(error: &ModuleBindingError) -> bool {
    matches!(
        error,
        ModuleBindingError::WeightStore(WeightStoreError::CapacityExhausted { .. })
            | ModuleBindingError::WeightRecipe(WeightRecipeError::WeightStore(
                WeightStoreError::CapacityExhausted { .. }
            ))
    )
}

/// Populates an unloaded module from materialized local-name bindings while
/// permitting an independently managed parameter class to remain unloaded.
pub fn populate_module_from_arrays_excluding<F>(
    module: &mut (impl ModuleParameters + ?Sized),
    arrays: &BTreeMap<String, Array>,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let expected = {
        let params = module.parameters().flatten();
        params
            .iter()
            .filter(|(name, parameter)| {
                !excluded(name) && is_materialized_module_parameter(name, parameter, &params)
            })
            .map(|(name, _)| name.to_string())
            .collect::<BTreeSet<_>>()
    };
    let mut canonical_expected = BTreeMap::<String, Vec<String>>::new();
    for name in &expected {
        canonical_expected
            .entry(canonical_checkpoint_name(name))
            .or_default()
            .push(name.clone());
    }
    let mut resolved = BTreeMap::<String, &Array>::new();
    let mut unexpected = Vec::new();
    for (name, value) in arrays {
        let target = if expected.contains(name) {
            Some(name.clone())
        } else {
            canonical_expected
                .get(&canonical_checkpoint_name(name))
                .and_then(|candidates| (candidates.len() == 1).then(|| candidates[0].clone()))
        };
        match target {
            Some(target) => {
                if resolved.insert(target, value).is_some() {
                    unexpected.push(name.clone());
                }
            }
            None => unexpected.push(name.clone()),
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
    let mut params = module.parameters_mut().flatten();
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
pub fn populate_module_from_dense_arrays_quantized_excluding<F>(
    module: &mut (impl ModuleParameters + ?Sized),
    arrays: &BTreeMap<String, Array>,
    quantization: WeightQuantization,
    stream: &Stream,
    excluded: F,
) -> Result<(), Error>
where
    F: Fn(&str) -> bool,
{
    quantization.validate()?;
    let config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    {
        let mut parameters = module.parameters_mut().flatten();
        for (name, value) in arrays {
            load_array_quantized_strict(
                &mut parameters,
                name.clone(),
                value.clone(),
                stream,
                quantization,
                &config,
                &mut report,
            )?;
        }
    }
    report.finish_excluding(module, &config, excluded)
}

pub fn build_module_bindings_with_recipes_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, exclude)?
        .build_bindings(store)
}

pub fn build_module_binding_plan_with_recipes_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<ModuleBindingPlan, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    build_flattened_module_binding_plan_with_recipes_excluding(
        module.parameters().flatten(),
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
        .filter(|(name, parameter)| {
            !exclude(name) && is_materialized_module_parameter(name, parameter, &params)
        })
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    local_names.sort();
    recipes.retain(|name, _| {
        if exclude(name) {
            return false;
        }
        params
            .get(name.as_str())
            .is_none_or(|parameter| is_materialized_module_parameter(name, parameter, &params))
    });
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
        let canonical = canonical_checkpoint_name(&destination);
        let authoritative_key = if store.is_authoritative_materialized_key(&destination) {
            Some(destination.clone())
        } else if store.is_authoritative_materialized_key(&canonical) {
            Some(canonical.clone())
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
        } else if keys.contains(&canonical) {
            canonical
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

        let metadata = store
            .source_metadata(&checkpoint_key)
            .map_err(crate::backend::mlx::runtime::checkpoint::store::neutral_store_error)?;
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
pub fn populate_module_from_lease(
    module: &mut impl ModuleParameters,
    lease: &ResidentUnitLease,
) -> Result<(), ModuleBindingError> {
    populate_module_from_lease_excluding(module, lease, |_| false)
}

/// Assigns non-excluded module parameters from a protected resident unit.
pub fn populate_module_from_lease_excluding<F>(
    module: &mut impl ModuleParameters,
    lease: &ResidentUnitLease,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let resident_names = lease.binding_names().collect::<BTreeSet<_>>();
    let expected_names = {
        let params = module.parameters().flatten();
        params
            .iter()
            .filter(|(name, parameter)| {
                !excluded(name) && is_materialized_module_parameter(name, parameter, &params)
            })
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

    let mut params = module.parameters_mut().flatten();
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
        // Derived bindings (including key rewrites and expert stacking) must
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
    /// A semantic binding did not identify exactly one module-local parameter.
    #[error("binding {binding:?} resolves to module-local parameters {candidates:?}, expected exactly one")]
    LocalParameterTarget {
        /// Fully qualified semantic binding.
        binding: String,
        /// Matching module-local parameter names.
        candidates: Vec<String>,
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
    /// Persistent checkpoint inspection failed.
    #[error(transparent)]
    WeightStore(#[from] crate::backend::mlx::runtime::checkpoint::store::WeightStoreError),
    /// Derived-weight metadata validation failed.
    #[error(transparent)]
    WeightRecipe(#[from] crate::backend::mlx::runtime::checkpoint::recipe::WeightRecipeError),
    /// Backend-neutral recipe validation failed.
    #[error(transparent)]
    NeutralRecipe(#[from] eredu_checkpoint::recipe::RecipeError),
    /// A backend-neutral residency declaration was invalid.
    #[error(transparent)]
    ResidencyDeclaration(#[from] eredu_runtime::ResidencyDeclarationError),
    /// Residency binding or lookup failed.
    #[error(transparent)]
    Residency(#[from] crate::backend::mlx::runtime::residency::manager::ResidencyError),
}

impl From<BindingPlanError> for ModuleBindingError {
    fn from(error: BindingPlanError) -> Self {
        Self::BindingPlan(error.to_string())
    }
}
