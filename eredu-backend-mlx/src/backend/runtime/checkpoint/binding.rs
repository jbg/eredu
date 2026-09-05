//! Canonical checkpoint bindings for unloaded module parameter trees.

//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use eredu_checkpoint::store::{ReadPolicy, TensorReadRequest};
use eredu_nn::{ParameterId, Parameterized};
use eredu_runtime::{
    ModuleBindingPlan, ParameterBindingTarget, ReplicatedTextMaterializationTask, WeightBinding,
    WeightBindingPlan,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use safemlx::{Array, Stream};

use crate::{
    backend::nn::shared::{neutral_parameter_refs, MlxNeuralBackend},
    backend::runtime::checkpoint::recipe::{
        recipe_dtype_from_mlx, MlxWeightRecipeExt, WeightRecipeError,
    },
    backend::runtime::checkpoint::store::{
        MlxParameterMaterializationContext, WeightMaterialization,
    },
    backend::runtime::residency::manager::ResidentUnitLease,
};
use eredu_checkpoint::recipe::DerivedWeightRecipe;

const MODEL_LOAD_MATERIALIZATION_BUFFERS: usize = 2;

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
    build_module_binding_plan_with_recipes(module, prefix, store, recipes)?
        .build_bindings(store)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
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
    build_module_binding_plan_with_recipes_excluding(module, "", store, selected, exclude)?
        .build_bindings(store)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
}

/// Resolves one exact replicated-text task partition to native module handles.
///
/// The task partition is authoritative for source identity, recipes, lowering,
/// and transformed companion identities. Module traversal only verifies that
/// every selected destination has one native handle, while the checkpoint
/// source supplies byte metadata without opening payloads.
/// Adapts MLX parameter geometry and the intrinsic MXFP4 recipe conversion to
/// the canonical backend-neutral exact-task binder.
pub fn build_mlx_exact_replicated_text_bindings<M>(
    module: &M,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    tasks: &[&ReplicatedTextMaterializationTask],
    addressable_parameters: &BTreeSet<String>,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    eredu_runtime::build_exact_replicated_text_bindings(
        module,
        store,
        tasks,
        addressable_parameters,
        mlx_parameter_binding_target,
        |_task, recipe, source| {
            crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
        },
    )
    .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
}

/// Materializes a complete binding unit with a fixed two-completion window.
///
/// The canonical neutral preflight completes before any payload lease or MLX
/// operation; publication remains atomic in the caller's module binder.
pub fn materialize_module_bindings(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<BTreeMap<String, Array>, ModuleBindingError> {
    eredu_runtime::preflight_bindings::<MlxNeuralBackend>(store, bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    let plan = WeightBindingPlan::new(bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
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
    let weights = arrays
        .iter()
        .map(|(name, value)| {
            let id = ParameterId::new(name.clone())
                .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
            Ok((id, crate::MlxTensor::from_array(value.clone())))
        })
        .collect::<Result<Vec<_>, ModuleBindingError>>()?;
    let unit = eredu_runtime::MaterializedUnit::<MlxNeuralBackend>::try_from_weights(weights)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    eredu_runtime::bind_materialized_unit_excluding::<MlxNeuralBackend, M, _>(module, unit, |id| {
        excluded(id.as_str())
    })
    .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
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
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
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
    eredu_runtime::build_module_binding_plan(
        module,
        prefix,
        store,
        recipes,
        BTreeSet::new(),
        exclude,
        mlx_parameter_binding_target,
    )
    .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
}

fn mlx_parameter_binding_target(parameter: &crate::MlxTensor) -> Option<ParameterBindingTarget> {
    let shape = parameter
        .as_array()
        .shape()
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    Some(ParameterBindingTarget {
        shape,
        dtype: recipe_dtype_from_mlx(parameter.as_array().dtype()),
        permitted_source_dtypes: Vec::new(),
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
    let weights = lease
        .binding_names()
        .map(|name| {
            let id = ParameterId::new(name)
                .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
            let value = lease.device_value(name)?.clone();
            Ok((id, crate::MlxTensor::from_array(value)))
        })
        .collect::<Result<Vec<_>, ModuleBindingError>>()?;
    let unit = eredu_runtime::MaterializedUnit::<MlxNeuralBackend>::try_from_weights(weights)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    eredu_runtime::bind_materialized_unit_excluding::<MlxNeuralBackend, M, _>(module, unit, |id| {
        excluded(id.as_str())
    })
    .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
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

#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_os = "macos"))
))]
#[allow(
    clippy::items_after_test_module,
    reason = "binding tests stay adjacent to the binding planners they exercise"
)]
mod tests {
    use super::*;
    use eredu_checkpoint::store::MemoryWeightStore;
    use eredu_nn::{ParameterMetadata, ParameterSpec, ParameterVisitor, ParameterVisitorMut};

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
}
