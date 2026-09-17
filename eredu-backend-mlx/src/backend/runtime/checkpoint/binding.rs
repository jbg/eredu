//! Canonical checkpoint bindings for unloaded module parameter trees.

//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use eredu_checkpoint::store::{ReadPolicy, TensorReadRequest};
use eredu_nn::{ParameterId, Parameterized};
use eredu_runtime::{
    ModuleBindingPlan, ParameterBindingTarget, ReplicatedTextMaterializationTask, WeightBinding,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use safemlx::{Array, Stream};

use crate::{
    backend::nn::shared::{neutral_parameter_refs, MlxNeuralBackend},
    backend::runtime::checkpoint::recipe::{
        plan_binding_reads, recipe_dtype_from_mlx, BindingReadBatch, DirectRecipeRead,
        WeightRecipeError,
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
    local_layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    eredu_runtime::build_exact_replicated_text_bindings(
        module,
        store,
        tasks,
        addressable_parameters,
        local_layout,
        mlx_parameter_binding_target,
        |_task, recipe, source| {
            crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(recipe, source)
        },
    )
    .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))
}

/// Materializes a complete binding unit with at most two pending allocation groups.
///
/// The canonical neutral preflight completes before any payload lease or MLX
/// operation; publication remains atomic in the caller's module binder.
pub fn materialize_module_bindings(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<BTreeMap<String, Array>, ModuleBindingError> {
    let plan = eredu_runtime::preflight_bindings::<MlxNeuralBackend>(store, bindings)
        .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
    let mut arrays = BTreeMap::new();
    let mut pending = VecDeque::with_capacity(MODEL_LOAD_MATERIALIZATION_BUFFERS);
    let context = MlxParameterMaterializationContext::new(source_stream, execution_stream);
    for batch in plan_binding_reads(store, plan.owners())? {
        let (names, materialization) = match batch {
            BindingReadBatch::Direct { bindings, reads } => {
                let inputs = DirectRecipeRead::materialize_many(reads)?;
                let mut prepared = WeightMaterialization::prepare_retained(inputs, Vec::new())?;
                for index in 0..prepared.inputs().len() {
                    let input = &prepared.inputs()[index];
                    let output = if source_stream == execution_stream {
                        input.clone()
                    } else {
                        input
                            .copy(execution_stream)
                            .map_err(WeightRecipeError::from)?
                    };
                    prepared.retain_output(output);
                }
                let names = bindings
                    .iter()
                    .map(|binding| {
                        (
                            binding.name().to_owned(),
                            binding.checkpoint_key().to_owned(),
                        )
                    })
                    .collect();
                (names, prepared.submit_prepared_outputs()?)
            }
            BindingReadBatch::Ordinary(binding) => {
                let materialization = loop {
                    match submit_module_binding(
                        store,
                        binding,
                        source_stream,
                        execution_stream,
                        &context,
                    ) {
                        Ok(materialization) => break materialization,
                        Err(error)
                            if !pending.is_empty() && is_shard_cache_capacity_error(&error) =>
                        {
                            finish_module_binding(&mut pending, &mut arrays)?;
                        }
                        Err(error) => return Err(error),
                    }
                };
                (
                    vec![(
                        binding.name().to_owned(),
                        binding.checkpoint_key().to_owned(),
                    )],
                    materialization,
                )
            }
        };
        pending.push_back((names, materialization));
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
        let pending = super::recipe::prepare_ordinary_recipe(recipe, store, context, false)?;
        let (source, sources) = pending.into_parts();
        let prepared = WeightMaterialization::prepare_retained(vec![source], sources)?;
        let output = if source_stream == execution_stream {
            prepared.inputs()[0].clone()
        } else {
            prepared.inputs()[0]
                .copy(execution_stream)
                .map_err(WeightRecipeError::from)?
        };
        Ok(prepared.submit_outputs(vec![output])?)
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

type PendingModuleBinding = (Vec<(String, String)>, WeightMaterialization);

fn finish_module_binding(
    pending: &mut VecDeque<PendingModuleBinding>,
    arrays: &mut BTreeMap<String, Array>,
) -> Result<(), ModuleBindingError> {
    let (names, materialization) = pending
        .pop_front()
        .expect("non-empty model-loading window has a front");
    let values = materialization.synchronize_many()?;
    assert_eq!(
        names.len(),
        values.len(),
        "one initialized array per planned parameter"
    );
    for ((name, checkpoint_key), value) in names.into_iter().zip(values) {
        if arrays.insert(name.clone(), value).is_some() {
            return Err(ModuleBindingError::DuplicateCheckpointBinding {
                checkpoint_key,
                first: name.clone(),
                second: name,
            });
        }
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
    Some(mlx_binding_target(shape, parameter.as_array().dtype()))
}

fn mlx_binding_target(shape: Vec<usize>, dtype: safemlx::Dtype) -> ParameterBindingTarget {
    ParameterBindingTarget {
        shape,
        dtype: recipe_dtype_from_mlx(dtype),
        // Floating unloaded slots are replaced by the checkpoint value; byte
        // slots accept the same FP8/exponent encodings as actual native slots.
        permitted_source_dtypes: match dtype {
            safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float32 => vec![
                eredu_checkpoint::recipe::RecipeDtype::F16,
                eredu_checkpoint::recipe::RecipeDtype::BF16,
                eredu_checkpoint::recipe::RecipeDtype::F32,
            ],
            safemlx::Dtype::Uint8 => vec![
                eredu_checkpoint::recipe::RecipeDtype::U8,
                eredu_checkpoint::recipe::RecipeDtype::F8E4M3,
                eredu_checkpoint::recipe::RecipeDtype::F8E8M0,
            ],
            _ => Vec::new(),
        },
    }
}

/// Scalar translation of architecture-created metadata slots through the same
/// binding-target worker used by the eventual native parameter visitor.
pub(crate) fn mlx_workspace_binding_targets(
    layouts: &BTreeMap<String, eredu_nn::workspace::WorkspaceLayout>,
) -> Option<BTreeMap<String, ParameterBindingTarget>> {
    use eredu_nn::workspace::WorkspaceDtype;
    layouts
        .iter()
        .map(|(name, layout)| {
            let dtype = match layout.dtype() {
                WorkspaceDtype::Float32 => safemlx::Dtype::Float32,
                WorkspaceDtype::Int32 => safemlx::Dtype::Int32,
                WorkspaceDtype::Uint32 => safemlx::Dtype::Uint32,
                WorkspaceDtype::Uint8 => safemlx::Dtype::Uint8,
                WorkspaceDtype::Bool => safemlx::Dtype::Bool,
            };
            let shape = layout
                .shape()
                .iter()
                .map(|n| usize::try_from(*n).ok())
                .collect::<Option<Vec<_>>>()?;
            Some((name.clone(), mlx_binding_target(shape, dtype)))
        })
        .collect()
}

#[cfg(test)]
mod dtype_tests;

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
    populate_module_from_lease_values(module, lease, excluded, |lease, name| {
        match crate::backend::runtime::residency::manager::clone_original_source_value(lease, name)? {
            Some(value) => Ok(value),
            None => Ok(lease.device_value(name)?.clone()),
        }
    })
}

/// Rebinds an ordinary invocation from the retained values. Its native handle
/// allocations belong to ordinary execution; the finite source-construction
/// aliases remain reserved for the initial admitted parameter binding.
pub(crate) fn populate_module_from_ordinary_lease<M>(
    module: &mut M,
    lease: &ResidentUnitLease,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
{
    populate_module_from_lease_values(module, lease, |_| false, |lease, name| {
        Ok(lease.device_value(name)?.clone())
    })
}

fn populate_module_from_lease_values<M, F, V>(
    module: &mut M,
    lease: &ResidentUnitLease,
    excluded: F,
    clone_value: V,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
    V: Fn(&ResidentUnitLease, &str) -> Result<Array, ModuleBindingError>,
{
    let weights = lease
        .binding_names()
        .map(|name| {
            let id = ParameterId::new(name)
                .map_err(|error| ModuleBindingError::BindingPlan(error.to_string()))?;
            let value = clone_value(lease, name)?;
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

/// Final prepared host binding rows. The caller authenticates the accepted
/// per-unit row limit and owns Q through this call and the populated unit.
pub(crate) fn populate_module_from_original_lease_excluding<M, F>(
    module: &mut M,
    lease: &ResidentUnitLease,
    limit: usize,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    M: Parameterized<crate::MlxTensor>,
    F: Fn(&str) -> bool,
{
    let count = lease.binding_names().count();
    if count > limit {
        return Err(ModuleBindingError::PreparedBindingCapacity);
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .map_err(ModuleBindingError::PreparedBindingReserve)?;
    for name in lease.binding_names() {
        if rows.len() == count {
            return Err(ModuleBindingError::PreparedBindingCapacity);
        }
        let value = lease
            .device_value(name)?
            .try_clone_handle()
            .map_err(ModuleBindingError::PreparedBindingNative)?;
        rows.push(eredu_runtime::PreparedParameterBinding::new(
            name,
            crate::MlxTensor::from_array(value),
        ));
    }
    if rows.len() != count {
        return Err(ModuleBindingError::PreparedBindingCapacity);
    }
    eredu_runtime::bind_prepared_parameter_values(
        module,
        &mut rows,
        |id| excluded(id.as_str()),
        MlxNeuralBackend::validate_prepared_bind,
        <MlxNeuralBackend as eredu_runtime::ParameterBackend>::bind,
    )
    .map_err(ModuleBindingError::PreparedBinding)
}

pub(crate) fn original_parameter_binding_control_bytes(rows: usize) -> Option<usize> {
    use std::mem::size_of;
    eredu_runtime::prepared_parameter_binding_control_bytes::<crate::MlxTensor,crate::MlxTensor,
        crate::backend::nn::shared::MlxParameterError>(rows)?
        .checked_add(MlxNeuralBackend::prepared_binding_visit_control_bytes(rows)?)?
        .checked_add(size_of::<std::collections::TryReserveError>())?
        .checked_add(size_of::<Result<(),std::collections::TryReserveError>>())?
        .checked_add(size_of::<Result<(),ModuleBindingError>>())?
        .checked_add(size_of::<Result<Array,safemlx::error::Exception>>())?
        .checked_add(size_of::<&ResidentUnitLease>())?
        .checked_add(size_of::<&mut crate::MlxTensor>())?
        .checked_add(size_of::<&str>())?
        .checked_add(size_of::<[usize;3]>())?
        .checked_add(size_of::<&dyn Fn(&str)->bool>())?
        .checked_add(size_of::<<crate::backend::runtime::residency::manager::ResidentLeaseStorage
            as eredu_runtime::ResidencyLeaseStorage>::BindingNames<'static>>())
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
    /// The accepted finite binding source changed population.
    #[error("prepared parameter binding capacity differs")]
    PreparedBindingCapacity,
    /// Final finite row storage could not be reserved.
    #[error("prepared parameter binding allocation failed")]
    PreparedBindingReserve(#[source] std::collections::TryReserveError),
    /// Native handle sharing failed under the current scope.
    #[error(transparent)]
    PreparedBindingNative(safemlx::error::Exception),
    /// Shared prepublication validation refused the finite binding source.
    #[error(transparent)]
    PreparedBinding(
        eredu_runtime::PreparedParameterBindingError<crate::backend::nn::shared::MlxParameterError>,
    ),

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
