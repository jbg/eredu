//! Canonical checkpoint bindings for unloaded module parameter trees.
//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use safemlx::{
    module::{FlattenedModuleParamRef, ModuleParameters},
    Array, Dtype, Stream,
};

use crate::{
    error::Error,
    runtime::checkpoint::binding_plan::{BindingPlan, BindingPlanError, PlannedBinding},
    runtime::checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype, WeightRecipeError},
    runtime::checkpoint::store::{
        TensorSelection, WeightMaterialization, WeightReadPolicy, WeightStore, WeightStoreError,
    },
    runtime::checkpoint::{
        load::{load_array_quantized_strict, StrictLoadConfig, StrictLoadReport},
        quantization::WeightQuantization,
    },
    runtime::residency::manager::{ResidentUnitLease, WeightBinding},
};

const MODEL_LOAD_MATERIALIZATION_BUFFERS: usize = 2;

/// Converts a module parameter name to its canonical checkpoint spelling.
///
/// Quantized modules wrap their packed matrix (and ordinary biased modules
/// wrap their bias) in an `inner` module. MLX-compatible checkpoints omit that
/// implementation detail while retaining companion `.scales` and `.biases`
/// tensors unchanged.
pub fn canonical_checkpoint_name(parameter_name: &str) -> String {
    parameter_name
        .replace(".inner.weight", ".weight")
        .replace(".inner.bias", ".bias")
}

/// Maps custom packed-bank companion fields onto the canonical overlay name.
pub(crate) fn packed_companion_checkpoint_name(parameter_name: &str) -> Option<String> {
    [("_scales", "scales"), ("_biases", "biases")]
        .into_iter()
        .find_map(|(runtime_suffix, checkpoint_component)| {
            parameter_name
                .strip_suffix(runtime_suffix)
                .map(|prefix| format!("{prefix}.{checkpoint_component}"))
        })
}

/// Whether a flattened module parameter has checkpoint-backed storage.
///
/// Native GGML modules expose a one-element `scales` parameter solely to keep
/// the public quantized-module shape compatible with affine quantization. The
/// packed `u8` weight contains its own quantization metadata, so that sentinel
/// must not become a residency or distributed-planning member. Affine packed
/// weights use a non-`u8` storage dtype and retain their real companions.
pub(crate) fn is_materialized_module_parameter(
    name: &str,
    parameter: &Array,
    parameters: &FlattenedModuleParamRef<'_>,
) -> bool {
    let weight_name = if name == "scales" {
        Some("inner.weight".to_string())
    } else {
        name.strip_suffix(".scales")
            .map(|prefix| format!("{prefix}.inner.weight"))
    };
    let weight_dtype = weight_name
        .as_deref()
        .and_then(|weight_name| parameters.get(weight_name))
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
    store: &dyn WeightStore,
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
    store: &dyn WeightStore,
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
    store: &dyn WeightStore,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    build_module_binding_plan_with_recipes(module, prefix, store, recipes)?.build_bindings(store)
}

/// A declarative module binding plan plus fully qualified logical targets.
pub(crate) struct ModuleBindingPlan {
    plan: BindingPlan,
    logical_targets: BTreeMap<String, String>,
}

impl ModuleBindingPlan {
    pub(crate) fn build_bindings(
        &self,
        store: &dyn WeightStore,
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

pub(crate) fn build_module_binding_plan_with_recipes(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<ModuleBindingPlan, ModuleBindingError> {
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, |_| false)
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
pub(crate) fn materialize_module_bindings(
    store: &dyn WeightStore,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<BTreeMap<String, Array>, ModuleBindingError> {
    let mut arrays = BTreeMap::new();
    let mut pending = VecDeque::with_capacity(MODEL_LOAD_MATERIALIZATION_BUFFERS);
    for binding in bindings {
        let materialization = loop {
            match submit_module_binding(store, binding, source_stream, execution_stream) {
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
    Ok(arrays)
}

fn submit_module_binding(
    store: &dyn WeightStore,
    binding: &WeightBinding,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<WeightMaterialization, ModuleBindingError> {
    if let Some(recipe) = binding.recipe() {
        let pending = recipe.prepare_materialization(store, source_stream)?;
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
        Ok(store
            .acquire_with_policy(
                binding.checkpoint_key(),
                binding.selection().clone(),
                WeightReadPolicy::RequireBounded,
            )?
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
pub(crate) fn populate_module_from_arrays_excluding<F>(
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
    let actual = arrays.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).cloned().collect::<Vec<_>>();
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
        let value = arrays
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
pub(crate) fn populate_module_from_dense_arrays_quantized_excluding<F>(
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

pub(crate) fn build_module_bindings_with_recipes_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    build_module_binding_plan_with_recipes_excluding(module, prefix, store, recipes, exclude)?
        .build_bindings(store)
}

pub(crate) fn build_module_binding_plan_with_recipes_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<ModuleBindingPlan, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let params = module.parameters().flatten();
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
                let expected_dtype = RecipeDtype::from(parameter.dtype());
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

        let metadata = store.metadata(&checkpoint_key)?;
        let expected_shape = parameter
            .shape()
            .iter()
            .map(|&dimension| usize::try_from(dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ModuleBindingError::InvalidModuleShape {
                parameter: destination.clone(),
                shape: parameter.shape().to_vec(),
            })?;
        if metadata.shape != expected_shape {
            return Err(ModuleBindingError::ShapeMismatch {
                checkpoint_key,
                parameter: destination,
                expected: expected_shape,
                actual: metadata.shape,
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
        let value = lease.array(name)?;
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
    WeightStore(#[from] crate::runtime::checkpoint::store::WeightStoreError),
    /// Derived-weight metadata validation failed.
    #[error(transparent)]
    WeightRecipe(#[from] crate::runtime::checkpoint::recipe::WeightRecipeError),
    /// Residency binding or lookup failed.
    #[error(transparent)]
    Residency(#[from] crate::runtime::residency::manager::ResidencyError),
}

impl From<BindingPlanError> for ModuleBindingError {
    fn from(error: BindingPlanError) -> Self {
        Self::BindingPlan(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use safemlx::{
        module::{ModuleParameters, Param},
        native_quantization::NativeQuantizationFormat,
        nn,
        ops::GgufCheckpoint,
        Array, Device, DeviceType, Dtype, ExecutionContext,
    };
    use safemlx_gguf::GgmlType;

    use super::*;
    use crate::{
        nn::linear::unloaded_maybe_quantized_linear,
        runtime::checkpoint::quantization::AffineQuantization,
        runtime::checkpoint::store::{GgufWeightStore, SafetensorsWeightStore},
        test_utils::SyntheticGguf,
    };

    fn cpu() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn packed_e4m3_recipe_matches_u8_runtime_storage_only() {
        assert!(recipe_dtype_matches(&RecipeDtype::U8, &RecipeDtype::F8E4M3));
        assert!(!recipe_dtype_matches(
            &RecipeDtype::U8,
            &RecipeDtype::F8E5M2
        ));
        assert!(!recipe_dtype_matches(
            &RecipeDtype::F32,
            &RecipeDtype::F8E4M3
        ));
        assert!(recipe_dtype_matches(&RecipeDtype::F32, &RecipeDtype::BF16));
        assert!(recipe_dtype_matches(&RecipeDtype::BF16, &RecipeDtype::F16));
    }

    #[test]
    fn only_native_u8_scale_sentinels_are_non_materialized() {
        assert!(is_native_scale_sentinel("scales", &[1], Some(Dtype::Uint8)));
        assert!(is_native_scale_sentinel(
            "projection.scales",
            &[1],
            Some(Dtype::Uint8)
        ));
        assert!(!is_native_scale_sentinel(
            "scales",
            &[1],
            Some(Dtype::Uint32)
        ));
        assert!(!is_native_scale_sentinel(
            "scales",
            &[2],
            Some(Dtype::Uint8)
        ));
        assert!(!is_native_scale_sentinel(
            "weight_scale_inv",
            &[1],
            Some(Dtype::Uint8)
        ));
    }

    #[test]
    fn dense_binding_validates_names_shapes_and_exact_bytes() {
        let context = cpu();
        let linear = nn::Linear::unloaded(2, 3, true, Dtype::Float32, context.stream()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(&[0.1f32; 6], &[3, 2]);
        let bias = Array::from_slice(&[0.2f32; 3], &[3]);
        Array::save_safetensors(
            [("proj.weight", &weight), ("proj.bias", &bias)],
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let bindings = build_module_bindings(&linear, "proj", &store).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(binding_bytes(&bindings).unwrap(), 36);
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.checkpoint_key())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["proj.bias", "proj.weight"])
        );
    }

    #[test]
    fn quantized_binding_preserves_packed_companions() {
        let context = cpu();
        let quantization = AffineQuantization::new(16, 4).unwrap().into();
        let module =
            unloaded_maybe_quantized_linear(32, 2, true, Some(quantization), context.stream())
                .unwrap();
        let params = module.parameters().flatten();
        let arrays = params
            .iter()
            .map(|(name, value)| (canonical_checkpoint_name(&format!("proj.{name}")), *value))
            .collect::<Vec<_>>();
        let dir = tempfile::tempdir().unwrap();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let bindings = build_module_bindings(&module, "proj", &store).unwrap();
        let keys = bindings
            .iter()
            .map(|binding| binding.checkpoint_key())
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("proj.weight"));
        assert!(keys.contains("proj.scales"));
        assert!(keys.contains("proj.biases"));
        assert!(keys.contains("proj.bias"));
    }

    #[test]
    fn native_q4k_binding_omits_compatibility_scale_sentinel() {
        let context = cpu();
        let mut module = nn::QuantizedLinear {
            group_size: 256,
            bits: 144,
            mode: safemlx::ops::QuantizationMode::Affine,
            native: None,
            native_format: Some(NativeQuantizationFormat::GgufQ4K),
            native_endian: safemlx_gguf::Endian::Little,
            native_columns: 256,
            scales: Param::new(Array::from_slice(&[0.0f32], &[1])),
            biases: Param::new(None),
            inner: nn::Linear {
                weight: Param::new(Array::from_slice(&[0u8; 288], &[2, 144])),
                bias: Param::new(None),
            },
        };
        let arrays = HashMap::from([(
            "proj.weight".to_string(),
            Array::from_slice(&[0.0f32; 512], &[2, 256]),
        )]);
        let fixture = SyntheticGguf::with_packed_tensors(&arrays, &HashMap::new(), |name, _| {
            (name == "proj.weight").then_some(GgmlType::Q4K)
        });
        let checkpoint = GgufCheckpoint::open(fixture.path()).unwrap();
        let store = GgufWeightStore::new_for_test(checkpoint, str::to_string).unwrap();

        assert_eq!(full_parameter_names(&module, "proj"), ["proj.inner.weight"]);
        let bindings = build_module_bindings(&module, "proj", &store).unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].name(), "inner.weight");
        assert_eq!(bindings[0].checkpoint_key(), "proj.weight");

        let materialized =
            materialize_module_bindings(&store, &bindings, context.stream(), context.stream())
                .unwrap();
        populate_module_from_arrays_excluding(&mut module, &materialized, |_| false).unwrap();
        assert_eq!(module.inner.weight.value.dtype(), Dtype::Uint8);
        assert_eq!(module.scales.value.shape(), &[1]);
    }

    #[test]
    fn model_loading_window_drains_at_a_one_shard_mapping_bound() {
        let context = cpu();
        let linear = nn::Linear::unloaded(2, 3, true, Dtype::Float32, context.stream()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(&[0.1f32; 6], &[3, 2]);
        let bias = Array::from_slice(&[0.2f32; 3], &[3]);
        Array::save_safetensors(
            [("proj.weight", &weight)],
            None,
            dir.path().join("weight.safetensors"),
        )
        .unwrap();
        Array::save_safetensors(
            [("proj.bias", &bias)],
            None,
            dir.path().join("bias.safetensors"),
        )
        .unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"proj.bias":"bias.safetensors","proj.weight":"weight.safetensors"}}"#,
        )
        .unwrap();
        let store = SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap();
        let bindings = build_module_bindings(&linear, "proj", &store).unwrap();

        let arrays =
            materialize_module_bindings(&store, &bindings, context.stream(), context.stream())
                .unwrap();
        assert_eq!(
            arrays["weight"].evaluated().unwrap().as_slice::<f32>(),
            &[0.1; 6]
        );
        assert_eq!(
            arrays["bias"].evaluated().unwrap().as_slice::<f32>(),
            &[0.2; 3]
        );
        assert_eq!(store.diagnostics().unwrap().currently_mapped_shards, 1);
    }

    #[test]
    fn missing_and_shape_mismatch_are_rejected() {
        let context = cpu();
        let linear = nn::Linear::unloaded(2, 3, true, Dtype::Float32, context.stream()).unwrap();
        let missing = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(&[0.1f32; 6], &[3, 2]);
        Array::save_safetensors(
            [("proj.weight", &weight)],
            None,
            missing.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(missing.path()).unwrap();
        assert!(matches!(
            build_module_bindings(&linear, "proj", &store),
            Err(ModuleBindingError::MissingParameter { .. })
        ));

        let mismatch = tempfile::tempdir().unwrap();
        let wrong = Array::from_slice(&[0.1f32; 4], &[2, 2]);
        let bias = Array::from_slice(&[0.2f32; 3], &[3]);
        Array::save_safetensors(
            [("proj.weight", &wrong), ("proj.bias", &bias)],
            None,
            mismatch.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(mismatch.path()).unwrap();
        assert!(matches!(
            build_module_bindings(&linear, "proj", &store),
            Err(ModuleBindingError::ShapeMismatch { .. })
        ));
    }
}
