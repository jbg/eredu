//! Backend-neutral checkpoint materialization and parameter binding.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeCatalog, RecipeError},
    store::{CheckpointSource, ReadPolicy, StoreError, TensorReadRequest},
};
use eredu_nn::{
    ParameterId, ParameterMetadata, ParameterVisitor, ParameterVisitorMut, Parameterized,
};

use crate::{ParameterBackend, ResidencyDeclarationError, WeightBinding, WeightBindingPlan};

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

impl<B: ParameterBackend> MaterializedUnit<B> {
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
    mut unit: MaterializedUnit<B>,
) -> Result<(), ParameterOrchestrationError<B::ParameterError>>
where
    B: ParameterBackend,
    M: Parameterized<B::Parameter>,
{
    struct Validator<'a, B: ParameterBackend> {
        weights: &'a BTreeMap<ParameterId, B::MaterializedWeight>,
        visited: BTreeMap<ParameterId, ()>,
        error: Option<ParameterOrchestrationError<B::ParameterError>>,
    }

    impl<'a, 'value, B: ParameterBackend> ParameterVisitor<'value, B::Parameter> for Validator<'a, B> {
        fn visit(&mut self, metadata: ParameterMetadata, parameter: &'value B::Parameter) {
            if self.error.is_some() {
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
                return;
            }
            self.visited.insert(metadata.id, ());
        }
    }

    let mut validator = Validator::<B> {
        weights: &unit.weights,
        visited: BTreeMap::new(),
        error: None,
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

    struct Binder<'a, B: ParameterBackend> {
        weights: &'a mut BTreeMap<ParameterId, B::MaterializedWeight>,
        error: Option<ParameterOrchestrationError<B::ParameterError>>,
    }

    impl<'a, 'value, B: ParameterBackend> ParameterVisitorMut<'value, B::Parameter> for Binder<'a, B> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, parameter: &'value mut B::Parameter) {
            if self.error.is_some() {
                return;
            }
            let Some(weight) = self.weights.remove(&metadata.id) else {
                self.error = Some(ParameterOrchestrationError::MissingBinding {
                    parameter: metadata.id,
                });
                return;
            };
            if let Err(error) = B::bind(parameter, weight) {
                self.error = Some(ParameterOrchestrationError::Backend(error));
            }
        }
    }

    let mut binder = Binder::<B> {
        weights: &mut unit.weights,
        error: None,
    };
    module.visit_parameters_mut(&mut binder);
    if let Some(error) = binder.error {
        return Err(error);
    }
    if !unit.weights.is_empty() {
        return Err(ParameterOrchestrationError::UnexpectedBindings {
            parameters: unit.weights.into_keys().collect(),
        });
    }
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
