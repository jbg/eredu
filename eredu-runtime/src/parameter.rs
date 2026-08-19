//! Backend-neutral checkpoint materialization and parameter binding.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::RecipeError,
    store::{CheckpointSource, ReadPolicy, StoreError, TensorReadRequest},
};
use eredu_nn::{ParameterId, ParameterMetadata, ParameterVisitorMut, Parameterized};

use crate::{ParameterBackend, WeightBinding};

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
    let mut weights = BTreeMap::new();
    for binding in bindings {
        let inferred = binding.source_recipe().infer(source)?;
        if inferred.byte_len() != binding.expected_bytes() {
            return Err(ParameterOrchestrationError::ByteMismatch {
                parameter: binding.name().to_owned(),
                expected: binding.expected_bytes(),
                actual: inferred.byte_len(),
            });
        }
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
