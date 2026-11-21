//! Installs protected weights and already admitted immutable replacements.

use eredu_core::{checkpoint_mechanisms::CheckpointParameterBackend, BackendFailure};
use eredu_nn::Parameterized;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Unit policy with independent parameter owners and shareable replacements.
#[derive(Clone)]
pub struct SelectiveUnitParameters<T> {
    excluded: Arc<BTreeSet<String>>,
    replacements: Arc<BTreeMap<String, T>>,
}

impl<T> SelectiveUnitParameters<T> {
    /// Creates a policy using the exclusions from portable checkpoint binding.
    pub fn new(excluded: BTreeSet<String>) -> Self {
        Self {
            excluded: Arc::new(excluded),
            replacements: Arc::default(),
        }
    }
}

impl<T: Clone> SelectiveUnitParameters<T> {
    pub(super) fn publish(&mut self, values: &BTreeMap<String, T>, active: bool) {
        self.replacements = if active {
            Arc::new(values.clone())
        } else {
            Arc::default()
        };
    }
}

pub(super) fn install_unit<B, U>(
    unit: &mut U,
    weights: crate::MaterializedUnit<B>,
    policy: Option<&SelectiveUnitParameters<B::Parameter>>,
) -> Result<(), BackendFailure>
where
    B: CheckpointParameterBackend,
    U: Parameterized<B::Parameter>,
    B::Parameter: eredu_nn::Tensor,
{
    crate::bind_materialized_unit_excluding(unit, weights, |id| {
        policy.is_some_and(|policy| policy.excluded.contains(id.as_str()))
    })
    .map_err(BackendFailure::from_error)?;
    if let Some(policy) = policy {
        eredu_nn::publish_parameter_replacements(unit, &policy.replacements);
    }
    Ok(())
}
