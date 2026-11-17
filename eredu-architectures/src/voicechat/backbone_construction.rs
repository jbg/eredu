//! Exact retained binding plans for the ordinary Nemotron runtime.

use eredu_core::residency::ParameterConversionRetentionEligibility;
use eredu_nn::{AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor};
use eredu_runtime::{
    ComponentDefinition, LayerRuntimeState, LayerWeightResidency, LocalComponentPolicy,
    PreparedComponent, PreparedLocalComponent, RuntimeStateComponents,
};

use super::*;

/// A construction failure returns no partially loaded runtime.
#[derive(Debug, thiserror::Error)]
pub enum BackboneConstructionError {
    /// Retained preparation does not describe the requested target.
    #[error(transparent)]
    Preparation(#[from] BackbonePreparationError),
    /// Ordinary Nemotron module or runtime construction failed.
    #[error(transparent)]
    Architecture(#[from] eredu_nn::Error),
}

impl PreparedBackboneParameters {
    /// Prepares exact target resources from the source-free Nemotron declaration.
    /// No native module, tensor, device or checkpoint payload is required.
    pub fn prepare_bindings(
        &self,
    ) -> Result<PreparedComponent<Arc<ComponentDefinition>>, BackbonePreparationError> {
        PreparedComponent::prepare(
            "voicechat/backbone/target",
            Arc::clone(self.declaration.definition()),
            self.source.clone(),
            |member| {
                self.parameters
                    .get(member.target())
                    .map(|parameter| parameter.recipe.clone())
                    .ok_or_else(|| {
                        BackbonePreparationError::Configuration(format!(
                            "missing backbone parameter {}",
                            member.target()
                        ))
                    })
            },
        )
    }

    /// Constructs the local backbone from retained resources and the admitted residency.
    pub fn prepare_local<B, S>(
        &self,
        residency: LayerWeightResidency,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedLocalComponent<nemotron_h::LayeredModel<B>>, BackboneConstructionError>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let prepared = self.prepare_bindings()?;
        let model = nemotron_h::CompleteNemotronKernel::<B>::from_declaration(
            Arc::clone(&self.declaration),
            context,
        )?;
        let prepared = prepared.with_storage_preferences::<B, S, _>(&model, context)?;
        Ok(PreparedLocalComponent::new(
            model,
            prepared,
            LocalComponentPolicy {
                residency,
                state: None,
                conversion_retention: None,
                conversion_eligibility: ParameterConversionRetentionEligibility::Eligible,
            },
        )?)
    }
}
