//! Cold-path architecture/backend composition selected by public loaders.

#[cfg(test)]
pub(crate) mod grouped_provider;
// Standalone exchange harnesses are available only to crate-internal validation.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod expert_dispatch;

use safemlx::error::Exception;
use safemlx::Array;

pub(crate) use crate::backend::nn::shared::MlxNeuralBackend;

impl From<eredu_runtime::ParameterBankLoadOptions>
    for crate::backend::runtime::residency::parameter_bank::ParameterBankOptions
{
    fn from(options: eredu_runtime::ParameterBankLoadOptions) -> Self {
        Self::new(
            options.offload(),
            options.compact_bank_scratch_bytes(),
            options.prefill_compact_bank_target_bytes(),
        )
        .expect("architecture-owned expert-cache options were already validated")
    }
}

/// Adapts public MLX-array observation to the neutral tensor/error contract.
pub(crate) struct NeutralActivationObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl<'a> NeutralActivationObserver<'a> {
    pub(crate) fn new(
        inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Self {
        Self { inner }
    }
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralActivationObserver<'_>
{
    fn observe(&mut self, path: &str, value: &crate::MlxTensor) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value.as_array())
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        self.inner
            .intervene(path, value.as_array())
            .map(|value| value.map(crate::MlxTensor::from_array))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: routing.selected_experts.as_array(),
                selected_scores: routing.selected_scores.as_array(),
                coefficients: routing.coefficients.as_array(),
                routed_output: routing.routed_output.as_array(),
                local_routed_output: routing.local_routed_output.map(crate::MlxTensor::as_array),
                reduced_routed_output: routing
                    .reduced_routed_output
                    .map(crate::MlxTensor::as_array),
                shared_output: routing.shared_output.map(crate::MlxTensor::as_array),
                combined_output: routing.combined_output.map(crate::MlxTensor::as_array),
                expert_count: routing.expert_count,
            })
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

#[cfg(test)]
fn select_architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    mut owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
    mut owns_expert: impl FnMut(eredu_runtime::ParameterBankKey) -> bool,
) -> impl Iterator<Item = eredu_architectures::ExpertResidencyUnit> {
    units.into_iter().filter(move |unit| {
        owns_unit(unit.owner_group(), unit.owner_unit())
            && match unit.distribution() {
                eredu_architectures::ExpertResidencyDistribution::ExpertParallel => {
                    owns_expert(unit.identity())
                }
                eredu_architectures::ExpertResidencyDistribution::Replicated => true,
                _ => false,
            }
    })
}

pub mod mlx;
pub mod moshi;

#[cfg(test)]
pub(crate) mod checkpoint_fixtures;

#[cfg(test)]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;

#[cfg(test)]
mod expert_selection_tests {
    use eredu_architectures::{
        ExpertParameterRecipe, ExpertParameterRole, ExpertResidencyDistribution,
        ExpertResidencyUnit,
    };
    use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
    use eredu_runtime::{ExecutionGroupId, ParameterBankKey};

    fn expert_unit(
        identity: ParameterBankKey,
        group: &str,
        owner_unit: usize,
        distribution: ExpertResidencyDistribution,
    ) -> ExpertResidencyUnit {
        let parameter = ExpertParameterRecipe::new(
            "weight",
            format!("{group}.{owner_unit}.weight"),
            DerivedWeightRecipe::source("source", TensorSelection::Full),
            ExpertParameterRole::Preserved,
        )
        .unwrap();
        ExpertResidencyUnit::new(
            identity,
            ExecutionGroupId::new(group).unwrap(),
            owner_unit,
            format!("{group}.{owner_unit}"),
            distribution,
            [parameter],
        )
        .unwrap()
    }

    #[test]
    fn expert_selection_uses_owner_address_and_distribution_before_lowering() {
        let units = vec![
            expert_unit(
                ParameterBankKey::new(1, 0),
                "target",
                7,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(7, 1),
                "mtp.0",
                1,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(9, 2),
                "mtp.0",
                1,
                ExpertResidencyDistribution::Replicated,
            ),
        ];

        let selected = super::select_architecture_expert_units(
            units,
            |group, unit| group.as_str() == "mtp.0" && unit == 1,
            |identity| identity.member() == 1,
        )
        .map(|unit| unit.identity())
        .collect::<Vec<_>>();

        assert_eq!(
            selected,
            vec![ParameterBankKey::new(7, 1), ParameterBankKey::new(9, 2)]
        );
    }
}
