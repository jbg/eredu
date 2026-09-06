use super::*;
use eredu_core::cache::{StateComponentPolicy, StateTensorDtype};
use eredu_runtime::{
    synthesize_replicated_text_capabilities, BackendMechanismFacts, ReplicatedTextMechanismSupport,
    StateLifecycleCapabilities,
};

pub(crate) const GROUPED_OPERATION_CAPABILITIES: [GroupedOperationRequirement; 4] = [
    GroupedOperationRequirement::GatedProduct,
    GroupedOperationRequirement::GatedProductTensorParallelPartial,
    GroupedOperationRequirement::Relu2,
    GroupedOperationRequirement::Relu2TensorParallelPartial,
];

/// Exact, side-effect-free MLX mechanism facts.
pub(crate) struct MlxReplicatedTextSupport;

impl ReplicatedTextMechanismSupport for MlxReplicatedTextSupport {
    fn facts(&self, state_policy: &CacheResidencyPolicy) -> BackendMechanismFacts {
        BackendMechanismFacts::new(
            MlxNeuralBackend::OPERATOR_CAPABILITIES,
            [
                WeightResidencyMechanism::Resident,
                WeightResidencyMechanism::Windowed,
                WeightResidencyMechanism::DiskStreamed,
            ],
            StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_prompt_cache(matches!(state_policy, CacheResidencyPolicy::Paged(_)))
                .with_observation_retention(true),
        )
        .with_session(eredu_core::SessionCapabilities::new(true, true, true))
        .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
        .with_indexed_movement(true)
        .with_addressable_storage(
            eredu_runtime::AddressableStorageCapabilities::new(true, true, true, u64::MAX)
                .with_tiers(eredu_runtime::AddressableStorageTiers::new(
                    true, true, true,
                )),
        )
        .with_prompt_cache(true)
        .with_exact_completion(true)
    }

    fn supports_direct(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        supports_direct(descriptor)
    }

    fn supports_transform(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        supports_transform(descriptor)
    }

    fn supports_state_component(
        &self,
        component: &StateComponentPolicy,
        placement: StateComponentPlacement,
    ) -> bool {
        matches!(
            placement,
            StateComponentPlacement::Device | StateComponentPlacement::Paged
        ) && matches!(
            component.dtype(),
            StateTensorDtype::Floating
                | StateTensorDtype::Float32
                | StateTensorDtype::Int32
                | StateTensorDtype::Uint32
        )
    }
}

pub(crate) fn capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
) -> BackendMechanismCapabilities {
    synthesize_replicated_text_capabilities(requirements, request, &MlxReplicatedTextSupport)
}
