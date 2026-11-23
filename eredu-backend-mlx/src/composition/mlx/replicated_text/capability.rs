use super::*;
use eredu_core::cache::StateTensorDimension;
use eredu_core::preparation::StateComponentPlacement;
use eredu_core::preparation::{
    BackendMechanismFacts, ReplicatedTextMechanismSupport, StateLifecycleCapabilities,
    StateStorageDtype,
};

pub(crate) const GROUPED_OPERATION_CAPABILITIES: [GroupedOperationRequirement; 5] = [
    GroupedOperationRequirement::Linear,
    GroupedOperationRequirement::GatedProduct,
    GroupedOperationRequirement::GatedProductTensorParallelPartial,
    GroupedOperationRequirement::Relu2,
    GroupedOperationRequirement::Relu2TensorParallelPartial,
];

/// Exact, side-effect-free MLX mechanism facts.
pub(crate) struct MlxReplicatedTextSupport;

impl ReplicatedTextMechanismSupport for MlxReplicatedTextSupport {
    fn facts(&self, _: StateComponentPlacement) -> BackendMechanismFacts {
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
                .with_prompt_cache(true)
                .with_observation_retention(true),
        )
        .with_session(eredu_core::SessionCapabilities::new(true, true, true))
        .with_grouped_operations(GROUPED_OPERATION_CAPABILITIES)
        .with_indexed_movement(true)
        .with_addressable_storage(
            eredu_core::preparation::rows::NativeRowLookupSupport::row_storage(
                &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
            ),
        )
        .with_prompt_cache(true)
        .with_exact_completion(true)
        .with_chunked_prefill(true)
    }

    fn supports_direct(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        supports_direct(descriptor)
    }

    fn prefers_original_encoding(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        !cfg!(feature = "cuda")
            && descriptor.logical_shape().len() == 2
            && matches!(
                descriptor.executable(),
                eredu_checkpoint::LinearFormat::GgufIQuant {
                    ggml_type,
                    endian: eredu_gguf::Endian::Little,
                } if ggml_type != eredu_gguf::GgmlType::NvFp4
                    && eredu_gguf::packed_dot::supports_packed_dot(ggml_type)
            )
    }

    fn supports_transform(&self, descriptor: &WeightLoweringDescriptor) -> bool {
        cfg!(target_endian = "little") && descriptor.supports_word_transform()
    }

    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<StateStorageDtype> {
        StateStorageDtype::from_floating_source(source, |_| true)
    }

    fn supports_state_component(
        &self,
        _shape: &[StateTensorDimension],
        _storage_dtype: StateStorageDtype,
        placement: StateComponentPlacement,
    ) -> bool {
        matches!(
            placement,
            StateComponentPlacement::Device | StateComponentPlacement::Paged
        )
    }
}
