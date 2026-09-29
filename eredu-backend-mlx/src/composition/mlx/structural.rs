//! MLX architecture binding against portable checkpoint catalogs.

/// Cold MLX facts supplied to the portable admission selector.
pub(crate) const fn preparation_mechanism_capabilities(
) -> eredu_core::PreparationMechanismCapabilities {
    let modalities = eredu_core::InputModalities {
        text: true,
        image: cfg!(feature = "image"),
        audio: cfg!(feature = "audio"),
        video: cfg!(feature = "image"),
    };
    eredu_core::PreparationMechanismCapabilities::new(true, true)
        .with_safetensors_quantization(true, true)
        .with_gguf_quantized_loading(true)
        .with_residency(eredu_core::ResidencyRequest::FullyResident, true)
        .with_residency(eredu_core::ResidencyRequest::LayerwiseHost, true)
        .with_residency(eredu_core::ResidencyRequest::DenseDiskStream, true)
        .with_residency(
            eredu_core::ResidencyRequest::AddressableParameterBanks,
            true,
        )
        .with_parallel_axis(eredu_core::ParallelAxis::Tensor, true)
        .with_parallel_axis(eredu_core::ParallelAxis::Pipeline, true)
        .with_parallel_axis(eredu_core::ParallelAxis::Expert, true)
        .with_parallel_axis(eredu_core::ParallelAxis::Data, true)
        .with_input_modalities(modalities)
        .with_exact_completion(true)
        .with_session(eredu_core::SessionCapabilities::new(true, true, true))
}
