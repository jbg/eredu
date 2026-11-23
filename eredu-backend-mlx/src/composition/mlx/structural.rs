//! MLX architecture binding against portable checkpoint catalogs.

/// Cold MLX facts supplied to the portable admission selector.
pub(crate) fn preparation_mechanism_capabilities() -> eredu_core::PreparationMechanismCapabilities {
    use eredu_core::preparation::{ReplicatedTextMechanismSupport, StateComponentPlacement};

    let distributed = cfg!(feature = "distributed");
    let modalities = eredu_core::InputModalities {
        text: true,
        image: cfg!(feature = "image"),
        audio: cfg!(feature = "audio"),
        video: cfg!(feature = "image"),
    };
    eredu_core::PreparationMechanismCapabilities::new(true, true)
        .with_safetensors_quantization(true, true)
        .with_gguf_quantized_loading(true)
        .with_parallel_axis(eredu_core::ParallelAxis::Tensor, distributed)
        .with_parallel_axis(eredu_core::ParallelAxis::Pipeline, distributed)
        .with_parallel_axis(eredu_core::ParallelAxis::Expert, distributed)
        .with_parallel_axis(eredu_core::ParallelAxis::Data, distributed)
        .with_input_modalities(modalities)
        .with_replicated_text_facts(
            &super::replicated_text::MlxReplicatedTextSupport
                .facts(StateComponentPlacement::Device),
        )
}
