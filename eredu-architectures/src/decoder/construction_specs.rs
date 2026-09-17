//! Closed neural specification copies through the caller's metadata context.
use eredu_nn::{Error, LinearSpec, NeuralBackend, NormalizationConstructionSpec, Tensor};
macro_rules! copy_spec {
    ($name:ident,$ty:ty) => {
        pub(crate) fn $name<B: NeuralBackend>(
            spec: &$ty,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<$ty, Error> {
            match B::construction_metadata(context) {
                Some(metadata) => spec.clone_with_metadata(metadata),
                None => Ok(spec.clone()),
            }
        }
    };
}
copy_spec!(copy_normalization, NormalizationConstructionSpec);
copy_spec!(copy_linear, LinearSpec);
copy_spec!(copy_selector, eredu_nn::TopKGroupSelectorSpec);
copy_spec!(copy_grouped, eredu_nn::GroupedGatedProductSpec);
copy_spec!(copy_relu2, eredu_nn::GroupedRelu2Spec);
copy_spec!(copy_low_rank, eredu_nn::LowRankProjectionSpec);

copy_spec!(copy_parameter, eredu_nn::ParameterSpec);
copy_spec!(copy_convolution, eredu_nn::CausalDepthwiseConvolutionSpec);
pub(crate) fn require_source_compiler<B: NeuralBackend>(
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), Error> {
    if B::construction_metadata(context).is_some_and(|context| context.uses_checked_metadata()) {
        return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
    }
    Ok(())
}

copy_spec!(copy_embedding, eredu_nn::EmbeddingSpec);
copy_spec!(copy_hyper_connection, eredu_nn::HyperConnectionSpec);
copy_spec!(copy_hyper_head, eredu_nn::HyperHeadSpec);
