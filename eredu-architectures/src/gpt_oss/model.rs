//! Backend-neutral layered GPT-OSS model assembly.

use eredu_nn::{Error, RoutedNeuralBackend, Tensor};

use super::{block::GptOssBlockFactory, config::ModelArgs};

/// Shared layered lifecycle specialized to GPT-OSS blocks.
pub type LayeredModel<B> = crate::decoder::LayeredModel<B, ModelArgs, GptOssBlockFactory>;

/// Builds one layered GPT-OSS model with pinned static modules.
pub fn new_layered_model<B: RoutedNeuralBackend>(
    args: ModelArgs,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<LayeredModel<B>, Error> {
    LayeredModel::new(args, context)
}
