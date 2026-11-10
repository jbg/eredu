//! Ordinary non-gated affine/activation/affine feed-forward execution.
use super::ComponentInstrumentation;
use eredu_nn::{Error, LinearOperator, NeuralBackend, Tensor};

/// Discovery equation for the independently declared executable activation.
pub fn component(
    activation: eredu_nn::UnaryActivation,
) -> eredu_core::component::ComponentNonlinearity {
    use eredu_core::component::ComponentNonlinearity as C;
    match activation {
        eredu_nn::UnaryActivation::Gelu => C::Gelu,
        eredu_nn::UnaryActivation::GeluApproximate => C::GeluApproximate,
        eredu_nn::UnaryActivation::Silu => C::Silu {
            multiplier: eredu_core::component::ComponentScalar::new(1.0),
        },
        eredu_nn::UnaryActivation::Relu => C::Relu,
        eredu_nn::UnaryActivation::ReluSquared => C::ReluSquared,
    }
}

/// Executes a non-gated FFN through the same component boundary in every mode.
/// The optional parallel context changes only the output reduction mechanism.
#[allow(clippy::too_many_arguments)]
pub fn forward<B: NeuralBackend>(
    input: &B::Tensor,
    read: &mut B::Linear,
    write: &mut B::Linear,
    activation: eredu_nn::UnaryActivation,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let projected = read.forward(input, context)?;
    let units = activation.apply::<B>(projected, context)?;
    let units = instrumentation.apply("feed_forward.units", units)?;
    instrumentation.project::<B>("feed_forward.write_input", write, &units, parallel, context)
}
