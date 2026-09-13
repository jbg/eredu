//! Ordinary non-gated affine/activation/affine feed-forward execution.
use super::ComponentInstrumentation;
use eredu_nn::{Error, LinearOperator, NeuralBackend, Tensor};

/// Exact pointwise equation between an ordinary FFN's read and write projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussian-CDF GELU.
    Gelu,
    /// Tanh-approximated GELU.
    GeluApproximate,
    /// Positive part.
    Relu,
    /// Squared positive part.
    ReluSquared,
}
impl Activation {
    /// The discovery declaration corresponding to this executable equation.
    pub fn component(self) -> eredu_core::component::ComponentNonlinearity {
        use eredu_core::component::ComponentNonlinearity as C;
        match self {
            Self::Gelu => C::Gelu,
            Self::GeluApproximate => C::GeluApproximate,
            Self::Relu => C::Relu,
            Self::ReluSquared => C::ReluSquared,
        }
    }
}

/// Executes a non-gated FFN through the same component boundary in every mode.
/// The optional parallel context changes only the output reduction mechanism.
#[allow(clippy::too_many_arguments)]
pub fn forward<B: NeuralBackend>(
    input: &B::Tensor,
    read: &mut B::Linear,
    write: &mut B::Linear,
    activation: Activation,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let projected = read.forward(input, context)?;
    let units = match activation {
        Activation::Gelu => B::Tensor::gelu(&projected, context)?,
        Activation::GeluApproximate => B::gelu_approximate(projected, context)?,
        Activation::Relu => projected.maximum_scalar(0.0, context)?,
        Activation::ReluSquared => projected.maximum_scalar(0.0, context)?.square(context)?,
    };
    let units = instrumentation.apply("feed_forward.units", units)?;
    instrumentation.project::<B>("feed_forward.write_input", write, &units, parallel, context)
}
