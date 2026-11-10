//! Ordinary affine/activation/affine construction without family naming policy.

use crate::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, NeuralBackend, Tensor,
};

/// Exact pointwise equation between ordinary feed-forward read and write projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryActivation {
    /// Gaussian-CDF GELU.
    Gelu,
    /// Tanh-approximated GELU.
    GeluApproximate,
    /// Sigmoid-weighted linear unit.
    Silu,
    /// Positive part.
    Relu,
    /// Squared positive part.
    ReluSquared,
}
impl UnaryActivation {
    /// Applies the declared equation using ordinary neutral backend primitives.
    pub fn apply<B: NeuralBackend>(
        self,
        input: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Gelu => B::Tensor::gelu(&input, context),
            Self::GeluApproximate => B::gelu_approximate(input, context),
            Self::Silu => B::silu(input, context),
            Self::Relu => input.maximum_scalar(0.0, context),
            Self::ReluSquared => input.maximum_scalar(0.0, context)?.square(context),
        }
    }
}

/// Independent affine read and write slots with one intervening pointwise equation.
/// Input and output widths may differ; enclosing residual scopes validate their coupling.
#[derive(Debug, Clone)]
pub struct UnaryFeedForwardSpec {
    /// Input-to-intermediate read projection.
    pub up: LinearSpec,
    /// Intermediate-to-output write projection.
    pub down: LinearSpec,
    /// Pointwise equation applied before the write projection.
    pub activation: UnaryActivation,
}
impl UnaryFeedForwardSpec {
    /// Builds and validates a complete parameter scope before allocation.
    pub fn new(
        up: LinearSpec,
        down: LinearSpec,
        activation: UnaryActivation,
    ) -> Result<Self, Error> {
        let spec = Self {
            up,
            down,
            activation,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Affine declarations in read, write allocation order.
    pub fn projections(&self) -> [&LinearSpec; 2] {
        [&self.up, &self.down]
    }

    /// Rejects incompatible geometry, conflicting identities and unresolved sharing.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for UnaryFeedForwardSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.up.output != self.down.input {
            return Err(Error::backend("incompatible unary feed-forward geometry"));
        }
        let mut parameters = self.up.logical_parameters()?;
        parameters.extend(self.down.logical_parameters()?);
        Ok(parameters)
    }
}
