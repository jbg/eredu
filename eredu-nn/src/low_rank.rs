//! Low-rank projection parameters, kept separate from the base matrix.
use crate::{
    DeclareLogicalParameters, Error, LinearOperator, LinearSpec, LogicalParameterDeclaration,
    NeuralBackend, Parameterized, Tensor,
};

/// Exact bias-free affine slots for an additive low-rank projection.
#[derive(Debug, Clone)]
pub struct LowRankProjectionSpec {
    /// Input-to-rank matrix, with invocation-local input width.
    pub down: LinearSpec,
    /// Rank-to-output matrix, with complete replicated rank width.
    pub up: LinearSpec,
    /// Finite multiplier applied in the output's floating precision.
    pub scale: f32,
}
impl LowRankProjectionSpec {
    /// Rejects incompatible geometry, identities and sharing before backend allocation.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for LowRankProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.down.output != self.up.input
            || !self.scale.is_finite()
            || [&self.down, &self.up].iter().any(|spec| {
                spec.bias.is_some()
                    || spec.format.encoding() != eredu_checkpoint::LinearFormat::Dense
                    || spec.format.arithmetic() != crate::LinearArithmetic::Native
            })
        {
            return Err(Error::backend(
                "invalid low-rank projection geometry, scale or affine policy",
            ));
        }
        let mut parameters = self.down.logical_parameters()?;
        parameters.extend(self.up.logical_parameters()?);
        Ok(parameters)
    }
}

/// A bias-free additive projection `scale * up(down(input))`.
/// Invocation owners select whether to borrow this module; it holds no enabled flag.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LowRankProjection<B: NeuralBackend> {
    /// Input-to-rank matrix, row-parallel when the base input is sharded.
    pub down: B::Linear,
    /// Rank-to-output matrix, replicated after reduction of rank activations.
    pub up: B::Linear,
    #[parameter(skip)]
    scale: f32,
}
impl<B: NeuralBackend> LowRankProjection<B> {
    /// Constructs the declared affine slots after validating their shared rank.
    pub fn new(
        spec: LowRankProjectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            down: B::linear(spec.down, context)?,
            up: B::linear(spec.up, context)?,
            scale: spec.scale,
        })
    }

    /// Applies only the additive delta. With TP, input-sharded rank contributions
    /// are reduced before the replicated up projection, so its output is not
    /// multiplied by the number of ranks.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let rank = match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.down, input, parallel, context)?,
            None => self.down.forward(input, context)?,
        };
        let output = self.up.forward(&rank, context)?;
        let dtype = output
            .element_type()
            .ok_or_else(|| Error::backend("unknown low-rank output precision"))?;
        let scale =
            B::Tensor::from_f32_slice(&[self.scale], &[], context)?.cast_float(dtype, context)?;
        output.multiply(&scale, context)
    }
}
