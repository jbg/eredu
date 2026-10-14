//! Low-rank projection parameters, kept separate from the base matrix.
use crate::{
    Error, LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec,
    Parameterized, Tensor,
};

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
    /// Declares separately named parameters. `input` is the invocation's local
    /// input width, `output` the complete output width; rank is replicated.
    pub fn new(
        input: i32,
        output: i32,
        rank: i32,
        scale: f32,
        down_weight: ParameterSpec,
        up_weight: ParameterSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if input <= 0 || output <= 0 || rank <= 0 || !scale.is_finite() {
            return Err(Error::backend(
                "invalid low-rank projection geometry or scale",
            ));
        }
        let projection = |input, output, weight| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight,
                    bias: None,
                    format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
                },
                context,
            )
        };
        Ok(Self {
            down: projection(input, rank, down_weight)?,
            up: projection(rank, output, up_weight)?,
            scale,
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
