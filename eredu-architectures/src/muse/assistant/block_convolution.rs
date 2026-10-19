//! Dynamic input/output kernels whose history is confined to one proposal block.

use eredu_nn::{
    Error, Index, LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, PadMode, Parameter,
    ParameterSpec, Parameterized, Tensor,
};

/// Geometry of a grouped dynamic convolution around one assistant sublayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockConvolutionGeometry {
    /// Hidden channels.
    pub hidden: i32,
    /// Independent proposal positions; convolution never crosses this boundary.
    pub block_size: i32,
    /// Taps ordered current to oldest.
    pub taps: i32,
    /// Channels sharing one input-dependent coefficient.
    pub group_size: i32,
}

impl BlockConvolutionGeometry {
    fn projection_width(self) -> Result<i32, Error> {
        if self.hidden <= 0
            || self.block_size <= 0
            || self.taps <= 0
            || self.group_size <= 0
            || self.hidden % self.group_size != 0
        {
            return Err(Error::backend(
                "invalid assistant block-convolution geometry",
            ));
        }
        self.taps
            .checked_mul(self.hidden / self.group_size)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| Error::backend("assistant dynamic-kernel width exceeds i32"))
    }
}

/// Output-side coefficients projected from the sublayer's original input.
/// Consuming this value in `finish` prevents accidentally reprojecting the branch
/// output or carrying a kernel into the next proposal.
#[derive(Debug)]
pub struct PreparedBlockKernel<T> {
    coefficients: T,
    shape: Vec<i32>,
    owner: String,
}

/// Published DFlash grouped dynamic convolution. Parameters and projection use
/// ordinary neutral binding; there is no persistent convolution carry.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct BlockConvolution<B: NeuralBackend> {
    /// `[input/output, tap, channel]` learned kernels.
    pub base_kernel: Parameter<B::Tensor>,
    /// Projects both input-side and output-side grouped coefficients at once.
    pub kernel_projection: B::Linear,
    #[parameter(skip)]
    geometry: BlockConvolutionGeometry,
    #[parameter(skip)]
    owner: String,
}

impl<B: NeuralBackend> BlockConvolution<B> {
    /// Constructs unloaded parameters under the supplied sublayer prefix.
    pub fn new(
        geometry: BlockConvolutionGeometry,
        prefix: impl Into<String>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_format(
            geometry,
            prefix,
            eredu_checkpoint::LinearFormat::Dense,
            context,
        )
    }

    /// Constructs the dynamic projection with its exact selected physical encoding.
    pub fn new_with_format(
        geometry: BlockConvolutionGeometry,
        prefix: impl Into<String>,
        format: eredu_checkpoint::LinearFormat,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let output = geometry.projection_width()?;
        B::require_operator_capabilities(
            "DFlash block convolution",
            eredu_nn::NeuralOperatorCapabilities::ZEROS_LIKE,
        )?;
        let owner = prefix.into();
        if owner.is_empty() {
            return Err(Error::backend(
                "assistant convolution requires a parameter prefix",
            ));
        }
        Ok(Self {
            base_kernel: Parameter::unloaded(
                ParameterSpec::trainable(format!("{owner}.base_kernel")).map_err(Error::backend)?,
                &[2, geometry.taps, geometry.hidden],
                context,
            )?,
            kernel_projection: B::linear(
                LinearSpec {
                    input: geometry.hidden,
                    output,
                    weight: ParameterSpec::trainable(format!("{owner}.kernel_projection.weight"))
                        .map_err(Error::backend)?,
                    bias: None,
                    format: LinearFormatSpec::unscaled(format)?,
                },
                context,
            )?,
            geometry,
            owner,
        })
    }

    /// Convolves normalized sublayer input and retains its output-side kernel.
    pub fn prepare(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, PreparedBlockKernel<B::Tensor>), Error> {
        self.validate(hidden)?;
        let g = self.geometry;
        let projected = self.kernel_projection.forward(hidden, context)?.reshape(
            &[
                hidden.dim(0),
                hidden.dim(1),
                2,
                g.taps,
                g.hidden / g.group_size,
            ],
            context,
        )?;
        let side = |index| {
            projected.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::At(index),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )
        };
        let output = self.convolve(hidden, &side(0)?, 0, context)?;
        Ok((
            output,
            PreparedBlockKernel {
                coefficients: side(1)?,
                shape: hidden.shape().to_vec(),
                owner: self.owner.clone(),
            },
        ))
    }

    /// Convolves the sublayer output using the original input's retained kernel.
    pub fn finish(
        &self,
        hidden: &B::Tensor,
        kernel: PreparedBlockKernel<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.validate(hidden)?;
        if hidden.shape() != kernel.shape || self.owner != kernel.owner {
            return Err(Error::backend(
                "assistant output kernel belongs to a different sublayer or proposal geometry",
            ));
        }
        self.convolve(hidden, &kernel.coefficients, 1, context)
    }

    fn validate(&self, hidden: &B::Tensor) -> Result<(), Error> {
        if hidden.shape().len() != 3
            || hidden.dim(0) <= 0
            || hidden.dim(1) <= 0
            || hidden.dim(2) != self.geometry.hidden
        {
            return Err(Error::backend(
                "assistant convolution expects nonempty [batch, positions, hidden] input",
            ));
        }
        Ok(())
    }

    fn convolve(
        &self,
        hidden: &B::Tensor,
        delta: &B::Tensor,
        side: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let g = self.geometry;
        let groups = g.hidden / g.group_size;
        let base = self
            .base_kernel
            .as_ref()
            .index(&[Index::At(side), Index::Full, Index::Full], context)?
            .reshape(&[1, 1, g.taps, groups, g.group_size], context)?;
        // Coefficient addition, each product, and each accumulated tap retain
        // activation precision, as in the pinned serving equation.
        let coefficients = base.add(&delta.expand_dims(-1, context)?, context)?;
        let mut outputs = Vec::new();
        for first in (0..hidden.dim(1)).step_by(g.block_size as usize) {
            let end = first.saturating_add(g.block_size).min(hidden.dim(1));
            let length = end - first;
            let block = hidden
                .index(
                    &[Index::Full, Index::Range(first, end), Index::Full],
                    context,
                )?
                .reshape(&[hidden.dim(0), length, groups, g.group_size], context)?;
            let coefficient = |tap| {
                coefficients.index(
                    &[
                        Index::Full,
                        Index::Range(first, end),
                        Index::At(tap),
                        Index::Full,
                        Index::Full,
                    ],
                    context,
                )
            };
            let mut output = coefficient(0)?.multiply(&block, context)?;
            for tap in 1..g.taps {
                let shifted = if tap >= length {
                    block.zeros_like(context)?
                } else {
                    let prefix = block.index(
                        &[
                            Index::Full,
                            Index::Range(0, length - tap),
                            Index::Full,
                            Index::Full,
                        ],
                        context,
                    )?;
                    B::Tensor::pad(
                        &prefix,
                        &[(0, 0), (tap, 0), (0, 0), (0, 0)],
                        PadMode::Constant,
                        context,
                    )?
                };
                output = output.add(&coefficient(tap)?.multiply(&shifted, context)?, context)?;
            }
            outputs.push(output.reshape(&[hidden.dim(0), length, g.hidden], context)?);
        }
        B::Tensor::concatenate(&outputs, 1, context)
    }
}
