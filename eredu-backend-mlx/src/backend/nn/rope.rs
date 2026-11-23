//! Rotary position embedding initialization and variants.

use eredu_backend_mlx_macros::PhysicalParameters;
use eredu_nn::{PreparedRotaryAlgorithm, PreparedRotarySpec};
use safemlx::{
    error::Exception,
    ops::{
        concatenate_axis, cos,
        indexing::{NewAxis, TryIndexOp},
        sin,
    },
    Array, Stream,
};

use crate::{module::Module, nn};

/// Piecewise wavelength-based RoPE frequency scaling.
///
/// Applies piecewise frequency scaling based on wavelength cutoffs derived from
/// `low_freq_factor`, `high_freq_factor`, `factor`, and `original_max_position_embeddings`.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct FrequencyScaledRope {
    /// Number of rotary dimensions.
    pub dimensions: i32,
    /// Whether to use traditional pair ordering.
    pub traditional: bool,
    /// Runtime scale factor passed to MLX RoPE.
    pub scale: f32,
    /// Pre-computed scaled frequencies. Not a module parameter.
    pub freqs: Array,
}

impl<'a, Input> Module<Input> for FrequencyScaledRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        if !self.traditional {
            let seq_len = x.dim(-2);
            let half_dims = self.dimensions / 2;
            let positions = Array::arange::<_, f32>(Some(offset), offset + seq_len, None, stream)?
                .try_index_device((.., NewAxis), stream)?;
            let freqs = self.freqs.try_index_device(NewAxis, stream)?;
            let angles = positions
                .divide(&freqs, stream)?
                .multiply(Array::from_f32(self.scale), stream)?;
            let cos = cos(&angles, stream)?.try_index_device((NewAxis, .., ..), stream)?;
            let sin = sin(&angles, stream)?.try_index_device((NewAxis, .., ..), stream)?;
            let x1 = x.try_index_device((.., .., ..half_dims), stream)?;
            let x2 = x.try_index_device((.., .., half_dims..self.dimensions), stream)?;
            let out1 = x1
                .multiply(&cos, stream)?
                .subtract(x2.multiply(&sin, stream)?, stream)?;
            let out2 = x2
                .multiply(cos, stream)?
                .add(x1.multiply(sin, stream)?, stream)?;
            let rotated =
                concatenate_axis(&[out1, out2], -1, stream)?.as_dtype(x.dtype(), stream)?;
            let output = if self.dimensions == x.dim(-1) {
                rotated
            } else {
                concatenate_axis(
                    &[
                        rotated,
                        x.try_index_device((.., .., self.dimensions..), stream)?,
                    ],
                    -1,
                    stream,
                )?
            };
            return output.reshape(shape, stream);
        }
        let x = safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
            stream,
        )?;
        x.reshape(shape, stream)
    }
}

/// Proportional RoPE over a configurable prefix of each attention head.
///
/// Proportional RoPE keeps the full head dimension so non-traditional
/// half-rotation pairs match Hugging Face's layout. Frequency slots outside the
/// configured rotary proportion use the largest finite `f32` denominator, which
/// MLX reciprocates to an effectively zero inverse frequency and therefore
/// leaves unchanged for real context lengths.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct ProportionalRope {
    /// Head dimension passed to MLX RoPE.
    pub dimensions: i32,
    /// Whether to use traditional pair ordering.
    pub traditional: bool,
    /// Runtime scale factor passed to MLX RoPE.
    pub scale: f32,
    /// Frequency vector for the full half-head dimension.
    pub freqs: Array,
}

/// YaRN rotary embeddings with frequency interpolation and attention scaling.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct YarnRope {
    /// Number of rotary dimensions.
    pub dimensions: i32,
    /// Whether adjacent pairs, rather than split halves, are rotated.
    pub traditional: bool,
    /// Multiplicative attention scale applied to the rotated inputs.
    pub concentration: f32,
    /// Pre-computed YaRN frequency denominators.
    pub freqs: Array,
}

impl YarnRope {
    /// Constructs a YaRN RoPE module.
    pub fn new(dims: i32, traditional: bool, values: &[f32], amplitude: f32) -> Self {
        Self {
            dimensions: dims,
            traditional,
            concentration: amplitude,
            freqs: Array::from_slice(
                &values.iter().map(|v| 1.0 / v).collect::<Vec<_>>(),
                &[dims / 2],
            ),
        }
    }
}

impl<'a, Input> Module<Input> for YarnRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let dtype = x.dtype();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        let rotary = x
            .try_index_device((.., .., ..self.dimensions), stream)?
            .multiply(Array::from_f32(self.concentration), stream)?;
        let rotated = safemlx::fast::rope(
            rotary,
            self.dimensions,
            self.traditional,
            None::<f32>,
            1.0,
            offset,
            &self.freqs,
            stream,
        )?
        .as_dtype(dtype, stream)?;
        let output = if self.dimensions == x.dim(-1) {
            rotated
        } else {
            concatenate_axis(
                &[
                    rotated,
                    x.try_index_device((.., .., self.dimensions..), stream)?,
                ],
                -1,
                stream,
            )?
        };
        output.reshape(shape, stream)
    }
}

fn proportional_rotary_dims(dims: i32, proportion: f32) -> (i32, i32) {
    let half_dims = dims / 2;
    let rotary_dims = ((proportion * dims as f32).round() as i32).clamp(2, dims);
    let rope_angles = (rotary_dims / 2).clamp(1, half_dims);
    (rope_angles * 2, rope_angles)
}

fn proportional_frequency_values(dims: i32, base: f32, factor: f32, proportion: f32) -> Vec<f32> {
    let (_, rope_angles) = proportional_rotary_dims(dims, proportion);
    let half_dims = dims / 2;
    let mut freqs = Vec::with_capacity(half_dims as usize);
    for index in 0..half_dims {
        if index < rope_angles {
            freqs.push(base.powf(2.0 * index as f32 / dims as f32) / factor);
        } else {
            freqs.push(f32::MAX);
        }
    }
    freqs
}

impl ProportionalRope {
    /// Builds proportional RoPE for partial-rotary models.
    pub fn new(
        dims: i32,
        traditional: bool,
        base: f32,
        factor: f32,
        proportion: f32,
        _stream: &Stream,
    ) -> Result<Self, Exception> {
        let freqs = proportional_frequency_values(dims, base, factor, proportion);
        let freqs = Array::from_slice(&freqs, &[dims / 2]);
        Ok(Self {
            dimensions: dims,
            traditional,
            scale: 1.0,
            freqs,
        })
    }
}

impl<'a, Input> Module<Input> for ProportionalRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)], stream)?;
        let x = safemlx::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
            stream,
        )?;
        x.reshape(shape, stream)
    }
}

/// Enum wrapping different RoPE variants so that `initialize_rope` can return
/// either a standard RoPE or a piecewise frequency-scaled RoPE.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub enum RopeVariant {
    /// Standard MLX RoPE.
    Default(nn::RotaryPositionalEncoding),
    /// Piecewise wavelength-scaled RoPE.
    FrequencyScaled(FrequencyScaledRope),
    /// Proportional-prefix RoPE.
    Proportional(ProportionalRope),
    /// YaRN scaled RoPE.
    Yarn(YarnRope),
}

mod input_products;

/// Explicit table and product rounding over precomputed frequency denominators.
#[derive(Debug, Clone)]
pub(super) struct ElementwiseRotary {
    dimensions: i32,
    traditional: bool,
    inverse_frequencies: Array,
    amplitude: f32,
    arithmetic: eredu_nn::RotaryArithmetic,
    recurrence: Option<(f32, f32)>,
}

impl ElementwiseRotary {
    pub(super) fn new(
        spec: &PreparedRotarySpec,
        native: &RopeVariant,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let (inverse_frequencies, amplitude) = match native {
            RopeVariant::Yarn(rope) => {
                let PreparedRotaryAlgorithm::Yarn {
                    ref inverse_frequencies,
                    ..
                } = spec.algorithm
                else {
                    return Err(Exception::custom(
                        "rotary algorithm and materialization disagree",
                    ));
                };
                (
                    Array::from_slice(inverse_frequencies, &[spec.dimensions / 2]),
                    rope.concentration,
                )
            }
            RopeVariant::FrequencyScaled(rope) => (rope.freqs.reciprocal(stream)?, 1.0),
            RopeVariant::Proportional(rope) => (rope.freqs.reciprocal(stream)?, 1.0),
            RopeVariant::Default(_) => {
                let factor = match spec.algorithm {
                    PreparedRotaryAlgorithm::Standard { factor } => factor,
                    _ => 1.0,
                };
                let values = (0..spec.dimensions / 2)
                    .map(|index| {
                        (1.0 / spec.base.powf(2.0 * index as f32 / spec.dimensions as f32)) / factor
                    })
                    .collect::<Vec<_>>();
                (Array::from_slice(&values, &[spec.dimensions / 2]), 1.0)
            }
        };
        let recurrence = match (&spec.algorithm, spec.arithmetic) {
            (
                PreparedRotaryAlgorithm::Standard { factor },
                eredu_nn::RotaryArithmetic::RecurrentFloat32,
            ) => Some((spec.base, factor.recip())),
            _ => None,
        };
        Ok(Self {
            dimensions: spec.dimensions,
            traditional: spec.traditional,
            recurrence,
            inverse_frequencies,
            amplitude,
            arithmetic: spec.arithmetic,
        })
    }

    pub(super) fn forward(
        &self,
        input: &Array,
        offset: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if let Some(output) = input_products::forward(
            input,
            &self.inverse_frequencies,
            self.dimensions,
            self.traditional,
            offset,
            self.amplitude,
            self.arithmetic,
            stream,
        )? {
            return Ok(output);
        }
        let shape = input.shape();
        let length = input.dim(-2);
        let width = input.dim(-1);
        let x = input.reshape(&[-1, length, width], stream)?;
        let (cos, sin) = if let Some((base, scale)) = self.recurrence {
            let (sin, cos): (Vec<f32>, Vec<f32>) = eredu_nn::arithmetic::recurrent_rotary_tables(
                length as usize,
                self.dimensions as usize,
                base,
                scale,
                offset,
            )
            .into_iter()
            .unzip();
            let shape = [length, self.dimensions / 2];
            (
                Array::from_slice(&cos, &shape),
                Array::from_slice(&sin, &shape),
            )
        } else {
            let positions = Array::arange::<_, f32>(Some(offset), offset + length, None, stream)?
                .expand_dims(-1, stream)?;
            let angles = positions.multiply(&self.inverse_frequencies, stream)?;
            let amplitude = Array::from_f32(self.amplitude);
            let table_type = if self.arithmetic == eredu_nn::RotaryArithmetic::Float32Products {
                safemlx::Dtype::Float32
            } else {
                input.dtype()
            };
            (
                cos(&angles, stream)?
                    .multiply(&amplitude, stream)?
                    .as_dtype(table_type, stream)?,
                sin(&angles, stream)?
                    .multiply(&amplitude, stream)?
                    .as_dtype(table_type, stream)?,
            )
        };
        let (cos, sin, rotary) = match self.arithmetic {
            eredu_nn::RotaryArithmetic::InputTableFloat32Products
            | eredu_nn::RotaryArithmetic::Float32Products
            | eredu_nn::RotaryArithmetic::RecurrentFloat32 => (
                cos.as_dtype(safemlx::Dtype::Float32, stream)?,
                sin.as_dtype(safemlx::Dtype::Float32, stream)?,
                x.try_index_device((.., .., ..self.dimensions), stream)?
                    .as_dtype(safemlx::Dtype::Float32, stream)?,
            ),
            _ => (
                cos,
                sin,
                x.try_index_device((.., .., ..self.dimensions), stream)?,
            ),
        };
        let half = self.dimensions / 2;
        let (first, second) = if self.traditional {
            let pairs = rotary.reshape(&[-1, length, half, 2], stream)?;
            (
                pairs.try_index_device((.., .., .., 0), stream)?,
                pairs.try_index_device((.., .., .., 1), stream)?,
            )
        } else {
            (
                rotary.try_index_device((.., .., ..half), stream)?,
                rotary.try_index_device((.., .., half..), stream)?,
            )
        };
        let (left, right) = if self.recurrence.is_some() {
            let left_add = second.multiply(&sin, stream)?.negative(stream)?;
            let right_add = second.multiply(&cos, stream)?;
            (
                super::arithmetic::fused_f32(&first, &cos, &left_add, stream)?,
                super::arithmetic::fused_f32(&first, &sin, &right_add, stream)?,
            )
        } else {
            (
                first
                    .multiply(&cos, stream)?
                    .subtract(second.multiply(&sin, stream)?, stream)?,
                second
                    .multiply(&cos, stream)?
                    .add(first.multiply(&sin, stream)?, stream)?,
            )
        };
        let rotated = if self.traditional {
            concatenate_axis(
                &[
                    left.expand_dims(-1, stream)?,
                    right.expand_dims(-1, stream)?,
                ],
                -1,
                stream,
            )?
            .reshape(&[-1, length, self.dimensions], stream)?
        } else {
            concatenate_axis(&[left, right], -1, stream)?
        };
        let rotated = rotated.as_dtype(input.dtype(), stream)?;
        let output = if width == self.dimensions {
            rotated
        } else {
            concatenate_axis(
                &[
                    rotated,
                    x.try_index_device((.., .., self.dimensions..), stream)?,
                ],
                -1,
                stream,
            )?
        };
        output.reshape(shape, stream)
    }
}

impl<'a, Input> Module<Input> for RopeVariant
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error> {
        match self {
            RopeVariant::Default(rope) => rope.forward(input, stream),
            RopeVariant::FrequencyScaled(rope) => rope.forward(input, stream),
            RopeVariant::Proportional(rope) => rope.forward(input, stream),
            RopeVariant::Yarn(rope) => rope.forward(input, stream),
        }
    }
}

/// Materializes a portable rotary construction specification.
pub fn initialize_rope(
    spec: &PreparedRotarySpec,
    stream: &Stream,
) -> Result<RopeVariant, Exception> {
    let (dims, base, traditional) = (spec.dimensions, spec.base, spec.traditional);
    match spec.algorithm {
        PreparedRotaryAlgorithm::FrequencyTable(ref denominators) => {
            Ok(RopeVariant::FrequencyScaled(FrequencyScaledRope {
                dimensions: dims,
                traditional,
                scale: 1.0,
                freqs: Array::from_slice(denominators, &[dims / 2]),
            }))
        }
        PreparedRotaryAlgorithm::Standard { factor } => Ok(RopeVariant::Default(
            nn::RotaryPositionalEncodingBuilder::new(dims)
                .traditional(traditional)
                .base(base)
                .scale(factor.recip())
                .build()
                .expect("infallible MLX RoPE construction"),
        )),
        PreparedRotaryAlgorithm::Proportional {
            factor,
            rotary_fraction,
        } => Ok(RopeVariant::Proportional(ProportionalRope::new(
            dims,
            traditional,
            base,
            factor,
            rotary_fraction,
            stream,
        )?)),
        PreparedRotaryAlgorithm::Yarn {
            ref inverse_frequencies,
            amplitude,
        } => Ok(RopeVariant::Yarn(YarnRope::new(
            dims,
            traditional,
            inverse_frequencies,
            amplitude,
        ))),
    }
}
