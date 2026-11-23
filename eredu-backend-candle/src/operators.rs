use crate::block_projection::BlockProjection;
use crate::{
    quantized::{matrix_slot, PackedMatrix},
    word_projection::WordProjection,
};
use crate::{tensor::rounded_bf16_binary, CandleContext, CandleTensor};
use candle_core::{DType, Tensor as NativeTensor, D};
use eredu_nn::{
    operation_geometry::NormalizationGeometry, EmbeddingOperator, EmbeddingSpec, Error,
    LinearArithmetic, LinearOperator, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationReduction, NormalizationScale, Parameter, Parameterized,
    PreparedRotaryAlgorithm, PreparedRotarySpec, RotaryArithmetic, RotaryOperator, RotaryPosition,
    Tensor, TensorElementType,
};
use half::bf16;
use rayon::prelude::*;

/// Normalizes BF16 score rows in FP32 before rounding probabilities.
pub(crate) fn bf16_softmax(scores: &NativeTensor) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if scores.device().is_metal() {
        return crate::metal_bf16::softmax(scores);
    }
    let scores = scores.contiguous()?;
    let width = scores.dim(D::Minus1)?;
    if width == 0 {
        candle_core::bail!("BF16 softmax requires a nonempty score axis");
    }
    let (storage, layout) = scores.storage_and_layout();
    let candle_core::Storage::Cpu(candle_core::CpuStorage::BF16(values)) = &*storage else {
        candle_core::bail!("BF16 softmax requires CPU storage");
    };
    let start = layout.start_offset();
    let values = &values[start..start + scores.elem_count()];
    let mut probabilities = vec![bf16::ZERO; values.len()];
    values
        .par_chunks(width)
        .zip(probabilities.par_chunks_mut(width))
        .for_each_init(
            || vec![0f32; width],
            |scratch, (row, output)| {
                let max = row
                    .iter()
                    .map(|value| value.to_f32())
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut lanes = [0.; 4];
                for (k, (&score, exp)) in row.iter().zip(scratch.iter_mut()).enumerate() {
                    *exp = eredu_nn::arithmetic::stable_exp_f32(score.to_f32() - max);
                    lanes[k % 4] += *exp;
                }
                let inverse = bf16_softmax_inverse(lanes, width);
                for (out, &exp) in output.iter_mut().zip(scratch.iter()) {
                    *out = bf16::from_f32(exp * inverse);
                }
            },
        );
    NativeTensor::from_vec(probabilities, scores.shape(), scores.device())
}

pub(crate) fn bf16_softmax_inverse(lanes: [f32; 4], width: usize) -> f32 {
    let sum = if width < 4 {
        lanes[..width]
            .iter()
            .copied()
            .reduce(|a, b| a + b)
            .unwrap_or(0.)
    } else {
        (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
    };
    sum.recip()
}

/// Affine projection with a dense or packed weight.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleLinear {
    pub(crate) weight: Parameter<CandleTensor>,
    pub(crate) bias: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    packed: Option<PackedMatrix>,
    words: Option<WordProjection>,
    block: Option<BlockProjection>,
    #[parameter(skip)]
    arithmetic: LinearArithmetic,
}

impl CandleLinear {
    pub(crate) fn new(spec: LinearSpec, context: &CandleContext) -> Result<Self, Error> {
        let arithmetic = spec.format.arithmetic();
        let (weight, packed) = matrix_slot(&spec.format, spec.output, spec.input, context)?;
        let words = WordProjection::new(&spec.format, spec.output, spec.input, None, context)?;
        let block = BlockProjection::new(&spec.format, spec.output, spec.input, None, context)?;
        let bias = spec
            .bias
            .map(|specification| {
                CandleTensor::unloaded_f32(&[spec.output], context)
                    .map(|value| Parameter::new(specification, value))
            })
            .transpose()?;
        Ok(Self {
            weight: Parameter::new(spec.weight, weight),
            bias,
            packed,
            words,
            block,
            arithmetic,
        })
    }

    fn project(
        &self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let weight = self.weight.as_ref();
        let bias = self.bias.as_ref().map(Parameter::as_ref);
        if let Some(words) = &self.words {
            return words.project(input, weight, bias, None);
        }
        match &self.packed {
            Some(packed) => packed.linear(input, weight, bias),
            None => CandleTensor::linear(input, weight, bias, context),
        }
    }
}

impl LinearOperator<CandleTensor> for CandleLinear {
    fn parameter_element_type(&self) -> Option<TensorElementType> {
        (self.packed.is_some() || self.words.is_some() || self.block.is_some())
            .then_some(TensorElementType::F32)
            .or_else(|| self.weight.as_ref().element_type())
    }

    fn forward_with_rounded_weight_inputs(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let weight = self.weight.as_ref();
        if self.packed.is_some()
            || self.words.is_some()
            || self.block.is_some()
            || input.value.dtype() != DType::F32
        {
            return self.forward(input, context);
        }
        let dtype = weight.value.dtype();
        let device = input.value.device();
        if dtype == DType::BF16
            && (device.is_cpu()
                || (cfg!(all(feature = "metal", target_os = "macos")) && device.is_metal()))
            && device.same_device(weight.value.device())
        {
            return CandleTensor::wrap((|| {
                let result =
                    crate::matmul::rounded_operand_projection(&input.value, &weight.value, dtype)?;
                match &self.bias {
                    Some(bias) => result.broadcast_add(&bias.as_ref().value.to_dtype(DType::F32)?),
                    None => Ok(result),
                }
            })());
        }
        if matches!(dtype, DType::F16 | DType::BF16) {
            let rounded = CandleTensor::wrap(
                input
                    .value
                    .to_dtype(dtype)
                    .and_then(|value| value.to_dtype(DType::F32)),
            )?;
            return self.forward(&rounded, context);
        }
        self.forward(input, context)
    }

    fn set_compute_element(&mut self, element: Option<TensorElementType>) -> Result<(), Error> {
        self.arithmetic =
            eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?
                .with_compute_element(element)?
                .arithmetic();
        Ok(())
    }

    fn forward(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        self.forward_with_input_observer(input, context, None)
    }

    fn forward_with_input_observer(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
        observer: Option<&mut dyn eredu_nn::ProjectionInputObserver<CandleTensor>>,
    ) -> Result<CandleTensor, Error> {
        let mut arithmetic = self.arithmetic;
        if arithmetic == LinearArithmetic::ChannelMajorAffine {
            let &[batch, time, channels] = input.shape() else {
                return Err(Error::backend(
                    "channel-major affine requires [batch, time, channel]",
                ));
            };
            if batch <= 0 || time <= 0 || channels <= 0 {
                return Err(Error::backend(
                    "channel-major affine requires positive extents",
                ));
            }
            if time == 1 || channels == 1 {
                arithmetic = LinearArithmetic::RowMajorAffine;
            }
        }
        let compute_element = match arithmetic {
            LinearArithmetic::RowMajorAffine => Some(TensorElementType::F32),
            policy => policy.compute_element(),
        };
        let compute = compute_element
            .map(|element| input.cast_float(element, context))
            .transpose()?;
        let source = compute.as_ref().unwrap_or(input);
        let activation = self
            .block
            .as_ref()
            .map(|block| block.input(source, context))
            .transpose()?
            .flatten();
        if let Some(observer) = observer {
            observer.observe(activation.as_ref().unwrap_or(source))?;
        }
        let weight = self.weight.as_ref();
        let bias = self.bias.as_ref().map(|bias| &bias.as_ref().value);
        let output = if let Some(block) = &self.block {
            block.project(
                source,
                activation.as_ref(),
                weight,
                self.bias.as_ref().map(Parameter::as_ref),
                self.packed.as_ref(),
                None,
            )?
        } else if input.value.dtype() == DType::BF16 && weight.value.dtype() == DType::BF16 {
            match arithmetic {
                LinearArithmetic::RowMajorAffine => CandleTensor::wrap(
                    crate::matmul::ordered_bf16_affine(&source.value, &weight.value, bias, false),
                )?,
                LinearArithmetic::ChannelMajorAffine => CandleTensor::wrap((|| {
                    let value = crate::matmul::ordered_bf16_affine(
                        &source.value,
                        &weight.value,
                        None,
                        true,
                    )?
                    .to_dtype(DType::BF16)?;
                    match bias {
                        Some(bias) => {
                            rounded_bf16_binary(&value, bias, NativeTensor::broadcast_add)
                        }
                        None => Ok(value),
                    }
                })())?,
                _ => self.project(source, context)?,
            }
        } else {
            self.project(source, context)?
        };
        if compute_element.is_some() {
            CandleTensor::wrap(output.value.to_dtype(input.value.dtype()))
        } else {
            Ok(output)
        }
    }
}

/// Vocabulary embedding and tied readout with a dense or packed weight.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleEmbedding {
    weight: Parameter<CandleTensor>,
    #[parameter(skip)]
    packed: Option<PackedMatrix>,
    words: Option<WordProjection>,
    block: Option<BlockProjection>,
}

impl CandleEmbedding {
    pub(crate) fn new(spec: EmbeddingSpec, context: &CandleContext) -> Result<Self, Error> {
        if spec.format.arithmetic() != LinearArithmetic::Native {
            return Err(Error::backend(
                "embedding requires native projection arithmetic",
            ));
        }
        let (value, packed) = matrix_slot(&spec.format, spec.vocabulary, spec.dimensions, context)?;
        let words = WordProjection::new(
            &spec.format,
            spec.vocabulary,
            spec.dimensions,
            None,
            context,
        )?;
        let block = BlockProjection::new(
            &spec.format,
            spec.vocabulary,
            spec.dimensions,
            None,
            context,
        )?;
        Ok(Self {
            weight: Parameter::new(spec.weight, value),
            packed,
            words,
            block,
        })
    }
}

impl EmbeddingOperator<CandleTensor> for CandleEmbedding {
    fn output_element_type(&self) -> Option<TensorElementType> {
        (self.packed.is_some() || self.words.is_some() || self.block.is_some())
            .then_some(TensorElementType::F32)
            .or_else(|| self.weight.as_ref().element_type())
    }

    fn weighted_sum(
        &mut self,
        coefficients: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let vocabulary = self.weight.as_ref().dim(0);
        if coefficients.shape().last() != Some(&vocabulary) || vocabulary <= 0 {
            return Err(Error::backend(
                "weighted embedding requires a complete vocabulary axis",
            ));
        }
        if self.packed.is_none() && self.words.is_none() && self.block.is_none() {
            let weight = self.weight.as_ref();
            let dtype = weight
                .element_type()
                .ok_or_else(|| Error::backend("unknown embedding dtype"))?;
            return CandleTensor::matmul(
                &coefficients.cast_float(dtype, context)?,
                weight,
                context,
            );
        }
        eredu_nn::weighted_embedding::weighted_embedding_rows(
            coefficients,
            vocabulary,
            context,
            |ids| self.forward(ids, context),
            |_| context.synchronize().map_err(Error::backend_source),
        )
    }

    fn forward(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let weight = self.weight.as_ref();
        if let Some(block) = &self.block {
            return block.embed(input, weight, self.packed.as_ref(), context);
        }
        if let Some(words) = &self.words {
            return words.embed(input, weight);
        }
        match &self.packed {
            Some(packed) => packed.embed(input, weight, context),
            None => weight.take_axis(input, 0, context),
        }
    }

    fn as_linear(
        &mut self,
        input: &CandleTensor,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        let weight = self.weight.as_ref();
        if let Some(block) = &self.block {
            let activation = block.input(input, context)?;
            return block.project(
                input,
                activation.as_ref(),
                weight,
                None,
                self.packed.as_ref(),
                None,
            );
        }
        if let Some(words) = &self.words {
            return words.project(input, weight, None, None);
        }
        match &self.packed {
            Some(packed) => packed.linear(input, weight, None),
            None => CandleTensor::linear(input, weight, None, context),
        }
    }
}

/// RMS normalization with a learned, learned-offset, or unit scale.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleNormalization {
    weight: Option<Parameter<CandleTensor>>,
    #[parameter(skip)]
    offset: Option<f32>,
    #[parameter(skip)]
    width: i32,
    #[parameter(skip)]
    groups: Option<i32>,
    #[parameter(skip)]
    epsilon: f32,
    #[parameter(skip)]
    reduction: NormalizationReduction,
}

impl CandleNormalization {
    pub(crate) fn new(
        spec: NormalizationConstructionSpec,
        context: &CandleContext,
    ) -> Result<Self, Error> {
        spec.validate()?;
        let (weight, offset) = match spec.scale {
            NormalizationScale::Learned(weight) => (Some(weight), None),
            NormalizationScale::LearnedOffset { weight, offset } => (Some(weight), Some(offset)),
            NormalizationScale::Unit => (None, None),
        };
        let weight = weight
            .map(|specification| {
                CandleTensor::unloaded_f32(&[spec.dimensions], context)
                    .map(|value| Parameter::new(specification, value))
            })
            .transpose()?;
        Ok(Self {
            weight,
            offset,
            width: spec.dimensions,
            groups: spec.groups,
            epsilon: spec.epsilon,
            reduction: spec.reduction,
        })
    }
}

impl NormalizationOperator<CandleTensor> for CandleNormalization {
    fn forward(&mut self, input: &CandleTensor, _: &CandleContext) -> Result<CandleTensor, Error> {
        let geometry = NormalizationGeometry::new(input.shape(), self.epsilon)?;
        if geometry.width() != self.width {
            return Err(Error::backend(
                "input width differs from RMS normalization width",
            ));
        }
        let weight = self.weight.as_ref().map(Parameter::as_ref);
        if let Some(value) = ordered_rms(input, weight, self.offset, self.epsilon, self.reduction)?
        {
            return Ok(value);
        }
        rms_norm(input, weight, self.offset, self.groups, self.epsilon)
    }
}

fn ordered_rms(
    input: &CandleTensor,
    weight: Option<&CandleTensor>,
    offset: Option<f32>,
    epsilon: f32,
    reduction: NormalizationReduction,
) -> Result<Option<CandleTensor>, Error> {
    use eredu_nn::arithmetic::{cascaded_sum_squares, channel_major_rms_sum};

    let time = match reduction {
        NormalizationReduction::Native => return Ok(None),
        NormalizationReduction::ChannelMajor => {
            let &[batch, time, width] = input.shape() else {
                return Err(Error::backend(
                    "channel-major RMS requires [batch, time, channel]",
                ));
            };
            if batch <= 0 || time <= 0 || width <= 0 {
                return Err(Error::backend(
                    "channel-major RMS requires positive extents",
                ));
            }
            if time == 1 || width == 1 {
                return Ok(None);
            }
            time as usize
        }
        _ => 0,
    };
    if !matches!(
        input.value.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    ) {
        return Err(Error::backend(
            "ordered RMS normalization requires floating input",
        ));
    }
    CandleTensor::wrap((|| {
        let dtype = input.value.dtype();
        let width = input.value.dim(candle_core::D::Minus1)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if reduction == NormalizationReduction::EightLaneInputPrecision
            && input.value.device().is_metal()
        {
            let wide = input.value.to_dtype(DType::F32)?;
            let inverse = crate::metal_bf16::rms_inverse(&wide, epsilon, true)?;
            let normalized = wide.broadcast_mul(&inverse)?.to_dtype(dtype)?;
            return match weight {
                Some(weight) => {
                    rounded_bf16_binary(&normalized, &weight.value, NativeTensor::broadcast_mul)
                }
                None => Ok(normalized),
            };
        }
        let values = input
            .value
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let round_before_gain = reduction == NormalizationReduction::EightLaneInputPrecision;
        let gain = weight
            .filter(|_| !round_before_gain)
            .map(|weight| weight.value.to_dtype(DType::F32)?.to_vec1::<f32>())
            .transpose()?;
        let mut output = Vec::with_capacity(values.len());
        for (row, values) in values.chunks_exact(width).enumerate() {
            let sum = match reduction {
                NormalizationReduction::ChannelMajor => channel_major_rms_sum(values, time, row),
                NormalizationReduction::Cascaded => cascaded_sum_squares::<4>(values),
                NormalizationReduction::EightLaneInputPrecision => {
                    cascaded_sum_squares::<8>(values)
                }
                NormalizationReduction::Native => unreachable!(),
            };
            let inverse = 1.0 / (sum / width as f32 + epsilon).sqrt();
            for (column, value) in values.iter().enumerate() {
                let normalized = value * inverse;
                output.push(match &gain {
                    Some(gain) => normalized * (gain[column] + offset.unwrap_or(0.0)),
                    None => normalized,
                });
            }
        }
        let result = NativeTensor::from_vec(output, input.value.shape(), input.value.device())?
            .to_dtype(dtype)?;
        match weight.filter(|_| round_before_gain) {
            Some(weight) => {
                rounded_bf16_binary(&result, &weight.value, NativeTensor::broadcast_mul)
            }
            None => Ok(result),
        }
    })())
    .map(Some)
}

pub(crate) fn rms_norm(
    input: &CandleTensor,
    weight: Option<&CandleTensor>,
    offset: Option<f32>,
    groups: Option<i32>,
    epsilon: f32,
) -> Result<CandleTensor, Error> {
    let width = NormalizationGeometry::new(input.shape(), epsilon)?.width();
    CandleTensor::wrap((|| {
        let dtype = input.value.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            candle_core::bail!("RMS normalization requires floating input");
        }
        if weight.is_some_and(|weight| weight.shape() != [width]) {
            candle_core::bail!("RMS scale width differs from input");
        }
        let ordinary_gain = offset.is_none() && groups.is_none();
        let groups = groups.unwrap_or(1);
        let rows = input.value.elem_count() / width as usize;
        let compute = if dtype == DType::F64 {
            dtype
        } else {
            DType::F32
        };
        let grouped = input.value.to_dtype(compute)?.reshape((
            rows,
            groups as usize,
            (width / groups) as usize,
        ))?;
        let inverse = rms_inverse(&grouped, &input.value, epsilon)?;
        let normalized = grouped
            .broadcast_mul(&inverse)?
            .reshape(input.value.shape())?;
        let scaled = match weight {
            Some(weight) if ordinary_gain => {
                return rounded_bf16_binary(
                    &normalized.to_dtype(dtype)?,
                    &weight.value.to_dtype(dtype)?,
                    NativeTensor::broadcast_mul,
                );
            }
            Some(weight) => {
                let gain = weight
                    .value
                    .to_dtype(compute)?
                    .affine(1., offset.unwrap_or(0.).into())?;
                normalized.broadcast_mul(&gain)?
            }
            None => normalized,
        };
        scaled.to_dtype(dtype)
    })())
}

fn rms_inverse(
    grouped: &NativeTensor,
    input: &NativeTensor,
    epsilon: f32,
) -> candle_core::Result<NativeTensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if crate::metal_f32::applies(input) {
        return crate::metal_f32::rms_inverse(grouped, epsilon);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input.dtype() == DType::BF16 && input.device().is_metal() {
        return crate::metal_bf16::rms_inverse(grouped, epsilon, false);
    }
    let squared = grouped.sqr()?;
    let mean = if crate::float_cpu::applies(input) {
        // Round each FP32 square, then accumulate the mean in FP64.
        squared
            .to_dtype(DType::F64)?
            .mean_keepdim(D::Minus1)?
            .to_dtype(DType::F32)?
    } else {
        squared.mean_keepdim(D::Minus1)?
    };
    mean.affine(1., epsilon.into())?.sqrt()?.recip()
}

/// Native rotary-position operator.
#[derive(Clone, Debug, Parameterized)]
#[parameterized(tensor = "CandleTensor")]
pub struct CandleRotary {
    #[parameter(skip)]
    spec: PreparedRotarySpec,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    inverse_frequencies: Option<Vec<f32>>,
}

impl CandleRotary {
    pub(crate) fn new(spec: PreparedRotarySpec) -> Result<Self, Error> {
        if !spec.base.is_finite()
            || spec.base <= 0.
            || spec.dimensions <= 0
            || spec.dimensions % 2 != 0
        {
            return Err(Error::backend("invalid rotary specification"));
        }
        let pairs = spec.dimensions as usize / 2;
        let inverse_frequencies = match &spec.algorithm {
            PreparedRotaryAlgorithm::Standard { factor }
                if factor.is_finite() && *factor > 0. && factor.recip().is_finite() =>
            {
                None
            }
            PreparedRotaryAlgorithm::FrequencyTable(denominators)
                if denominators.len() == pairs
                    && denominators
                        .iter()
                        .all(|&frequency| frequency > 0. && frequency.recip().is_finite()) =>
            {
                Some(
                    denominators
                        .iter()
                        .map(|frequency| frequency.recip())
                        .collect(),
                )
            }
            PreparedRotaryAlgorithm::Proportional {
                factor,
                rotary_fraction,
            } if factor.is_finite()
                && *factor > 0.
                && *rotary_fraction > 0.
                && *rotary_fraction <= 1. =>
            {
                let active = (((rotary_fraction * spec.dimensions as f32).round() as usize) / 2)
                    .clamp(1, pairs);
                Some(
                    (0..pairs)
                        .map(|pair| {
                            if pair < active {
                                (spec.base.powf(pair as f32 / pairs as f32) / factor).recip()
                            } else {
                                0.
                            }
                        })
                        .collect(),
                )
            }
            PreparedRotaryAlgorithm::Yarn {
                inverse_frequencies,
                amplitude,
            } if inverse_frequencies.len() == pairs
                && inverse_frequencies
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.)
                && amplitude.is_finite()
                && *amplitude > 0. =>
            {
                Some(inverse_frequencies.clone())
            }
            _ => {
                return Err(Error::backend(
                    "requested rotary frequency algorithm is unsupported or invalid",
                ))
            }
        };
        let scale = match spec.algorithm {
            PreparedRotaryAlgorithm::Standard { factor } => factor.recip(),
            _ => 1.,
        };
        Ok(Self {
            spec,
            scale,
            inverse_frequencies,
        })
    }
}

impl RotaryOperator<CandleTensor> for CandleRotary {
    fn forward(
        &mut self,
        input: &CandleTensor,
        position: RotaryPosition<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        match position {
            RotaryPosition::Offset(offset) => crate::tensor::apply_rotary(
                input,
                self.spec.dimensions,
                self.spec.traditional,
                self.spec.base,
                self.scale,
                match self.spec.algorithm {
                    PreparedRotaryAlgorithm::Yarn { amplitude, .. } => Some(amplitude),
                    _ => None,
                },
                offset,
                self.inverse_frequencies.as_deref(),
                self.spec.arithmetic,
                context,
            ),
            RotaryPosition::Embeddings { .. }
                if self.spec.arithmetic == RotaryArithmetic::RecurrentFloat32 =>
            {
                Err(Error::backend(
                    "recurrent rotary requires an absolute offset",
                ))
            }
            RotaryPosition::Embeddings { cosine, sine } => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if input.value.device().is_metal() {
                    return CandleTensor::wrap(crate::metal_f32::explicit_rotary(
                        &input.value,
                        &cosine.value,
                        &sine.value,
                        self.spec.dimensions as usize,
                        self.spec.traditional,
                        self.spec.arithmetic,
                    ));
                }
                CandleTensor::wrap(crate::float_cpu::explicit_rotary(
                    &input.value,
                    &cosine.value,
                    &sine.value,
                    self.spec.dimensions as usize,
                    self.spec.traditional,
                    self.spec.arithmetic,
                ))
            }
        }
    }
}

pub(crate) fn silu(input: &NativeTensor) -> candle_core::Result<NativeTensor> {
    let dtype = input.dtype();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if dtype == DType::BF16 && input.device().is_metal() {
        return crate::metal_bf16::silu(input);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if crate::metal_f32::applies(input) {
        return crate::metal_f32::silu(input);
    }
    if crate::float_cpu::applies(input) {
        crate::float_cpu::silu(input)
    } else if matches!(dtype, DType::BF16 | DType::F16) {
        input.to_dtype(DType::F32)?.silu()?.to_dtype(dtype)
    } else {
        input.silu()
    }
}

pub(crate) fn gelu_approximate(input: &NativeTensor) -> candle_core::Result<NativeTensor> {
    let dtype = input.dtype();
    if matches!(dtype, DType::BF16 | DType::F16) {
        input.to_dtype(DType::F32)?.gelu()?.to_dtype(dtype)
    } else {
        input.gelu()
    }
}

pub(crate) fn gelu_approximate_f16(input: &NativeTensor) -> candle_core::Result<NativeTensor> {
    if input.dtype() != DType::F32 {
        return gelu_approximate(input);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if input.device().is_metal() {
        return crate::metal_f32::gelu_approximate_f16(input);
    }
    let rows = if input.rank() == 0 {
        input.reshape((1,))?
    } else {
        input.clone()
    };
    crate::float_cpu::map_rows(&rows, |row| {
        for value in row {
            *value = eredu_nn::activations::gelu_approximate_f16(*value);
        }
    })?
    .reshape(input.shape())
}

pub(crate) fn gated_product(
    gate: &CandleTensor,
    up: &CandleTensor,
    policy: eredu_nn::GatedProductPolicy,
) -> Result<CandleTensor, Error> {
    use eredu_nn::GatedProductActivation;
    policy.validate()?;
    CandleTensor::wrap((|| {
        let gate = match policy.gate_upper_bound() {
            Some(bound) => gate.value.minimum(bound)?,
            None => gate.value.clone(),
        };
        let up = match policy.up_absolute_bound() {
            Some(bound) => up.value.clamp(-bound, bound)?,
            None => up.value.clone(),
        };
        let up = if policy.up_offset() == 0. {
            up
        } else {
            up.affine(1., policy.up_offset().into())?
        };
        let activation = match policy.activation() {
            GatedProductActivation::Silu
                if policy.sigmoid_multiplier() == 1. && !policy.multiply_before_sigmoid() =>
            {
                silu(&gate)?
            }
            GatedProductActivation::Silu => {
                let sigmoid =
                    candle_nn::ops::sigmoid(&gate.affine(policy.sigmoid_multiplier().into(), 0.)?)?;
                if policy.multiply_before_sigmoid() {
                    let product = rounded_bf16_binary(&gate, &up, NativeTensor::broadcast_mul)?;
                    return rounded_bf16_binary(&product, &sigmoid, NativeTensor::broadcast_mul);
                }
                rounded_bf16_binary(&gate, &sigmoid, NativeTensor::broadcast_mul)?
            }
            GatedProductActivation::Gelu => gate.gelu_erf()?,
            GatedProductActivation::GeluApproximate => gate.gelu()?,
            GatedProductActivation::GeluApproximateF16 => gelu_approximate_f16(&gate)?,
            _ => candle_core::bail!("unsupported gated-product activation"),
        };
        rounded_bf16_binary(&activation, &up, NativeTensor::broadcast_mul)
    })())
}

pub(crate) fn causal_mask(
    sequence: i32,
    offset: i32,
    window: Option<i32>,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    let geometry = eredu_nn::operation_geometry::CausalMaskGeometry::new(sequence, offset, window)?;
    CandleTensor::wrap(causal_visibility(geometry, window, &context.device))
}

pub(crate) fn causal_visibility(
    geometry: eredu_nn::operation_geometry::CausalMaskGeometry,
    window: Option<i32>,
    device: &candle_core::Device,
) -> candle_core::Result<NativeTensor> {
    // Native Metal comparison kernels accept I64 positions.
    let query = NativeTensor::arange(
        i64::from(geometry.offset()),
        i64::from(geometry.keys()),
        device,
    )?
    .unsqueeze(1)?;
    let key = NativeTensor::arange(0i64, i64::from(geometry.keys()), device)?.unsqueeze(0)?;
    let visible = query.broadcast_ge(&key)?;
    match window {
        Some(distance) => {
            let earliest = query.broadcast_sub(&NativeTensor::new(i64::from(distance), device)?)?;
            visible.mul(&key.broadcast_ge(&earliest)?)
        }
        None => Ok(visible),
    }
}
