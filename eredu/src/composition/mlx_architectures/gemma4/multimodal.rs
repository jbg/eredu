use eredu_checkpoint::WeightQuantization;

use std::ops::Range;

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{clip, indexing::TryIndexOp, rsqrt, sum_axis},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

use super::model::{maybe_quantized_linear_with_bias, rms_norm_without_scale};
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Gemma4ClippedLinear {
    pub input_dim: i32,
    pub output_dim: i32,
    #[param]
    pub linear: MaybeQuantized<nn::Linear>,
    #[param]
    pub input_min: Param<Array>,
    #[param]
    pub input_max: Param<Array>,
    #[param]
    pub output_min: Param<Array>,
    #[param]
    pub output_max: Param<Array>,
}

impl Gemma4ClippedLinear {
    pub(crate) fn new_quantized(
        input: i32,
        output: i32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            input_dim: input,
            output_dim: output,
            linear: maybe_quantized_linear_with_bias(quantization, input, output, bias, stream)?,
            input_min: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            input_max: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            output_min: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
            output_max: Param::<Array>::unloaded(&[], Dtype::Float32, stream)?,
        })
    }

    pub(crate) fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let x = clip(x, (&*self.input_min, &*self.input_max), stream)?;
        let output = self.linear.forward(&x, stream)?;
        clip(output, (&*self.output_min, &*self.output_max), stream)
    }

    pub(crate) fn forward_row_parallel(
        &mut self,
        x: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let x = clip(x, (&*self.input_min, &*self.input_max), stream)?;
        let mut partial = self.linear.forward(&x, stream)?;
        if let Some(bias) = maybe_quantized_linear_bias(&self.linear) {
            partial = partial.subtract(bias, stream)?;
        }
        let mut output = safemlx::distributed::all_sum(&partial, group, stream)?;
        if let Some(bias) = maybe_quantized_linear_bias(&self.linear) {
            output = output.add(bias, stream)?;
        }
        clip(output, (&*self.output_min, &*self.output_max), stream)
    }
}

pub(crate) fn maybe_quantized_linear_bias(linear: &MaybeQuantized<nn::Linear>) -> Option<&Array> {
    match linear {
        MaybeQuantized::Original(linear) => linear.bias.as_ref().as_ref(),
        MaybeQuantized::Quantized(linear) => linear.inner.bias.as_ref().as_ref(),
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub(crate) struct Gemma4ModalityEmbedder {
    pub eps: f32,
    pub input_size: i32,
    pub parallel_input_range: Option<Range<usize>>,
    #[quantizable]
    #[param]
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl Gemma4ModalityEmbedder {
    pub(crate) fn new(
        input_size: i32,
        output_size: i32,
        eps: f32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        Ok(Self {
            eps,
            input_size,
            parallel_input_range: None,
            embedding_projection: maybe_quantized_linear_with_bias(
                quantization,
                input_size,
                output_size,
                bias,
                stream,
            )?,
        })
    }

    pub(crate) fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        self.embedding_projection
            .forward(&rms_norm_without_scale(x, self.eps, stream)?, stream)
    }

    pub(crate) fn new_tensor_parallel(
        input_size: i32,
        output_size: i32,
        eps: f32,
        bias: bool,
        quantization: Option<WeightQuantization>,
        local_input_range: Range<usize>,
        stream: &Stream,
    ) -> Result<Self, crate::backend::mlx::error::Error> {
        let input = usize::try_from(input_size).map_err(|_| {
            crate::backend::mlx::error::Error::Parallel("negative modality width".into())
        })?;
        if local_input_range.start >= local_input_range.end || local_input_range.end > input {
            return Err(crate::backend::mlx::error::Error::Parallel(format!(
                "invalid Gemma modality input range {local_input_range:?} for width {input}"
            )));
        }
        let local = local_input_range.len();
        Ok(Self {
            eps,
            input_size,
            parallel_input_range: Some(local_input_range),
            embedding_projection: maybe_quantized_linear_with_bias(
                quantization,
                i32::try_from(local).map_err(|_| {
                    crate::backend::mlx::error::Error::Parallel(
                        "local modality width exceeds i32".into(),
                    )
                })?,
                output_size,
                bias,
                stream,
            )?,
        })
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        x: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let Some(range) = &self.parallel_input_range else {
            let normalized = rms_norm_without_scale(x, self.eps, stream)?;
            return self.embedding_projection.forward(&normalized, stream);
        };
        let local = if x.dim(-1) as usize == range.len() && range.len() < self.input_size as usize {
            let local_sum = sum_axis(&x.square(stream)?, -1, true, stream)?;
            let global_sum = safemlx::distributed::all_sum(&local_sum, group, stream)?;
            x.multiply(
                rsqrt(
                    global_sum
                        .divide(Array::from_f32(self.input_size as f32), stream)?
                        .add(Array::from_f32(self.eps), stream)?,
                    stream,
                )?,
                stream,
            )?
        } else {
            rms_norm_without_scale(x, self.eps, stream)?
                .try_index_device((.., .., range.start as i32..range.end as i32), stream)?
        };
        crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut self.embedding_projection,
            &local,
            group,
            stream,
        )
    }
}
