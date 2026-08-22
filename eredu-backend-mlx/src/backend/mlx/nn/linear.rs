//! Linear layers, embeddings, and language-model output heads.

use eredu_checkpoint::{BlockFp8ScaleEncoding, LinearFormat, WeightQuantization};

use safemlx::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    native_quantization::NativeQuantizedTensor,
    nn,
    ops::{matmul, quantized_matmul_with_mode, quantized_packed_dimension, QuantizationMode},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};

fn ceil_div(value: i32, divisor: i32) -> i32 {
    (value + divisor - 1) / divisor
}

/// Backend-owned linear materialization for every neutral physical format.
#[derive(Debug, Clone, ModuleParameters)]
pub struct PhysicalLinear {
    pub input_dimensions: i32,
    pub output_dimensions: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub weight_scale_inv: Param<Option<Array>>,
    #[param]
    pub scales: Param<Option<Array>>,
    #[param]
    pub biases: Param<Option<Array>>,
    #[param]
    pub bias: Param<Option<Array>>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: QuantizationMode,
    pub gguf: Option<WeightQuantization>,
}

impl PhysicalLinear {
    /// Allocates unloaded backend parameters for one typed neutral format.
    pub fn unloaded(
        input_dimensions: i32,
        output_dimensions: i32,
        bias: bool,
        format: LinearFormat,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        format
            .validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        if input_dimensions <= 0 || output_dimensions <= 0 {
            return Err(Exception::custom(format!(
                "linear dimensions must be positive, got input={input_dimensions} output={output_dimensions}"
            )));
        }
        let quantization = format.weight_quantization();
        if let Some(quantization) = quantization.filter(|value| value.gguf_iquant().is_none()) {
            if input_dimensions % quantization.group_size() != 0 {
                return Err(Exception::custom(format!(
                    "linear input dimension {input_dimensions} is not divisible by physical group size {}",
                    quantization.group_size()
                )));
            }
        }
        let (weight_shape, weight_dtype) = match format {
            LinearFormat::Dense => (vec![output_dimensions, input_dimensions], Dtype::Float32),
            LinearFormat::Affine(config) => (
                vec![
                    output_dimensions,
                    quantized_packed_dimension(input_dimensions, config.bits),
                ],
                Dtype::Uint32,
            ),
            LinearFormat::MxFp4 => (
                vec![
                    output_dimensions,
                    quantized_packed_dimension(input_dimensions, WeightQuantization::MXFP4_BITS),
                ],
                Dtype::Uint32,
            ),
            LinearFormat::GgufIQuant { ggml_type, .. } => {
                let (block_values, block_bytes) = ggml_type
                    .block_and_bytes()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                if input_dimensions % block_values as i32 != 0 {
                    return Err(Exception::custom(format!(
                        "linear input dimension {input_dimensions} is not divisible by GGUF block size {block_values}"
                    )));
                }
                (
                    vec![
                        output_dimensions,
                        input_dimensions / block_values as i32 * block_bytes as i32,
                    ],
                    Dtype::Uint8,
                )
            }
            LinearFormat::E4M3BlockFp8(fp8) => {
                if fp8.block_rows != 128 || fp8.block_columns != 128 {
                    return Err(Exception::custom(format!(
                        "MLX block-FP8 kernels require [128, 128] blocks, got [{}, {}]",
                        fp8.block_rows, fp8.block_columns
                    )));
                }
                (vec![output_dimensions, input_dimensions], Dtype::Uint8)
            }
        };
        let fp8 = match format {
            LinearFormat::E4M3BlockFp8(value) => Some(value),
            _ => None,
        };
        let affine = quantization.filter(|value| value.gguf_iquant().is_none());
        Ok(Self {
            input_dimensions,
            output_dimensions,
            weight: Param::<Array>::unloaded(&weight_shape, weight_dtype, stream)?,
            weight_scale_inv: if let Some(fp8) = fp8 {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        ceil_div(output_dimensions, fp8.block_rows),
                        ceil_div(input_dimensions, fp8.block_columns),
                    ],
                    match fp8.scale_encoding {
                        BlockFp8ScaleEncoding::FloatingPoint => Dtype::Float32,
                        BlockFp8ScaleEncoding::Ue8m0 => Dtype::Uint8,
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            scales: if let Some(quantization) = affine {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        output_dimensions,
                        input_dimensions / quantization.group_size(),
                    ],
                    if quantization == WeightQuantization::MxFp4 {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                Param::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|value| value.has_biases()) {
                Param::<Option<Array>>::unloaded_some(
                    &[
                        output_dimensions,
                        input_dimensions / quantization.group_size(),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                Param::new(None)
            },
            bias: if bias {
                Param::<Option<Array>>::unloaded_some(&[output_dimensions], Dtype::Float32, stream)?
            } else {
                Param::new(None)
            },
            group_size: affine.map_or(0, WeightQuantization::group_size),
            bits: affine.map_or(0, WeightQuantization::bits),
            mode: affine.map_or(QuantizationMode::Affine, |value| {
                crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode(value)
            }),
            gguf: quantization.filter(|value| value.gguf_iquant().is_some()),
        })
    }

    /// Applies the selected dense or packed projection without materializing
    /// weights on the host.
    pub fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut output = if let Some(quantization) = self.gguf {
            let (ggml_type, endian) = quantization.gguf_iquant().expect("GGUF format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.output_dimensions, self.input_dimensions],
                ggml_type,
                endian,
            )?
            .linear(input, true, stream)?
        } else if let Some(scales) = self.scales.as_ref() {
            quantized_matmul_with_mode(
                input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                true,
                self.group_size,
                self.bits,
                self.mode,
                stream,
            )?
        } else if let Some(scale) = self.weight_scale_inv.as_ref() {
            super::fp8::linear(input, self.weight.as_ref(), scale, stream)?
        } else {
            matmul(input, self.weight.as_ref().transpose(stream)?, stream)?
        };
        if let Some(bias) = self.bias.as_ref() {
            output = output.add(bias, stream)?;
        }
        Ok(output)
    }

    /// Applies a row-sharded projection, reducing partials before adding its
    /// optional replicated output bias exactly once.
    pub fn forward_row_parallel(
        &mut self,
        input: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let bias = self.bias.value.take();
        let partial = self.forward(input, stream);
        self.bias.value = bias;
        let output = safemlx::distributed::all_sum(&partial?, group, stream)?;
        match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }
}

/// Builds an initialized untied language-model head.
pub fn build_lm_head(hidden_size: i32, vocab_size: i32) -> Result<nn::Linear, Exception> {
    nn::LinearBuilder::new(hidden_size, vocab_size)
        .bias(false)
        .build()
}

/// Builds an unloaded untied language-model head.
pub fn build_unloaded_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<nn::Linear, Exception> {
    nn::Linear::unloaded(hidden_size, vocab_size, false, Dtype::Float32, stream)
}

/// Builds an initialized language-model head wrapped for optional quantization.
pub fn build_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    Ok(MaybeQuantized::Original(build_lm_head(
        hidden_size,
        vocab_size,
    )?))
}

/// Builds an unloaded language-model head wrapped for optional quantization.
pub fn build_unloaded_maybe_quantized_lm_head(
    hidden_size: i32,
    vocab_size: i32,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, None, stream)
}

/// Creates an unloaded linear using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_linear(
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear_with_dtype(
        input_dims,
        output_dims,
        bias,
        quantization,
        Dtype::Float32,
        stream,
    )
}

/// Creates an unloaded linear using the requested dense dtype or a quantized parameter tree.
pub fn unloaded_maybe_quantized_linear_with_dtype(
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    quantization: Option<WeightQuantization>,
    dense_dtype: Dtype,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => {
            Ok(MaybeQuantized::Quantized(nn::QuantizedLinear::unloaded_iq(
                input_dims,
                output_dims,
                ggml_type,
                endian,
                bias,
                stream,
            )?))
        }
        Some(config) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinear::unloaded_with_mode(
                input_dims,
                output_dims,
                config.group_size(),
                config.bits(),
                crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode(
                    config,
                ),
                bias,
                stream,
            )?,
        )),
        None => Ok(MaybeQuantized::Original(nn::Linear::unloaded(
            input_dims,
            output_dims,
            bias,
            dense_dtype,
            stream,
        )?)),
    }
}

/// Creates an unloaded embedding using the standard dense or affine parameter tree.
pub fn unloaded_maybe_quantized_embedding(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Embedding>, Exception> {
    unloaded_maybe_quantized_embedding_with_dtype(
        embedding_count,
        dimensions,
        quantization,
        Dtype::Float32,
        stream,
    )
}

/// Creates an unloaded embedding using the requested dense dtype or a quantized parameter tree.
pub fn unloaded_maybe_quantized_embedding_with_dtype(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    dense_dtype: Dtype,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Embedding>, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => Ok(
            MaybeQuantized::Quantized(nn::QuantizedEmbedding::unloaded_iq(
                embedding_count,
                dimensions,
                ggml_type,
                endian,
                stream,
            )?),
        ),
        Some(config) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedEmbedding::unloaded_with_mode(
                embedding_count,
                dimensions,
                config.group_size(),
                config.bits(),
                crate::backend::mlx::runtime::checkpoint::quantization::mlx_quantization_mode(
                    config,
                ),
                stream,
            )?,
        )),
        None => Ok(MaybeQuantized::Original(nn::Embedding::unloaded(
            embedding_count,
            dimensions,
            dense_dtype,
            stream,
        )?)),
    }
}

/// Builds an unloaded language-model head with optional affine quantization.
pub fn build_unloaded_maybe_quantized_lm_head_with_quantization(
    hidden_size: i32,
    vocab_size: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    unloaded_maybe_quantized_linear(hidden_size, vocab_size, false, quantization, stream)
}

/// Projects hidden states to logits, using tied embeddings when `lm_head` is absent.
pub fn project_logits_maybe_quantized(
    lm_head: &mut Option<MaybeQuantized<nn::Linear>>,
    embed_tokens: &mut MaybeQuantized<nn::Embedding>,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => match embed_tokens {
            MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(hidden_states, stream),
            MaybeQuantized::Quantized(q_embed_tokens) => {
                q_embed_tokens.as_linear(hidden_states, stream)
            }
        },
    }
}

/// Projects hidden states to logits for dense, non-quantized heads.
pub fn project_logits_dense(
    lm_head: &mut Option<nn::Linear>,
    embed_tokens: &nn::Embedding,
    hidden_states: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    match lm_head.as_mut() {
        Some(lm_head) => lm_head.forward(hidden_states, stream),
        None => embed_tokens.as_linear(hidden_states, stream),
    }
}
