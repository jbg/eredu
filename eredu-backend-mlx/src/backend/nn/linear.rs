//! Linear layers, embeddings, and language-model output heads.

use eredu_checkpoint::{BlockFp8ScaleEncoding, LinearFormat, WeightQuantization};

use eredu_backend_mlx_macros::ModuleParameters;
use safemlx::{
    error::Exception,
    ops::{matmul, quantized_matmul_with_mode, quantized_packed_dimension, QuantizationMode},
    Array, Dtype, Stream,
};

use crate::{
    module::Param, native_quantization::NativeQuantizedTensor, nn, quantization::MaybeQuantized,
};

fn ceil_div(value: i32, divisor: i32) -> i32 {
    (value + divisor - 1) / divisor
}

/// Backend-owned linear materialization for every neutral physical format.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct PhysicalLinear {
    /// Logical input width.
    pub input_dimensions: i32,
    /// Logical output width.
    pub output_dimensions: i32,
    #[param]
    /// Dense or packed weight array.
    pub weight: Param<Array>,
    #[param]
    /// Inverse scale for native block-float formats.
    pub weight_scale_inv: Param<Option<Array>>,
    #[param]
    /// Per-group affine scales, when applicable.
    pub scales: Param<Option<Array>>,
    #[param]
    /// Per-group affine biases, when applicable.
    pub biases: Param<Option<Array>>,
    #[param]
    /// Optional additive output bias.
    pub bias: Param<Option<Array>>,
    /// Number of logical values in each quantization group.
    pub group_size: i32,
    /// Number of bits in each affine-quantized value.
    pub bits: i32,
    /// Native MLX quantized-matmul mode.
    pub mode: QuantizationMode,
    /// Checkpoint-native GGUF format, when values remain block encoded.
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
        let mode = affine
            .map(crate::backend::runtime::checkpoint::quantization::mlx_quantization_mode)
            .transpose()
            .map_err(|error| Exception::custom(error.to_string()))?
            .unwrap_or(QuantizationMode::Affine);
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
            mode,
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
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let bias = self.bias.value.take();
        let partial = self.forward(input, stream);
        self.bias.value = bias;
        let output = crate::backend::runtime::distributed::all_sum(&partial?, group, stream)?;
        match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }
}

/// Creates an unloaded embedding using the standard dense or affine parameter tree.
pub(crate) fn unloaded_maybe_quantized_embedding(
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
fn unloaded_maybe_quantized_embedding_with_dtype(
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
                crate::backend::runtime::checkpoint::quantization::mlx_quantization_mode(config)
                    .map_err(|error| Exception::custom(error.to_string()))?,
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
