//! Linear layers, embeddings, and language-model output heads.

use eredu_checkpoint::{BlockFp8ScaleEncoding, LinearFormat, WeightQuantization};

use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::{
    error::Exception,
    ops::{matmul, quantized_matmul_with_mode, quantized_packed_dimension, QuantizationMode},
    Array, Dtype, Stream,
};

use crate::{
    module::{Module, ModuleParamMut, ModuleParamRef, PhysicalParam, PhysicalParameters},
    native_quantization::NativeQuantizedTensor,
    nn,
};

/// Shared dense dispatch for ordinary and restored packed parameters.
pub(crate) fn dense_projection(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    if let Some(output) = super::mixed_projection::project(input, weight, stream)? {
        #[cfg(feature = "projection-profiling")]
        super::projection_profile::record(
            input,
            weight,
            &output,
            "dense",
            super::mixed_projection::profile_path(input),
            stream,
        );
        return Ok(output);
    }
    if let Some(output) = super::matrix::bf16_row_projection(input, weight, None, stream)? {
        return Ok(output);
    }
    let promoted = super::parameter_conversion::promoted_weight(input, weight, stream)?;
    let output = matmul(
        input,
        promoted.as_ref().unwrap_or(weight).transpose(stream)?,
        stream,
    )?;
    #[cfg(feature = "projection-profiling")]
    super::projection_profile::record(
        input,
        weight,
        &output,
        "dense",
        if promoted.is_some() {
            "native_with_retained_conversion"
        } else {
            "native_fallback"
        },
        stream,
    );
    Ok(output)
}

fn ceil_div(value: i32, divisor: i32) -> i32 {
    (value + divisor - 1) / divisor
}

/// Backend-owned dense or checkpoint-quantized embedding materialization.
#[derive(Debug, Clone)]
pub(crate) enum PhysicalEmbedding {
    /// Dense embedding weights.
    Dense(nn::Embedding),
    /// Packed affine or checkpoint-native embedding weights.
    Quantized(nn::QuantizedEmbedding),
}

impl PhysicalEmbedding {
    /// Weighted row access without retaining a fully dequantized packed table.
    pub(crate) fn weighted_sum(
        &mut self,
        coefficients: &Array,
        vocabulary: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if coefficients.ndim() == 0 || coefficients.dim(-1) != vocabulary || vocabulary <= 0 {
            return Err(Exception::custom(
                "weighted embedding requires a complete vocabulary axis",
            ));
        }
        if let Self::Dense(embedding) = self {
            return matmul(
                coefficients.as_dtype(embedding.weight.dtype(), stream)?,
                embedding.weight.as_ref(),
                stream,
            );
        }
        use safemlx::ops::indexing::TryIndexOp;
        let shape = coefficients.shape();
        let flat = coefficients.reshape(&[-1, vocabulary], stream)?;
        let mut result: Option<Array> = None;
        let mut output_dtype = None;
        for start in (0..vocabulary).step_by(512) {
            let end = (start + 512).min(vocabulary);
            let ids = Array::arange::<i32, i32>(Some(start), end, None, stream)?;
            let rows = self.forward(&ids, stream)?;
            output_dtype = Some(rows.dtype());
            // Round coefficients to table arithmetic once, then accumulate
            // across blocks in FP32. Rounding each block's output changes the
            // embedding with the chosen residency granularity.
            let weights = flat
                .try_index_device((.., start..end), stream)?
                .as_dtype(rows.dtype(), stream)?
                .as_dtype(Dtype::Float32, stream)?;
            let contribution = matmul(weights, rows.as_dtype(Dtype::Float32, stream)?, stream)?;
            let value = match result {
                Some(previous) => previous.add(contribution, stream)?,
                None => contribution,
            };
            // Completion bounds decoded rows and the accumulation graph to one
            // block. The resulting embedding remains a native tensor.
            safemlx::transforms::eval([&value])?;
            result = Some(value);
        }
        let value = result
            .ok_or_else(|| Exception::custom("weighted embedding has an empty vocabulary"))?;
        let mut output_shape = shape.to_vec();
        *output_shape.last_mut().unwrap() = value.dim(-1);
        value
            .as_dtype(output_dtype.unwrap(), stream)?
            .reshape(&output_shape, stream)
    }

    pub(crate) fn as_linear(&self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        match self {
            Self::Dense(embedding) => embedding.as_linear(input, stream),
            Self::Quantized(embedding) => embedding.as_linear(input, stream),
        }
    }
}

impl PhysicalParameters for PhysicalEmbedding {
    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(embedding) => embedding.parameters(),
            Self::Quantized(embedding) => embedding.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Dense(embedding) => embedding.parameters_mut(),
            Self::Quantized(embedding) => embedding.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Dense(embedding) => embedding.trainable_parameters(),
            Self::Quantized(embedding) => embedding.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(embedding) => embedding.freeze_parameters(recursive),
            Self::Quantized(embedding) => embedding.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(embedding) => embedding.unfreeze_parameters(recursive),
            Self::Quantized(embedding) => embedding.unfreeze_parameters(recursive),
        }
    }
}

impl Module<&Array> for PhysicalEmbedding {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        match self {
            Self::Dense(embedding) => embedding.forward(input, stream),
            Self::Quantized(embedding) => embedding.forward(input, stream),
        }
    }
}

/// Native projection-input evidence; factories run only under the observer's reservation.
pub(crate) trait NativeProjectionInputObserver {
    fn observe(&mut self, input: &Array) -> Result<(), Exception>;
    fn observe_generated(
        &mut self,
        prototype: &Array,
        source: &eredu_nn::GeneratedTensorSource,
        generate: &mut dyn FnMut() -> Result<Array, Exception>,
    ) -> Result<(), Exception>;
}

/// Backend-owned linear materialization for every neutral physical format.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct PhysicalLinear {
    /// Logical input width.
    pub input_dimensions: i32,
    /// Logical output width.
    pub output_dimensions: i32,
    #[param]
    /// Dense or packed weight array.
    pub weight: PhysicalParam<Array>,
    #[param]
    /// Inverse scale for native block-float formats.
    pub weight_scale_inv: PhysicalParam<Option<Array>>,
    #[param]
    /// Per-group affine scales, when applicable.
    pub scales: PhysicalParam<Option<Array>>,
    #[param]
    /// Per-group affine biases, when applicable.
    pub biases: PhysicalParam<Option<Array>>,
    #[param]
    /// F32 multiplier for retained NVFP4 weights.
    pub global_scale: PhysicalParam<Option<Array>>,
    #[param]
    /// Optional additive output bias.
    pub bias: PhysicalParam<Option<Array>>,
    /// Number of logical values in each quantization group.
    pub group_size: i32,
    /// Number of bits in each affine-quantized value.
    pub bits: i32,
    /// Native MLX quantized-matmul mode.
    pub mode: QuantizationMode,
    /// Checkpoint-native GGUF format, when values remain block encoded.
    pub gguf: Option<WeightQuantization>,
    /// Portable exact-zero input hint; permits no approximate pruning.
    pub exact_zero_inputs: bool,
}

impl PhysicalLinear {
    /// Geometry where the sparse generic-QMV arithmetic has measured utility.
    /// Widths selecting MLX's differently ordered fast QMV retain that mechanism.
    pub(crate) fn prefers_transposed_storage(&self) -> bool {
        cfg!(all(feature = "metal", not(feature = "cuda")))
            && self.exact_zero_inputs
            && self.bits == 8
            && self.group_size == 64
            && self.gguf.is_none()
            && self.mode == QuantizationMode::Affine
            && self.input_dimensions >= 4096
            && self.output_dimensions >= 1024
            && self.output_dimensions <= 65536
            && self.input_dimensions > self.output_dimensions
            && self.input_dimensions % 512 != 0
    }

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
            LinearFormat::MxFp4 | LinearFormat::NvFp4 => (
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
            weight: PhysicalParam::<Array>::unloaded(&weight_shape, weight_dtype, stream)?,
            weight_scale_inv: if let Some(fp8) = fp8 {
                PhysicalParam::<Option<Array>>::unloaded_some(
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
                PhysicalParam::new(None)
            },
            scales: if let Some(quantization) = affine {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[
                        output_dimensions,
                        input_dimensions / quantization.group_size(),
                    ],
                    if matches!(
                        quantization,
                        WeightQuantization::MxFp4 | WeightQuantization::NvFp4
                    ) {
                        Dtype::Uint8
                    } else {
                        Dtype::Float32
                    },
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            biases: if let Some(quantization) = affine.filter(|value| value.has_biases()) {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[
                        output_dimensions,
                        input_dimensions / quantization.group_size(),
                    ],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            global_scale: if format == LinearFormat::NvFp4 {
                PhysicalParam::<Option<Array>>::unloaded_some(&[], Dtype::Float32, stream)?
            } else {
                PhysicalParam::new(None)
            },
            bias: if bias {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[output_dimensions],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            group_size: affine.map_or(0, WeightQuantization::group_size),
            bits: affine.map_or(0, WeightQuantization::bits),
            mode,
            gguf: quantization.filter(|value| value.gguf_iquant().is_some()),
            exact_zero_inputs: false,
        })
    }

    /// Applies the selected dense or packed projection without materializing
    /// weights on the host.
    pub fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        self.forward_with_input_observer(input, stream, None)
    }

    pub(crate) fn forward_with_input_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        observer: Option<&mut dyn NativeProjectionInputObserver>,
    ) -> Result<Array, Exception> {
        let output = self.project_without_bias(input, stream, observer)?;
        match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        }
    }

    /// Both ordinary and distributed execution use the same selected projection.
    /// Keeping the bias out of this equation avoids mutating retained parameters
    /// while invoking fallible, user-controlled input observers.
    fn project_without_bias(
        &self,
        input: &Array,
        stream: &Stream,
        mut observer: Option<&mut dyn NativeProjectionInputObserver>,
    ) -> Result<Array, Exception> {
        let floating = self.weight.as_ref().dtype().is_float();
        let transforms_input = !floating
            && self.gguf.is_none()
            && self.scales.as_ref().is_none()
            && self.weight_scale_inv.as_ref().is_some();
        if !transforms_input {
            if let Some(observer) = observer.as_mut() {
                observer.observe(input)?;
            }
        }
        let output = if matches!(
            self.weight.as_ref().dtype(),
            Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
        ) {
            // Admitted packed edits publish only the affected parameter as F32.
            // Ordinary dense parameters and restored packed parameters take their
            // original paths; companions are retained unchanged for restoration.
            dense_projection(input, self.weight.as_ref(), stream)?
        } else if let Some(quantization) = self.gguf {
            let (ggml_type, endian) = quantization.gguf_iquant().expect("GGUF format");
            NativeQuantizedTensor::from_iq_array(
                self.weight.value.clone(),
                &[self.output_dimensions, self.input_dimensions],
                ggml_type,
                endian,
            )?
            .linear(input, true, stream)?
        } else if let Some(global) = self.global_scale.as_ref() {
            super::nvfp4::linear(
                input,
                self.weight.as_ref(),
                self.scales
                    .as_ref()
                    .as_ref()
                    .ok_or_else(|| Exception::custom("NVFP4 block scales are absent"))?,
                global,
                eredu_nn::LinearRowLayout::Contiguous,
                stream,
            )?
        } else if let Some(scales) = self.scales.as_ref() {
            let sparse = if self.prefers_transposed_storage() {
                self.biases
                    .as_ref()
                    .as_ref()
                    .map(|biases| {
                        super::sparse_q8::project(
                            input,
                            self.weight.as_ref(),
                            scales,
                            biases,
                            stream,
                        )
                    })
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let output = match sparse {
                Some(output) => output,
                None => quantized_matmul_with_mode(
                    input,
                    self.weight.as_ref(),
                    scales,
                    self.biases.as_ref().as_ref(),
                    true,
                    self.group_size,
                    self.bits,
                    self.mode,
                    stream,
                )?,
            };
            #[cfg(feature = "projection-profiling")]
            super::projection_profile::affine::record(
                input,
                self.weight.as_ref(),
                scales,
                self.biases.as_ref().as_ref(),
                &output,
                self.group_size,
                self.bits,
                self.mode,
                "linear",
                stream,
            );
            output
        } else if let Some(scale) = self.weight_scale_inv.as_ref() {
            super::fp8::linear_with_input_observer(
                input,
                self.weight.as_ref(),
                scale,
                stream,
                observer,
            )?
        } else {
            dense_projection(input, self.weight.as_ref(), stream)?
        };
        Ok(output)
    }

    /// Reduces row-sharded partials before adding a replicated output bias once.
    pub(crate) fn forward_row_parallel_with_compute(
        &mut self,
        input: &Array,
        group: &crate::backend::runtime::distributed::Group,
        stream: &Stream,
        observer: Option<&mut dyn NativeProjectionInputObserver>,
        compute_dtype: Option<Dtype>,
    ) -> Result<Array, Exception> {
        let guarded = group.has_local_phase_agreement();
        let prepared = (|| {
            let cast = compute_dtype
                .map(|dtype| input.as_dtype(dtype, stream))
                .transpose()?;
            let partial =
                self.project_without_bias(cast.as_ref().unwrap_or(input), stream, observer)?;
            if compute_dtype.is_some() {
                partial.as_dtype(input.dtype(), stream)
            } else {
                Ok(partial)
            }
        })();
        let partial = if guarded {
            let prepared = prepared.and_then(|partial| {
                // Admit the exact immutable data contract before any participant
                // enters the reduction. Native completion and transport failures
                // remain owned by the collective itself.
                use eredu_runtime::CommunicationOperation;
                group.completion_policy().ok_or_else(|| {
                    Exception::custom("guarded row projection requires bounded completion")
                })?;
                group.validate_tensor(CommunicationOperation::AllReduceSum, &partial, false)?;
                group.validate_expected_output(
                    CommunicationOperation::AllReduceSum,
                    partial.dtype(),
                    partial.ndim(),
                    partial.size(),
                )?;
                Ok(partial)
            });
            agree_row_parallel_result(group, 0, prepared, self.bias.as_ref().as_ref(), stream)?
        } else {
            prepared?
        };
        // No recovery status exchange follows a failed data collective: its
        // retained authority establishes completion or fences further work.
        let output = crate::backend::runtime::distributed::all_sum(&partial, group, stream)?;
        let output = match self.bias.as_ref() {
            Some(bias) => output.add(bias, stream),
            None => Ok(output),
        };
        if guarded {
            agree_row_parallel_result(group, 1, output, None, stream)
        } else {
            output
        }
    }
}

/// Agrees only replicated output geometry. Input feature widths may differ
/// across balanced row shards; bias broadcasting keeps its native semantics.
fn agree_row_parallel_result(
    group: &crate::backend::runtime::distributed::Group,
    phase: u64,
    local: Result<Array, Exception>,
    bias: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    const PROTOCOL: u64 = 0x524f57504152414c;
    let mut words = vec![u64::from(bias.is_some())];
    for value in local.as_ref().ok().into_iter().chain(bias) {
        words.extend([value.dtype() as u64, value.ndim() as u64]);
        words.extend(value.shape().iter().map(|&extent| extent as u64));
    }
    // All participants enter the fixed-size header before geometry-dependent
    // exchanges, including when graph construction or observation failed.
    let accepted = group.agree_local_phase(
        [PROTOCOL, phase, words.len() as u64, 0, 0, 0, 0, 0],
        local.is_ok(),
        stream,
    );
    let value = local?;
    if !accepted.map_err(Exception::from_source)? {
        return Err(Exception::custom(
            "a row-projection participant failed or selected different output geometry",
        ));
    }
    for (index, chunk) in words.chunks(6).enumerate() {
        let mut descriptor = [u64::MAX; 8];
        descriptor[0] = PROTOCOL;
        descriptor[1] = index as u64;
        descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
        if !group
            .agree_local_phase(descriptor, true, stream)
            .map_err(Exception::from_source)?
        {
            return Err(Exception::custom(
                "row-projection participant output or bias geometry differs",
            ));
        }
    }
    Ok(value)
}

/// Creates an unloaded embedding using its checkpoint-declared physical format.
pub(crate) fn unloaded_embedding(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
) -> Result<PhysicalEmbedding, Exception> {
    unloaded_embedding_with_dtype(
        embedding_count,
        dimensions,
        quantization,
        Dtype::Float32,
        stream,
    )
}

/// Creates an unloaded embedding using the requested dense dtype or packed parameter tree.
fn unloaded_embedding_with_dtype(
    embedding_count: i32,
    dimensions: i32,
    quantization: Option<WeightQuantization>,
    dense_dtype: Dtype,
    stream: &Stream,
) -> Result<PhysicalEmbedding, Exception> {
    match quantization {
        Some(WeightQuantization::GgufIQuant { ggml_type, endian }) => Ok(
            PhysicalEmbedding::Quantized(nn::QuantizedEmbedding::unloaded_iq(
                embedding_count,
                dimensions,
                ggml_type,
                endian,
                stream,
            )?),
        ),
        Some(config) => Ok(PhysicalEmbedding::Quantized(
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
        None => Ok(PhysicalEmbedding::Dense(nn::Embedding::unloaded(
            embedding_count,
            dimensions,
            dense_dtype,
            stream,
        )?)),
    }
}
