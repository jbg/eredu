use std::iter::once;

use eredu_backend_mlx_macros::PhysicalParameters;
use eredu_gguf::{Endian as GgufEndian, GgmlType};
use safemlx::{
    error::Exception,
    ops::indexing::TryIndexOp,
    ops::{
        self, dequantize_with_mode, quantized_matmul_with_mode, quantized_packed_dimension,
        QuantizationMode,
    },
    Array, Dtype, Stream,
};

use crate::{
    module::{Module, PhysicalParam, PhysicalParameters},
    native_quantization::{NativeQuantizationFormat, NativeQuantizedTensor},
    nn::Embedding,
};

/// The same as ``Embedding`` but with a quantized weight matrix.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct QuantizedEmbedding {
    /// Quantization group size. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    pub group_size: i32,

    /// Bits per parameter. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub bits: i32,

    /// Quantized weight encoding.
    pub mode: QuantizationMode,

    /// Optional checkpoint-native storage used instead of affine parameters.
    pub native: Option<NativeQuantizedTensor>,

    /// Checkpoint-native IQ format stored in `inner.weight`.
    pub native_format: Option<NativeQuantizationFormat>,

    /// Byte order of checkpoint-native IQ blocks.
    pub native_endian: GgufEndian,

    /// Logical input width represented by checkpoint-native IQ blocks.
    pub native_columns: i32,

    /// Scales
    #[param]
    pub scales: PhysicalParam<Array>,

    /// Biases
    #[param]
    pub biases: PhysicalParam<Option<Array>>,

    /// Inner embedding
    #[param]
    pub inner: Embedding,
}

fn build_quantized_embedding_inner(
    weight: Array,
    group_size: i32,
    bits: i32,
    mode: QuantizationMode,
    stream: &crate::Stream,
) -> Result<QuantizedEmbedding, Exception> {
    let arrays = ops::quantize_with_mode(&weight, group_size, bits, mode, stream)?;

    let inner = Embedding {
        weight: PhysicalParam::new(arrays.weight),
    };

    let mut qe = QuantizedEmbedding {
        group_size,
        bits,
        mode,
        native: None,
        native_format: None,
        native_endian: GgufEndian::Little,
        native_columns: 0,
        scales: PhysicalParam::new(arrays.scales),
        biases: PhysicalParam::new(arrays.biases),
        inner,
    };

    // Freeze all parameters
    qe.freeze_parameters(true);

    Ok(qe)
}

impl QuantizedEmbedding {
    /// Default group size
    pub const DEFAULT_GROUP_SIZE: i32 = 64;

    /// Default bits
    pub const DEFAULT_BITS: i32 = 4;

    /// Creates an unloaded quantized embedding with an explicit encoding.
    pub fn unloaded_with_mode(
        embedding_count: i32,
        dimensions: i32,
        group_size: i32,
        bits: i32,
        mode: QuantizationMode,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        mode.validate(group_size, bits)?;
        let stream = stream.as_ref();
        let scale_dtype = if mode == QuantizationMode::MxFp4 {
            Dtype::Uint8
        } else {
            Dtype::Float32
        };
        let inner = Embedding {
            weight: PhysicalParam::<Array>::unloaded(
                &[
                    embedding_count,
                    quantized_packed_dimension(dimensions, bits),
                ],
                Dtype::Uint32,
                stream,
            )?,
        };
        let mut qe = Self {
            group_size,
            bits,
            mode,
            native: None,
            native_format: None,
            native_endian: GgufEndian::Little,
            native_columns: 0,
            scales: PhysicalParam::<Array>::unloaded(
                &[embedding_count, dimensions / group_size],
                scale_dtype,
                stream,
            )?,
            biases: if mode.has_biases() {
                PhysicalParam::<Option<Array>>::unloaded_some(
                    &[embedding_count, dimensions / group_size],
                    Dtype::Float32,
                    stream,
                )?
            } else {
                PhysicalParam::new(None)
            },
            inner,
        };
        qe.freeze_parameters(true);
        Ok(qe)
    }

    /// Creates an unloaded embedding backed by checkpoint-native GGML IQ rows.
    pub fn unloaded_iq(
        embedding_count: i32,
        dimensions: i32,
        ggml_type: GgmlType,
        endian: GgufEndian,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        let format = NativeQuantizationFormat::from_ggml_type(ggml_type)
            .ok_or_else(|| Exception::custom(format!("{ggml_type:?} is not an IQ type")))?;
        let (block_values, block_bytes) = format.block_geometry();
        if dimensions <= 0 || dimensions % block_values != 0 {
            return Err(Exception::custom(format!(
                "IQ embedding width {dimensions} is not divisible by {block_values}"
            )));
        }
        let stream = stream.as_ref();
        let mut embedding = Self {
            group_size: block_values,
            bits: block_bytes,
            mode: QuantizationMode::Affine,
            native: None,
            native_format: Some(format),
            native_endian: endian,
            native_columns: dimensions,
            scales: PhysicalParam::<Array>::unloaded(&[1], Dtype::Float32, stream)?,
            biases: PhysicalParam::new(None),
            inner: Embedding {
                weight: PhysicalParam::<Array>::unloaded(
                    &[embedding_count, dimensions / block_values * block_bytes],
                    Dtype::Uint8,
                    stream,
                )?,
            },
        };
        embedding.freeze_parameters(true);
        Ok(embedding)
    }

    /// Convert an embedding layer to a quantized embedding layer.
    ///
    /// # Params
    ///
    /// - `embedding`: The embedding layer to convert.
    /// - `group_size`: The group size to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_GROUP_SIZE`]
    /// - `bits`: The bit width to use for the quantized weight. Default to [`QuantizedEmbedding::DEFAULT_BITS`]
    pub fn try_from_embedding(
        embedding: Embedding,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
        stream: &crate::Stream,
    ) -> Result<Self, Exception> {
        let group_size = group_size.into().unwrap_or(Self::DEFAULT_GROUP_SIZE);
        let bits = bits.into().unwrap_or(Self::DEFAULT_BITS);
        build_quantized_embedding_inner(
            embedding.weight.value,
            group_size,
            bits,
            QuantizationMode::Affine,
            stream,
        )
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(
        &self,
        x: impl AsRef<Array>,
        stream: &crate::Stream,
    ) -> Result<Array, Exception> {
        if let Some(native) = &self.native {
            return native.linear(x.as_ref(), true, stream);
        }
        if let Some(format) = self.native_format {
            let native = NativeQuantizedTensor::from_iq_array(
                self.inner.weight.value.clone(),
                &[self.inner.weight.value.dim(0), self.native_columns],
                format.ggml_type().expect("IQ format"),
                self.native_endian,
            )?;
            return native.linear(x.as_ref(), true, stream);
        }
        quantized_matmul_with_mode(
            x.as_ref(),
            &self.inner.weight,
            &self.scales,
            self.biases.value.as_ref(),
            true,
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )
    }
}

impl Module<&Array> for QuantizedEmbedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array, Self::Error> {
        if let Some(native) = &self.native {
            return native.embedding(x, stream);
        }
        if let Some(format) = self.native_format {
            let native = NativeQuantizedTensor::from_iq_array(
                self.inner.weight.value.clone(),
                &[self.inner.weight.value.dim(0), self.native_columns],
                format.ggml_type().expect("IQ format"),
                self.native_endian,
            )?;
            return native.embedding(x, stream);
        }
        let s = x.shape();
        let x = x.flatten(None, None, stream)?;
        let w = self.inner.weight.try_index_device(&x, stream)?;
        let scales = self.scales.try_index_device(&x, stream)?;
        let biases = self
            .biases
            .value
            .as_ref()
            .map(|biases| biases.try_index_device(&x, stream))
            .transpose()?;

        let out = dequantize_with_mode(
            &w,
            &scales,
            biases.as_ref(),
            self.group_size,
            self.bits,
            self.mode,
            stream,
        )?;

        let ret_shape = s.iter().copied().chain(once(-1)).collect::<Vec<_>>();
        out.reshape(&ret_shape, stream)
    }
}
