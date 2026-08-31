//! Embedding layer.

use crate::module::Module;
use crate::module::PhysicalParam;
use crate::ops::indexing::TryIndexOp;
use crate::quantization::Quantizable;
use crate::{Array, Dtype, Stream};
use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::error::Exception;

use super::QuantizedEmbedding;

/// Implements a simple lookup table that maps each input integer to a high-dimensional vector.
///
/// Typically used to embed discrete tokens for processing by neural networks.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct Embedding {
    /// The weight of the
    #[param]
    pub weight: PhysicalParam<Array>,
}

impl Embedding {
    /// Creates an embedding layer whose weight carries only shape metadata.
    ///
    /// This is intended for modules that will immediately load real
    /// checkpoint weights before any forward pass.
    pub fn unloaded(
        embedding_count: i32,
        dimensions: i32,
        dtype: Dtype,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: PhysicalParam::<Array>::unloaded(
                &[embedding_count, dimensions],
                dtype,
                stream,
            )?,
        })
    }

    /// Call the embedding layer as a linear layer.
    ///
    /// Use this for example when input embedding and output projection
    /// weights are tied.
    pub fn as_linear(&self, x: &Array, stream: &crate::Stream) -> Result<Array, Exception> {
        crate::ops::matmul(x, self.weight.value.transpose(stream)?, stream)
    }
}

impl Quantizable for Embedding {
    type Quantized = QuantizedEmbedding;

    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &crate::Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        QuantizedEmbedding::try_from_embedding(self, group_size, bits, stream)
    }
}

impl Module<&Array> for Embedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array, Self::Error> {
        self.weight.try_index_device(x, stream)
    }
}
