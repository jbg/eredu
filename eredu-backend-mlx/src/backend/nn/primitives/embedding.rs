//! Embedding layer.

use crate::module::Module;
use crate::module::PhysicalParam;
use crate::ops::indexing::TryIndexOp;
use crate::{Array, Dtype, Stream};
use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::error::Exception;

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
        if let Some(output) =
            crate::backend::nn::mixed_projection::project(x, self.weight.as_ref(), stream)?
        {
            #[cfg(feature = "projection-profiling")]
            crate::backend::nn::projection_profile::record(
                x,
                self.weight.as_ref(),
                &output,
                "tied_embedding",
                crate::backend::nn::mixed_projection::profile_path(x),
                stream,
            );
            return Ok(output);
        }
        let promoted = crate::backend::nn::parameter_conversion::promoted_weight(
            x,
            self.weight.as_ref(),
            stream,
        )?;
        let weight = promoted.as_ref().unwrap_or(self.weight.as_ref());
        let output = crate::ops::matmul(x, weight.transpose(stream)?, stream)?;
        #[cfg(feature = "projection-profiling")]
        crate::backend::nn::projection_profile::record(
            x,
            self.weight.as_ref(),
            &output,
            "tied_embedding",
            if promoted.is_some() {
                "native_with_retained_conversion"
            } else {
                "native_fallback"
            },
            stream,
        );
        Ok(output)
    }
}

impl Module<&Array> for Embedding {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array, Self::Error> {
        self.weight.try_index_device(x, stream)
    }
}
