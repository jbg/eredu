//! Backend-owned RMS normalization.

use eredu_backend_mlx_macros::PhysicalParameters;
use safemlx::{error::Exception, Array, Dtype, Stream};

use crate::module::{Module, PhysicalParam};

/// MLX RMS normalization with a checkpoint-bound scale parameter.
#[derive(Debug, Clone, PhysicalParameters)]
#[module(root = crate)]
pub struct RmsNorm {
    /// Learned scale.
    #[param]
    pub weight: PhysicalParam<Array>,
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl RmsNorm {
    /// Creates an unloaded RMS normalization parameter.
    pub fn unloaded(
        dimensions: i32,
        eps: f32,
        dtype: Dtype,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Ok(Self {
            weight: PhysicalParam::<Array>::unloaded(&[dimensions], dtype, stream)?,
            eps,
        })
    }
}

impl Module<&Array> for RmsNorm {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        safemlx::fast::rms_norm(input, self.weight.as_ref(), self.eps, stream)
    }
}
