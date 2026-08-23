//! Activation functions.

use safemlx::{
    error::Exception,
    ops::{maximum, sigmoid},
    Array, Stream,
};

/// Applies the SiLU activation function.
pub fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    x.multiply(sigmoid(&x, stream)?, stream)
}

/// Applies the squared rectified-linear activation.
pub fn relu2(x: Array, stream: &Stream) -> Result<Array, Exception> {
    maximum(&x, Array::from_f32(0.0), stream)?.square(stream)
}
