//! Activation functions.

use std::f32::consts::PI;

use safemlx::{
    array,
    error::{Exception, Result as MlxResult},
    ops::{erf, exp, maximum, r#where, sqrt, tanh},
    Array, Stream,
};

/// Applies the SiLU activation function.
pub fn silu(x: Array, stream: &Stream) -> Result<Array, Exception> {
    let dtype = x.dtype();
    // SiLU is one pointwise operation. Keep its intermediate sigmoid in FP32
    // for low-precision input so it is rounded only once at the output.
    let work = match dtype {
        safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16 => {
            x.as_dtype(safemlx::Dtype::Float32, stream)?
        }
        _ => x,
    };
    if let Some(output) =
        super::arithmetic::f32_pointwise(&work, super::arithmetic::Pointwise::Silu, stream)?
    {
        return output.as_dtype(dtype, stream);
    }
    work.divide(
        work.negative(stream)?
            .exp(stream)?
            .add(Array::from_f32(1.0), stream)?,
        stream,
    )?
    .as_dtype(dtype, stream)
}

/// Applies sigmoid with FP32 exponential and quotient rounding.
pub fn sigmoid(input: Array, stream: &Stream) -> Result<Array, Exception> {
    match super::arithmetic::f32_pointwise(&input, super::arithmetic::Pointwise::Sigmoid, stream)? {
        Some(output) => Ok(output),
        None => safemlx::ops::sigmoid(input, stream),
    }
}

/// Applies the squared rectified-linear activation.
pub fn relu2(x: Array, stream: &Stream) -> Result<Array, Exception> {
    maximum(&x, Array::from_f32(0.0), stream)?.square(stream)
}

/// Applies the exponential linear unit.
pub fn elu(
    x: impl AsRef<Array>,
    alpha: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> MlxResult<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    let alpha = array!(alpha.into().unwrap_or(1.0));
    r#where(
        &x.gt(array!(0.0), stream)?,
        x,
        alpha.multiply(exp(x, stream)?.subtract(array!(1.0), stream)?, stream)?,
        stream,
    )
}

/// Applies softplus.
pub fn softplus(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    safemlx::ops::logaddexp(x.as_ref(), array!(0), stream)
}

/// Applies log-sigmoid.
pub fn log_sigmoid(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    let stream = stream.as_ref();
    softplus(x.as_ref().negative(stream)?, stream)?.negative(stream)
}

/// Applies exact GELU.
pub fn gelu(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    let stream = stream.as_ref();
    x.as_ref()
        .multiply(
            array!(1).add(
                erf(x.as_ref().divide(array!(2f32.sqrt()), stream)?, stream)?,
                stream,
            )?,
            stream,
        )?
        .divide(array!(2.0), stream)
}

/// Applies the tanh GELU approximation.
pub fn gelu_approximate(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    array!(0.5).multiply(x, stream)?.multiply(
        array!(1.0).add(
            tanh(
                sqrt(array!(2.0 / PI), stream)?.multiply(
                    x.add(
                        array!(0.044715).multiply(x.power(array!(3), stream)?, stream)?,
                        stream,
                    )?,
                    stream,
                )?,
                stream,
            )?,
            stream,
        )?,
        stream,
    )
}
