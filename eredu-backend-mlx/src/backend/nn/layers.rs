//! Activation functions.

use std::f32::consts::PI;

use safemlx::{
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
            .add(Array::try_from_f32(1.0)?, stream)?,
        stream,
    )?
    .as_dtype(dtype, stream)
}

/// Applies sigmoid with FP32 exponential and quotient rounding.
pub fn sigmoid(input: Array, stream: &Stream) -> Result<Array, Exception> {
    let dtype = input.dtype();
    // Sigmoid is one pointwise operation. Reduced-precision exp/add/divide
    // intermediates would introduce additional rounding before its result.
    let work = match dtype {
        safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16 => {
            input.as_dtype(safemlx::Dtype::Float32, stream)?
        }
        _ => input,
    };
    let output = match super::arithmetic::f32_pointwise(
        &work,
        super::arithmetic::Pointwise::Sigmoid,
        stream,
    )? {
        Some(output) => output,
        None => safemlx::ops::sigmoid(work, stream)?,
    };
    output.as_dtype(dtype, stream)
}

/// Applies the squared rectified-linear activation.
pub fn relu2(x: Array, stream: &Stream) -> Result<Array, Exception> {
    maximum(&x, Array::try_from_f32(0.0)?, stream)?.square(stream)
}

/// Applies the exponential linear unit.
pub fn elu(
    x: impl AsRef<Array>,
    alpha: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> MlxResult<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    let alpha = Array::try_from_f32(alpha.into().unwrap_or(1.0))?;
    r#where(
        &x.gt(Array::try_from_f32(0.0)?, stream)?,
        x,
        alpha.multiply(
            exp(x, stream)?.subtract(Array::try_from_f32(1.0)?, stream)?,
            stream,
        )?,
        stream,
    )
}

/// Applies softplus.
pub fn softplus(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    safemlx::ops::logaddexp(x.as_ref(), Array::try_from_scalar(0)?, stream)
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
            Array::try_from_scalar(1)?.add(
                erf(
                    x.as_ref()
                        .divide(Array::try_from_scalar(2f32.sqrt())?, stream)?,
                    stream,
                )?,
                stream,
            )?,
            stream,
        )?
        .divide(Array::try_from_f32(2.0)?, stream)
}

/// Applies the tanh GELU approximation.
pub fn gelu_approximate(x: impl AsRef<Array>, stream: impl AsRef<Stream>) -> MlxResult<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    Array::try_from_f32(0.5)?.multiply(x, stream)?.multiply(
        Array::try_from_f32(1.0)?.add(
            tanh(
                sqrt(Array::try_from_scalar(2.0 / PI)?, stream)?.multiply(
                    x.add(
                        Array::try_from_f32(0.044715)?
                            .multiply(x.power(Array::try_from_scalar(3)?, stream)?, stream)?,
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
