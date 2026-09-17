//! The existing convolution worker with an exact original source/control profile.
use crate::MlxTensor;
use eredu_nn::Error;
use safemlx::{Array, Dtype, OriginalScopeObserver, Stream, error::Exception};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum SourceRefusal {
    #[error("original convolution requires the selected GPU stream")]
    Stream,
    #[error("original convolution requires F32, F16 or BF16 source scalars")]
    Dtype,
    #[error("original convolution has no complete selected native receipt")]
    Geometry,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct RetainedRefusal {
    #[source]
    cause: SourceRefusal,
    // Release the error payload/erasure before the paying native domain.
    _custody: Exception,
}
fn error(cause: Exception) -> Error {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_source(cause),
        _ => Error::backend_retained_source(cause),
    }
}
fn validate(
    input: &Array,
    weight: &Array,
    stride: &[i32],
    padding: &[i32],
    dilation: &[i32],
    groups: i32,
    stream: &Stream,
) -> Result<(), Error> {
    let Some(observer) = OriginalScopeObserver::try_current().map_err(error)? else {
        return Ok(());
    };
    let refusal = if stream.device_type().map_err(error)? != safemlx::DeviceType::Gpu {
        Some(SourceRefusal::Stream)
    } else if [input.dtype(), weight.dtype()]
        .iter()
        .any(|dtype| !matches!(dtype, Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16))
    {
        Some(SourceRefusal::Dtype)
    } else if safemlx::ops::OriginalConvolutionLayout::inspect(
        input.shape(),
        weight.shape(),
        stride,
        padding,
        dilation,
        groups,
    )
    .is_none()
    {
        Some(SourceRefusal::Geometry)
    } else {
        None
    };
    match refusal {
        None => Ok(()),
        Some(cause) => Err(Error::backend_retained_source(RetainedRefusal {
            cause,
            _custody: observer.invalid_input_error(),
        })),
    }
}
pub(crate) fn conv1d(
    input: &MlxTensor,
    weight: &MlxTensor,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    validate(
        input.as_array(),
        weight.as_array(),
        &[stride],
        &[padding],
        &[dilation],
        groups,
        stream,
    )?;
    safemlx::ops::conv1d(
        input.as_array(),
        weight.as_array(),
        stride,
        padding,
        dilation,
        groups,
        stream,
    )
    .map(MlxTensor::from_array)
    .map_err(error)
}
pub(crate) fn conv2d(
    input: &MlxTensor,
    weight: &MlxTensor,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    validate(
        input.as_array(),
        weight.as_array(),
        &[stride.0, stride.1],
        &[padding.0, padding.1],
        &[dilation.0, dilation.1],
        groups,
        stream,
    )?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let dtype = input.as_array().dtype();
        if dtype == weight.as_array().dtype()
            && matches!(dtype, Dtype::Float16 | Dtype::Bfloat16)
            && super::uses_metal_winograd_2d(
                input.as_array().shape(),
                weight.as_array().shape(),
                stride,
                dilation,
                groups,
            )
        {
            // Preserve the ordinary precision worker exactly. The current
            // original profile rejects this algorithm before either cast.
            let input = input
                .as_array()
                .as_dtype(Dtype::Float32, stream)
                .map_err(error)?;
            let weight = weight
                .as_array()
                .as_dtype(Dtype::Float32, stream)
                .map_err(error)?;
            let output = safemlx::ops::conv2d(
                &input,
                &weight,
                Some(stride),
                Some(padding),
                Some(dilation),
                Some(groups),
                stream,
            )
            .map_err(error)?;
            return output
                .as_dtype(dtype, stream)
                .map(MlxTensor::from_array)
                .map_err(error);
        }
    }
    safemlx::ops::conv2d(
        input.as_array(),
        weight.as_array(),
        Some(stride),
        Some(padding),
        Some(dilation),
        Some(groups),
        stream,
    )
    .map(MlxTensor::from_array)
    .map_err(error)
}
pub(crate) fn control_bytes(layout: safemlx::ops::OriginalConvolutionLayout) -> Option<usize> {
    let frames = [
        layout.control_bytes()?,
        size_of::<[(&MlxTensor, &MlxTensor, &Stream); 2]>(),
        size_of::<[&Array; 2]>(),
        size_of::<[&[i32]; 3]>(),
        size_of::<[(i32, i32); 3]>(),
        size_of::<[i32; 4]>(),
        size_of::<Result<MlxTensor, Error>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Option<SourceRefusal>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<[Dtype; 2]>(),
        size_of::<Result<safemlx::DeviceType, Exception>>(),
        OriginalScopeObserver::control_bytes()?,
        Stream::device_type_control_bytes()?,
        Error::retained_source_control_bytes::<Exception>()?,
        Error::retained_source_control_bytes::<RetainedRefusal>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
