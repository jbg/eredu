use super::worker::{Binary, Unary, Worker};
use super::*;
use crate::backend::array_copy::CaptureCompletion;
use safemlx::{OriginalScopeObserver, PreparedArrayClone};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Validation(&'static str);
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Refusal<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    _custody: Exception,
}
// The selected original path retains a fixed typed cause and its actual domain.
// Ordinary diagnostics retain their existing detailed shapes and text.
pub(super) fn invalid(reason: &'static str, ordinary: impl FnOnce() -> String) -> Exception {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Exception::custom(ordinary()),
        Ok(Some(observer)) => Exception::from_retained_source(Refusal {
            cause: Validation(reason),
            _custody: observer.invalid_input_error(),
        }),
        Err(custody) => Exception::from_retained_source(Refusal {
            cause: Validation(reason),
            _custody: custody,
        }),
    }
}
pub(super) fn profile(
    values: &[&Array],
    prefix_rank: usize,
    stream: &Stream,
) -> Result<(), Exception> {
    if let Some(observer) = OriginalScopeObserver::try_current()? {
        if stream.device_type()? != safemlx::DeviceType::Gpu
            || prefix_rank > 2
            || values.iter().any(|array| {
                !matches!(
                    array.dtype(),
                    Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
                )
            })
        {
            return Err(observer.invalid_input_error());
        }
    }
    Ok(())
}
pub(super) fn dimensions(residual: &Array) -> Result<[i32; 4], Exception> {
    let shape: [i32; 4] = residual.shape().try_into().map_err(|_| {
        invalid("hyper residual rank", || {
            format!(
                "hyper residual requires four dimensions, got {:?}",
                residual.shape()
            )
        })
    })?;
    if shape[2] <= 0
        || shape[3] <= 0
        || shape[0].checked_mul(shape[1]).is_none()
        || shape[2].checked_mul(shape[3]).is_none()
    {
        return Err(invalid("hyper residual geometry", || {
            "hyper residual geometry overflow".into()
        }));
    }
    Ok(shape)
}
pub(super) fn parameters(
    dimensions: [i32; 4],
    rows: i32,
    scales: i32,
    function: &Array,
    base: &Array,
    scale: &Array,
) -> Result<(), Exception> {
    let width = dimensions[2]
        .checked_mul(dimensions[3])
        .ok_or_else(|| invalid("hyper width overflow", || "hyper width overflow".into()))?;
    if function.shape() != [rows, width] || base.shape() != [rows] || scale.shape() != [scales] {
        return Err(invalid("hyper parameter geometry", || {
            "hyper parameters differ from declared geometry".into()
        }));
    }
    Ok(())
}
pub(super) struct Native<'a>(pub &'a Stream);
impl Worker for Native<'_> {
    type Value = Array;
    type Error = Exception;
    fn shape<'a>(&self, value: &'a Array) -> &'a [i32] {
        value.shape()
    }
    fn alias(&mut self, value: &Array) -> Result<Array, Exception> {
        match OriginalScopeObserver::try_current()? {
            Some(observer) => CaptureCompletion::Original(&observer)
                .clone_array(value)
                .map_err(|cause| {
                    Exception::from_retained_source(Refusal {
                        cause,
                        _custody: observer.invalid_input_error(),
                    })
                }),
            None => Ok(value.clone()),
        }
    }
    fn f32(&mut self, value: &Array) -> Result<Array, Exception> {
        value.as_dtype(Dtype::Float32, self.0)
    }
    fn cast_like(&mut self, value: &Array, source: &Array) -> Result<Array, Exception> {
        value.as_dtype(source.dtype(), self.0)
    }
    fn scalar(&mut self, value: f32) -> Result<Array, Exception> {
        Array::try_from_f32(value)
    }
    fn zeros_like(&mut self, shape: &[i32], source: &Array) -> Result<Array, Exception> {
        safemlx::ops::zeros_dtype(shape, source.dtype(), self.0)
    }
    fn reshape(&mut self, value: &Array, shape: &[i32]) -> Result<Array, Exception> {
        value.reshape(shape, self.0)
    }
    fn transpose(&mut self, value: &Array, axes: &[i32]) -> Result<Array, Exception> {
        value.transpose_axes(axes, self.0)
    }
    fn slice(
        &mut self,
        value: &Array,
        axis: usize,
        start: i32,
        end: i32,
    ) -> Result<Array, Exception> {
        // Shared split slices only its flattened matrix and one-dimensional
        // scale/base. No general indexing plan or dynamic shape storage.
        let rank = value.ndim();
        assert!(rank <= 2 && axis < rank);
        let mut starts = [0; 2];
        let mut ends = [0; 2];
        ends[..rank].copy_from_slice(value.shape());
        starts[axis] = start;
        ends[axis] = end;
        value.try_slice(&starts[..rank], &ends[..rank], &[1, 1][..rank], self.0)
    }
    fn unary(&mut self, kind: Unary, value: &Array) -> Result<Array, Exception> {
        match kind {
            Unary::Square => value.square(self.0),
            Unary::Rsqrt => safemlx::ops::rsqrt(value, self.0),
            Unary::Sigmoid => safemlx::ops::sigmoid(value, self.0),
        }
    }
    fn binary(&mut self, kind: Binary, a: &Array, b: &Array) -> Result<Array, Exception> {
        match kind {
            Binary::Add => a.add(b, self.0),
            Binary::Multiply => a.multiply(b, self.0),
            Binary::Divide => a.divide(b, self.0),
        }
    }
    fn matmul(&mut self, a: &Array, b: &Array) -> Result<Array, Exception> {
        safemlx::ops::matmul(a, b, self.0)
    }
    fn mean_last(&mut self, value: &Array) -> Result<Array, Exception> {
        value.mean_axis(-1, true, self.0)
    }
    fn sum(&mut self, value: &Array, axis: usize, keep: bool) -> Result<Array, Exception> {
        value.sum_axis(axis as i32, keep, self.0)
    }
    fn softmax_last(&mut self, value: &Array) -> Result<Array, Exception> {
        safemlx::ops::softmax_axis(value, -1, true, self.0)
    }
}

pub(crate) fn control_bytes(handles: usize, aliases: usize) -> Option<usize> {
    let original = [
        Exception::retained_source_control_bytes::<Refusal<Validation>>()?,
        Exception::retained_source_control_bytes::<
            Refusal<crate::backend::array_copy::CaptureTensorNativeError>,
        >()?,
        OriginalScopeObserver::control_bytes()?,
        PreparedArrayClone::control_bytes()?
            .checked_add(Array::inspection_clone_handle_bytes())?
            .checked_mul(aliases)?,
        Array::as_dtype_control_bytes()?,
        Stream::device_type_control_bytes()?,
    ];
    original.into_iter().try_fold(
        ordinary_control_bytes(handles)?.checked_add(size_of_val(&original))?,
        usize::checked_add,
    )
}

/// Fixed shared equation transports, with no Original clone/observer owners.
/// Actual safe call wrappers and aliases are counted by the same Worker
/// traversal; rank-four selection keeps both SmallVec shapes inline.
pub(crate) fn ordinary_control_bytes(handles: usize) -> Option<usize> {
    let frames = [
        size_of::<Native<'_>>(),
        std::alloc::Layout::array::<(Array, Result<Array, Exception>)>(handles)
            .ok()?
            .size(),
        size_of::<[smallvec::SmallVec<[i32; 4]>; 2]>(),
        size_of::<[i32; 4]>() * 5,
        size_of::<[&Array; 4]>() * 2,
        size_of::<[usize; 10]>(),
        size_of::<[f32; 3]>(),
        size_of::<[bool; 3]>(),
        size_of::<[(bool, &'static str); 3]>(),
        size_of::<[i32; 2]>() * 3,
        size_of::<[&[i32]; 3]>(),
        size_of::<Result<i32, Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<safemlx::DeviceType, Exception>>(),
        size_of::<Result<Array, crate::backend::array_copy::CaptureTensorNativeError>>(),
        size_of::<Result<[i32; 4], Exception>>(),
        size_of::<Result<worker::Split<Array>, Exception>>(),
        size_of::<Result<(Array, worker::Split<Array>), Exception>>(),
        size_of::<Result<(Array, Array), Exception>>(),
        size_of::<Result<HyperConnectionSplit, Exception>>(),
        size_of::<Option<&mut dyn FnMut(&Array) -> Result<(), Exception>>>(),
        size_of::<Binary>(),
        size_of::<Unary>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<eredu_nn::HyperConnectionState<crate::MlxTensor>>(),
        size_of::<Result<eredu_nn::HyperConnectionState<crate::MlxTensor>, eredu_nn::Error>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        eredu_nn::Error::retained_source_construction_bytes::<Exception>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
