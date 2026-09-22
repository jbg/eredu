//! The numerical realization of the shared rectangular update worker.
use super::*;
use eredu_nn::parameter_values::ParameterUpdateMechanism;

pub(super) struct NativeUpdate<'a> {
    update: &'a ParameterUpdate,
    stream: &'a Stream,
    starts: Vec<i32>,
    ends: Vec<i32>,
    strides: Vec<i32>,
    shape: Vec<i32>,
}
impl<'a> NativeUpdate<'a> {
    pub(super) fn new(
        region: &'a ParameterRegion,
        update: &'a ParameterUpdate,
        stream: &'a Stream,
    ) -> Result<Self, Error> {
        let rank = region.shape.len();
        if region.starts.len() != rank {
            return Err(missing());
        }
        let mut starts = Vec::with_capacity(rank);
        let mut ends = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        let mut shape = Vec::with_capacity(rank);
        for (&start, &count) in region.starts.iter().zip(&region.shape) {
            starts.push(i32::try_from(start).map_err(|_| missing())?);
            ends.push(
                i32::try_from(start.checked_add(count).ok_or_else(missing)?)
                    .map_err(|_| missing())?,
            );
            strides.push(1);
            shape.push(i32::try_from(count).map_err(|_| missing())?);
        }
        Ok(Self {
            update,
            stream,
            starts,
            ends,
            strides,
            shape,
        })
    }
    pub(super) fn control_bytes(region: &ParameterRegion) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Array>().checked_mul(5)?,
            size_of::<Result<Option<Array>, Error>>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),
            size_of::<bool>(),
            region
                .shape
                .len()
                .checked_mul(size_of::<i32>())?
                .checked_mul(4)?,
            Array::static_slice_control_bytes()?,
            Array::static_slice_update_control_bytes()?,
            safemlx::EvaluatedArray::completed_readback_control_bytes::<f32>()?,
            crate::backend::runtime::cache::completed_borrow_control_bytes()?.checked_mul(2)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
impl ParameterUpdateMechanism for NativeUpdate<'_> {
    type Value = Array;
    type Error = Error;
    fn values(&self) -> Result<Array, Error> {
        Ok(Array::try_from_slice(self.update.values(), &self.shape)?)
    }
    fn select(&self, source: &Array) -> Result<Array, Error> {
        Ok(source.try_slice(&self.starts, &self.ends, &self.strides, self.stream)?)
    }
    fn cast_f32(&self, value: Array) -> Result<Array, Error> {
        Ok(value.as_dtype(Dtype::Float32, self.stream)?)
    }
    fn add(&self, left: &Array, right: &Array) -> Result<Array, Error> {
        Ok(left.add(right, self.stream)?)
    }
    fn cast_like(&self, value: Array, source: &Array) -> Result<Array, Error> {
        Ok(value.as_dtype(source.dtype(), self.stream)?)
    }
    fn finite_completed(&self, value: &Array) -> Result<bool, Error> {
        let checked = value
            .as_dtype(Dtype::Float32, self.stream)?
            .contiguous(false, self.stream)?;
        Ok(
            crate::backend::runtime::cache::complete_and_borrow(&checked, self.stream)?
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()),
        )
    }
    fn replace(&self, source: &Array, value: &Array) -> Result<Array, Error> {
        Ok(source.try_slice_update(value, &self.starts, &self.ends, &self.strides, self.stream)?)
    }
    fn complete(&self, value: &Array) -> Result<(), Error> {
        crate::backend::runtime::cache::complete_and_borrow(value, self.stream)?;
        Ok(())
    }
}
