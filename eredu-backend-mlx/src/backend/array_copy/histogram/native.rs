use super::*;
use safemlx::{Dtype, ops::indexing::TryIndexOp};
pub(super) struct Native<'a, 'o> {
    pub(super) stream: &'a Stream,
    pub(super) completion: CaptureCompletion<'o>,
    pub(super) retain: &'a mut dyn FnMut(&Array) -> Result<(), Error>,
    pub(super) host_read: &'a mut dyn FnMut(usize),
}
impl Native<'_, '_> {
    fn retained(&mut self, value: Array) -> Result<Array, Error> {
        (self.retain)(&value)?;
        Ok(value)
    }
}
impl Kernel for Native<'_, '_> {
    type Value = Array;
    fn interval(&mut self, source: &Array, start: i32, end: i32) -> Result<Array, Error> {
        self.retained(source.try_index_device(start..end, self.stream)?)
    }
    fn unary(&mut self, kind: Unary, source: &Array) -> Result<Array, Error> {
        self.retained(match kind {
            Unary::CastF32 => source.as_dtype(Dtype::Float32, self.stream)?,
            Unary::Finite => source.is_finite(self.stream)?,
            Unary::Not => source.logical_not(self.stream)?,
            Unary::MaskU32 => source.as_dtype(Dtype::Uint32, self.stream)?,
            Unary::Sum => source.sum(false, self.stream)?,
        })
    }
    fn scalar(&mut self, value: f32) -> Result<Array, Error> {
        self.retained(Array::try_from_f32(value)?)
    }
    fn binary(&mut self, kind: Binary, left: &Array, right: &Array) -> Result<Array, Error> {
        self.retained(match kind {
            Binary::Less => left.lt(right, self.stream)?,
            Binary::Greater => left.gt(right, self.stream)?,
            Binary::GreaterEqual => left.ge(right, self.stream)?,
            Binary::LessEqual => left.le(right, self.stream)?,
            Binary::And => left.logical_and(right, self.stream)?,
        })
    }
    fn read(&mut self, value: &Array) -> Result<u32, Error> {
        (self.host_read)(1);
        self.completion
            .settle(value, self.stream)?
            .try_iter::<u32>()?
            .next()
            .ok_or(Error::ShapeMismatch)
    }
}
