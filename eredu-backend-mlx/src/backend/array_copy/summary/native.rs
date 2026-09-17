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
        let value = match kind {
            Unary::CastF32 => source.as_dtype(Dtype::Float32, self.stream)?,
            Unary::Finite => source.is_finite(self.stream)?,
            Unary::Nan => source.is_nan(self.stream)?,
            Unary::PositiveInfinity => source.is_pos_inf(self.stream)?,
            Unary::NegativeInfinity => source.is_neg_inf(self.stream)?,
            Unary::MaskU32 => source.as_dtype(Dtype::Uint32, self.stream)?,
            Unary::Sum => source.sum(false, self.stream)?,
            Unary::Minimum => source.min(false, self.stream)?,
            Unary::Maximum => source.max(false, self.stream)?,
        };
        self.retained(value)
    }
    fn scalar(&mut self, value: f32) -> Result<Array, Error> {
        self.retained(Array::try_from_f32(value)?)
    }
    fn select(&mut self, mask: &Array, source: &Array, other: &Array) -> Result<Array, Error> {
        self.retained(safemlx::ops::r#where(mask, source, other, self.stream)?)
    }
    fn binary(&mut self, kind: Binary, left: &Array, right: &Array) -> Result<Array, Error> {
        self.retained(match kind {
            Binary::Divide => left.divide(right, self.stream)?,
            Binary::Multiply => left.multiply(right, self.stream)?,
        })
    }
    fn read_u32(&mut self, value: &Array, _: u32) -> Result<u32, Error> {
        (self.host_read)(1);
        self.completion
            .settle(value, self.stream)?
            .try_iter::<u32>()?
            .next()
            .ok_or(Error::ShapeMismatch)
    }
    fn read_f32(&mut self, value: &Array, _: ScalarRead) -> Result<f64, Error> {
        (self.host_read)(1);
        self.completion
            .settle(value, self.stream)?
            .try_iter::<f32>()?
            .next()
            .map(f64::from)
            .ok_or(Error::ShapeMismatch)
    }
}
