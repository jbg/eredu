use super::*;
use safemlx::{
    Dtype,
    ops::indexing::{TryIndexOp, TryIndexMutOp},
};

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
    fn flatten(&mut self, source: &Array, vocabulary: i32) -> Result<Array, Error> {
        self.retained(source.reshape(&[-1, vocabulary], self.stream)?)
    }
    fn terminal(&mut self, source: &Array) -> Result<Array, Error> {
        self.retained(source.try_index_device((-1, ..), self.stream)?)
    }
    fn unary(&mut self, kind: Unary, source: &Array) -> Result<Array, Error> {
        let value = match kind {
            Unary::CastF32 => source.as_dtype(Dtype::Float32, self.stream)?,
            Unary::Finite => source.is_finite(self.stream)?,
            Unary::MaskU32 => source.as_dtype(Dtype::Uint32, self.stream)?,
            Unary::Sum => source.sum(false, self.stream)?,
            Unary::Maximum => source.max(false, self.stream)?,
            Unary::Exp => source.exp(self.stream)?,
            Unary::Argmax => safemlx::ops::indexing::argmax(source, false, self.stream)?,
        };
        self.retained(value)
    }
    fn scalar(&mut self, value: f32) -> Result<Array, Error> {
        self.retained(Array::try_from_f32(value)?)
    }
    fn interval(&mut self, row: &Array, start: i32, end: i32) -> Result<Array, Error> {
        self.retained(row.try_index_device(start..end, self.stream)?)
    }
    fn at(&mut self, row: &Array, id: u32) -> Result<Array, Error> {
        // A host scalar selects a static Slice/Reshape, without uploading an
        // index array or invoking the array-valued Gather route.
        self.retained(row.try_index_device((id as i32,), self.stream)?)
    }
    fn binary(&mut self, kind: Binary, left: &Array, right: &Array) -> Result<Array, Error> {
        self.retained(match kind {
            Binary::Subtract => left.subtract(right, self.stream)?,
            Binary::Greater => left.gt(right, self.stream)?,
        })
    }
    fn exclude(&mut self, row: &Array, id: u32, value: &Array) -> Result<Array, Error> {
        if row.ndim() != 1 || value.ndim() != 0 || id >= row.dim(0) as u32 {
            return Err(Error::ShapeMismatch);
        }
        let mut result = self.completion.clone_array(row)?;
        result.try_index_mut_device((id as i32,), value, self.stream)?;
        self.retained(result)
    }
    fn read_f32(&mut self, value: &Array) -> Result<f64, Error> {
        (self.host_read)(1);
        let completed = self.completion.settle(value, self.stream)?;
        let result = completed
            .try_iter::<f32>()?
            .next()
            .ok_or(Error::ShapeMismatch)?;
        Ok(f64::from(result))
    }
    fn read_u32(&mut self, value: &Array, _: Read) -> Result<u32, Error> {
        (self.host_read)(1);
        let completed = self.completion.settle(value, self.stream)?;
        let result = completed
            .try_iter::<u32>()?
            .next()
            .ok_or(Error::ShapeMismatch)?;
        Ok(result)
    }
}
