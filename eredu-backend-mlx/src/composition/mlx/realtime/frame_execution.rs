use super::*;

/// Family-blind MLX materialization and tensor operations for prepared frames.
#[doc(hidden)]
pub struct MlxRealtimeFrameTensorMechanisms<'a> {
    stream: &'a Stream,
}

impl<'a> MlxRealtimeFrameTensorMechanisms<'a> {
    /// Binds portable frame tensor operations to one selected MLX stream.
    pub(super) const fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }
}

impl RealtimeHostTokenMaterializer for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error> {
        let shape = shape
            .into_iter()
            .map(|dimension| {
                i32::try_from(dimension)
                    .map_err(|_| Error::Parallel("realtime tensor dimension exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Array::from_slice(values, &shape)
            .copy(self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

impl RealtimeFrameTensorMechanisms for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
        let batch = i32::try_from(batch)
            .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
        Array::full::<i32>(&[batch, 1], Array::from_int(token), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}
