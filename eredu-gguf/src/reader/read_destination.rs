//! Storage adapters for the original ordered payload reads.
use super::*;

/// Failure while filling a caller-owned encoded-byte destination.
#[derive(Debug)]
pub enum ReadDestinationError {
    /// The original GGUF processing error, with its original context.
    Gguf(Error),
    /// A prepared conversion binding or storage check failed.
    Conversion(crate::ConversionDestinationError),
    /// A prepared result metadata binding or storage check failed.
    Metadata(crate::MetadataDestinationError),
    /// The destination must have exactly the physical read's encoded extent.
    Length {
        /// Encoded byte extent required by the actual descriptor or plan.
        expected: u64,
        /// Number of bytes supplied by the caller.
        actual: usize,
    },
}
impl From<Error> for ReadDestinationError {
    fn from(error: Error) -> Self {
        Self::Gguf(error)
    }
}
impl std::fmt::Display for ReadDestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(error) => error.fmt(f),
            Self::Conversion(error) => error.fmt(f),
            Self::Metadata(error) => error.fmt(f),
            Self::Length { expected, actual } => write!(
                f,
                "GGUF raw destination has {actual} bytes; expected {expected}"
            ),
        }
    }
}
impl std::error::Error for ReadDestinationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(error) => Some(error),
            Self::Conversion(error) => Some(error),
            Self::Metadata(error) => Some(error),
            Self::Length { .. } => None,
        }
    }
}
impl ReadDestinationError {
    pub(crate) fn ordinary(self) -> Error {
        match self {
            Self::Gguf(error) => error,
            Self::Length { .. } | Self::Conversion(_) | Self::Metadata(_) => {
                unreachable!("ordinary raw storage sizes itself")
            }
        }
    }
    pub(crate) fn with_shard(self, path: &Path) -> Self {
        match self {
            Self::Gguf(source) => Self::Gguf(Error::Shard {
                path: path.to_path_buf(),
                source: Box::new(source),
            }),
            error => error,
        }
    }
}

pub(crate) enum RawStorage<'a> {
    Ordinary,
    Borrowed(&'a mut [u8]),
}
pub(super) enum RawBuffer<'a> {
    Owned(Vec<u8>),
    Borrowed { bytes: &'a mut [u8], used: usize },
}
impl<'a> RawStorage<'a> {
    pub(super) fn validate(&self, expected: u64) -> std::result::Result<(), ReadDestinationError> {
        if let Self::Borrowed(bytes) = self {
            if u64::try_from(bytes.len()).ok() != Some(expected) {
                return Err(ReadDestinationError::Length {
                    expected,
                    actual: bytes.len(),
                });
            }
        }
        Ok(())
    }
    pub(super) fn full(self, len: usize) -> RawBuffer<'a> {
        match self {
            Self::Ordinary => RawBuffer::Owned(vec![0; len]),
            Self::Borrowed(bytes) => {
                bytes.fill(0);
                RawBuffer::Borrowed { bytes, used: len }
            }
        }
    }
    pub(super) fn selected(self, len: usize) -> RawBuffer<'a> {
        match self {
            Self::Ordinary => RawBuffer::Owned(Vec::with_capacity(len)),
            Self::Borrowed(bytes) => RawBuffer::Borrowed { bytes, used: 0 },
        }
    }
}
impl RawBuffer<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Owned(bytes) => bytes.len(),
            Self::Borrowed { used, .. } => *used,
        }
    }
    pub(super) fn resize(&mut self, end: usize) -> std::result::Result<(), ReadDestinationError> {
        match self {
            Self::Owned(bytes) => bytes.resize(end, 0),
            Self::Borrowed { bytes, used } => {
                let actual = bytes.len();
                let range = bytes
                    .get_mut(*used..end)
                    .ok_or(ReadDestinationError::Length {
                        expected: u64::try_from(end).unwrap_or(u64::MAX),
                        actual,
                    })?;
                range.fill(0);
                *used = end;
            }
        }
        Ok(())
    }
    pub(super) fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Borrowed { bytes, used } => &bytes[..*used],
        }
    }
    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Borrowed { bytes, used } => &mut bytes[..*used],
        }
    }
    pub(super) fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Borrowed { .. } => unreachable!("ordinary read returns owned bytes"),
        }
    }
}

impl<R: Read + Seek> Reader<R> {
    /// Reads and converts a full tensor using an exact borrowed raw buffer.
    /// Conversion still owns its ordinary output allocations. On error the
    /// buffer may contain partial data and remains exclusively owned by caller.
    pub fn read_tensor_with_raw_destination(
        &mut self,
        tensor: &TensorDescriptor,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_tensor_with_storage(tensor, RawStorage::Borrowed(raw), None)
    }
    /// Reads a physical axis plan in its original span order into exact raw storage.
    pub fn read_tensor_plan_with_raw_destination(
        &mut self,
        plan: &TensorSelectionPlan,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_tensor_plan_with_storage(plan.view(), RawStorage::Borrowed(raw), None)
    }
    /// Reads a block-aligned contiguous span into exact borrowed raw storage.
    pub fn read_dense_tensor_span_with_raw_destination(
        &mut self,
        plan: &DenseTensorSpanPlan,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_dense_tensor_span_with_storage(plan.view(), RawStorage::Borrowed(raw), None)
    }
    pub(crate) fn read_tensor_with_storage<C: read_destination::ConversionDestination>(
        &mut self,
        tensor: &TensorDescriptor,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_tensor_with_storage(tensor.view(), storage, conversion)
    }
}

#[cfg(test)]
mod tests;

impl From<crate::ConversionDestinationError> for ReadDestinationError {
    fn from(error: crate::ConversionDestinationError) -> Self {
        match error {
            crate::ConversionDestinationError::Gguf(error) => Self::Gguf(error),
            error => Self::Conversion(error),
        }
    }
}
pub(crate) trait ConversionDestination {
    type Output;
    fn convert(
        self,
        descriptor: crate::TensorDescriptorView<'_>,
        raw: &[u8],
        endian: Endian,
    ) -> std::result::Result<Self::Output, ReadDestinationError>;
}
impl ConversionDestination for Option<&mut crate::PreparedConversion> {
    type Output = ConvertedTensor;
    fn convert(
        self,
        descriptor: crate::TensorDescriptorView<'_>,
        raw: &[u8],
        endian: Endian,
    ) -> std::result::Result<Self::Output, ReadDestinationError> {
        match self {
            None => crate::convert::convert_view(descriptor, raw, endian).map_err(Into::into),
            Some(destination) => destination
                .fill_view(raw, descriptor, endian)
                .map_err(Into::into),
        }
    }
}
impl<F: crate::StorageFamily> ConversionDestination for &mut crate::StoredConversion<F> {
    type Output = ();
    fn convert(
        self,
        descriptor: crate::TensorDescriptorView<'_>,
        raw: &[u8],
        endian: Endian,
    ) -> std::result::Result<(), ReadDestinationError> {
        self.fill(raw, descriptor, endian).map_err(Into::into)
    }
}
pub(super) fn convert_destination<C: ConversionDestination>(
    descriptor: crate::TensorDescriptorView<'_>,
    raw: &[u8],
    endian: Endian,
    conversion: C,
) -> std::result::Result<C::Output, ReadDestinationError> {
    conversion.convert(descriptor, raw, endian)
}

impl<R: Read + Seek> Reader<R> {
    /// Executes the original read and conversion using both prepared destinations.
    /// Metadata, reader and cache storage keep their ordinary ownership.
    pub fn read_tensor_with_destinations(
        &mut self,
        tensor: &TensorDescriptor,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_tensor_with_storage(tensor, RawStorage::Borrowed(raw), Some(conversion))
    }
    /// Executes the original read and conversion using both prepared destinations.
    /// Metadata, reader and cache storage keep their ordinary ownership.
    pub fn read_tensor_plan_with_destinations(
        &mut self,
        plan: &TensorSelectionPlan,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_tensor_plan_with_storage(plan.view(), RawStorage::Borrowed(raw), Some(conversion))
    }
    /// Executes the original read and conversion using both prepared destinations.
    /// Metadata, reader and cache storage keep their ordinary ownership.
    pub fn read_dense_tensor_span_with_destinations(
        &mut self,
        plan: &DenseTensorSpanPlan,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedTensor, ReadDestinationError> {
        self.read_dense_tensor_span_with_storage(
            plan.view(),
            RawStorage::Borrowed(raw),
            Some(conversion),
        )
    }
}

impl From<crate::MetadataDestinationError> for ReadDestinationError {
    fn from(error: crate::MetadataDestinationError) -> Self {
        match error {
            crate::MetadataDestinationError::Gguf(error) => Self::Gguf(error),
            error => Self::Metadata(error),
        }
    }
}
