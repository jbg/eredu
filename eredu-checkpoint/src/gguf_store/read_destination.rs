//! A source-owned raw buffer; reader and conversion storage remain ordinary.
use super::*;
use std::alloc::Layout;
use std::collections::TryReserveError;

#[derive(Debug)]
pub(super) enum Cause {
    Store(StoreError),
    Conversion(eredu_gguf::ConversionDestinationError),
    Metadata(eredu_gguf::MetadataDestinationError),
    Layout,
    Reserve(TryReserveError),
    Length { expected: u64, actual: usize },
}
impl From<StoreError> for Cause {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
impl Cause {
    pub(super) fn ordinary(self) -> StoreError {
        match self {
            Self::Store(error) => error,
            _ => unreachable!("ordinary GGUF materialization has no prepared raw storage"),
        }
    }
    pub(super) fn read(error: eredu_gguf::ReadDestinationError, key: &str) -> Self {
        match error {
            eredu_gguf::ReadDestinationError::Gguf(error) => {
                Self::Store(gguf_read_error(key, error))
            }
            eredu_gguf::ReadDestinationError::Conversion(error) => Self::Conversion(error),
            eredu_gguf::ReadDestinationError::Metadata(error) => Self::Metadata(error),
            eredu_gguf::ReadDestinationError::Length { expected, actual } => {
                Self::Length { expected, actual }
            }
        }
    }
}

// The boxed arm preserves the caller's final lease allocation. Existing public
// consuming APIs retain their by-value owner and original drop behavior.
#[derive(Debug)]
enum LeaseOwner {
    Value(GgufLease),
    Boxed(Box<GgufLease>),
}
impl std::ops::Deref for LeaseOwner {
    type Target = GgufLease;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Value(lease) => lease,
            Self::Boxed(lease) => lease,
        }
    }
}

/// Owns one exact raw-read destination and the genuine lease selecting its source.
///
/// Preparation does not open a reader or materialize a tensor. Reader/cache,
/// selection-plan, conversion-output and allocator controls remain separate.
#[derive(Debug)]
pub struct PreparedGgufRead<S: GgufRawStorage = Vec<u8>> {
    raw: S,
    layout: Layout,
    lease: LeaseOwner,
}

/// Retains the exact lease, buffer and actual cause after preparation or execution fails.
/// The raw buffer may contain a partially failed read; its length is not read credit.
#[derive(Debug)]
pub struct PreparedGgufReadFailure<S: GgufRawStorage = Vec<u8>> {
    cause: Cause,
    raw: S,
    lease: LeaseOwner,
}
impl<S: GgufRawStorage> PreparedGgufReadFailure<S> {
    /// Original checkpoint processing error, when processing caused failure.
    pub fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            _ => None,
        }
    }
    /// Actual reserve cause, including capacity overflow when reported by Vec.
    pub fn reserve_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Reserve(error) => Some(error),
            _ => None,
        }
    }
    /// The initialized destination, which can contain partial failed contents.
    pub fn raw_bytes(&self) -> &[u8] {
        self.raw.as_ref()
    }
    /// Actual buffer capacity retained by this failure.
    pub fn raw_capacity(&self) -> usize {
        self.raw.capacity()
    }
    /// Genuine source lease kept until this failure retires.
    pub fn lease(&self) -> &GgufLease {
        &self.lease
    }
}
impl<S: GgufRawStorage> std::fmt::Display for PreparedGgufReadFailure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Store(error) => error.fmt(f),
            Cause::Conversion(error) => error.fmt(f),
            Cause::Metadata(error) => error.fmt(f),
            Cause::Layout => f.write_str("GGUF raw destination layout is not representable"),
            Cause::Reserve(error) => error.fmt(f),
            Cause::Length { expected, actual } => write!(
                f,
                "GGUF raw destination has {actual} bytes; expected {expected}"
            ),
        }
    }
}
impl<S: GgufRawStorage> std::error::Error for PreparedGgufReadFailure<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            Cause::Conversion(error) => Some(error),
            Cause::Metadata(error) => Some(error),
            Cause::Reserve(error) => Some(error),
            _ => None,
        }
    }
}
impl GgufLease {
    /// Prepares only the raw encoded-byte destination, consuming this exact lease.
    /// The private retained physical proof supplies the extent; the actual read
    /// checks it again against its descriptor or plan before writing payloads.
    pub fn prepare_portable_read(self) -> Result<PreparedGgufRead, PreparedGgufReadFailure> {
        PreparedGgufRead::prepare(LeaseOwner::Value(self))
    }
}
impl PreparedGgufRead {
    fn prepare(lease: LeaseOwner) -> Result<Self, PreparedGgufReadFailure> {
        let mut raw = Vec::new();
        let prepared = (|| {
            let len = usize::try_from(lease.proof.length_bytes).map_err(|_| Cause::Layout)?;
            let layout = Layout::array::<u8>(len).map_err(|_| Cause::Layout)?;
            raw.try_reserve_exact(len).map_err(Cause::Reserve)?;
            raw.resize(len, 0);
            Ok(layout)
        })();
        match prepared {
            Ok(layout) => Ok(PreparedGgufRead { raw, layout, lease }),
            Err(cause) => Err(PreparedGgufReadFailure { cause, raw, lease }),
        }
    }
}
impl<S: GgufRawStorage> PreparedGgufRead<S> {
    /// Requested raw element layout, excluding allocator bookkeeping.
    pub fn raw_layout(&self) -> Layout {
        self.layout
    }
    /// Actual Vec capacity; it can exceed the requested raw byte count.
    pub fn raw_capacity(&self) -> usize {
        self.raw.capacity()
    }
    /// Genuine source lease determining the only permitted read.
    pub fn lease(&self) -> &GgufLease {
        &self.lease
    }
    /// Uses the same cache and conversion, consuming the prepared raw destination.
    /// A failed read is not retried; its buffer and source remain in the error.
    pub fn materialize(mut self) -> Result<ConvertedCheckpointTensor, PreparedGgufReadFailure<S>> {
        match materialize(&self.lease, Some(self.raw.as_mut())) {
            Ok(converted) => Ok(converted),
            Err(cause) => Err(PreparedGgufReadFailure {
                cause,
                raw: self.raw,
                lease: self.lease,
            }),
        }
    }
}

pub(super) fn materialize(
    lease: &GgufLease,
    raw: Option<&mut [u8]>,
) -> Result<ConvertedCheckpointTensor, Cause> {
    materialize_with_conversion(lease, raw, None)
}

fn materialize_with_conversion(
    lease: &GgufLease,
    raw: Option<&mut [u8]>,
    conversion: Option<&mut eredu_gguf::PreparedConversion>,
) -> Result<ConvertedCheckpointTensor, Cause> {
    materialize_with_metadata(lease, raw, conversion, None)
}

fn materialize_with_metadata(
    lease: &GgufLease,
    raw: Option<&mut [u8]>,
    conversion: Option<&mut eredu_gguf::PreparedConversion>,
    metadata: Option<&mut eredu_gguf::PreparedTensorMetadata>,
) -> Result<ConvertedCheckpointTensor, Cause> {
    materialize_worker(lease, |cache| {
        cache.materialize_with_destination(
            lease.entry.checkpoint,
            &lease.entry.physical_name,
            lease.identity.selection.as_ref(),
            lease.store.max_cached_readers,
            &lease.entry.metadata.name,
            raw,
            conversion,
            metadata,
        )
    })
}
fn materialize_worker<T>(
    lease: &GgufLease,
    execute: impl FnOnce(&mut ReaderCache) -> Result<T, Cause>,
) -> Result<T, Cause> {
    let converted = execute(
        &mut *lease
            .store
            .readers
            .lock()
            .map_err(|_| StoreError::Internal("GGUF reader cache is poisoned".into()))?,
    )?;
    lease
        .store
        .statistics
        .physical_reads
        .fetch_add(1, Ordering::Relaxed);
    lease
        .store
        .statistics
        .physical_read_bytes
        .fetch_add(lease.proof.length_bytes, Ordering::Relaxed);
    Ok(converted)
}

mod conversion_destination;
pub use conversion_destination::{
    PreparedGgufConversion, PreparedGgufConversionFailure, PreparedGgufTensor,
    PreparedGgufTensorFailure,
};

mod boxed;
pub use boxed::PreparedGgufBoxedFailure;

mod supplied;
pub use supplied::{
    GgufRawStorage, GgufRawStorageProvider, GgufRawStorageRequest, PreparedGgufSuppliedFailure,
};

mod stored;
pub use stored::{GgufStorageProvider, StoredGgufFailure};

impl std::fmt::Display for Cause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => e.fmt(f),
            Self::Conversion(e) => e.fmt(f),
            Self::Metadata(e) => e.fmt(f),
            Self::Reserve(e) => e.fmt(f),
            Self::Layout => f.write_str("GGUF raw destination layout is not representable"),
            Self::Length { expected, actual } => write!(
                f,
                "GGUF raw destination has {actual} bytes; expected {expected}"
            ),
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::Conversion(e) => Some(e),
            Self::Metadata(e) => Some(e),
            Self::Reserve(e) => Some(e),
            _ => None,
        }
    }
}
