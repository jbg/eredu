//! Cold source-owned reader inventory. This is not a request admission grant.
use super::*;
use eredu_gguf::{ReaderBuffer, ReaderBufferPreparationFailure};
use std::{alloc::Layout, collections::TryReserveError};

/// Fixed reader backing prepared for one builder's actual cache ceiling.
/// The immutable inventory includes both idle and attached buffers and the
/// bank's allocated element backing. Cold allocator/control costs are separate.
#[derive(Debug)]
pub struct GgufReaderBuffers {
    free: Vec<ReaderBuffer>,
    count: usize,
    storage_bytes: u64,
    prepared: bool,
}

/// Actual cold preparation refusal with every successfully prepared owner.
#[derive(Debug)]
pub struct GgufReaderBuffersFailure {
    cause: GgufReaderBuffersCause,
    buffers: GgufReaderBuffers,
}

/// The original failure from cold reader-bank preparation.
#[derive(Debug, thiserror::Error)]
pub enum GgufReaderBuffersCause {
    /// Fixed bank element allocation failed.
    #[error("{0}")]
    Bank(#[source] TryReserveError),
    /// One reader allocation failed; its partial owner is retained here.
    #[error("{0}")]
    Buffer(#[source] ReaderBufferPreparationFailure),
    /// Actual retained-capacity arithmetic could not be represented.
    #[error("GGUF reader buffer inventory overflow")]
    Inventory,
}
impl std::fmt::Display for GgufReaderBuffersFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for GgufReaderBuffersFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl GgufReaderBuffersFailure {
    /// Return the exact cause/failed buffer and successful bank prefix intact.
    pub fn into_parts(self) -> (GgufReaderBuffersCause, GgufReaderBuffers) {
        (self.cause, self.buffers)
    }
}

/// Original cause of cold source construction, retaining failed buffer storage.
#[derive(Debug, thiserror::Error)]
pub enum GgufSourcePreparationCause {
    /// Exact fresh reader/materializer construction with its original prefix.
    #[error("{0}")]
    Prepared(#[source] super::reader_compile::PreparedGgufSourceFailure),
    /// Existing limit, schema, mapping or final store-construction refusal.
    #[error("{0}")]
    Store(#[source] StoreError),
    /// Actual reader allocation refusal and its successfully prepared prefix.
    #[error("{0}")]
    Readers(#[source] GgufReaderBuffersFailure),
}

#[derive(Debug)]
struct SourcePreparationFailure {
    cause: GgufSourcePreparationCause,
    buffers: Option<GgufReaderBuffers>,
    builder: Option<GgufWeightStoreBuilder>,
}

/// Cold construction refusal with every input still owned by the convenience.
///
/// Before reader preparation the exact builder is retained. During preparation
/// the cause retains the failed buffer and successful prefix. After entering
/// the existing consuming build worker, failed catalogs/materializers retire
/// there; the complete returned bank is retained here. No checkpoint is cloned
/// for error retention. The error's Box is a separate, unqualified cold cost.
#[derive(Debug)]
pub struct GgufSourcePreparationFailure(Box<SourcePreparationFailure>);

impl std::fmt::Display for GgufSourcePreparationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.cause.fmt(f)
    }
}
impl std::error::Error for GgufSourcePreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0.cause {
            GgufSourcePreparationCause::Prepared(cause) => Some(cause),
            GgufSourcePreparationCause::Store(cause) => Some(cause),
            GgufSourcePreparationCause::Readers(cause) => Some(cause),
        }
    }
}
impl GgufSourcePreparationFailure {
    pub(super) fn validation(cause: StoreError) -> Self {
        Self(Box::new(SourcePreparationFailure {
            cause: GgufSourcePreparationCause::Store(cause),
            buffers: None,
            builder: None,
        }))
    }

    /// Return the exact cause, any untouched builder and any returned full bank.
    /// A partial bank belongs to the `Readers` cause instead.
    pub fn into_parts(
        self,
    ) -> (
        GgufSourcePreparationCause,
        Option<GgufWeightStoreBuilder>,
        Option<GgufReaderBuffers>,
    ) {
        let SourcePreparationFailure {
            cause,
            buffers,
            builder,
        } = *self.0;
        (cause, builder, buffers)
    }
}
impl GgufReaderBuffers {
    pub(super) fn preparation_controls() -> Option<usize> {
        use std::mem::size_of;
        let parts = [
            ReaderBuffer::preparation_request()?.1,
            size_of::<Self>(),
            size_of::<GgufReaderBuffersFailure>(),
            size_of::<GgufReaderBuffersCause>(),
            size_of::<Result<Self, GgufReaderBuffersFailure>>(),
            size_of::<Result<(), GgufReaderBuffersCause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<&mut Vec<ReaderBuffer>>(),
            size_of::<Layout>(),
            size_of::<u64>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    pub(super) fn prepare(count: usize) -> Result<Self, GgufReaderBuffersFailure> {
        let mut buffers = Self {
            free: Vec::new(),
            count,
            storage_bytes: 0,
            prepared: false,
        };
        let prepared = (|| {
            buffers
                .free
                .try_reserve_exact(count)
                .map_err(GgufReaderBuffersCause::Bank)?;
            for _ in 0..count {
                buffers
                    .free
                    .push(ReaderBuffer::prepare().map_err(GgufReaderBuffersCause::Buffer)?);
            }
            let headers = Layout::array::<ReaderBuffer>(buffers.free.capacity())
                .ok()
                .and_then(|layout| u64::try_from(layout.size()).ok())
                .ok_or(GgufReaderBuffersCause::Inventory)?;
            buffers.storage_bytes = buffers.free.iter().try_fold(headers, |bytes, buffer| {
                u64::try_from(buffer.capacity())
                    .ok()
                    .and_then(|capacity| bytes.checked_add(capacity))
                    .ok_or(GgufReaderBuffersCause::Inventory)
            })?;
            buffers.prepared = true;
            Ok(())
        })();
        match prepared {
            Ok(()) => Ok(buffers),
            Err(cause) => Err(GgufReaderBuffersFailure { cause, buffers }),
        }
    }
    /// Number of buffer owners this source's cache ceiling requires.
    pub fn count(&self) -> usize {
        self.count
    }
    /// Total actual backing after successful preparation. A failed prefix has
    /// no qualified inventory and returns `None`, even if some buffers exist.
    pub fn storage_bytes(&self) -> Option<u64> {
        self.prepared.then_some(self.storage_bytes)
    }
    fn recycle(&mut self, buffer: ReaderBuffer) {
        assert!(
            self.free.len() < self.count,
            "source cache returns its own vacated buffer"
        );
        assert!(
            self.free.len() < self.free.capacity(),
            "cold bank backing cannot grow"
        );
        self.free.push(buffer);
    }
}
impl GgufWeightStoreBuilder {
    pub(super) fn reader_buffer_count(&self) -> usize {
        let maximum = if self.max_cached_readers == 0 {
            DEFAULT_MAX_CACHED_SHARDS
        } else {
            self.max_cached_readers
        };
        maximum.min(self.checkpoints.len())
    }
    /// Allocate the exact fixed reader inventory before publishing this source.
    /// These are preexisting source buffers, not admitted request preparation.
    pub fn prepare_reader_buffers(&self) -> Result<GgufReaderBuffers, GgufReaderBuffersFailure> {
        GgufReaderBuffers::prepare(self.reader_buffer_count())
    }

    /// Constructs the exact source-owned reader bank and coordinate index cold.
    ///
    /// The actual builder supplies the count. Existing validation precedes the
    /// first new buffer allocation. This is not original-request preparation;
    /// the resulting immutable inventory describes preexisting source storage.
    pub fn build_with_prepared_reader_buffers(
        self,
    ) -> Result<GgufWeightStore, GgufSourcePreparationFailure> {
        if let Err(cause) = self.validate_build() {
            return Err(GgufSourcePreparationFailure(Box::new(
                SourcePreparationFailure {
                    cause: GgufSourcePreparationCause::Store(cause),
                    buffers: None,
                    builder: Some(self),
                },
            )));
        }
        super::reader_compile::prepare(self, None, None).map_err(|cause| {
            GgufSourcePreparationFailure(Box::new(SourcePreparationFailure {
                cause: GgufSourcePreparationCause::Prepared(cause),
                buffers: None,
                builder: None,
            }))
        })
    }

    /// Build with actual cold reader storage. Every refusal returns the complete
    /// supplied bank intact; ordinary `build` keeps lazy std reader allocation.
    pub fn build_with_reader_buffers(
        self,
        buffers: GgufReaderBuffers,
    ) -> Result<GgufWeightStore, (StoreError, GgufReaderBuffers)> {
        if buffers.count != self.reader_buffer_count() || buffers.storage_bytes().is_none() {
            return Err((
                reader_refusal("", "complete source-derived reader inventory"),
                buffers,
            ));
        }
        let mut supplied = Some(buffers);
        self.build_inner(&mut supplied).map_err(|error| {
            (
                error,
                supplied.expect("bank moves only at successful publication"),
            )
        })
    }
}
impl ReaderCache {
    pub(super) fn prepared_materializer_storage_bytes(&self) -> Result<u64, StoreError> {
        let overflow = || StoreError::Overflow {
            context: "GGUF prepared materializer storage".into(),
        };
        let materializers = Layout::array::<TensorMaterializer>(self.materializers.capacity())
            .map_err(|_| overflow())?;
        let last_used = Layout::array::<u64>(self.last_used.capacity()).map_err(|_| overflow())?;
        let bytes = u64::try_from(materializers.size())
            .ok()
            .and_then(|bytes| {
                u64::try_from(last_used.size())
                    .ok()
                    .and_then(|n| bytes.checked_add(n))
            })
            .ok_or_else(overflow)?;
        self.materializers
            .iter()
            .try_fold(bytes, |bytes, materializer| {
                materializer
                    .prepared_index_layout()
                    .and_then(|layout| u64::try_from(layout.size()).ok())
                    .and_then(|n| bytes.checked_add(n))
                    .ok_or_else(overflow)
            })
    }
    pub(super) fn recycle_reader_buffer(&mut self, index: usize) {
        if let Some(buffer) = self.materializers[index].take_idle_reader_buffer() {
            self.buffers
                .as_mut()
                .expect("prepared materializer belongs to source bank")
                .recycle(buffer);
        }
    }
    pub(super) fn supply_reader_buffer(
        &mut self,
        index: usize,
        key: &str,
    ) -> Result<(), StoreError> {
        let Some(bank) = self.buffers.as_mut() else {
            return Ok(());
        };
        let buffer = bank
            .free
            .pop()
            .ok_or_else(|| reader_refusal(key, "available source reader buffer"))?;
        if let Err(buffer) = self.materializers[index].supply_reader_buffer(buffer) {
            bank.recycle(buffer);
            return Err(reader_refusal(key, "empty materializer reader slot"));
        }
        Ok(())
    }
}
fn reader_refusal(key: &str, resource: &'static str) -> StoreError {
    gguf_read_error(key, eredu_gguf::Error::PreparedReaderStorage { resource })
}

#[cfg(test)]
mod tests;
