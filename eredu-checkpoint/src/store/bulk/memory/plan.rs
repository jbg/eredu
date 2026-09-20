//! Original memory read records, sized before their shared constructor runs.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};

/// Allocation-free refusal while inspecting the actual immutable memory source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MemoryEncodedReadPlanError {
    /// The requested occurrence is absent from this exact store.
    #[error("memory read source occurrence {index} is absent")]
    UnknownTensor { index: usize },
    /// A source length cannot be represented by this host.
    #[error("memory encoded source length overflows")]
    SourceLength,
    /// Encoded metadata and the retained payload disagree.
    #[error("memory encoded source geometry differs")]
    Geometry,
    /// Consecutive destination coordinates overflow this host.
    #[error("memory encoded destination overflows")]
    DestinationLength,
    /// Construction storage cannot be represented by this host.
    #[error("memory encoded read metadata layout overflows")]
    Layout,
}
impl MemoryEncodedReadPlanError {
    pub(super) fn into_ordinary(self, keys: &[String]) -> StoreError {
        match self {
            Self::UnknownTensor { index } => StoreError::UnknownTensor {
                key: keys[index].clone(),
            },
            Self::Geometry => StoreError::Internal("memory encoded source geometry differs".into()),
            Self::SourceLength | Self::DestinationLength | Self::Layout => StoreError::Overflow {
                context: match self {
                    Self::SourceLength => "memory encoded source length",
                    Self::DestinationLength => "memory encoded destination",
                    _ => "memory encoded read metadata",
                }
                .into(),
            },
        }
    }
}

/// A fixed refusal from an authorized memory-read route or its source inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MemoryEncodedReadRouteError {
    /// An actual enclosing source view excludes this requested occurrence.
    #[error("memory read source occurrence {index} is not authorized")]
    UnauthorizedTensor { index: usize },
    /// The selected source cannot describe this batch.
    #[error(transparent)]
    Source(#[from] MemoryEncodedReadPlanError),
}

enum PlanSource<'a> {
    Borrowed(&'a MemoryWeightStore),
    Retained(Arc<MemoryWeightStore>),
}
impl std::ops::Deref for PlanSource<'_> {
    type Target = MemoryWeightStore;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(source) => source,
            Self::Retained(source) => source,
        }
    }
}

/// An original constructor over one immutable store and borrowed ordered keys.
/// Repeated keys retain separate metadata/span occurrences and share the same
/// original source owner. A routed plan retains its selected concrete store.
pub struct MemoryEncodedReadPlan<'a> {
    store: PlanSource<'a>,
    keys: &'a [String],
    byte_len: usize,
    backing_bytes: usize,
}
impl<'a> MemoryEncodedReadPlan<'a> {
    /// Validate every source and size the original destinations before cloning
    /// metadata or source handles. Source payload admission remains separate.
    pub fn new(
        store: &'a MemoryWeightStore,
        keys: &'a [String],
    ) -> Result<Self, MemoryEncodedReadPlanError> {
        Self::inspect(PlanSource::Borrowed(store), keys)
    }

    /// Follow actual built-in source owners and validate their batch visibility
    /// before inspection. The plan retains the selected store, so construction
    /// invokes no routing callbacks and does not need the enclosing views alive.
    /// `None` means no concrete memory route, including mixed composite children;
    /// it never invokes ordinary acquisition or accepts a forwarded owner identity.
    /// Built-in traversal clones existing handles without allocating route rows.
    pub fn from_source(
        source: &RetainedCheckpointSource,
        keys: &'a [String],
    ) -> Result<Option<Self>, MemoryEncodedReadRouteError> {
        let Some(store) = acquisition::retained_route::encoded_memory_source(source, keys)? else {
            return Ok(None);
        };
        Self::inspect(PlanSource::Retained(store), keys)
            .map(Some)
            .map_err(Into::into)
    }

    fn inspect(
        store: PlanSource<'a>,
        keys: &'a [String],
    ) -> Result<Self, MemoryEncodedReadPlanError> {
        use MemoryEncodedReadPlanError as E;
        let mut backing_bytes = Layout::array::<TensorMetadata>(keys.len())
            .map_err(|_| E::Layout)?
            .size()
            .checked_add(
                Layout::array::<ReadMemory>(keys.len())
                    .map_err(|_| E::Layout)?
                    .size(),
            )
            .ok_or(E::Layout)?;
        let mut byte_len = 0usize;
        for (index, key) in keys.iter().enumerate() {
            let tensor = store.tensors.get(key).ok_or(E::UnknownTensor { index })?;
            let length =
                usize::try_from(tensor.metadata.encoded_byte_len).map_err(|_| E::SourceLength)?;
            if length != tensor.bytes.len() {
                return Err(E::Geometry);
            }
            byte_len = byte_len.checked_add(length).ok_or(E::DestinationLength)?;
            backing_bytes = backing_bytes
                .checked_add(size_of::<ReadSpan>())
                .and_then(|n| {
                    n.checked_add(MetadataCloneLayout::of(&tensor.metadata)?.payload_bytes()?)
                })
                .ok_or(E::Layout)?;
        }
        Ok(Self {
            store,
            keys,
            byte_len,
            backing_bytes,
        })
    }

    /// Requested backing extents and named constructor/result controls. This
    /// excludes existing keys, source payloads, allocator-private bookkeeping,
    /// and any storage needed to create the caller's custody. It is not a grant.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<PreparedMemoryEncodedRead<C>>(),
            size_of::<MemoryEncodedReadBuildError<C>>(),
            size_of::<Result<PreparedMemoryEncodedRead<C>, MemoryEncodedReadBuildError<C>>>(),
            size_of::<ReadMemory>(),
            size_of::<ReadSpan>(),
            size_of::<TensorMetadata>(),
            size_of::<storage::SourceHandle<MemoryTensor>>(),
            size_of::<Vec<ReadSpan>>(),
            size_of::<Result<(), TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(self.backing_bytes, usize::checked_add)
    }

    /// Build the actual original read batch after the caller admits its storage.
    /// Every completed prefix precedes custody during success, failure and unwind.
    /// This uses fresh exact-capacity vectors and ordinary metadata field clones.
    pub fn construct<C>(
        self,
        custody: C,
    ) -> Result<PreparedMemoryEncodedRead<C>, MemoryEncodedReadBuildError<C>> {
        let mut owner = PreparedMemoryEncodedRead {
            batch: EncodedReadBatch {
                tensors: Vec::new(),
                shards: Vec::new(),
                memory: Vec::new(),
                byte_len: 0,
                telemetry: None,
                cache: None,
            },
            _custody: custody,
        };
        let result = (|| {
            owner.batch.tensors.try_reserve_exact(self.keys.len())?;
            owner.batch.memory.try_reserve_exact(self.keys.len())?;
            for key in self.keys {
                let tensor = &self.store.tensors[key];
                // Inspection binds an immutable source and validates the whole sum.
                let end = owner.batch.byte_len + tensor.bytes.len();
                let mut spans = Vec::new();
                spans.try_reserve_exact(1)?;
                spans.push(ReadSpan {
                    source: 0..tensor.metadata.encoded_byte_len,
                    destination: owner.batch.byte_len..end,
                });
                owner.batch.memory.push(ReadMemory {
                    tensor: tensor.clone(),
                    spans,
                });
                owner.batch.tensors.push(tensor.metadata.clone());
                owner.batch.byte_len = end;
            }
            debug_assert_eq!(owner.batch.byte_len, self.byte_len);
            Ok(())
        })();
        match result {
            Ok(()) => Ok(owner),
            Err(cause) => Err(MemoryEncodedReadBuildError {
                cause,
                partial: owner,
            }),
        }
    }
}

/// Move-only original read records with caller custody retained after all source
/// and metadata fields. Only metadata loans and direct destination reads escape.
///
/// ```compile_fail
/// use eredu_checkpoint::store::PreparedMemoryEncodedRead;
/// fn copy(read: PreparedMemoryEncodedRead<()>) { let _ = read.clone(); }
/// ```
pub struct PreparedMemoryEncodedRead<C> {
    pub(super) batch: EncodedReadBatch,
    _custody: C,
}
impl<C> PreparedMemoryEncodedRead<C> {
    /// Immutable metadata in requested occurrence order.
    pub fn tensors(&self) -> &[TensorMetadata] {
        self.batch.tensors()
    }
    /// Required caller-owned payload destination length, not constructor storage.
    pub fn byte_len(&self) -> usize {
        self.batch.byte_len()
    }
    /// Read the original immutable payloads without allocating scratch or staging.
    /// A wrong destination length refuses before any output byte is changed.
    pub fn read_into(&self, destination: &mut [u8]) -> Result<(), EncodedReadFailure> {
        if destination.len() != self.byte_len() {
            return Err(EncodedReadFailure {
                batch: None,
                shard: None,
                completed_shards: 0,
                cause: EncodedReadFailureCause::DestinationLengths,
            });
        }
        for source in &self.batch.memory {
            source.copy_into(destination)?;
        }
        Ok(())
    }
}
impl<C> fmt::Debug for PreparedMemoryEncodedRead<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedMemoryEncodedRead")
            .field("tensors", &self.batch.tensors.len())
            .field("byte_len", &self.byte_len())
            .finish_non_exhaustive()
    }
}

/// A failed reserve retains the actual constructed batch prefix and custody.
pub struct MemoryEncodedReadBuildError<C> {
    cause: TryReserveError,
    partial: PreparedMemoryEncodedRead<C>,
}
impl<C> MemoryEncodedReadBuildError<C> {
    /// Metadata rows completely constructed before the failed reserve.
    pub fn completed_tensors(&self) -> usize {
        self.partial.batch.tensors.len()
    }
}
impl<C> fmt::Debug for MemoryEncodedReadBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryEncodedReadBuildError")
            .field("cause", &self.cause)
            .field("partial", &self.partial)
            .finish()
    }
}
impl<C> fmt::Display for MemoryEncodedReadBuildError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C> std::error::Error for MemoryEncodedReadBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[cfg(test)]
mod route_tests;
#[cfg(test)]
mod tests;
