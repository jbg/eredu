//! Writable tensor construction followed by immutable checkpoint publication.

use super::*;
use std::{any::Any, collections::TryReserveError, fmt};

/// Unique writable encoded storage with validated SafeTensors metadata.
///
/// Consuming this buffer into a [`MemoryWeightStore`] publishes its existing
/// bytes without copying them. It cannot yield owned bytes or be cloned; mutable
/// access ends before any checkpoint lease or detached reader can be created.
#[derive(Debug)]
pub struct MemoryTensorBuffer {
    tensor: MemoryTensor,
    custody: Option<storage::SourceControl>,
}

impl MemoryTensorBuffer {
    /// Allocates zeroed storage after validating its complete encoded geometry.
    pub fn allocate(
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        byte_len: u64,
    ) -> Result<Self, MemoryTensorBufferError> {
        Self::allocate_inner(name, dtype, shape, byte_len, None)
    }

    /// Retains constructor custody before allocating new metadata and payload.
    ///
    /// The caller must arrange admission before calling this method. Custody is
    /// preserved by each resulting tensor, including independently retained
    /// leases, pinned bytes and detached readers, and by a construction error.
    /// This does not adopt preexisting bytes, certify funding, or price later
    /// catalog construction, selection buffers or weak-reference controls.
    pub fn allocate_with_custody<C: Any + fmt::Debug + Send + Sync>(
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        byte_len: u64,
        custody: C,
    ) -> Result<Self, MemoryTensorBufferError> {
        Self::allocate_inner(
            name,
            dtype,
            shape,
            byte_len,
            Some(storage::SourceControl::new(custody)),
        )
    }

    fn allocate_inner(
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        byte_len: u64,
        custody: Option<storage::SourceControl>,
    ) -> Result<Self, MemoryTensorBufferError> {
        let construct = || {
            let byte_len = usize::try_from(byte_len).map_err(|_| StoreError::Overflow {
                context: "in-memory tensor payload length".into(),
            })?;
            let mut metadata =
                metadata_for_parts(name, Path::new("<memory>"), dtype, shape, byte_len)?;
            metadata.backing_shard = None;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(byte_len)?;
            bytes.resize(byte_len, 0);
            Ok::<_, BufferFailure>((metadata, bytes))
        };
        match construct() {
            Ok((metadata, bytes)) => Ok(Self {
                tensor: MemoryTensor {
                    metadata,
                    dtype,
                    bytes,
                },
                custody,
            }),
            Err(cause) => Err(MemoryTensorBufferError {
                cause,
                _custody: custody,
            }),
        }
    }

    /// Borrows the final byte destination while its ownership is still unique.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.tensor.bytes
    }
}

#[derive(Debug, thiserror::Error)]
enum BufferFailure {
    #[error(transparent)]
    Metadata(#[from] StoreError),
    #[error("in-memory tensor payload allocation failed: {0}")]
    Allocation(#[from] TryReserveError),
}

/// A rejected tensor allocation retaining its original constructor custody.
#[derive(Debug)]
pub struct MemoryTensorBufferError {
    cause: BufferFailure,
    _custody: Option<storage::SourceControl>,
}
impl fmt::Display for MemoryTensorBufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for MemoryTensorBufferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            BufferFailure::Metadata(cause) => cause,
            BufferFailure::Allocation(cause) => cause,
        })
    }
}

/// Rejected publication retaining both its constructed prefix and duplicate.
pub struct MemoryWeightStoreBuildError {
    cause: StoreError,
    _partial: MemoryWeightStore,
    _rejected: MemoryTensorBuffer,
}
impl fmt::Debug for MemoryWeightStoreBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryWeightStoreBuildError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl fmt::Display for MemoryWeightStoreBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for MemoryWeightStoreBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl MemoryWeightStore {
    /// Publishes completed buffers without copying or detaching their payloads.
    ///
    /// Catalog nodes and tensor sharing controls are created here; their funding
    /// remains the caller's responsibility. A duplicate name preserves the
    /// already-published prefix and rejected buffer in the returned error.
    pub fn from_buffers(
        buffers: impl IntoIterator<Item = MemoryTensorBuffer>,
    ) -> Result<Self, MemoryWeightStoreBuildError> {
        let mut store = Self::default();
        for buffer in buffers {
            let name = &buffer.tensor.metadata.name;
            if store.tensors.contains_key(name) {
                return Err(MemoryWeightStoreBuildError {
                    cause: StoreError::Internal(format!("duplicate in-memory tensor {name:?}")),
                    _partial: store,
                    _rejected: buffer,
                });
            }
            store.tensors.insert(
                name.clone(),
                storage::SourceHandle::new(buffer.tensor, buffer.custody),
            );
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests;
