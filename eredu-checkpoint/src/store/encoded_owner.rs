//! Allocation-free retained ownership for an already-acquired encoded byte span.
use super::*;
use std::alloc::Layout;

/// A pin of actual source-owned bytes, without cloning request or lease metadata.
///
/// Construction only clones existing source/data Arcs. It does not acquire a
/// source, read a payload, copy bytes, change selection, or grant admission. Keep
/// this owner alive while a view borrows its bytes. Shared allocation and cache
/// ownership remain those of the original source, independently of its inline layout.
///
/// ```compile_fail
/// use eredu_checkpoint::store::PinnedEncodedBytes;
/// fn escape(owner: PinnedEncodedBytes) -> &'static [u8] {
///     owner.encoded_bytes().unwrap()
/// }
/// ```
#[derive(Clone, Debug)]
pub struct PinnedEncodedBytes {
    owner: Owner,
}
#[derive(Clone, Debug)]
enum Owner {
    Safetensors {
        bytes: Arc<Vec<u8>>,
        shard: Arc<CachedShard>,
    },
    Memory {
        selected: Option<Arc<[u8]>>,
        tensor: storage::SourceHandle<MemoryTensor>,
        span: Range<usize>,
    },
}
pub(super) fn memory_bytes<'a>(
    tensor: &'a MemoryTensor,
    selected: &'a Option<Arc<[u8]>>,
    span: &Range<usize>,
) -> Option<&'a [u8]> {
    match selected {
        Some(bytes) => Some(bytes.as_ref()),
        None => tensor.bytes.get(span.clone()),
    }
}
impl PinnedEncodedBytes {
    /// Exact existing encoded span. No materialization or selection is performed.
    pub fn encoded_bytes(&self) -> Option<&[u8]> {
        match &self.owner {
            Owner::Safetensors { bytes, .. } => Some(bytes.as_slice()),
            Owner::Memory {
                selected,
                tensor,
                span,
            } => memory_bytes(tensor, selected, span),
        }
    }
    /// The retained real shard path, if this source is file-backed.
    pub fn backing_path(&self) -> Option<&Path> {
        match &self.owner {
            Owner::Safetensors { shard, .. } => Some(&shard.path),
            Owner::Memory { .. } => None,
        }
    }
    /// Inline owner only; existing shared allocations and allocator charges are excluded.
    pub const fn owner_layout() -> Layout {
        Layout::new::<Self>()
    }
}
impl MemoryLease {
    /// Retain this actual memory span and source without cloning metadata/selection.
    pub fn pin_encoded_bytes(&self) -> PinnedEncodedBytes {
        PinnedEncodedBytes {
            owner: Owner::Memory {
                selected: self.selected_bytes.clone(),
                tensor: self.tensor.clone(),
                span: self.span.clone(),
            },
        }
    }
}
impl SafetensorsLease {
    /// Retain this actual byte buffer and cache pin without cloning lease metadata.
    pub fn pin_encoded_bytes(&self) -> PinnedEncodedBytes {
        PinnedEncodedBytes {
            owner: Owner::Safetensors {
                bytes: Arc::clone(&self.bytes),
                shard: Arc::clone(&self.shard),
            },
        }
    }
}
impl CheckpointLease {
    /// Pin an already byte-addressable lease. Lazy GGUF leases keep their separate
    /// actual read/conversion path and return `None`; this does not read them.
    pub fn pin_encoded_bytes(&self) -> Option<PinnedEncodedBytes> {
        match self {
            Self::Safetensors(lease) => Some(lease.pin_encoded_bytes()),
            Self::Memory(lease) => Some(lease.pin_encoded_bytes()),
            Self::Gguf(_) => None,
        }
    }
}
#[cfg(test)]
mod tests;
