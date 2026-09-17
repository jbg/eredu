//! Read-only catalog loans used by closed source storage planning.
//! Metadata and overlay membership are descriptions, never admission authority.
use super::{EncodedTensorLease, StoreError, TensorMetadata, TensorSelection};
use std::alloc::Layout;

/// Whether the actual retained source supersedes source-side semantic recipes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKeyAuthority {
    /// Ordinary catalog entry (or absent key for the authority-only query).
    Ordinary,
    /// Entry materialized by the actual selected source overlay.
    Materialized,
}

/// Allocation-free catalog-loan refusal. Existing retained errors remain owned
/// by the source; lending one does not format, clone, move, or discard it.
#[derive(Clone, Copy, Debug)]
pub enum SourceMetadataBorrowError<'a> {
    /// This source has not implemented the borrowed companion.
    Unavailable,
    /// The source's existing metadata header has not been prepared.
    HeaderUnavailable,
    /// The requested key is absent.
    UnknownTensor,
    /// This authorization view does not expose the requested key.
    UnauthorizedTensor,
    /// A prior header preparation error remains retained in its exact source.
    Retained(&'a StoreError),
}
impl std::fmt::Display for SourceMetadataBorrowError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("borrowed source metadata is unavailable"),
            Self::HeaderUnavailable => f.write_str("source header is not prepared"),
            Self::UnknownTensor => f.write_str("source tensor is absent"),
            Self::UnauthorizedTensor => f.write_str("source tensor is not authorized"),
            Self::Retained(source) => std::fmt::Display::fmt(source, f),
        }
    }
}
impl std::error::Error for SourceMetadataBorrowError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Retained(source) => Some(*source),
            _ => None,
        }
    }
}

/// The source owns all metadata and retains it for the returned reference.
/// Alias identity and payload/cache/lease ownership remain separate facts.
pub type SourceMetadataLoan<'a> = Result<&'a TensorMetadata, SourceMetadataBorrowError<'a>>;

/// Requested backing layouts for one clone of the catalog metadata fields.
/// These are payload requests, not allocator charges or source/lease ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataCloneLayout {
    /// The inline metadata value, separate from all backing allocations.
    pub value: Layout,
    /// UTF-8 name bytes.
    pub name: Layout,
    /// Logical shape elements.
    pub logical_shape: Layout,
    /// Physical shape elements.
    pub physical_shape: Layout,
    /// Bytes of an unrecognized dtype name, when present.
    pub dtype_name: Option<Layout>,
    /// Platform-encoded path bytes, when present.
    pub backing_path: Option<Layout>,
}
impl MetadataCloneLayout {
    /// Describes the actual retained fields without cloning or formatting them.
    pub fn of(metadata: &TensorMetadata) -> Option<Self> {
        let dtype_name = match &metadata.stored_dtype {
            crate::StoredDtype::Other(name) => Some(Layout::array::<u8>(name.len()).ok()?),
            _ => None,
        };
        let backing_path = match &metadata.backing_shard {
            Some(path) => {
                Some(Layout::array::<u8>(path.as_os_str().as_encoded_bytes().len()).ok()?)
            }
            None => None,
        };
        Some(Self {
            value: Layout::new::<TensorMetadata>(),
            name: Layout::array::<u8>(metadata.name.len()).ok()?,
            logical_shape: Layout::array::<usize>(metadata.logical_shape.len()).ok()?,
            physical_shape: Layout::array::<usize>(metadata.physical_shape.len()).ok()?,
            dtype_name,
            backing_path,
        })
    }

    /// Sum of requested backing bytes, excluding the inline value and allocator
    /// bookkeeping. Empty backing requests do not require an allocation.
    pub fn payload_bytes(self) -> Option<usize> {
        self.name
            .size()
            .checked_add(self.logical_shape.size())?
            .checked_add(self.physical_shape.size())?
            .checked_add(self.dtype_name.map_or(0, |layout| layout.size()))?
            .checked_add(self.backing_path.map_or(0, |layout| layout.size()))
    }
}

/// Requested storage for cloning one actual selection value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionCloneLayout {
    /// Inline selection value.
    pub value: Layout,
    /// The indices or contiguous-shape Vec, when that variant owns one.
    pub elements: Option<Layout>,
}
impl SelectionCloneLayout {
    /// No selection validation, allocation, or output-shape computation occurs.
    pub fn of(selection: &TensorSelection) -> Option<Self> {
        let elements = match selection {
            TensorSelection::Full | TensorSelection::Range { .. } => None,
            TensorSelection::Indices { indices, .. } => {
                Some(Layout::array::<usize>(indices.len()).ok()?)
            }
            TensorSelection::Contiguous { shape, .. } => {
                Some(Layout::array::<usize>(shape.len()).ok()?)
            }
        };
        Some(Self {
            value: Layout::new::<TensorSelection>(),
            elements,
        })
    }
}

/// Named copies made by adapters that own metadata, selection, and output shape.
/// This is not the complete layout of a concrete lease: memory leases share the
/// source metadata, while GGUF leases retain additional catalog and read identity
/// fields. Source owners, read buffers, cache entries, and native handles are not
/// included here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedMetadataCloneLayout {
    /// Metadata fields when the adapter clones them.
    pub metadata: MetadataCloneLayout,
    /// Selection fields when the adapter clones them.
    pub selection: SelectionCloneLayout,
    /// Requested output-shape backing storage.
    pub output_shape: Layout,
}
impl SelectedMetadataCloneLayout {
    /// Describes a real selected lease, using its retained output-shape loan.
    pub fn from_lease<L: EncodedTensorLease + ?Sized>(lease: &L) -> Option<Self> {
        Some(Self {
            metadata: MetadataCloneLayout::of(lease.metadata())?,
            selection: SelectionCloneLayout::of(lease.selection())?,
            output_shape: Layout::array::<usize>(lease.output_shape().len()).ok()?,
        })
    }

    /// Describes the named copy destinations before acquisition from actual
    /// catalog metadata and the actual requested selection. Full, range, and
    /// indices preserve rank; contiguous selections own their declared rank.
    /// This does not validate dimensions, indices, read bounds, or source policy.
    /// Invalid requests still require the ordinary validator and its error path.
    pub fn for_selection(metadata: &TensorMetadata, selection: &TensorSelection) -> Option<Self> {
        let rank = match selection {
            TensorSelection::Full
            | TensorSelection::Range { .. }
            | TensorSelection::Indices { .. } => metadata.logical_shape.len(),
            TensorSelection::Contiguous { shape, .. } => shape.len(),
        };
        Some(Self {
            metadata: MetadataCloneLayout::of(metadata)?,
            selection: SelectionCloneLayout::of(selection)?,
            output_shape: Layout::array::<usize>(rank).ok()?,
        })
    }

    /// Requested backing bytes for the three named copies, excluding their
    /// inline storage and every other source/provider/native owner.
    pub fn payload_bytes(self) -> Option<usize> {
        self.metadata
            .payload_bytes()?
            .checked_add(self.selection.elements.map_or(0, |layout| layout.size()))?
            .checked_add(self.output_shape.size())
    }
}

#[cfg(test)]
mod tests;
