//! Concrete provider loans and requested storage for actual retained lease clones.
use super::{
    CheckpointLease, EncodedTensorLease, MetadataCloneLayout, SelectionCloneLayout,
    SelectionValidationPlan, SourceMetadataBorrowError, TensorMetadata, TensorReadRequest,
    TensorSelection,
};
use std::alloc::Layout;

/// Concrete source selected by the retained authorization/overlay path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseProvider {
    /// Immutable memory tensor; leases share its metadata and payload owner.
    Memory,
    /// Admitted SafeTensors shard; leases own metadata and share shard/read owners.
    Safetensors,
    /// GGUF catalog; leases own catalog/read identity and share the lazy reader store.
    Gguf,
}

/// Refusal to describe the actual selected provider without ordinary fallback.
#[derive(Clone, Copy, Debug)]
pub enum LeaseControlBorrowError<'a> {
    /// The exact source's existing borrowed refusal or retained error.
    Source(SourceMetadataBorrowError<'a>),
    /// Nested forwarding counts cannot be represented.
    Overflow,
}
impl<'a> From<SourceMetadataBorrowError<'a>> for LeaseControlBorrowError<'a> {
    fn from(value: SourceMetadataBorrowError<'a>) -> Self {
        Self::Source(value)
    }
}
impl std::fmt::Display for LeaseControlBorrowError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(source) => std::fmt::Display::fmt(source, f),
            Self::Overflow => f.write_str("checkpoint lease forwarding count overflow"),
        }
    }
}
impl std::error::Error for LeaseControlBorrowError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(SourceMetadataBorrowError::Retained(source)) => Some(*source),
            _ => None,
        }
    }
}

/// Loan of an actual concrete catalog entry after selected source routing.
/// Private construction prevents numerical/provider tags from manufacturing a
/// catalog loan. This is a description, not an acquisition or admission token.
#[derive(Clone, Copy, Debug)]
pub struct SourceLeaseControls<'a> {
    provider: LeaseProvider,
    metadata: &'a TensorMetadata,
    request_key: &'a str,
    prepared_request_clones: usize,
    safetensors_reads: Option<super::SafetensorsReadSource<'a>>,
    safetensors_lease: Option<super::SafetensorsLeaseSource<'a>>,
}
impl<'a> SourceLeaseControls<'a> {
    pub(crate) fn new(
        provider: LeaseProvider,
        metadata: &'a TensorMetadata,
        request_key: &'a str,
    ) -> Self {
        Self {
            provider,
            metadata,
            request_key,
            prepared_request_clones: 0,
            safetensors_reads: None,
            safetensors_lease: None,
        }
    }
    pub(super) fn with_safetensors_lease(
        mut self,
        source: super::SafetensorsLeaseSource<'a>,
    ) -> Self {
        self.safetensors_lease = Some(source);
        self
    }
    /// The actual selected SafeTensors source can prepare final owned lease
    /// destinations without acquiring its payload shards. Other format workers
    /// retain their distinct preparation and conversion mechanisms.
    pub fn safetensors_lease_source(&self) -> Option<super::SafetensorsLeaseSource<'a>> {
        self.safetensors_lease
    }
    pub(super) fn with_safetensors_reads(
        mut self,
        reads: super::SafetensorsReadSource<'a>,
    ) -> Self {
        self.safetensors_reads = Some(reads);
        self
    }
    /// Actual admitted SafeTensors header inputs. Memory and GGUF use distinct
    /// acquisition workers and do not manufacture this format-specific view.
    pub fn safetensors_reads(&self) -> Option<super::SafetensorsReadSource<'a>> {
        self.safetensors_reads
    }
    pub(super) fn through_prepared(mut self) -> Result<Self, LeaseControlBorrowError<'a>> {
        self.prepared_request_clones = self
            .prepared_request_clones
            .checked_add(1)
            .ok_or(LeaseControlBorrowError::Overflow)?;
        Ok(self)
    }
    /// Concrete leaf selected by the real source path.
    pub fn provider(&self) -> LeaseProvider {
        self.provider
    }
    /// Actual concrete provider metadata, rather than a Prepared snapshot copy.
    pub fn metadata(&self) -> &'a TensorMetadata {
        self.metadata
    }
    /// Exact requested key; it need not be reconstructed from the metadata name.
    pub fn request_key(&self) -> &'a str {
        self.request_key
    }
    /// Prepared forwarding frames that each clone the owned read request.
    /// Their original requests remain alive until their child acquisition returns.
    pub fn prepared_request_clones(&self) -> usize {
        self.prepared_request_clones
    }
    /// Both loans refer to the same retained concrete catalog entry.
    /// Borrowing keeps both owners live; no pointer value is exported as authority.
    pub fn same_entry(&self, other: &Self) -> bool {
        self.provider == other.provider && std::ptr::eq(self.metadata, other.metadata)
    }
    /// One actual request clone's named payload layouts. The caller-created
    /// request and the forwarding multiplicity remain separate populations.
    pub fn request_clone_layout(&self, selection: &TensorSelection) -> Option<RequestCloneLayout> {
        Some(RequestCloneLayout {
            value: Layout::new::<TensorReadRequest>(),
            key: Layout::array::<u8>(self.request_key.len()).ok()?,
            selection: SelectionCloneLayout::of(selection)?,
        })
    }
    /// Uses the genuine selected entry's geometry in the shared destination worker.
    pub fn selection_validation<'s>(
        &'s self,
        selection: &'s TensorSelection,
    ) -> Option<SelectionValidationPlan<'s>> {
        SelectionValidationPlan::new(self.request_key, &self.metadata.logical_shape, selection)
    }
}

/// One requested read-request clone; the inline selection is inside `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestCloneLayout {
    /// Inline request value.
    pub value: Layout,
    /// Owned key bytes.
    pub key: Layout,
    /// Selection backing, if any.
    pub selection: SelectionCloneLayout,
}
impl RequestCloneLayout {
    /// Requested backing bytes only, without inline double counting.
    pub fn payload_bytes(&self) -> Option<usize> {
        self.key
            .size()
            .checked_add(self.selection.elements.map_or(0, |value| value.size()))
    }
}

/// Additional named owned fields copied by an actual GGUF lease clone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgufLeaseCloneLayout {
    /// Catalog's physical tensor name.
    pub physical_name: Layout,
    /// Catalog's original tensor name.
    pub original_name: Layout,
    /// Physical descriptor name.
    pub descriptor_name: Layout,
    /// Physical descriptor's dimensions in GGML order.
    pub descriptor_dimensions: Layout,
    /// Unknown scalar encoding name, if the retained encoding owns one.
    pub encoding_name: Option<Layout>,
    /// Independent name retained by the physical-read identity.
    pub identity_name: Layout,
    /// Actual physical index or dense-span-shape backing, when present.
    pub physical_selection: Option<Layout>,
}
impl GgufLeaseCloneLayout {
    /// Requested backing bytes; shared reader/conversion/cache owners are excluded.
    pub fn payload_bytes(&self) -> Option<usize> {
        self.physical_name
            .size()
            .checked_add(self.original_name.size())?
            .checked_add(self.descriptor_name.size())?
            .checked_add(self.descriptor_dimensions.size())?
            .checked_add(self.encoding_name.map_or(0, |value| value.size()))?
            .checked_add(self.identity_name.size())?
            .checked_add(self.physical_selection.map_or(0, |value| value.size()))
    }
}

/// Requested owned storage for `CheckpointLease::clone` on one actual lease.
/// Shared payload/source/cache Arc owners are retained, not copied or repriced.
/// This does not describe acquisition scratch, allocator overhead, or native conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseCloneLayout {
    /// Inline enum value, including the inline SafeTensors/Memory lease.
    pub value: Layout,
    /// GGUF's separate Box allocation, if selected.
    pub boxed_lease: Option<Layout>,
    /// Owned metadata copy (absent for a memory lease).
    pub metadata: Option<MetadataCloneLayout>,
    /// Owned logical selection copy.
    pub selection: SelectionCloneLayout,
    /// Actual retained output-shape copy.
    pub output_shape: Layout,
    /// GGUF-only owned catalog and physical-read identity fields.
    pub gguf: Option<GgufLeaseCloneLayout>,
}
impl LeaseCloneLayout {
    /// Requested backing allocation bytes, including GGUF's Box but excluding
    /// inline controls and all already-shared owner allocations.
    pub fn payload_bytes(&self) -> Option<usize> {
        self.boxed_lease
            .map_or(0, |value| value.size())
            .checked_add(match self.metadata {
                Some(value) => value.payload_bytes()?,
                None => 0,
            })?
            .checked_add(self.selection.elements.map_or(0, |value| value.size()))?
            .checked_add(self.output_shape.size())?
            .checked_add(match self.gguf {
                Some(value) => value.payload_bytes()?,
                None => 0,
            })
    }
}
impl CheckpointLease {
    /// Describes the fields that this actual lease's derived Clone owns.
    /// Does not clone a field, lock a cache, read a file, or infer a GGUF selection.
    pub fn clone_control_layout(&self) -> Option<LeaseCloneLayout> {
        let (boxed_lease, metadata, gguf) = match self {
            Self::Memory(_) => (None, None, None),
            Self::Safetensors(lease) => {
                (None, Some(MetadataCloneLayout::of(lease.metadata())?), None)
            }
            Self::Gguf(lease) => (
                Some(Layout::new::<crate::gguf_store::GgufLease>()),
                Some(MetadataCloneLayout::of(lease.metadata())?),
                Some(lease.owned_clone_layout()?),
            ),
        };
        Some(LeaseCloneLayout {
            value: Layout::new::<Self>(),
            boxed_lease,
            metadata,
            selection: SelectionCloneLayout::of(self.selection())?,
            output_shape: Layout::array::<usize>(self.output_shape().len()).ok()?,
            gguf,
        })
    }
}

#[cfg(test)]
mod tests;
