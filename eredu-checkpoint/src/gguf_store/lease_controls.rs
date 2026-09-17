//! Actual GGUF fields; no reimplementation of physical-selection planning.
use super::*;
use crate::store::{
    GgufLeaseCloneLayout, LeaseControlBorrowError, LeaseProvider, SourceLeaseControls,
    SourceMetadataBorrowError,
};
use std::alloc::Layout;

pub(super) fn source<'a>(
    store: &'a GgufWeightStore,
    key: &'a str,
) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
    let entry = store
        .inner
        .catalog
        .get(key)
        .ok_or(SourceMetadataBorrowError::UnknownTensor)?;
    Ok(SourceLeaseControls::new(
        LeaseProvider::Gguf,
        &entry.metadata,
        key,
    ))
}
impl GgufLease {
    pub(crate) fn owned_clone_layout(&self) -> Option<GgufLeaseCloneLayout> {
        physical_clone_layout(&self.entry, &self.identity)
    }
}
pub(super) fn physical_clone_layout(
    entry: &CatalogEntry,
    identity: &GgufLeaseIdentity,
) -> Option<GgufLeaseCloneLayout> {
    let physical_selection = match identity.selection.as_ref() {
        Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices { indices, .. })) => {
            Some(Layout::array::<usize>(indices.len()).ok()?)
        }
        Some(GgufPhysicalSelection::DenseSpan(span)) => {
            Some(Layout::array::<u64>(span.shape().len()).ok()?)
        }
        Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Range { .. })) | None => None,
    };
    let encoding_name = match &entry.source_encoding {
        crate::SourceTensorEncoding::Safetensors(StoredDtype::Other(name))
        | crate::SourceTensorEncoding::RecipeOutput(StoredDtype::Other(name)) => {
            Some(Layout::array::<u8>(name.len()).ok()?)
        }
        crate::SourceTensorEncoding::Safetensors(_)
        | crate::SourceTensorEncoding::RecipeOutput(_)
        | crate::SourceTensorEncoding::Gguf { .. } => None,
    };
    Some(GgufLeaseCloneLayout {
        physical_name: Layout::array::<u8>(entry.physical_name.len()).ok()?,
        original_name: Layout::array::<u8>(entry.original_name.len()).ok()?,
        descriptor_name: Layout::array::<u8>(entry.physical_descriptor.name.len()).ok()?,
        descriptor_dimensions: Layout::array::<u64>(entry.physical_descriptor.dimensions.len())
            .ok()?,
        encoding_name,
        identity_name: Layout::array::<u8>(identity.physical_name.len()).ok()?,
        physical_selection,
    })
}
