//! Observed owned buffer capacities. Cache/source/allocator ownership is separate.
use super::*;
use std::alloc::Layout;
pub(crate) fn vector<T>(capacity: usize) -> Option<usize> {
    Some(Layout::array::<T>(capacity).ok()?.size())
}
pub(crate) fn selection(value: &TensorSelection) -> Option<usize> {
    match value {
        TensorSelection::Indices { indices, .. } => vector::<usize>(indices.capacity()),
        TensorSelection::Contiguous { shape, .. } => vector::<usize>(shape.capacity()),
        _ => Some(0),
    }
}
pub(crate) fn metadata(value: &TensorMetadata) -> Option<usize> {
    value
        .name
        .capacity()
        .checked_add(vector::<usize>(value.logical_shape.capacity())?)?
        .checked_add(vector::<usize>(value.physical_shape.capacity())?)?
        .checked_add(match &value.stored_dtype {
            crate::StoredDtype::Other(name) => name.capacity(),
            _ => 0,
        })?
        .checked_add(
            value
                .backing_shard
                .as_ref()
                .map_or(0, |path| path.capacity()),
        )
}
impl PreparedCheckpointAcquisition {
    pub(super) fn payload_storage(&self) -> Option<(usize, usize)> {
        let request = self
            .request
            .key
            .capacity()
            .checked_add(selection(&self.request.selection)?)?;
        let (bytes, arcs) = match &self.destination {
            Destination::Safetensors(value) => value.retained_payload_capacity()?,
            Destination::Memory(value) => value.retained_payload_capacity()?,
            Destination::Gguf(value) => (value.retained_payload_capacity()?, 0),
        };
        Some((request.checked_add(bytes)?, arcs))
    }
}
