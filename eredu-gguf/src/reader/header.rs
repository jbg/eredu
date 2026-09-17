//! Closed capture/verification of the bytes and storage events of one real header.
use crate::{Error, Result};
use std::{alloc::Layout, sync::Arc};

/// A changed byte in a successfully read, explicitly prepared GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("prepared GGUF header changed at byte offset {offset}")]
pub struct PreparedHeaderChanged {
    pub(crate) offset: u64,
}
impl PreparedHeaderChanged {
    /// Absolute offset of the first mismatch in the successful parser read.
    pub const fn offset(self) -> u64 {
        self.offset
    }
}

/// The actual parser operation represented by a storage step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderStorageKind {
    /// Initialized bytes, later moved into a String.
    StringBytes,
    /// A successful metadata key clone before insertion.
    MetadataKey,
    /// A metadata map insertion; private node storage is not represented.
    MetadataEntries,
    /// The raw tensor descriptor Vec.
    RawDescriptors,
    /// Initial HashSet capacity request; buckets/seeds are not represented.
    TensorNames,
    /// A tensor name cloned for duplicate detection.
    TensorName,
    /// One dimension Vec.
    Dimensions,
    /// The final tensor descriptor Vec, overlapping the raw Vec.
    FinalDescriptors,
    /// One concrete metadata array Vec.
    Array,
    /// Observed final range collection capacity; collect growth/sort remain separate.
    Ranges,
}

/// A sealed event from actual successful header parsing.
///
/// Requested typed layouts are distinct from observed cold capacities. Neither
/// includes allocator charge, std map/set internals or stable-sort scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderStorageStep {
    kind: HeaderStorageKind,
    elements: usize,
    requested: Option<Layout>,
    capacity: Option<usize>,
    capacity_layout: Option<Layout>,
}
impl HeaderStorageStep {
    /// Parser operation, in execution order.
    pub const fn kind(&self) -> HeaderStorageKind {
        self.kind
    }
    /// Requested elements, or actual population for a collection observation.
    pub const fn elements(&self) -> usize {
        self.elements
    }
    /// Actual typed reserve request when the parser makes one explicitly.
    pub const fn requested_layout(&self) -> Option<Layout> {
        self.requested
    }
    /// Capacity observed in the cold parse, absent for a map with no capacity API.
    /// This is not a future allocator bound.
    pub const fn observed_capacity(&self) -> Option<usize> {
        self.capacity
    }
    /// Observed typed backing layout, when its element representation is known.
    pub const fn observed_capacity_layout(&self) -> Option<Layout> {
        self.capacity_layout
    }
}

/// Source-owned header image and schedule produced only by a successful parser.
///
/// This fixes the bytes observed at reopen, not the filesystem object or payload.
/// It is not a complete parser allocation, source or request bound.
#[derive(Debug)]
pub struct PreparedHeader {
    image: Vec<u8>,
    steps: Vec<HeaderStorageStep>,
    pub(super) metadata_order: Vec<String>,
    pub(super) parsed: Option<super::storage::FinalHeader>,
}
impl PartialEq for PreparedHeader {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image && self.steps == other.steps
    }
}
impl Eq for PreparedHeader {}
impl PreparedHeader {
    pub(super) fn final_header(&self) -> Result<&super::storage::FinalHeader> {
        self.parsed
            .as_ref()
            .ok_or_else(|| super::storage::refused("final header owner"))
    }
    pub(crate) fn scratch_len(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.kind == HeaderStorageKind::StringBytes)
            .map(|s| s.elements)
            .max()
            .unwrap_or(0)
    }

    /// Actual retained String/Vec backing layouts, including final header values.
    /// This excludes BTreeMap nodes, Arc headers, allocator charge and cold transient overlap.
    /// Image/schedule capacities are included; this is not a complete source bound.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        fn vec_bytes<T>(values: &Vec<T>) -> Option<usize> {
            Some(Layout::array::<T>(values.capacity()).ok()?.size())
        }
        fn string(value: &String) -> Option<usize> {
            Some(Layout::array::<u8>(value.capacity()).ok()?.size())
        }
        fn array(value: &crate::MetadataArray) -> Option<usize> {
            use crate::MetadataArray::*;
            match value {
                Uint8(v) => vec_bytes(v),
                Int8(v) => vec_bytes(v),
                Uint16(v) => vec_bytes(v),
                Int16(v) => vec_bytes(v),
                Uint32(v) => vec_bytes(v),
                Int32(v) => vec_bytes(v),
                Float32(v) => vec_bytes(v),
                Bool(v) => vec_bytes(v),
                Uint64(v) => vec_bytes(v),
                Int64(v) => vec_bytes(v),
                Float64(v) => vec_bytes(v),
                String(v) => v
                    .iter()
                    .try_fold(vec_bytes(v)?, |n, s| n.checked_add(string(s)?)),
                Array(v) => v
                    .iter()
                    .try_fold(vec_bytes(v)?, |n, a| n.checked_add(array(a)?)),
            }
        }
        let mut bytes = vec_bytes(&self.image)?
            .checked_add(vec_bytes(&self.steps)?)?
            .checked_add(vec_bytes(&self.metadata_order)?)?;
        for key in &self.metadata_order {
            bytes = bytes.checked_add(string(key)?)?;
        }
        if let Some(parsed) = &self.parsed {
            bytes = bytes
                .checked_add(vec_bytes(&parsed.tensors)?)?
                .checked_add(vec_bytes(&parsed.ranges)?)?;
            for (key, value) in &parsed.metadata {
                bytes = bytes.checked_add(string(key)?)?;
                bytes = bytes.checked_add(match value {
                    crate::MetadataValue::String(v) => string(v)?,
                    crate::MetadataValue::Array(v) => array(v)?,
                    _ => 0,
                })?;
            }
            for descriptor in &parsed.tensors {
                bytes = bytes
                    .checked_add(string(&descriptor.name)?)?
                    .checked_add(vec_bytes(&descriptor.dimensions)?)?;
            }
        }
        Some(bytes)
    }
    /// Exact parsed prefix, excluding alignment padding and tensor payloads.
    pub fn byte_len(&self) -> usize {
        self.image.len()
    }
    /// Actual retained image capacity.
    pub fn image_capacity(&self) -> usize {
        self.image.capacity()
    }
    /// Ordered real parser storage events.
    pub fn storage_steps(&self) -> &[HeaderStorageStep] {
        &self.steps
    }
    /// Actual retained schedule capacity.
    pub fn schedule_capacity(&self) -> usize {
        self.steps.capacity()
    }
    /// Logical minimum layouts at current lengths, not the capture Vec reserve requests.
    pub fn logical_minimum_layouts(&self) -> Option<[Layout; 2]> {
        Some([
            Layout::array::<u8>(self.image.len()).ok()?,
            Layout::array::<HeaderStorageStep>(self.steps.len()).ok()?,
        ])
    }
    /// Actual image/schedule backing layouts only; final value buffers are
    /// included separately in [`Self::retained_payload_bytes`].
    pub fn retained_layouts(&self) -> Option<[Layout; 2]> {
        Some([
            Layout::array::<u8>(self.image.capacity()).ok()?,
            Layout::array::<HeaderStorageStep>(self.steps.capacity()).ok()?,
        ])
    }
}

#[derive(Debug)]
pub(super) enum HeaderPolicy {
    Ordinary,
    Capture(PreparedHeader),
    Compare {
        header: Arc<PreparedHeader>,
        bytes: usize,
        steps: usize,
    },
}
impl HeaderPolicy {
    pub(super) fn capture() -> Self {
        Self::Capture(PreparedHeader {
            image: Vec::new(),
            steps: Vec::new(),
            metadata_order: Vec::new(),
            parsed: None,
        })
    }
    pub(super) fn compare(header: &Arc<PreparedHeader>) -> Self {
        Self::Compare {
            header: Arc::clone(header),
            bytes: 0,
            steps: 0,
        }
    }
    pub(super) fn keep_metadata_key(&mut self, key: super::storage::Name<'_>) -> Result<()> {
        if let Self::Capture(header) = self {
            header.metadata_order.push(super::storage::owned(key)?);
        }
        Ok(())
    }
    pub(super) fn read(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Ordinary => Ok(()),
            Self::Capture(header) => {
                if usize::try_from(offset).ok() != Some(header.image.len()) {
                    return Err(Error::Overflow("prepared header capture position"));
                }
                header.image.extend_from_slice(bytes);
                Ok(())
            }
            Self::Compare {
                header,
                bytes: consumed,
                ..
            } => {
                let start = usize::try_from(offset).map_err(|_| changed(offset))?;
                let end = start
                    .checked_add(bytes.len())
                    .ok_or_else(|| changed(offset))?;
                if start != *consumed {
                    return Err(changed(offset));
                }
                let expected = header
                    .image
                    .get(start..end)
                    .ok_or_else(|| changed(offset))?;
                if let Some(index) = bytes.iter().zip(expected).position(|(a, b)| a != b) {
                    return Err(changed(offset + index as u64));
                }
                *consumed = end;
                Ok(())
            }
        }
    }
    pub(super) fn typed<T>(
        &mut self,
        kind: HeaderStorageKind,
        count: usize,
        capacity: usize,
    ) -> Result<()> {
        if matches!(self, Self::Ordinary) {
            return Ok(());
        }
        self.step(HeaderStorageStep {
            kind,
            elements: count,
            requested: Some(
                Layout::array::<T>(count)
                    .map_err(|_| Error::Overflow("header requested layout"))?,
            ),
            capacity: Some(capacity),
            capacity_layout: Some(
                Layout::array::<T>(capacity)
                    .map_err(|_| Error::Overflow("header observed layout"))?,
            ),
        })
    }
    pub(super) fn collection(
        &mut self,
        kind: HeaderStorageKind,
        count: usize,
        capacity: Option<usize>,
    ) -> Result<()> {
        if matches!(self, Self::Ordinary) {
            return Ok(());
        }
        self.step(HeaderStorageStep {
            kind,
            elements: count,
            requested: None,
            capacity,
            capacity_layout: None,
        })
    }
    pub(super) fn ranges<T>(&mut self, values: &Vec<T>) -> Result<()> {
        if matches!(self, Self::Ordinary) {
            return Ok(());
        }
        self.step(HeaderStorageStep {
            kind: HeaderStorageKind::Ranges,
            elements: values.len(),
            requested: None,
            capacity: Some(values.capacity()),
            capacity_layout: Some(
                Layout::array::<T>(values.capacity())
                    .map_err(|_| Error::Overflow("header range capacity"))?,
            ),
        })
    }
    fn step(&mut self, step: HeaderStorageStep) -> Result<()> {
        match self {
            Self::Ordinary => Ok(()),
            Self::Capture(header) => {
                header.steps.push(step);
                Ok(())
            }
            Self::Compare {
                header,
                bytes,
                steps,
            } => {
                let expected = header
                    .steps
                    .get(*steps)
                    .ok_or_else(|| changed(*bytes as u64))?;
                // Future allocator capacities are observations, never authenticated bounds.
                if expected.kind != step.kind
                    || expected.elements != step.elements
                    || expected.requested != step.requested
                {
                    return Err(changed(*bytes as u64));
                }
                *steps += 1;
                Ok(())
            }
        }
    }
    pub(super) fn finish(self) -> Result<Option<Arc<PreparedHeader>>> {
        match self {
            Self::Ordinary => Ok(None),
            Self::Capture(header) => Ok(Some(Arc::new(header))),
            Self::Compare {
                header,
                bytes,
                steps,
            } => {
                if bytes != header.image.len() || steps != header.steps.len() {
                    return Err(changed(bytes as u64));
                }
                Ok(Some(header))
            }
        }
    }
}
fn changed(offset: u64) -> Error {
    Error::PreparedHeaderChanged(PreparedHeaderChanged { offset })
}

#[cfg(test)]
mod tests;
