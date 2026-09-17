//! Borrowed metadata encoding through the existing SafeTensors/JSON schema.
use safetensors::tensor::TensorInfo;
use serde::{Serialize, Serializer, ser::SerializeMap};
use std::{
    io::{self, Write},
    mem::{size_of, size_of_val},
};

/// Fixed rejection of source geometry or an exact header destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SafetensorsHeaderError {
    /// Tensor names must be unique and may not use the reserved metadata member.
    #[error("invalid or duplicate SafeTensors tensor name")]
    Name,
    /// Canonical dtype alignment/name order differs from the actual source rows.
    #[error("SafeTensors source rows are not in canonical dtype/name order")]
    Order,
    /// Source byte offsets are not contiguous and monotonically increasing.
    #[error("SafeTensors source offsets are not contiguous")]
    Offset,
    /// Declared shape/dtype does not produce the exact source extent.
    #[error("SafeTensors shape/dtype and source byte extent disagree")]
    Geometry,
    /// Tensor bit counts are not byte aligned.
    #[error("SafeTensors tensor bit extent is not byte aligned")]
    Alignment,
    /// A checked extent or host layout cannot be represented.
    #[error("SafeTensors header or payload extent overflows")]
    Overflow,
    /// The header exceeds the same format limit as the ordinary serializer.
    #[error("SafeTensors header exceeds its format limit")]
    HeaderTooLarge,
    /// The caller did not supply the queried exact initialized destination.
    #[error("SafeTensors header destination has the wrong length")]
    Destination,
}

/// A nonallocating plan borrowing exact ordered `TensorInfo` declarations.
///
/// The owning caller pays and preserves those declarations and output storage.
/// Encoding delegates each row to SafeTensors' actual `TensorInfo::serialize`
/// and uses serde_json's same compact serializer. No arbitrary serializer, map
/// population, data buffer, path or execution authority can enter this plan.
pub struct SafetensorsHeaderPlan<'a> {
    entries: &'a [(&'a str, TensorInfo)],
    json_bytes: usize,
    aligned_bytes: usize,
    file_bytes: usize,
}
struct Header<'a>(&'a [(&'a str, TensorInfo)]);
impl Serialize for Header<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, info) in self.0 {
            map.serialize_entry(name, info)?;
        }
        map.end()
    }
}
#[derive(Default)]
struct Count {
    bytes: usize,
    overflow: bool,
}
impl Write for Count {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self.bytes.checked_add(bytes.len()) {
            Some(total) => self.bytes = total,
            None => self.overflow = true,
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
struct Destination<'a> {
    bytes: &'a mut [u8],
    position: usize,
}
impl Write for Destination<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .position
            .checked_add(bytes.len())
            .expect("counted header offset");
        self.bytes[self.position..end].copy_from_slice(bytes);
        self.position = end;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl<'a> SafetensorsHeaderPlan<'a> {
    /// Checks exact geometry, distinct names and the ordinary descending dtype
    /// alignment/name order before counting the immutable JSON representation.
    pub fn prepare(entries: &'a [(&'a str, TensorInfo)]) -> Result<Self, SafetensorsHeaderError> {
        let mut offset = 0usize;
        for (index, (name, info)) in entries.iter().enumerate() {
            if *name == "__metadata__" || entries[..index].iter().any(|(prior, _)| prior == name) {
                return Err(SafetensorsHeaderError::Name);
            }
            if let Some((prior_name, prior)) = index.checked_sub(1).and_then(|i| entries.get(i)) {
                if info
                    .dtype
                    .cmp(&prior.dtype)
                    .then(prior_name.cmp(name))
                    .is_gt()
                {
                    return Err(SafetensorsHeaderError::Order);
                }
            }
            let (start, end) = info.data_offsets;
            if start != offset || end < start {
                return Err(SafetensorsHeaderError::Offset);
            }
            let elements = info
                .shape
                .iter()
                .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                .ok_or(SafetensorsHeaderError::Overflow)?;
            let bits = elements
                .checked_mul(info.dtype.bitsize())
                .ok_or(SafetensorsHeaderError::Overflow)?;
            if bits % 8 != 0 {
                return Err(SafetensorsHeaderError::Alignment);
            }
            if bits / 8 != end - start {
                return Err(SafetensorsHeaderError::Geometry);
            }
            offset = end;
        }
        let mut count = Count::default();
        // Closed TensorInfo/string/integer serialization has no semantic error,
        // and this concrete writer always succeeds. It never constructs a JSON
        // error or an owned map even when the checked count overflows.
        Header(entries)
            .serialize(&mut serde_json::Serializer::new(&mut count))
            .expect("closed infallible SafeTensors header schema");
        if count.overflow {
            return Err(SafetensorsHeaderError::Overflow);
        }
        let aligned_bytes = count
            .bytes
            .checked_add(7)
            .ok_or(SafetensorsHeaderError::Overflow)?
            & !7usize;
        if aligned_bytes as u128 > super::MAX_HEADER_BYTES as u128 {
            return Err(SafetensorsHeaderError::HeaderTooLarge);
        }
        let file_bytes = 8usize
            .checked_add(aligned_bytes)
            .and_then(|n| n.checked_add(offset))
            .ok_or(SafetensorsHeaderError::Overflow)?;
        std::alloc::Layout::array::<u8>(file_bytes)
            .map_err(|_| SafetensorsHeaderError::Overflow)?;
        Ok(Self {
            entries,
            json_bytes: count.bytes,
            aligned_bytes,
            file_bytes,
        })
    }
    /// Exact header destination including its little-endian length and padding.
    pub fn header_bytes(&self) -> usize {
        8 + self.aligned_bytes
    }
    /// Whole file extent: exact encoded header followed by the declared payload.
    pub fn file_bytes(&self) -> usize {
        self.file_bytes
    }
    /// Encodes into one exact initialized caller-owned destination. The source
    /// remains immutably borrowed from the count through the final JSON write.
    pub fn write_header(&self, output: &mut [u8]) -> Result<(), SafetensorsHeaderError> {
        if output.len() != self.header_bytes() {
            return Err(SafetensorsHeaderError::Destination);
        }
        output[..8].copy_from_slice(&(self.aligned_bytes as u64).to_le_bytes());
        let mut writer = Destination {
            bytes: &mut output[8..8 + self.json_bytes],
            position: 0,
        };
        Header(self.entries)
            .serialize(&mut serde_json::Serializer::new(&mut writer))
            .expect("same closed counted SafeTensors header");
        debug_assert_eq!(writer.position, self.json_bytes);
        output[8 + self.json_bytes..].fill(b' ');
        Ok(())
    }
    /// Concrete fixed serializer/validation/destination frames, excluding the
    /// caller-owned source declarations and initialized encoded header buffer.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Header<'_>>(),
            size_of::<Count>(),
            size_of::<Destination<'_>>(),
            size_of::<serde_json::Serializer<&mut Count>>(),
            size_of::<serde_json::Serializer<&mut Destination<'_>>>(),
            size_of::<serde_json::ser::Compound<'_, &mut Count, serde_json::ser::CompactFormatter>>(
            ),
            size_of::<
                serde_json::ser::Compound<
                    '_,
                    &mut Destination<'_>,
                    serde_json::ser::CompactFormatter,
                >,
            >(),
            size_of::<std::slice::Iter<'_, (&str, TensorInfo)>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, (&str, TensorInfo)>>>(),
            size_of::<std::slice::Iter<'_, usize>>(),
            size_of::<(&str, &TensorInfo)>(),
            size_of::<(usize, usize)>(),
            size_of::<Result<Self, SafetensorsHeaderError>>(),
            size_of::<Result<(), SafetensorsHeaderError>>(),
            size_of::<Result<(), serde_json::Error>>(),
            size_of::<Result<usize, io::Error>>(),
            size_of::<SafetensorsHeaderError>(),
            size_of::<[u8; 8]>(),
            // Compact serde_json integer formatting uses itoa's largest integer
            // buffer (40 bytes); string control escaping uses six fixed bytes.
            // This closed schema never enters floating-point formatting.
            size_of::<[u8; 40]>(),
            size_of::<[u8; 6]>(),
            size_of::<std::slice::Iter<'_, u8>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<(&str, &[u8])>(),
            size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
#[cfg(test)]
mod tests;
