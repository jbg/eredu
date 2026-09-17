use crate::format::{
    align_up, Endian, GgmlType, MetadataArray, MetadataValue, TensorDescriptor, DEFAULT_ALIGNMENT,
};
use crate::{ConvertedTensor, Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

mod catalog_reader;
mod prepared_file;
pub(crate) use prepared_file::FileReader;
pub use prepared_file::{ReaderBuffer, ReaderBufferPreparationFailure};
mod header;
mod parse;
mod payload;
mod storage;
pub(crate) use catalog_reader::CatalogReader;
use std::borrow::Cow;
pub(crate) mod plan_storage;
use header::HeaderPolicy;
pub use header::{HeaderStorageKind, HeaderStorageStep, PreparedHeader, PreparedHeaderChanged};
pub use plan_storage::MetadataDestinationError;
use std::sync::Arc;
mod read_destination;
use read_destination::RawBuffer;
pub(crate) use read_destination::RawStorage;
pub use read_destination::ReadDestinationError;

/// A non-empty selection along one logical row-major tensor axis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorSelection {
    /// Select the half-open range `start..end` on `axis`.
    Range {
        axis: usize,
        start: usize,
        end: usize,
    },
    /// Select indices on `axis` in caller-supplied order.
    Indices { axis: usize, indices: Vec<usize> },
}

impl TensorSelection {
    fn axis(&self) -> usize {
        match self {
            Self::Range { axis, .. } | Self::Indices { axis, .. } => *axis,
        }
    }
}

/// GGUF block geometry constraining a physical tensor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionAlignment {
    block_values: u64,
    block_bytes: u64,
    selected_axis_multiple: u64,
}

impl SelectionAlignment {
    /// Values represented by one native GGUF block.
    pub const fn block_values(&self) -> u64 {
        self.block_values
    }

    /// Encoded bytes occupied by one native GGUF block.
    pub const fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// Required selection boundary multiple on the selected row-major axis.
    ///
    /// This is the GGUF block length when selecting the fastest physical
    /// dimension and one for every other dimension.
    pub const fn selected_axis_multiple(&self) -> u64 {
        self.selected_axis_multiple
    }
}

/// One absolute, contiguous encoded file range required by a selection plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedSpan {
    offset: u64,
    byte_len: u64,
}

/// A non-empty row-major scalar span exposed with a new logical shape.
///
/// This selection is intentionally distinct from [`TensorSelection`]: it is
/// legal only for unquantized F32, F16, and BF16 tensors. Packed GGUF blocks
/// therefore cannot enter a dequantize/requantize path accidentally.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DenseTensorSpan {
    offset_elements: u64,
    shape: Vec<u64>,
}

impl DenseTensorSpan {
    /// Actual retained shape-buffer capacity, excluding inline controls and allocator charge.
    pub fn shape_storage_layout(&self) -> Option<std::alloc::Layout> {
        std::alloc::Layout::array::<u64>(self.shape.capacity()).ok()
    }

    /// Describes a contiguous scalar interval and its row-major output shape.
    pub fn new(offset_elements: u64, shape: Vec<u64>) -> Result<Self> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(Error::InvalidHeader(
                "dense tensor span shape must be non-empty and nonzero".into(),
            ));
        }
        shape.iter().try_fold(1u64, |elements, dimension| {
            elements
                .checked_mul(*dimension)
                .ok_or(Error::Overflow("dense tensor span element count"))
        })?;
        Ok(Self {
            offset_elements,
            shape,
        })
    }

    /// Scalar offset from the beginning of the logical row-major tensor.
    pub const fn offset_elements(&self) -> u64 {
        self.offset_elements
    }

    /// Row-major shape exposed by the selected interval.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn element_count(&self) -> Result<u64> {
        self.shape.iter().try_fold(1u64, |elements, dimension| {
            elements
                .checked_mul(*dimension)
                .ok_or(Error::Overflow("dense tensor span element count"))
        })
    }
}

/// Metadata-only physical read plan for one block-aligned contiguous tensor span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensorSpanPlan {
    selection: DenseTensorSpan,
    selected_descriptor: TensorDescriptor,
    encoded_span: EncodedSpan,
}

impl DenseTensorSpanPlan {
    /// Validates a block-aligned contiguous span without reading its payload.
    ///
    /// `selection` is expressed in the descriptor's physical scalar units.
    /// Native quantized spans must begin and end on complete GGML blocks.
    pub fn new(tensor: &TensorDescriptor, selection: DenseTensorSpan) -> Result<Self> {
        let plan = plan_storage::span_plan(tensor, &selection, None)
            .map_err(MetadataDestinationError::ordinary)?;
        Ok(Self {
            selection,
            selected_descriptor: plan.selected_descriptor.into_owned(),
            encoded_span: plan.encoded_span,
        })
    }

    /// Original logical span represented by this plan.
    pub const fn selection(&self) -> &DenseTensorSpan {
        &self.selection
    }

    /// Descriptor used to convert the compact selected payload.
    pub const fn selected_descriptor(&self) -> &TensorDescriptor {
        &self.selected_descriptor
    }

    /// Exact physical file range read by the plan.
    pub const fn encoded_span(&self) -> EncodedSpan {
        self.encoded_span
    }

    /// Exact number of encoded bytes read by the plan.
    pub const fn encoded_byte_len(&self) -> u64 {
        self.encoded_span.byte_len
    }
}

impl EncodedSpan {
    /// Absolute byte offset in the GGUF shard.
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Number of encoded payload bytes in this span.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelativeEncodedSpan {
    offset: u64,
    byte_len: u64,
}

/// Metadata-only physical read plan for a single-axis tensor selection.
///
/// GGUF dimensions are fastest-moving first while logical tensor dimensions
/// are row-major. The plan records that axis translation, validates native
/// block alignment, describes the exact encoded reads as a compact repeated
/// pattern, and carries the descriptor expected by conversion after compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSelectionPlan {
    selection: TensorSelection,
    gguf_dimension: usize,
    alignment: SelectionAlignment,
    selected_descriptor: TensorDescriptor,
    source_data_offset: u64,
    repetition_stride: u64,
    repetitions: u64,
    relative_spans: Vec<RelativeEncodedSpan>,
    encoded_byte_len: u64,
}

impl TensorSelectionPlan {
    /// Build and validate a physical selection plan without reading payloads.
    pub fn new(tensor: &TensorDescriptor, selection: TensorSelection) -> Result<Self> {
        let plan =
            plan_storage::axis_plan(tensor, &selection, plan_storage::AxisStorage::new(None))
                .map_err(MetadataDestinationError::ordinary)?;
        Ok(Self {
            selection,
            gguf_dimension: plan.gguf_dimension,
            alignment: plan.alignment,
            selected_descriptor: plan.selected_descriptor.into_owned(),
            source_data_offset: plan.source_data_offset,
            repetition_stride: plan.repetition_stride,
            repetitions: plan.repetitions,
            relative_spans: plan.relative_spans.into_owned(),
            encoded_byte_len: plan.encoded_byte_len,
        })
    }

    /// Original logical selection represented by this plan.
    pub const fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Selected logical row-major axis.
    pub fn logical_axis(&self) -> usize {
        self.selection.axis()
    }

    /// Corresponding fastest-first GGUF dimension.
    pub const fn gguf_dimension(&self) -> usize {
        self.gguf_dimension
    }

    /// Native block geometry and selected-axis alignment.
    pub const fn alignment(&self) -> SelectionAlignment {
        self.alignment
    }

    /// Descriptor used to convert the compacted selected payload.
    pub const fn selected_descriptor(&self) -> &TensorDescriptor {
        &self.selected_descriptor
    }

    /// Exact number of encoded bytes read by the plan.
    pub const fn encoded_byte_len(&self) -> u64 {
        self.encoded_byte_len
    }

    /// Exact absolute encoded file spans in output order.
    pub fn encoded_spans(&self) -> impl Iterator<Item = EncodedSpan> + '_ {
        self.view().encoded_spans()
    }
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_metadata_entries: u64,
    pub max_array_elements: u64,
    pub max_tensor_count: u64,
    pub max_rank: u32,
    pub max_string_bytes: u64,
    pub max_allocation_bytes: u64,
    pub max_metadata_depth: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_metadata_entries: 1_000_000,
            max_array_elements: 16_000_000,
            max_tensor_count: 1_000_000,
            max_rank: 8,
            max_string_bytes: 256 << 20,
            max_allocation_bytes: 2 << 30,
            max_metadata_depth: 16,
        }
    }
}

pub struct Reader<R> {
    inner: R,
    endian: Endian,
    version: u32,
    alignment: u64,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorDescriptor>,
    limits: Limits,
    prepared_header: Option<Arc<PreparedHeader>>,
}

impl Reader<BufReader<File>> {
    /// Fixed payload-reader buffer capacity used by file-backed readers. This
    /// supports source-owned reader-cache bounds without opening a new file.
    pub const fn file_buffer_capacity() -> usize {
        8192
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, Limits::default())
    }
    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        Self::open_with_header_policy(path, limits, HeaderPolicy::Ordinary)
    }
    pub(crate) fn open_captured(path: &Path, limits: Limits) -> Result<Self> {
        Self::open_with_header_policy(path, limits, HeaderPolicy::capture())
    }
    fn open_with_header_policy(
        path: impl AsRef<Path>,
        limits: Limits,
        policy: HeaderPolicy,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::Io { offset: 0, source })?;
        Self::with_header_policy(
            BufReader::with_capacity(Self::file_buffer_capacity(), file),
            limits,
            policy,
        )
    }
}

#[cfg(test)]
mod file_buffer_tests {
    use super::*;

    #[test]
    fn file_reader_uses_declared_storage_capacity_through_payload_reads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("buffer.gguf");
        let bytes = 1.25_f32.to_le_bytes();
        crate::Writer::default()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[crate::TensorInput {
                    name: "weight",
                    dimensions: &[1],
                    ggml_type: crate::GgmlType::F32,
                    data: &bytes,
                }],
            )
            .unwrap();
        let mut reader = Reader::open(&path).unwrap();
        assert_eq!(
            reader.inner.capacity(),
            Reader::<BufReader<File>>::file_buffer_capacity()
        );
        let tensor = reader.tensors()[0].clone();
        assert_eq!(reader.read_raw(&tensor).unwrap(), bytes);
        assert_eq!(
            reader.inner.capacity(),
            Reader::<BufReader<File>>::file_buffer_capacity()
        );
    }
}

impl<R: Read + Seek> Reader<R> {
    pub fn new(inner: R) -> Result<Self> {
        Self::with_limits(inner, Limits::default())
    }

    pub fn with_limits(inner: R, limits: Limits) -> Result<Self> {
        Self::with_header_policy(inner, limits, HeaderPolicy::Ordinary)
    }
    fn with_header_policy(inner: R, limits: Limits, policy: HeaderPolicy) -> Result<Self> {
        let parsed = parse::parse(inner, limits, policy, None, None)?;
        Ok(Self {
            inner: parsed.inner,
            endian: parsed.endian,
            version: parsed.version,
            alignment: parsed.alignment,
            metadata: storage::owned(parsed.metadata)?,
            tensors: storage::owned(parsed.tensors)?,
            limits: parsed.limits,
            prepared_header: parsed.prepared_header,
        })
    }

    pub(crate) fn captured_header(&self) -> Option<&Arc<PreparedHeader>> {
        self.prepared_header.as_ref()
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn endian(&self) -> Endian {
        self.endian
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }
    pub fn into_metadata(self) -> BTreeMap<String, MetadataValue> {
        self.metadata
    }

    pub fn read_raw(&mut self, tensor: &TensorDescriptor) -> Result<Vec<u8>> {
        self.read_raw_with_storage(tensor, RawStorage::Ordinary)
            .map(RawBuffer::into_owned)
            .map_err(ReadDestinationError::ordinary)
    }

    pub fn read_tensor(&mut self, tensor: &TensorDescriptor) -> Result<ConvertedTensor> {
        let raw = self.read_raw(tensor)?;
        crate::convert::convert(tensor, &raw, self.endian)
    }

    /// Execute a validated metadata-only physical selection plan.
    pub fn read_tensor_plan(&mut self, plan: &TensorSelectionPlan) -> Result<ConvertedTensor> {
        self.read_tensor_plan_with_storage(plan.view(), RawStorage::Ordinary, None)
            .map_err(ReadDestinationError::ordinary)
    }
    /// Execute a validated block-aligned contiguous-span plan.
    pub fn read_dense_tensor_span(
        &mut self,
        plan: &DenseTensorSpanPlan,
    ) -> Result<ConvertedTensor> {
        self.read_dense_tensor_span_with_storage(plan.view(), RawStorage::Ordinary, None)
            .map_err(ReadDestinationError::ordinary)
    }
    fn payload(&mut self) -> payload::Payload<'_, R> {
        payload::Payload {
            inner: &mut self.inner,
            endian: self.endian,
            limits: &self.limits,
        }
    }
    fn read_raw_with_storage<'a>(
        &mut self,
        tensor: &TensorDescriptor,
        storage: RawStorage<'a>,
    ) -> std::result::Result<RawBuffer<'a>, ReadDestinationError> {
        self.payload().read_raw_with_storage(tensor.view(), storage)
    }
    pub(crate) fn read_tensor_plan_with_storage<C: read_destination::ConversionDestination>(
        &mut self,
        plan: plan_storage::AxisView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_tensor_plan_with_storage(plan, storage, conversion)
    }
    pub(crate) fn read_dense_tensor_span_with_storage<
        C: read_destination::ConversionDestination,
    >(
        &mut self,
        plan: plan_storage::SpanView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        self.payload()
            .read_dense_tensor_span_with_storage(plan, storage, conversion)
    }
}

fn check_limit(resource: &'static str, actual: u64, limit: u64) -> Result<()> {
    if actual > limit {
        Err(Error::Limit {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

struct Parser<'a, 's, R> {
    inner: R,
    endian: Endian,
    version: u32,
    limits: &'a Limits,
    policy: HeaderPolicy,
    scratch: Option<&'s mut [u8]>,
}

impl<R: Read + Seek> Parser<'_, '_, R> {
    fn pos(&mut self) -> Result<u64> {
        self.inner
            .stream_position()
            .map_err(|source| Error::Io { offset: 0, source })
    }
    fn exact(&mut self, out: &mut [u8]) -> Result<()> {
        let offset = self.pos()?;
        self.inner
            .read_exact(out)
            .map_err(|source| Error::Io { offset, source })?;
        self.policy.read(offset, out)
    }
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0];
        self.exact(&mut b)?;
        Ok(b[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let mut b = [0; 2];
        self.exact(&mut b)?;
        Ok(self.endian.u16(b))
    }
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0; 4];
        self.exact(&mut b)?;
        Ok(self.endian.u32(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0; 8];
        self.exact(&mut b)?;
        Ok(self.endian.u64(b))
    }
    fn count(&mut self) -> Result<u64> {
        if self.version == 1 {
            self.u32().map(Into::into)
        } else {
            self.u64()
        }
    }
    fn dimension(&mut self) -> Result<u64> {
        self.count()
    }
    fn string<'h>(&mut self, expected: Option<&'h String>) -> Result<storage::Name<'h>> {
        let len = self.count()?;
        check_limit("string bytes", len, self.limits.max_string_bytes)?;
        let len = usize::try_from(len).map_err(|_| Error::Overflow("string length"))?;
        if let Some(expected) = expected {
            if expected.len() != len {
                return Err(storage::refused("string length"));
            }
            self.policy
                .typed::<u8>(HeaderStorageKind::StringBytes, len, len)?;
            let scratch = self
                .scratch
                .take()
                .ok_or_else(|| storage::refused("string scratch"))?;
            let result = match scratch.get_mut(..len) {
                Some(bytes) => {
                    bytes.fill(0);
                    self.exact(bytes)
                }
                None => Err(storage::refused("string scratch extent")),
            };
            self.scratch = Some(scratch);
            result?;
            // The successful original exact read authenticated all bytes against
            // this cold UTF-8 String. No temporary owning String is constructed.
            Ok(Cow::Borrowed(expected))
        } else {
            if self.scratch.is_some() {
                return Err(storage::refused("string source"));
            }
            let mut bytes = vec![0; len];
            self.policy.typed::<u8>(
                HeaderStorageKind::StringBytes,
                bytes.len(),
                bytes.capacity(),
            )?;
            self.exact(&mut bytes)?;
            String::from_utf8(bytes)
                .map(Cow::Owned)
                .map_err(|e| Error::InvalidHeader(format!("invalid UTF-8 string: {e}")))
        }
    }
    fn value<'h>(
        &mut self,
        ty: u32,
        depth: u32,
        expected: Option<&'h MetadataValue>,
    ) -> Result<Cow<'h, MetadataValue>> {
        let value = match ty {
            0 => MetadataValue::Uint8(self.u8()?),
            1 => MetadataValue::Int8(self.u8()? as i8),
            2 => MetadataValue::Uint16(self.u16()?),
            3 => MetadataValue::Int16(self.u16()? as i16),
            4 => MetadataValue::Uint32(self.u32()?),
            5 => MetadataValue::Int32(self.u32()? as i32),
            6 => MetadataValue::Float32(f32::from_bits(self.u32()?)),
            7 => MetadataValue::Bool(match self.u8()? {
                0 => false,
                1 => true,
                v => return Err(Error::InvalidHeader(format!("invalid boolean {v}"))),
            }),
            8 => {
                let source = match expected {
                    None => None,
                    Some(MetadataValue::String(v)) => Some(v),
                    _ => return Err(storage::refused("string value variant")),
                };
                let value = self.string(source)?;
                if let Some(expected) = expected {
                    return Ok(Cow::Borrowed(expected));
                }
                MetadataValue::String(storage::owned(value)?)
            }
            9 => {
                let source = match expected {
                    None => None,
                    Some(MetadataValue::Array(v)) => Some(v),
                    _ => return Err(storage::refused("array value variant")),
                };
                let value = self.array(depth + 1, source)?;
                if let Some(expected) = expected {
                    return Ok(Cow::Borrowed(expected));
                }
                MetadataValue::Array(storage::owned(value)?)
            }
            10 => MetadataValue::Uint64(self.u64()?),
            11 => MetadataValue::Int64(self.u64()? as i64),
            12 => MetadataValue::Float64(f64::from_bits(self.u64()?)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        };
        Ok(match expected {
            Some(expected) => Cow::Borrowed(expected),
            None => Cow::Owned(value),
        })
    }
    fn array<'h>(
        &mut self,
        depth: u32,
        expected: Option<&'h MetadataArray>,
    ) -> Result<Cow<'h, MetadataArray>> {
        if depth > self.limits.max_metadata_depth {
            return Err(Error::Limit {
                resource: "metadata nesting depth",
                actual: depth.into(),
                limit: self.limits.max_metadata_depth.into(),
            });
        }
        let ty = self.u32()?;
        let len = self.count()?;
        check_limit(
            "metadata array elements",
            len,
            self.limits.max_array_elements,
        )?;
        let n = usize::try_from(len).map_err(|_| Error::Overflow("array length"))?;
        macro_rules! vals {
            ($variant:ident, $item:ident, $expr:expr) => {{
                let source = match expected {
                    None => None,
                    Some(MetadataArray::$variant(v)) => Some(v),
                    _ => return Err(storage::refused("array variant")),
                };
                let mut values = storage::Vector::new(n, source)?;
                values.record_array(&mut self.policy, n)?;
                for index in 0..n {
                    let $item = source.map(|v| &v[index]);
                    values.push($expr)?;
                }
                let values = values.finish()?;
                if let Some(expected) = expected {
                    Cow::Borrowed(expected)
                } else {
                    Cow::Owned(MetadataArray::$variant(storage::owned(values)?))
                }
            }};
            ($variant:ident, $expr:expr) => {
                vals!($variant, _item, Cow::Owned($expr?))
            };
        }
        Ok(match ty {
            0 => vals!(Uint8, self.u8()),
            1 => vals!(Int8, self.u8().map(|v| v as i8)),
            2 => vals!(Uint16, self.u16()),
            3 => vals!(Int16, self.u16().map(|v| v as i16)),
            4 => vals!(Uint32, self.u32()),
            5 => vals!(Int32, self.u32().map(|v| v as i32)),
            6 => vals!(Float32, self.u32().map(f32::from_bits)),
            7 => vals!(
                Bool,
                Ok::<_, Error>(match self.u8()? {
                    0 => false,
                    1 => true,
                    x => return Err(Error::InvalidHeader(format!("invalid boolean {x}"))),
                })
            ),
            8 => vals!(String, item, self.string(item)?),
            9 => vals!(Array, item, self.array(depth + 1, item)?),
            10 => vals!(Uint64, self.u64()),
            11 => vals!(Int64, self.u64().map(|v| v as i64)),
            12 => vals!(Float64, self.u64().map(f64::from_bits)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        })
    }
}

pub use plan_storage::{StoredPhysicalDescriptor, StoredPhysicalFailure};

pub(crate) use read_destination::ConversionDestination;

impl RelativeEncodedSpan {
    pub(crate) const fn empty() -> Self {
        Self {
            offset: 0,
            byte_len: 0,
        }
    }
}
