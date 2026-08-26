use crate::format::{
    align_up, Endian, GgmlType, MetadataArray, MetadataValue, TensorDescriptor, DEFAULT_ALIGNMENT,
};
use crate::{ConvertedTensor, Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// A non-empty selection along one MLX/row-major tensor axis.
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

    /// Required selection boundary multiple on the selected MLX axis.
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
        let tensor_elements = tensor.element_count()?;
        let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
        if !tensor_elements.is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "tensor element count {tensor_elements} is not divisible by {:?} block length {block_values}",
                    tensor.ggml_type
                ),
            ));
        }
        let expected_source_bytes = tensor_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor byte length"))?;
        if expected_source_bytes != tensor.byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "descriptor declares {} encoded bytes but its dense shape and type require {expected_source_bytes}",
                    tensor.byte_len
                ),
            ));
        }
        let selected_elements = selection.element_count()?;
        let selected_end = selection
            .offset_elements
            .checked_add(selected_elements)
            .ok_or(Error::Overflow("dense tensor span end"))?;
        if selected_end > tensor_elements {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous scalar span {}..{selected_end} exceeds tensor element count {tensor_elements}",
                    selection.offset_elements
                ),
            ));
        }
        if !selection.offset_elements.is_multiple_of(block_values)
            || !selected_elements.is_multiple_of(block_values)
        {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous span {}..{selected_end} must align to {:?} block length {block_values}",
                    selection.offset_elements, tensor.ggml_type
                ),
            ));
        }
        let fastest = selection.shape.last().copied().ok_or_else(|| {
            Error::tensor(&tensor.name, "contiguous span has no fastest dimension")
        })?;
        if !fastest.is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous span fastest dimension {fastest} must align to {:?} block length {block_values}",
                    tensor.ggml_type
                ),
            ));
        }
        let byte_offset = selection
            .offset_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor span byte offset"))?;
        let byte_len = selected_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor span byte length"))?;
        let offset = tensor
            .data_offset
            .checked_add(byte_offset)
            .ok_or(Error::Overflow("dense tensor span file offset"))?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(Error::Overflow("dense tensor span file end"))?;
        let tensor_end = tensor
            .data_offset
            .checked_add(tensor.byte_len)
            .ok_or(Error::Overflow("tensor end offset"))?;
        if end > tensor_end {
            return Err(Error::tensor(
                &tensor.name,
                "contiguous scalar span exceeds the encoded tensor payload",
            ));
        }

        let mut selected_descriptor = tensor.clone();
        selected_descriptor.dimensions = selection.shape.iter().rev().copied().collect();
        selected_descriptor.relative_offset = selected_descriptor
            .relative_offset
            .checked_add(byte_offset)
            .ok_or(Error::Overflow("dense tensor span relative offset"))?;
        selected_descriptor.data_offset = offset;
        selected_descriptor.byte_len = byte_len;
        Ok(Self {
            selection,
            selected_descriptor,
            encoded_span: EncodedSpan { offset, byte_len },
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
struct RelativeEncodedSpan {
    offset: u64,
    byte_len: u64,
}

/// Metadata-only physical read plan for a single-axis tensor selection.
///
/// GGUF dimensions are fastest-moving first while MLX dimensions are
/// row-major. The plan records that axis translation, validates native block
/// alignment, describes the exact encoded reads as a compact repeated pattern,
/// and carries the descriptor expected by conversion after compaction.
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
        let rank = tensor.dimensions.len();
        let logical_axis = selection.axis();
        if logical_axis >= rank {
            return Err(Error::tensor(
                &tensor.name,
                format!("selection axis {logical_axis} is outside rank {rank}"),
            ));
        }
        if tensor.byte_len == 0 || tensor.dimensions.contains(&0) {
            return Err(Error::tensor(
                &tensor.name,
                "cannot select from an empty tensor",
            ));
        }

        let gguf_dimension = rank - 1 - logical_axis;
        let dimension_u64 = tensor.dimensions[gguf_dimension];
        let dimension = usize::try_from(dimension_u64)
            .map_err(|_| Error::Overflow("selected tensor dimension"))?;
        let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
        if !tensor.dimensions[0].is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "fastest dimension {} is not divisible by GGUF block length {block_values}",
                    tensor.dimensions[0]
                ),
            ));
        }
        let source_byte_len = tensor
            .element_count()?
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("tensor descriptor byte length"))?;
        if source_byte_len != tensor.byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "descriptor declares {} encoded bytes but its shape and type require {source_byte_len}",
                    tensor.byte_len
                ),
            ));
        }
        let selected_axis_multiple = if gguf_dimension == 0 { block_values } else { 1 };
        let selected_axis_multiple_usize = usize::try_from(selected_axis_multiple)
            .map_err(|_| Error::Overflow("selection alignment"))?;

        let (selected_values, encoded_ranges) = match &selection {
            TensorSelection::Range { start, end, .. } => {
                if start >= end || *end > dimension {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection range {start}..{end} exceeds MLX axis {logical_axis} dimension {dimension}"
                        ),
                    ));
                }
                if start % selected_axis_multiple_usize != 0
                    || end % selected_axis_multiple_usize != 0
                {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection range {start}..{end} on MLX axis {logical_axis} must align to {selected_axis_multiple}-value GGUF blocks"
                        ),
                    ));
                }
                let encoded_start = start / selected_axis_multiple_usize;
                let encoded_end = end / selected_axis_multiple_usize;
                (
                    end - start,
                    vec![(encoded_start, encoded_end - encoded_start)],
                )
            }
            TensorSelection::Indices { indices, .. } => {
                if indices.is_empty() || indices.iter().any(|index| *index >= dimension) {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection indices {indices:?} exceed MLX axis {logical_axis} dimension {dimension}"
                        ),
                    ));
                }
                let encoded_indices = if selected_axis_multiple_usize == 1 {
                    indices.clone()
                } else {
                    if !indices.len().is_multiple_of(selected_axis_multiple_usize) {
                        return Err(Error::tensor(
                            &tensor.name,
                            format!(
                                "selection indices on MLX axis {logical_axis} must contain complete {selected_axis_multiple}-value GGUF blocks"
                            ),
                        ));
                    }
                    let mut blocks =
                        Vec::with_capacity(indices.len() / selected_axis_multiple_usize);
                    for chunk in indices.chunks_exact(selected_axis_multiple_usize) {
                        let start = chunk[0];
                        if start % selected_axis_multiple_usize != 0
                            || chunk
                                .iter()
                                .copied()
                                .ne(start..start + selected_axis_multiple_usize)
                        {
                            return Err(Error::tensor(
                                &tensor.name,
                                format!(
                                    "selection indices on MLX axis {logical_axis} must preserve every complete aligned {selected_axis_multiple}-value GGUF block"
                                ),
                            ));
                        }
                        blocks.push(start / selected_axis_multiple_usize);
                    }
                    blocks
                };
                (indices.len(), coalesce_indices(&encoded_indices))
            }
        };

        let mut encoded_dimensions = tensor.dimensions.clone();
        encoded_dimensions[0] /= block_values;
        let inner_units =
            encoded_dimensions[..gguf_dimension]
                .iter()
                .try_fold(1u64, |product, dimension| {
                    product
                        .checked_mul(*dimension)
                        .ok_or(Error::Overflow("selection inner stride"))
                })?;
        let inner_bytes = inner_units
            .checked_mul(block_bytes)
            .ok_or(Error::Overflow("selection inner byte stride"))?;
        let repetition_stride = encoded_dimensions[gguf_dimension]
            .checked_mul(inner_bytes)
            .ok_or(Error::Overflow("selection repetition stride"))?;
        let mut repetitions = encoded_dimensions[gguf_dimension + 1..].iter().try_fold(
            1u64,
            |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or(Error::Overflow("selection repetition count"))
            },
        )?;
        let mut relative_spans = encoded_ranges
            .into_iter()
            .map(|(start, count)| {
                let start =
                    u64::try_from(start).map_err(|_| Error::Overflow("selection span offset"))?;
                let count =
                    u64::try_from(count).map_err(|_| Error::Overflow("selection span length"))?;
                Ok(RelativeEncodedSpan {
                    offset: start
                        .checked_mul(inner_bytes)
                        .ok_or(Error::Overflow("selection span offset"))?,
                    byte_len: count
                        .checked_mul(inner_bytes)
                        .ok_or(Error::Overflow("selection span length"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let full_canonical_selection = relative_spans.len() == 1
            && relative_spans[0].offset == 0
            && relative_spans[0].byte_len == repetition_stride;
        if full_canonical_selection {
            repetitions = 1;
            relative_spans[0].byte_len = tensor.byte_len;
        }
        let encoded_bytes_per_repetition =
            relative_spans.iter().try_fold(0u64, |total, span| {
                total
                    .checked_add(span.byte_len)
                    .ok_or(Error::Overflow("selected tensor byte length"))
            })?;
        let encoded_byte_len = encoded_bytes_per_repetition
            .checked_mul(repetitions)
            .ok_or(Error::Overflow("selected tensor byte length"))?;

        let mut selected_descriptor = tensor.clone();
        selected_descriptor.dimensions[gguf_dimension] = u64::try_from(selected_values)
            .map_err(|_| Error::Overflow("selected tensor dimension"))?;
        selected_descriptor.byte_len = encoded_byte_len;
        let expected_byte_len = selected_descriptor
            .element_count()?
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("selected tensor descriptor byte length"))?;
        if expected_byte_len != encoded_byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "selection plan produced {encoded_byte_len} encoded bytes but its rewritten descriptor requires {expected_byte_len}"
                ),
            ));
        }
        let maximum_relative_end = relative_spans.iter().try_fold(0u64, |maximum, span| {
            let end = span
                .offset
                .checked_add(span.byte_len)
                .ok_or(Error::Overflow("selection span end"))?;
            Ok::<_, Error>(maximum.max(end))
        })?;
        if maximum_relative_end > repetition_stride && !full_canonical_selection {
            return Err(Error::tensor(
                &tensor.name,
                "selection span exceeds its encoded repetition stride",
            ));
        }
        let final_span_end = tensor
            .data_offset
            .checked_add(
                repetition_stride
                    .checked_mul(repetitions.saturating_sub(1))
                    .ok_or(Error::Overflow("selection final span offset"))?,
            )
            .and_then(|offset| offset.checked_add(maximum_relative_end))
            .ok_or(Error::Overflow("selection final span end"))?;
        let tensor_end = tensor
            .data_offset
            .checked_add(tensor.byte_len)
            .ok_or(Error::Overflow("tensor end offset"))?;
        if final_span_end > tensor_end {
            return Err(Error::tensor(
                &tensor.name,
                "selection plan exceeds the encoded tensor payload",
            ));
        }

        Ok(Self {
            selection,
            gguf_dimension,
            alignment: SelectionAlignment {
                block_values,
                block_bytes,
                selected_axis_multiple,
            },
            selected_descriptor,
            source_data_offset: tensor.data_offset,
            repetition_stride,
            repetitions,
            relative_spans,
            encoded_byte_len,
        })
    }

    /// Original logical selection represented by this plan.
    pub const fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Selected MLX/row-major axis.
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
        (0..self.repetitions).flat_map(move |repetition| {
            self.relative_spans.iter().map(move |span| EncodedSpan {
                offset: self.source_data_offset + repetition * self.repetition_stride + span.offset,
                byte_len: span.byte_len,
            })
        })
    }
}

fn coalesce_indices(indices: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for &index in indices {
        match ranges.last_mut() {
            Some((start, count)) if *start + *count == index => *count += 1,
            _ => ranges.push((index, 1)),
        }
    }
    ranges
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
}

impl Reader<BufReader<File>> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, Limits::default())
    }
    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::Io { offset: 0, source })?;
        Self::with_limits(BufReader::new(file), limits)
    }
}

impl<R: Read + Seek> Reader<R> {
    pub fn new(inner: R) -> Result<Self> {
        Self::with_limits(inner, Limits::default())
    }

    pub fn with_limits(mut inner: R, limits: Limits) -> Result<Self> {
        let file_size = inner
            .seek(SeekFrom::End(0))
            .map_err(|source| Error::Io { offset: 0, source })?;
        inner
            .seek(SeekFrom::Start(0))
            .map_err(|source| Error::Io { offset: 0, source })?;
        let mut parser = Parser {
            inner,
            endian: Endian::Little,
            version: 0,
            limits: &limits,
        };
        let mut magic = [0; 4];
        parser.exact(&mut magic)?;
        parser.endian = match &magic {
            b"GGUF" => Endian::Little,
            b"FUGG" => Endian::Big,
            _ => return Err(Error::InvalidHeader(format!("invalid magic {magic:?}"))),
        };
        let version = parser.u32()?;
        if !(1..=3).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }
        parser.version = version;
        let tensor_count = parser.count()?;
        check_limit("tensor count", tensor_count, limits.max_tensor_count)?;
        let metadata_count = parser.count()?;
        check_limit(
            "metadata entries",
            metadata_count,
            limits.max_metadata_entries,
        )?;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = parser.string()?;
            if key.len() > u16::MAX as usize || !key.is_ascii() {
                return Err(Error::InvalidMetadata {
                    key,
                    reason: "keys must be ASCII and at most 65535 bytes".into(),
                });
            }
            let ty = parser.u32()?;
            let value = parser.value(ty, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(Error::DuplicateMetadata(key));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(MetadataValue::Uint32(v)) => u64::from(*v),
            Some(MetadataValue::Uint64(v)) => *v,
            Some(_) => {
                return Err(Error::InvalidMetadata {
                    key: "general.alignment".into(),
                    reason: "must be uint32 or uint64".into(),
                })
            }
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::InvalidHeader(format!(
                "invalid alignment {alignment}"
            )));
        }

        let tensor_capacity =
            usize::try_from(tensor_count).map_err(|_| Error::Overflow("tensor count"))?;
        let mut raw = Vec::with_capacity(tensor_capacity);
        let mut names = HashSet::with_capacity(tensor_capacity);
        for _ in 0..tensor_count {
            let name = parser.string()?;
            if name.is_empty() {
                return Err(Error::tensor(name, "empty tensor name"));
            }
            if !names.insert(name.clone()) {
                return Err(Error::DuplicateTensor(name));
            }
            let rank = parser.u32()?;
            if rank > limits.max_rank {
                return Err(Error::Limit {
                    resource: "tensor rank",
                    actual: rank.into(),
                    limit: limits.max_rank.into(),
                });
            }
            let mut dimensions = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                dimensions.push(parser.dimension()?);
            }
            let ggml_type = GgmlType::from_code(parser.u32()?);
            let relative_offset = parser.u64()?;
            raw.push((name, dimensions, ggml_type, relative_offset));
        }
        let descriptor_end = parser.pos()?;
        let data_start = align_up(descriptor_end, alignment)?;
        // A metadata-only GGUF has no tensor-data section, so writers are not
        // required to materialize padding up to the aligned data start.
        if tensor_count != 0 && data_start > file_size {
            return Err(Error::InvalidHeader(
                "tensor data starts beyond end of file".into(),
            ));
        }

        let mut tensors = Vec::with_capacity(raw.len());
        for (name, dimensions, ggml_type, relative_offset) in raw {
            if relative_offset % alignment != 0 {
                return Err(Error::tensor(
                    &name,
                    format!("relative offset {relative_offset} is not aligned to {alignment}"),
                ));
            }
            let elements = dimensions.iter().try_fold(1u64, |a, &b| {
                a.checked_mul(b)
                    .ok_or(Error::Overflow("tensor element count"))
            })?;
            let (block, bytes) = ggml_type.block_and_bytes()?;
            if elements != 0
                && (dimensions.first().copied().unwrap_or(1) % block != 0 || elements % block != 0)
            {
                return Err(Error::tensor(
                    &name,
                    format!("shape {dimensions:?} is not divisible by block size {block}"),
                ));
            }
            let byte_len = (elements / block)
                .checked_mul(bytes)
                .ok_or(Error::Overflow("tensor byte length"))?;
            check_limit("tensor allocation", byte_len, limits.max_allocation_bytes)?;
            let data_offset = data_start
                .checked_add(relative_offset)
                .ok_or(Error::Overflow("tensor offset"))?;
            let end = data_offset
                .checked_add(byte_len)
                .ok_or(Error::Overflow("tensor end offset"))?;
            if end > file_size {
                return Err(Error::tensor(
                    &name,
                    format!("data range {data_offset}..{end} exceeds file size {file_size}"),
                ));
            }
            tensors.push(TensorDescriptor {
                name,
                dimensions,
                ggml_type,
                relative_offset,
                data_offset,
                byte_len,
            });
        }
        let mut ranges: Vec<_> = tensors
            .iter()
            .filter(|t| t.byte_len != 0)
            .map(|t| (t.data_offset, t.data_offset + t.byte_len, &t.name))
            .collect();
        ranges.sort_by_key(|r| r.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(Error::tensor(
                    pair[1].2,
                    format!("data overlaps tensor {:?}", pair[0].2),
                ));
            }
        }

        Ok(Self {
            inner: parser.inner,
            endian: parser.endian,
            version,
            alignment,
            metadata,
            tensors,
            limits,
        })
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
        check_limit(
            "tensor allocation",
            tensor.byte_len,
            self.limits.max_allocation_bytes,
        )?;
        self.inner
            .seek(SeekFrom::Start(tensor.data_offset))
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        let len =
            usize::try_from(tensor.byte_len).map_err(|_| Error::Overflow("tensor allocation"))?;
        let mut data = vec![0; len];
        self.inner
            .read_exact(&mut data)
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        Ok(data)
    }

    pub fn read_tensor(&mut self, tensor: &TensorDescriptor) -> Result<ConvertedTensor> {
        let raw = self.read_raw(tensor)?;
        crate::convert::convert(tensor, &raw, self.endian)
    }

    /// Execute a validated metadata-only physical selection plan.
    pub fn read_tensor_plan(&mut self, plan: &TensorSelectionPlan) -> Result<ConvertedTensor> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let selected_len = usize::try_from(plan.encoded_byte_len())
            .map_err(|_| Error::Overflow("selected tensor allocation"))?;
        let mut raw = Vec::with_capacity(selected_len);
        for span in plan.encoded_spans() {
            let span_len = usize::try_from(span.byte_len())
                .map_err(|_| Error::Overflow("selection span allocation"))?;
            self.inner
                .seek(SeekFrom::Start(span.offset()))
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
            let start = raw.len();
            let end = start
                .checked_add(span_len)
                .ok_or(Error::Overflow("selected tensor allocation"))?;
            raw.resize(end, 0);
            self.inner
                .read_exact(&mut raw[start..end])
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
        }
        crate::convert::convert(plan.selected_descriptor(), &raw, self.endian)
    }

    /// Execute a validated block-aligned contiguous-span plan.
    pub fn read_dense_tensor_span(
        &mut self,
        plan: &DenseTensorSpanPlan,
    ) -> Result<ConvertedTensor> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let span = plan.encoded_span();
        self.inner
            .seek(SeekFrom::Start(span.offset()))
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        let len = usize::try_from(span.byte_len())
            .map_err(|_| Error::Overflow("dense tensor span allocation"))?;
        let mut raw = vec![0; len];
        self.inner
            .read_exact(&mut raw)
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        crate::convert::convert(plan.selected_descriptor(), &raw, self.endian)
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

struct Parser<'a, R> {
    inner: R,
    endian: Endian,
    version: u32,
    limits: &'a Limits,
}

impl<R: Read + Seek> Parser<'_, R> {
    fn pos(&mut self) -> Result<u64> {
        self.inner
            .stream_position()
            .map_err(|source| Error::Io { offset: 0, source })
    }
    fn exact(&mut self, out: &mut [u8]) -> Result<()> {
        let offset = self.pos()?;
        self.inner
            .read_exact(out)
            .map_err(|source| Error::Io { offset, source })
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
    fn string(&mut self) -> Result<String> {
        let len = self.count()?;
        check_limit("string bytes", len, self.limits.max_string_bytes)?;
        let mut bytes =
            vec![0; usize::try_from(len).map_err(|_| Error::Overflow("string length"))?];
        self.exact(&mut bytes)?;
        String::from_utf8(bytes)
            .map_err(|e| Error::InvalidHeader(format!("invalid UTF-8 string: {e}")))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<MetadataValue> {
        Ok(match ty {
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
            8 => MetadataValue::String(self.string()?),
            9 => MetadataValue::Array(self.array(depth + 1)?),
            10 => MetadataValue::Uint64(self.u64()?),
            11 => MetadataValue::Int64(self.u64()? as i64),
            12 => MetadataValue::Float64(f64::from_bits(self.u64()?)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        })
    }
    fn array(&mut self, depth: u32) -> Result<MetadataArray> {
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
            ($variant:ident,$expr:expr) => {{
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push($expr?);
                }
                MetadataArray::$variant(v)
            }};
        }
        Ok(match ty {
            0 => vals!(Uint8, self.u8()),
            1 => vals!(Int8, self.u8().map(|v| v as i8)),
            2 => vals!(Uint16, self.u16()),
            3 => vals!(Int16, self.u16().map(|v| v as i16)),
            4 => vals!(Uint32, self.u32()),
            5 => vals!(Int32, self.u32().map(|v| v as i32)),
            6 => vals!(Float32, self.u32().map(f32::from_bits)),
            7 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(match self.u8()? {
                        0 => false,
                        1 => true,
                        x => return Err(Error::InvalidHeader(format!("invalid boolean {x}"))),
                    })
                }
                MetadataArray::Bool(v)
            }
            8 => vals!(String, self.string()),
            9 => vals!(Array, self.array(depth + 1)),
            10 => vals!(Uint64, self.u64()),
            11 => vals!(Int64, self.u64().map(|v| v as i64)),
            12 => vals!(Float64, self.u64().map(f64::from_bits)),
            other => return Err(Error::UnsupportedMetadataType(other)),
        })
    }
}
