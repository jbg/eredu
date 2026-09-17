//! Storage policies for the original physical-plan equations.
use super::*;
use crate::{
    supplied_storage::{FixedDescriptor, FixedSlice},
    TensorDescriptorView,
};
use std::collections::TryReserveError;
use std::ops::{Deref, DerefMut};

/// A failure preparing or using the physical result-metadata destination.
#[derive(Debug)]
pub enum MetadataDestinationError {
    /// Original validation failure, preserving its context and precedence.
    Gguf(Error),
    /// A supplied storage provider refused; its enclosing owning failure retains the actual typed cause and prefix.
    StorageRefused,
    /// A requested typed allocation is not representable.
    Layout,
    /// A cold allocation failed; its owner retains the prepared prefix.
    Reserve(TryReserveError),
    /// The actual source, names, or selection differ from the retained binding.
    Binding,
    /// A destination has already been used.
    Used,
    /// The actual operation exceeded its prepared storage.
    Capacity {
        /// Storage whose capacity was insufficient.
        field: &'static str,
        /// Elements required by this operation.
        required: usize,
        /// Available element capacity.
        capacity: usize,
    },
}
impl From<Error> for MetadataDestinationError {
    fn from(error: Error) -> Self {
        Self::Gguf(error)
    }
}
impl std::fmt::Display for MetadataDestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(error) => error.fmt(f),
            Self::Reserve(error) => error.fmt(f),
            Self::StorageRefused => f.write_str("GGUF supplied planner storage refused"),
            Self::Layout => f.write_str("GGUF metadata layout is not representable"),
            Self::Binding => f.write_str("GGUF metadata destination does not match its source"),
            Self::Used => f.write_str("GGUF metadata destination was already used"),
            Self::Capacity {
                field,
                required,
                capacity,
            } => write!(
                f,
                "GGUF {field} requires {required} elements; destination has {capacity}"
            ),
        }
    }
}
impl std::error::Error for MetadataDestinationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gguf(error) => Some(error),
            Self::Reserve(error) => Some(error),
            _ => None,
        }
    }
}
impl MetadataDestinationError {
    pub(super) fn ordinary(self) -> Error {
        match self {
            Self::Gguf(error) => error,
            _ => unreachable!("ordinary plan owns its storage"),
        }
    }
}
type PlanResult<T> = std::result::Result<T, MetadataDestinationError>;

pub(crate) fn capacity(field: &'static str, required: usize, available: usize) -> PlanResult<()> {
    if required > available {
        Err(MetadataDestinationError::Capacity {
            field,
            required,
            capacity: available,
        })
    } else {
        Ok(())
    }
}
pub(crate) fn reserve<T>(values: &mut Vec<T>, count: usize) -> PlanResult<()> {
    std::alloc::Layout::array::<T>(count).map_err(|_| MetadataDestinationError::Layout)?;
    values
        .try_reserve_exact(count.saturating_sub(values.len()))
        .map_err(MetadataDestinationError::Reserve)
}
pub(crate) fn fill_descriptor(
    target: &mut TensorDescriptor,
    source: &TensorDescriptor,
) -> PlanResult<()> {
    fill_descriptor_view(target, source.view())
}
fn fill_descriptor_view(
    target: &mut TensorDescriptor,
    source: TensorDescriptorView<'_>,
) -> PlanResult<()> {
    capacity("descriptor name", source.name.len(), target.name.capacity())?;
    capacity(
        "descriptor dimensions",
        source.dimensions.len(),
        target.dimensions.capacity(),
    )?;
    target.name.clear();
    target.name.push_str(source.name);
    target.dimensions.clear();
    target.dimensions.extend_from_slice(source.dimensions);
    target.ggml_type = source.ggml_type;
    target.relative_offset = source.relative_offset;
    target.data_offset = source.data_offset;
    target.byte_len = source.byte_len;
    Ok(())
}
pub(crate) fn empty_descriptor() -> TensorDescriptor {
    TensorDescriptor {
        name: String::new(),
        dimensions: Vec::new(),
        ggml_type: GgmlType::F32,
        relative_offset: 0,
        data_offset: 0,
        byte_len: 0,
    }
}
pub(crate) fn prepare_descriptor(
    target: &mut TensorDescriptor,
    source: &TensorDescriptor,
    rank_capacity: usize,
) -> PlanResult<()> {
    std::alloc::Layout::array::<u8>(source.name.len())
        .map_err(|_| MetadataDestinationError::Layout)?;
    target
        .name
        .try_reserve_exact(source.name.len())
        .map_err(MetadataDestinationError::Reserve)?;
    reserve(&mut target.dimensions, rank_capacity)?;
    fill_descriptor(target, source)
}

pub(crate) enum PlanSlot<'a, T> {
    Vec(&'a mut Vec<T>),
    Fixed(FixedSlice<'a, T>),
    Lazy(&'a mut dyn PlanBuffer<T>),
}
pub(super) enum Vector<'a, T> {
    Owned(Vec<T>),
    Borrowed(&'a mut Vec<T>),
    Fixed(FixedSlice<'a, T>),
    Lazy(&'a mut dyn PlanBuffer<T>),
}
impl<T> Deref for Vector<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
            Self::Fixed(v) => v.as_slice(),
            Self::Lazy(v) => v.as_slice(),
        }
    }
}
impl<T> DerefMut for Vector<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
            Self::Fixed(v) => v.as_mut_slice(),
            Self::Lazy(v) => v.as_mut_slice(),
        }
    }
}
impl<'a, T> Vector<'a, T> {
    fn new(
        buffer: Option<PlanSlot<'a, T>>,
        requested: Option<usize>,
        field: &'static str,
    ) -> PlanResult<Self> {
        match buffer {
            None => Ok(Self::Owned(match requested {
                None => Vec::new(),
                Some(n) => Vec::with_capacity(n),
            })),
            Some(PlanSlot::Vec(v)) => {
                capacity(field, requested.unwrap_or(0), v.capacity())?;
                v.clear();
                Ok(Self::Borrowed(v))
            }
            Some(PlanSlot::Lazy(v)) => {
                v.prepare(requested, field)?;
                Ok(Self::Lazy(v))
            }
            Some(PlanSlot::Fixed(mut v)) => {
                capacity(field, requested.unwrap_or(0), v.limit.min(v.data.len()))?;
                v.clear();
                Ok(Self::Fixed(v))
            }
        }
    }
    fn push_checked(&mut self, value: T, field: &'static str) -> PlanResult<()> {
        match self {
            Self::Owned(v) => v.push(value),
            Self::Borrowed(v) => {
                capacity(
                    field,
                    v.len()
                        .checked_add(1)
                        .ok_or(MetadataDestinationError::Layout)?,
                    v.capacity(),
                )?;
                v.push(value)
            }
            Self::Lazy(v) => v.push(value, field)?,
            Self::Fixed(v) => {
                let n = v
                    .used
                    .checked_add(1)
                    .ok_or(MetadataDestinationError::Layout)?;
                capacity(field, n, v.limit.min(v.data.len()))?;
                v.data[*v.used] = value;
                *v.used = n;
            }
        };
        Ok(())
    }
    pub(super) fn into_owned(self) -> Vec<T> {
        match self {
            Self::Owned(v) => v,
            _ => unreachable!("ordinary plan owns vectors"),
        }
    }
}
impl<'a, T: Copy> Vector<'a, T> {
    fn copy(
        source: &[T],
        buffer: Option<PlanSlot<'a, T>>,
        field: &'static str,
    ) -> PlanResult<Self> {
        match buffer {
            None => Ok(Self::Owned(source.to_vec())),
            Some(slot) => {
                let mut v = Self::new(Some(slot), Some(source.len()), field)?;
                for &x in source {
                    v.push_checked(x, field)?;
                }
                Ok(v)
            }
        }
    }
}
pub(crate) enum DescriptorSlot<'a> {
    Vec(&'a mut TensorDescriptor),
    Fixed(FixedDescriptor<'a>),
    Lazy(&'a mut dyn DescriptorBuffer),
}
pub(crate) enum Descriptor<'a> {
    Owned(TensorDescriptor),
    Borrowed(&'a mut TensorDescriptor),
    Fixed(FixedDescriptor<'a>),
    Lazy(&'a mut dyn DescriptorBuffer),
}
impl Deref for Descriptor<'_> {
    type Target = TensorDescriptor;
    fn deref(&self) -> &TensorDescriptor {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
            Self::Fixed(_) | Self::Lazy(_) => unreachable!("ordinary descriptor adapter"),
        }
    }
}
impl DerefMut for Descriptor<'_> {
    fn deref_mut(&mut self) -> &mut TensorDescriptor {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
            Self::Fixed(_) | Self::Lazy(_) => unreachable!("ordinary descriptor adapter"),
        }
    }
}
impl<'a> Descriptor<'a> {
    pub(crate) fn copy(
        source: &TensorDescriptor,
        destination: Option<&'a mut TensorDescriptor>,
    ) -> PlanResult<Self> {
        Self::copy_view(source.view(), destination.map(DescriptorSlot::Vec))
    }
    pub(crate) fn copy_view(
        source: TensorDescriptorView<'_>,
        destination: Option<DescriptorSlot<'a>>,
    ) -> PlanResult<Self> {
        match destination {
            None => Ok(Self::Owned(source.to_owned())),
            Some(DescriptorSlot::Vec(d)) => {
                fill_descriptor_view(d, source)?;
                Ok(Self::Borrowed(d))
            }
            Some(DescriptorSlot::Lazy(d)) => {
                d.copy_from(source)?;
                Ok(Self::Lazy(d))
            }
            Some(DescriptorSlot::Fixed(mut d)) => {
                d.copy_from(source)?;
                Ok(Self::Fixed(d))
            }
        }
    }
    pub(crate) fn view(&self) -> TensorDescriptorView<'_> {
        match self {
            Self::Owned(d) => d.view(),
            Self::Borrowed(d) => d.view(),
            Self::Fixed(d) => d.view(),
            Self::Lazy(d) => d.view(),
        }
    }
    fn set_dimension(&mut self, index: usize, value: u64) {
        match self {
            Self::Owned(d) => d.dimensions[index] = value,
            Self::Borrowed(d) => d.dimensions[index] = value,
            Self::Fixed(d) => d.dimensions.as_mut_slice()[index] = value,
            Self::Lazy(d) => d.dimensions_mut()[index] = value,
        }
    }
    fn set_byte_len(&mut self, value: u64) {
        match self {
            Self::Owned(d) => d.byte_len = value,
            Self::Borrowed(d) => d.byte_len = value,
            Self::Fixed(d) => *d.byte_len = value,
            Self::Lazy(d) => d.set_byte_len(value),
        }
    }
    fn set_offsets(&mut self, relative: u64, data: u64) {
        match self {
            Self::Owned(d) => {
                d.relative_offset = relative;
                d.data_offset = data
            }
            Self::Borrowed(d) => {
                d.relative_offset = relative;
                d.data_offset = data
            }
            Self::Fixed(d) => {
                *d.relative_offset = relative;
                *d.data_offset = data
            }
            Self::Lazy(d) => d.set_offsets(relative, data),
        }
    }
    fn replace_dimensions(&mut self, shape: &[u64]) -> PlanResult<()> {
        match self {
            Self::Owned(d) => d.dimensions = shape.iter().rev().copied().collect(),
            Self::Borrowed(d) => {
                capacity("span dimensions", shape.len(), d.dimensions.capacity())?;
                d.dimensions.clear();
                d.dimensions.extend(shape.iter().rev().copied());
            }
            Self::Lazy(d) => d.replace_dimensions(shape)?,
            Self::Fixed(d) => {
                capacity(
                    "span dimensions",
                    shape.len(),
                    d.dimensions.limit.min(d.dimensions.data.len()),
                )?;
                for (out, value) in d.dimensions.data.iter_mut().zip(shape.iter().rev()) {
                    *out = *value;
                }
                *d.dimensions.used = shape.len();
            }
        }
        Ok(())
    }
    pub(super) fn into_owned(self) -> TensorDescriptor {
        match self {
            Self::Owned(d) => d,
            _ => unreachable!("ordinary plan owns descriptor"),
        }
    }
    pub(crate) fn finish(self) -> TensorDescriptor {
        match self {
            Self::Owned(d) => d,
            Self::Borrowed(d) => std::mem::replace(d, empty_descriptor()),
            Self::Fixed(_) | Self::Lazy(_) => unreachable!("supplied metadata retains its owner"),
        }
    }
}

pub(crate) fn empty_selection(source: &TensorSelection) -> TensorSelection {
    match source {
        TensorSelection::Range { axis, start, end } => TensorSelection::Range {
            axis: *axis,
            start: *start,
            end: *end,
        },
        TensorSelection::Indices { axis, .. } => TensorSelection::Indices {
            axis: *axis,
            indices: Vec::new(),
        },
    }
}
pub(crate) fn prepare_selection(
    target: &mut TensorSelection,
    source: &TensorSelection,
) -> PlanResult<()> {
    match (target, source) {
        (TensorSelection::Range { .. }, TensorSelection::Range { .. }) => Ok(()),
        (
            TensorSelection::Indices { indices, .. },
            TensorSelection::Indices {
                indices: source, ..
            },
        ) => {
            reserve(indices, source.len())?;
            indices.extend_from_slice(source);
            Ok(())
        }
        _ => unreachable!("selection created from actual variant"),
    }
}
pub(crate) fn empty_span(source: &DenseTensorSpan) -> DenseTensorSpan {
    DenseTensorSpan {
        offset_elements: source.offset_elements,
        shape: Vec::new(),
    }
}
pub(crate) fn prepare_span(
    target: &mut DenseTensorSpan,
    source: &DenseTensorSpan,
) -> PlanResult<()> {
    reserve(&mut target.shape, source.shape.len())?;
    target.shape.extend_from_slice(&source.shape);
    Ok(())
}
pub(crate) fn span_capacity_bytes(source: &DenseTensorSpan) -> Option<usize> {
    std::alloc::Layout::array::<u64>(source.shape.capacity())
        .ok()
        .map(|l| l.size())
}

#[derive(Debug)]
pub(crate) struct AxisScratch {
    pub(crate) indices: Vec<usize>,
    pub(crate) ranges: Vec<(usize, usize)>,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) spans: Vec<RelativeEncodedSpan>,
    pub(crate) selected: TensorDescriptor,
}
impl AxisScratch {
    pub(crate) fn empty() -> Self {
        Self {
            indices: Vec::new(),
            ranges: Vec::new(),
            dimensions: Vec::new(),
            spans: Vec::new(),
            selected: empty_descriptor(),
        }
    }
    pub(crate) fn prepare(
        &mut self,
        tensor: &TensorDescriptor,
        selection: &TensorSelection,
    ) -> PlanResult<()> {
        let (indices, ranges) = match selection {
            TensorSelection::Range { .. } => (0, 1),
            TensorSelection::Indices { indices, .. } => (indices.len(), indices.len()),
        };
        reserve(&mut self.indices, indices)?;
        reserve(&mut self.ranges, ranges)?;
        reserve(&mut self.dimensions, tensor.dimensions.len())?;
        reserve(&mut self.spans, ranges)?;
        prepare_descriptor(&mut self.selected, tensor, tensor.dimensions.len())
    }
}
pub(crate) struct AxisStorage<'a> {
    indices: Option<PlanSlot<'a, usize>>,
    ranges: Option<PlanSlot<'a, (usize, usize)>>,
    dimensions: Option<PlanSlot<'a, u64>>,
    spans: Option<PlanSlot<'a, RelativeEncodedSpan>>,
    selected: Option<DescriptorSlot<'a>>,
}
impl<'a> AxisStorage<'a> {
    pub(crate) fn new(scratch: Option<&'a mut AxisScratch>) -> Self {
        match scratch {
            None => Self {
                indices: None,
                ranges: None,
                dimensions: None,
                spans: None,
                selected: None,
            },
            Some(s) => Self {
                indices: Some(PlanSlot::Vec(&mut s.indices)),
                ranges: Some(PlanSlot::Vec(&mut s.ranges)),
                dimensions: Some(PlanSlot::Vec(&mut s.dimensions)),
                spans: Some(PlanSlot::Vec(&mut s.spans)),
                selected: Some(DescriptorSlot::Vec(&mut s.selected)),
            },
        }
    }
    pub(crate) fn lazy(
        indices: &'a mut dyn PlanBuffer<usize>,
        ranges: &'a mut dyn PlanBuffer<(usize, usize)>,
        dimensions: &'a mut dyn PlanBuffer<u64>,
        spans: &'a mut dyn PlanBuffer<RelativeEncodedSpan>,
        selected: &'a mut dyn DescriptorBuffer,
    ) -> Self {
        Self {
            indices: Some(PlanSlot::Lazy(indices)),
            ranges: Some(PlanSlot::Lazy(ranges)),
            dimensions: Some(PlanSlot::Lazy(dimensions)),
            spans: Some(PlanSlot::Lazy(spans)),
            selected: Some(DescriptorSlot::Lazy(selected)),
        }
    }
    pub(crate) fn fixed(
        indices: FixedSlice<'a, usize>,
        ranges: FixedSlice<'a, (usize, usize)>,
        dimensions: FixedSlice<'a, u64>,
        spans: FixedSlice<'a, RelativeEncodedSpan>,
        selected: FixedDescriptor<'a>,
    ) -> Self {
        Self {
            indices: Some(PlanSlot::Fixed(indices)),
            ranges: Some(PlanSlot::Fixed(ranges)),
            dimensions: Some(PlanSlot::Fixed(dimensions)),
            spans: Some(PlanSlot::Fixed(spans)),
            selected: Some(DescriptorSlot::Fixed(selected)),
        }
    }
    fn ranges(&mut self, indices: &[usize]) -> PlanResult<Vector<'a, (usize, usize)>> {
        let mut ranges = Vector::new(self.ranges.take(), None, "encoded ranges")?;
        for &index in indices {
            match ranges.last_mut() {
                Some((start, count)) if *start + *count == index => *count += 1,
                _ => ranges.push_checked((index, 1), "encoded ranges")?,
            }
        }
        Ok(ranges)
    }
    fn range(&mut self, start: usize, count: usize) -> PlanResult<Vector<'a, (usize, usize)>> {
        match self.ranges.take() {
            None => Ok(Vector::Owned(vec![(start, count)])),
            Some(slot) => {
                let mut v = Vector::new(Some(slot), Some(1), "encoded ranges")?;
                v.push_checked((start, count), "encoded ranges")?;
                Ok(v)
            }
        }
    }
    fn collect_spans(
        &mut self,
        ranges: Vector<'a, (usize, usize)>,
        f: impl FnMut((usize, usize)) -> Result<RelativeEncodedSpan>,
    ) -> PlanResult<Vector<'a, RelativeEncodedSpan>> {
        match (ranges, self.spans.take()) {
            (Vector::Owned(ranges), None) => Ok(Vector::Owned(
                ranges.into_iter().map(f).collect::<Result<Vec<_>>>()?,
            )),
            (ranges, Some(slot)) => {
                let mut spans = Vector::new(Some(slot), None, "relative spans")?;
                for span in ranges.iter().copied().map(f) {
                    spans.push_checked(span?, "relative spans")?;
                }
                Ok(spans)
            }
            _ => unreachable!("closed ordinary/fixed plan storage"),
        }
    }
}

pub(crate) struct AxisPlan<'a> {
    pub(super) gguf_dimension: usize,
    pub(super) alignment: SelectionAlignment,
    pub(crate) selected_descriptor: Descriptor<'a>,
    pub(super) source_data_offset: u64,
    pub(super) repetition_stride: u64,
    pub(super) repetitions: u64,
    pub(super) relative_spans: Vector<'a, RelativeEncodedSpan>,
    pub(super) encoded_byte_len: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct AxisView<'a> {
    descriptor: TensorDescriptorView<'a>,
    source_data_offset: u64,
    repetition_stride: u64,
    repetitions: u64,
    relative_spans: &'a [RelativeEncodedSpan],
    encoded_byte_len: u64,
}
impl AxisPlan<'_> {
    pub(crate) fn view(&self) -> AxisView<'_> {
        AxisView {
            descriptor: self.selected_descriptor.view(),
            source_data_offset: self.source_data_offset,
            repetition_stride: self.repetition_stride,
            repetitions: self.repetitions,
            relative_spans: &self.relative_spans,
            encoded_byte_len: self.encoded_byte_len,
        }
    }
}
impl TensorSelectionPlan {
    pub(crate) fn view(&self) -> AxisView<'_> {
        AxisView {
            descriptor: self.selected_descriptor.view(),
            source_data_offset: self.source_data_offset,
            repetition_stride: self.repetition_stride,
            repetitions: self.repetitions,
            relative_spans: &self.relative_spans,
            encoded_byte_len: self.encoded_byte_len,
        }
    }
}
impl<'a> AxisView<'a> {
    pub(super) fn selected_descriptor(&self) -> TensorDescriptorView<'_> {
        self.descriptor
    }
    pub(super) fn encoded_byte_len(&self) -> u64 {
        self.encoded_byte_len
    }
    pub(crate) fn encoded_spans(self) -> impl Iterator<Item = EncodedSpan> + 'a {
        (0..self.repetitions).flat_map(move |repetition| {
            self.relative_spans.iter().map(move |span| EncodedSpan {
                offset: self.source_data_offset + repetition * self.repetition_stride + span.offset,
                byte_len: span.byte_len,
            })
        })
    }
}
pub(crate) struct SpanPlan<'a> {
    pub(crate) selected_descriptor: Descriptor<'a>,
    pub(super) encoded_span: EncodedSpan,
}
#[derive(Clone, Copy)]
pub(crate) struct SpanView<'a> {
    descriptor: TensorDescriptorView<'a>,
    encoded_span: EncodedSpan,
}
impl SpanPlan<'_> {
    pub(crate) fn view(&self) -> SpanView<'_> {
        SpanView {
            descriptor: self.selected_descriptor.view(),
            encoded_span: self.encoded_span,
        }
    }
}
impl DenseTensorSpanPlan {
    pub(crate) fn view(&self) -> SpanView<'_> {
        SpanView {
            descriptor: self.selected_descriptor.view(),
            encoded_span: self.encoded_span,
        }
    }
}
impl SpanView<'_> {
    pub(super) fn selected_descriptor(&self) -> TensorDescriptorView<'_> {
        self.descriptor
    }
    pub(super) fn encoded_span(&self) -> EncodedSpan {
        self.encoded_span
    }
    pub(super) fn encoded_byte_len(&self) -> u64 {
        self.encoded_span.byte_len
    }
}

pub(crate) fn axis_plan<'a>(
    tensor: &TensorDescriptor,
    selection: &TensorSelection,
    storage: AxisStorage<'a>,
) -> PlanResult<AxisPlan<'a>> {
    axis_plan_view(tensor.view(), selection, storage)
}
pub(crate) fn axis_plan_view<'a>(
    tensor: TensorDescriptorView<'_>,
    selection: &TensorSelection,
    mut storage: AxisStorage<'a>,
) -> PlanResult<AxisPlan<'a>> {
    let rank = tensor.dimensions.len();
    let logical_axis = selection.axis();
    if logical_axis >= rank {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!("selection axis {logical_axis} is outside rank {rank}"),
        )));
    }
    if tensor.byte_len == 0 || tensor.dimensions.contains(&0) {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            "cannot select from an empty tensor",
        )));
    }

    let gguf_dimension = rank - 1 - logical_axis;
    let dimension_u64 = tensor.dimensions[gguf_dimension];
    let dimension =
        usize::try_from(dimension_u64).map_err(|_| Error::Overflow("selected tensor dimension"))?;
    let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
    if !tensor.dimensions[0].is_multiple_of(block_values) {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "fastest dimension {} is not divisible by GGUF block length {block_values}",
                tensor.dimensions[0]
            ),
        )));
    }
    let source_byte_len = tensor
        .element_count()?
        .checked_div(block_values)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(Error::Overflow("tensor descriptor byte length"))?;
    if source_byte_len != tensor.byte_len {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "descriptor declares {} encoded bytes but its shape and type require {source_byte_len}",
                tensor.byte_len
            ),
        )));
    }
    let selected_axis_multiple = if gguf_dimension == 0 { block_values } else { 1 };
    let selected_axis_multiple_usize = usize::try_from(selected_axis_multiple)
        .map_err(|_| Error::Overflow("selection alignment"))?;

    let (selected_values, encoded_ranges) = match selection {
        TensorSelection::Range { start, end, .. } => {
            if start >= end || *end > dimension {
                return Err(MetadataDestinationError::Gguf(Error::tensor(
                    tensor.name,
                    format!(
                        "selection range {start}..{end} exceeds row-major axis {logical_axis} dimension {dimension}"
                    ),
                )));
            }
            if start % selected_axis_multiple_usize != 0 || end % selected_axis_multiple_usize != 0
            {
                return Err(MetadataDestinationError::Gguf(Error::tensor(
                    tensor.name,
                    format!(
                        "selection range {start}..{end} on row-major axis {logical_axis} must align to {selected_axis_multiple}-value GGUF blocks"
                    ),
                )));
            }
            let encoded_start = start / selected_axis_multiple_usize;
            let encoded_end = end / selected_axis_multiple_usize;
            (
                end - start,
                storage.range(encoded_start, encoded_end - encoded_start)?,
            )
        }
        TensorSelection::Indices { indices, .. } => {
            if indices.is_empty() || indices.iter().any(|index| *index >= dimension) {
                return Err(MetadataDestinationError::Gguf(Error::tensor(
                    tensor.name,
                    format!(
                        "selection indices {indices:?} exceed row-major axis {logical_axis} dimension {dimension}"
                    ),
                )));
            }
            let encoded_indices = if selected_axis_multiple_usize == 1 {
                Vector::copy(indices, storage.indices.take(), "encoded indices")?
            } else {
                if !indices.len().is_multiple_of(selected_axis_multiple_usize) {
                    return Err(MetadataDestinationError::Gguf(Error::tensor(
                        tensor.name,
                        format!(
                            "selection indices on row-major axis {logical_axis} must contain complete {selected_axis_multiple}-value GGUF blocks"
                        ),
                    )));
                }
                let mut blocks = Vector::new(
                    storage.indices.take(),
                    Some(indices.len() / selected_axis_multiple_usize),
                    "encoded indices",
                )?;
                for chunk in indices.chunks_exact(selected_axis_multiple_usize) {
                    let start = chunk[0];
                    if start % selected_axis_multiple_usize != 0
                        || chunk
                            .iter()
                            .copied()
                            .ne(start..start + selected_axis_multiple_usize)
                    {
                        return Err(MetadataDestinationError::Gguf(Error::tensor(
                            tensor.name,
                            format!(
                                "selection indices on row-major axis {logical_axis} must preserve every complete aligned {selected_axis_multiple}-value GGUF block"
                            ),
                        )));
                    }
                    blocks.push_checked(start / selected_axis_multiple_usize, "encoded indices")?;
                }
                blocks
            };
            (indices.len(), storage.ranges(&encoded_indices)?)
        }
    };

    let mut encoded_dimensions = Vector::copy(
        &tensor.dimensions,
        storage.dimensions.take(),
        "encoded dimensions",
    )?;
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
    let mut repetitions =
        encoded_dimensions[gguf_dimension + 1..]
            .iter()
            .try_fold(1u64, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or(Error::Overflow("selection repetition count"))
            })?;
    let mut relative_spans = storage.collect_spans(encoded_ranges, |(start, count)| {
        let start = u64::try_from(start).map_err(|_| Error::Overflow("selection span offset"))?;
        let count = u64::try_from(count).map_err(|_| Error::Overflow("selection span length"))?;
        Ok(RelativeEncodedSpan {
            offset: start
                .checked_mul(inner_bytes)
                .ok_or(Error::Overflow("selection span offset"))?,
            byte_len: count
                .checked_mul(inner_bytes)
                .ok_or(Error::Overflow("selection span length"))?,
        })
    })?;

    let full_canonical_selection = relative_spans.len() == 1
        && relative_spans[0].offset == 0
        && relative_spans[0].byte_len == repetition_stride;
    if full_canonical_selection {
        repetitions = 1;
        relative_spans[0].byte_len = tensor.byte_len;
    }
    let encoded_bytes_per_repetition = relative_spans.iter().try_fold(0u64, |total, span| {
        total
            .checked_add(span.byte_len)
            .ok_or(Error::Overflow("selected tensor byte length"))
    })?;
    let encoded_byte_len = encoded_bytes_per_repetition
        .checked_mul(repetitions)
        .ok_or(Error::Overflow("selected tensor byte length"))?;

    let mut selected_descriptor = Descriptor::copy_view(tensor, storage.selected.take())?;
    selected_descriptor.set_dimension(
        gguf_dimension,
        u64::try_from(selected_values).map_err(|_| Error::Overflow("selected tensor dimension"))?,
    );
    selected_descriptor.set_byte_len(encoded_byte_len);
    let expected_byte_len = selected_descriptor
        .view()
        .element_count()?
        .checked_div(block_values)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(Error::Overflow("selected tensor descriptor byte length"))?;
    if expected_byte_len != encoded_byte_len {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "selection plan produced {encoded_byte_len} encoded bytes but its rewritten descriptor requires {expected_byte_len}"
            ),
        )));
    }
    let maximum_relative_end = relative_spans.iter().try_fold(0u64, |maximum, span| {
        let end = span
            .offset
            .checked_add(span.byte_len)
            .ok_or(Error::Overflow("selection span end"))?;
        Ok::<_, Error>(maximum.max(end))
    })?;
    if maximum_relative_end > repetition_stride && !full_canonical_selection {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            "selection span exceeds its encoded repetition stride",
        )));
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
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            "selection plan exceeds the encoded tensor payload",
        )));
    }

    Ok(AxisPlan {
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

pub(crate) fn span_plan<'a>(
    tensor: &TensorDescriptor,
    selection: &DenseTensorSpan,
    destination: Option<&'a mut TensorDescriptor>,
) -> PlanResult<SpanPlan<'a>> {
    span_plan_view(
        tensor.view(),
        selection,
        destination.map(DescriptorSlot::Vec),
    )
}
pub(crate) fn span_plan_view<'a>(
    tensor: TensorDescriptorView<'_>,
    selection: &DenseTensorSpan,
    destination: Option<DescriptorSlot<'a>>,
) -> PlanResult<SpanPlan<'a>> {
    let tensor_elements = tensor.element_count()?;
    let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
    if !tensor_elements.is_multiple_of(block_values) {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "tensor element count {tensor_elements} is not divisible by {:?} block length {block_values}",
                tensor.ggml_type
            ),
        )));
    }
    let expected_source_bytes = tensor_elements
        .checked_div(block_values)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(Error::Overflow("dense tensor byte length"))?;
    if expected_source_bytes != tensor.byte_len {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "descriptor declares {} encoded bytes but its dense shape and type require {expected_source_bytes}",
                tensor.byte_len
            ),
        )));
    }
    let selected_elements = selection.element_count()?;
    let selected_end = selection
        .offset_elements
        .checked_add(selected_elements)
        .ok_or(Error::Overflow("dense tensor span end"))?;
    if selected_end > tensor_elements {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "contiguous scalar span {}..{selected_end} exceeds tensor element count {tensor_elements}",
                selection.offset_elements
            ),
        )));
    }
    if !selection.offset_elements.is_multiple_of(block_values)
        || !selected_elements.is_multiple_of(block_values)
    {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "contiguous span {}..{selected_end} must align to {:?} block length {block_values}",
                selection.offset_elements, tensor.ggml_type
            ),
        )));
    }
    let fastest =
        selection.shape.last().copied().ok_or_else(|| {
            Error::tensor(tensor.name, "contiguous span has no fastest dimension")
        })?;
    if !fastest.is_multiple_of(block_values) {
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            format!(
                "contiguous span fastest dimension {fastest} must align to {:?} block length {block_values}",
                tensor.ggml_type
            ),
        )));
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
        return Err(MetadataDestinationError::Gguf(Error::tensor(
            tensor.name,
            "contiguous scalar span exceeds the encoded tensor payload",
        )));
    }

    let mut selected_descriptor = Descriptor::copy_view(tensor, destination)?;
    selected_descriptor.replace_dimensions(&selection.shape)?;
    let relative = selected_descriptor
        .view()
        .relative_offset
        .checked_add(byte_offset)
        .ok_or(Error::Overflow("dense tensor span relative offset"))?;
    selected_descriptor.set_offsets(relative, offset);
    selected_descriptor.set_byte_len(byte_len);
    Ok(SpanPlan {
        selected_descriptor,
        encoded_span: EncodedSpan { offset, byte_len },
    })
}

#[cfg(test)]
mod tests;

pub(crate) trait PlanBuffer<T> {
    fn prepare(&mut self, requested: Option<usize>, field: &'static str) -> PlanResult<()>;
    fn push(&mut self, value: T, field: &'static str) -> PlanResult<()>;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}
pub(crate) trait DescriptorBuffer {
    fn copy_from(&mut self, source: TensorDescriptorView<'_>) -> PlanResult<()>;
    fn view(&self) -> TensorDescriptorView<'_>;
    fn dimensions_mut(&mut self) -> &mut [u64];
    fn set_byte_len(&mut self, value: u64);
    fn set_offsets(&mut self, relative: u64, data: u64);
    fn replace_dimensions(&mut self, shape: &[u64]) -> PlanResult<()>;
}
mod supplied;
pub(crate) use supplied::physical_storage_bound;
pub use supplied::{StoredPhysicalDescriptor, StoredPhysicalFailure};
