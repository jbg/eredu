//! SafeTensors physical read geometry with ordinary and exact borrowed storage.
use super::{
    Dtype, ReadPolicy, SelectionValidationError, SelectionValidationPlan, StoreError,
    TensorSelection,
};
use std::{alloc::Layout, ops::Range};

#[derive(Clone, Copy, Debug)]
enum Overflow {
    ContiguousStart,
    ContiguousEnd,
    RowStart,
    RowEnd,
    SelectedBits,
    BlockBytes,
    SelectionStart,
    SelectionEnd,
}
#[derive(Clone, Copy, Debug)]
enum Cause<'a> {
    Selection(SelectionValidationError<'a>),
    Overflow(Overflow),
    Invalid(&'static str),
    Bounded(&'static str),
    Geometry(&'static str),
    Destination { expected: usize, actual: usize },
}
/// Fixed read-planning error borrowing the actual source key.
#[derive(Clone, Copy, Debug)]
pub struct SafetensorsReadError<'a> {
    key: &'a str,
    cause: Cause<'a>,
}
impl std::fmt::Display for SafetensorsReadError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cause {
            Cause::Selection(error) => std::fmt::Display::fmt(&error, f),
            Cause::Overflow(kind) => {
                let context = match kind {
                    Overflow::ContiguousStart => "contiguous byte start",
                    Overflow::ContiguousEnd => "contiguous byte end",
                    Overflow::RowStart => "row selection byte start",
                    Overflow::RowEnd => "row selection byte end",
                    Overflow::SelectedBits => "selected bit length",
                    Overflow::BlockBytes => "selection block bytes",
                    Overflow::SelectionStart => "selection byte start",
                    Overflow::SelectionEnd => "selection byte end",
                };
                write!(f, "checkpoint size overflow: {context} for {:?}", self.key)
            }
            Cause::Invalid(message) => {
                write!(f, "invalid selection for tensor {:?}: {message}", self.key)
            }
            Cause::Bounded(message) => write!(
                f,
                "bounded selection is unavailable for tensor {:?}: {message}",
                self.key
            ),
            Cause::Geometry(message) => write!(
                f,
                "fixed SafeTensors read geometry for {:?}: {message}",
                self.key
            ),
            Cause::Destination { expected, actual } => write!(
                f,
                "read range destination has {actual} entries; expected {expected}"
            ),
        }
    }
}
impl std::error::Error for SafetensorsReadError<'_> {}
impl From<SafetensorsReadError<'_>> for StoreError {
    fn from(error: SafetensorsReadError<'_>) -> Self {
        match error.cause {
            Cause::Selection(error) => error.into(),
            Cause::Overflow(kind) => {
                let key = error.key;
                // Literal format strings preserve the ordinary diagnostic's
                // existing allocation policy, not just its resulting text.
                let context = match kind {
                    Overflow::ContiguousStart => format!("contiguous byte start for {key:?}"),
                    Overflow::ContiguousEnd => format!("contiguous byte end for {key:?}"),
                    Overflow::RowStart => format!("row selection byte start for {key:?}"),
                    Overflow::RowEnd => format!("row selection byte end for {key:?}"),
                    Overflow::SelectedBits => format!("selected bit length for {key:?}"),
                    Overflow::BlockBytes => format!("selection block bytes for {key:?}"),
                    Overflow::SelectionStart => format!("selection byte start for {key:?}"),
                    Overflow::SelectionEnd => format!("selection byte end for {key:?}"),
                };
                StoreError::Overflow { context }
            }
            Cause::Invalid(message) => super::invalid_selection(error.key, message),
            Cause::Bounded(message) => StoreError::BoundedSelectionUnavailable {
                key: error.key.into(),
                message: message.into(),
            },
            Cause::Geometry(_) | Cause::Destination { .. } => {
                StoreError::Internal(error.to_string())
            }
        }
    }
}

/// Exact borrowed header inputs behind the selected SafeTensors provider.
/// Only the admitted source can construct this view; no header is initialized.
#[derive(Clone, Copy, Debug)]
pub struct SafetensorsReadSource<'a> {
    key: &'a str,
    metadata_shape: &'a [usize],
    shape: &'a [usize],
    dtype: Dtype,
    offsets: (usize, usize),
    file: Option<super::read_bytes::ReadFileSource<'a>>,
}
impl<'a> SafetensorsReadSource<'a> {
    pub(super) fn new(
        key: &'a str,
        metadata_shape: &'a [usize],
        shape: &'a [usize],
        dtype: Dtype,
        offsets: (usize, usize),
    ) -> Self {
        Self {
            key,
            metadata_shape,
            shape,
            dtype,
            offsets,
            file: None,
        }
    }
    pub(super) fn with_file(mut self, file: super::read_bytes::ReadFileSource<'a>) -> Self {
        self.file = Some(file);
        self
    }
    /// Validates the actual selection into caller shape scratch and counts the
    /// exact coalesced destination through the shared physical-read worker.
    pub fn plan<'s, 'd>(
        &'s self,
        selection: &'s TensorSelection,
        policy: ReadPolicy,
        initial_shape: &'d mut [usize],
        replacement_shape: &'d mut [usize],
    ) -> Result<SafetensorsReadDestinationPlan<'s, 'd>, SafetensorsReadError<'s>> {
        let error = |cause| SafetensorsReadError {
            key: self.key,
            cause,
        };
        let shape = SelectionValidationPlan::new(self.key, self.metadata_shape, selection)
            .ok_or_else(|| error(Cause::Geometry("shape destination layout overflow")))?;
        let output = shape
            .validate_into(initial_shape, replacement_shape)
            .map_err(|cause| error(Cause::Selection(cause)))?;
        let payload_len = self
            .offsets
            .1
            .checked_sub(self.offsets.0)
            .ok_or_else(|| error(Cause::Geometry("source payload offsets descend")))?;
        let ranges = SelectionReadDestinationPlan::new(
            self.key,
            self.dtype.bitsize(),
            self.shape,
            payload_len,
            selection,
            output,
            policy,
        )?;
        Ok(SafetensorsReadDestinationPlan {
            source: *self,
            ranges,
        })
    }
}

/// Exact range count bound to the borrowed source, selection and validated shape.
/// It retains no tensor payload, cache entry or native owner.
pub struct SafetensorsReadDestinationPlan<'s, 'd> {
    source: SafetensorsReadSource<'s>,
    ranges: SelectionReadDestinationPlan<'s, 'd>,
}
impl<'s> SafetensorsReadDestinationPlan<'s, '_> {
    /// Exact caller destination layout, checked against allocation representability.
    pub fn range_layout(&self) -> Layout {
        self.ranges.range_layout()
    }
    /// Number of coalesced ranges in the actual order.
    pub fn range_count(&self) -> usize {
        self.ranges.range_count()
    }
    /// Fill exact range storage and bind a byte destination to this same genuine
    /// admitted source. No file is opened and no payload storage is allocated.
    pub fn byte_plan<'r>(
        &self,
        ranges: &'r mut [Range<usize>],
    ) -> Result<super::SafetensorsBytePlan<'s, 'r>, SafetensorsReadError<'s>> {
        let error = |message| SafetensorsReadError {
            key: self.source.key,
            cause: Cause::Geometry(message),
        };
        let file = self
            .source
            .file
            .ok_or_else(|| error("source has no admitted file loan"))?;
        let filled = self.fill_into(ranges)?;
        let length = filled.ranges.iter().try_fold(0usize, |total, range| {
            total
                .checked_add(range.len())
                .ok_or_else(|| error("byte destination length overflow"))
        })?;
        let layout =
            Layout::array::<u8>(length).map_err(|_| error("byte destination layout overflow"))?;
        let start = file
            .header_payload_start
            .checked_add(self.source.offsets.0)
            .ok_or_else(|| error("tensor payload offset overflow"))?;
        Ok(super::SafetensorsBytePlan::new(
            file,
            self.source.key,
            start,
            self.ranges.payload_len,
            filled.ranges,
            layout,
            filled.physically_bounded,
        ))
    }
    /// Fills exactly the counted range destination without allocating or reading.
    /// Both short and long destinations are rejected before writes.
    pub fn fill_into<'r>(
        &self,
        ranges: &'r mut [Range<usize>],
    ) -> Result<SafetensorsReadRanges<'r>, SafetensorsReadError<'s>> {
        self.ranges.fill_into(ranges)
    }
}

/// Counted geometry for one already validated selection. Both the source and
/// inferred output shapes remain borrowed through the actual destination fill.
/// This crate-private worker grants neither source authority nor admission.
pub(crate) struct SelectionReadDestinationPlan<'s, 'd> {
    key: &'s str,
    bits: usize,
    shape: &'s [usize],
    payload_len: usize,
    selection: &'s TensorSelection,
    output: &'d [usize],
    policy: ReadPolicy,
    count: usize,
    layout: Layout,
}
impl<'s, 'd> SelectionReadDestinationPlan<'s, 'd> {
    fn new(
        key: &'s str,
        bits: usize,
        shape: &'s [usize],
        payload_len: usize,
        selection: &'s TensorSelection,
        output: &'d [usize],
        policy: ReadPolicy,
    ) -> Result<Self, SafetensorsReadError<'s>> {
        let counted = run(
            key,
            bits,
            shape,
            payload_len,
            selection,
            output,
            policy,
            Fixed::count(),
        )?;
        let count = counted.ranges.len();
        let layout = Layout::array::<Range<usize>>(count).map_err(|_| SafetensorsReadError {
            key,
            cause: Cause::Geometry("range destination layout overflow"),
        })?;
        Ok(Self {
            key,
            bits,
            shape,
            payload_len,
            selection,
            output,
            policy,
            count,
            layout,
        })
    }
    pub(crate) fn range_layout(&self) -> Layout {
        self.layout
    }
    pub(crate) fn range_count(&self) -> usize {
        self.count
    }
    pub(crate) fn fill_into<'r>(
        &self,
        ranges: &'r mut [Range<usize>],
    ) -> Result<SafetensorsReadRanges<'r>, SafetensorsReadError<'s>> {
        if ranges.len() != self.count {
            return Err(SafetensorsReadError {
                key: self.key,
                cause: Cause::Destination {
                    expected: self.count,
                    actual: ranges.len(),
                },
            });
        }
        let result = run(
            self.key,
            self.bits,
            self.shape,
            self.payload_len,
            self.selection,
            self.output,
            self.policy,
            Fixed::fill(ranges),
        )?;
        let Ranges::Fill { values, used } = result.ranges else {
            unreachable!("fill policy")
        };
        Ok(SafetensorsReadRanges {
            ranges: &values[..used],
            physically_bounded: result.physically_bounded,
        })
    }
}
/// Borrowed coalesced read ranges. This is geometry, not an acquired lease.
pub struct SafetensorsReadRanges<'a> {
    ranges: &'a [Range<usize>],
    physically_bounded: bool,
}
impl SafetensorsReadRanges<'_> {
    /// Ordered relative byte ranges, including non-adjacent repeated reads.
    pub fn ranges(&self) -> &[Range<usize>] {
        self.ranges
    }
    /// Whether this plan physically restricts the requested selection.
    pub fn physically_bounded(&self) -> bool {
        self.physically_bounded
    }
}

trait Indices {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> usize;
}
impl Indices for Vec<usize> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn get(&self, index: usize) -> usize {
        self[index]
    }
}
enum BorrowedIndices<'a> {
    Range(Range<usize>),
    Slice(&'a [usize]),
}
impl Indices for BorrowedIndices<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Range(value) => value.len(),
            Self::Slice(value) => value.len(),
        }
    }
    fn get(&self, index: usize) -> usize {
        match self {
            Self::Range(value) => value.start + index,
            Self::Slice(value) => value[index],
        }
    }
}
trait Policy<'a> {
    type Indices: Indices;
    type Ranges;
    type Output;
    type Error;
    fn indices(&self, selection: &'a TensorSelection) -> Self::Indices;
    fn product(&self, key: &'a str, values: &[usize], outer: bool) -> Result<usize, Self::Error>;
    fn index_offset(&self, key: &'a str, index: usize, inner: usize) -> Result<usize, Self::Error>;
    fn initial(&mut self) -> Self::Ranges;
    fn last(ranges: &mut Self::Ranges) -> Option<&mut Range<usize>>;
    fn push(
        &self,
        key: &'a str,
        ranges: &mut Self::Ranges,
        value: Range<usize>,
    ) -> Result<(), Self::Error>;
    fn is_empty(ranges: &Self::Ranges) -> bool;
    fn single(self, range: Range<usize>, bounded: bool) -> Self::Output;
    fn finish(self, ranges: Self::Ranges, bounded: bool) -> Self::Output;
    fn error(&self, key: &'a str, cause: Cause<'a>) -> Self::Error;
}
struct Ordinary;
impl<'a> Policy<'a> for Ordinary {
    type Indices = Vec<usize>;
    type Ranges = Vec<Range<usize>>;
    type Output = super::SafetensorsReadPlan;
    type Error = StoreError;
    fn indices(&self, selection: &'a TensorSelection) -> Self::Indices {
        match selection {
            TensorSelection::Range { start, end, .. } => (*start..*end).collect(),
            TensorSelection::Indices { indices, .. } => indices.clone(),
            _ => unreachable!(),
        }
    }
    fn product(&self, _: &'a str, values: &[usize], _: bool) -> Result<usize, Self::Error> {
        Ok(values.iter().product())
    }
    fn index_offset(&self, _: &'a str, index: usize, inner: usize) -> Result<usize, Self::Error> {
        Ok(index * inner)
    }
    fn initial(&mut self) -> Self::Ranges {
        Vec::new()
    }
    fn last(ranges: &mut Self::Ranges) -> Option<&mut Range<usize>> {
        ranges.last_mut()
    }
    fn push(
        &self,
        _: &'a str,
        ranges: &mut Self::Ranges,
        value: Range<usize>,
    ) -> Result<(), Self::Error> {
        ranges.push(value);
        Ok(())
    }
    fn is_empty(ranges: &Self::Ranges) -> bool {
        ranges.is_empty()
    }
    fn single(self, range: Range<usize>, bounded: bool) -> Self::Output {
        super::SafetensorsReadPlan::single(range, bounded)
    }
    fn finish(self, ranges: Self::Ranges, physically_bounded: bool) -> Self::Output {
        super::SafetensorsReadPlan {
            ranges,
            physically_bounded,
        }
    }
    fn error(&self, key: &'a str, cause: Cause<'a>) -> Self::Error {
        SafetensorsReadError { key, cause }.into()
    }
}
enum Ranges<'a> {
    Count {
        count: usize,
        last: Option<Range<usize>>,
    },
    Fill {
        values: &'a mut [Range<usize>],
        used: usize,
    },
}
impl Ranges<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Count { count, .. } => *count,
            Self::Fill { used, .. } => *used,
        }
    }
}
struct Fixed<'a> {
    destination: Option<&'a mut [Range<usize>]>,
}
struct FixedResult<'a> {
    ranges: Ranges<'a>,
    physically_bounded: bool,
}
impl<'a> Fixed<'a> {
    fn count() -> Self {
        Self { destination: None }
    }
    fn fill(values: &'a mut [Range<usize>]) -> Self {
        Self {
            destination: Some(values),
        }
    }
}
impl<'a, 'd> Policy<'a> for Fixed<'d> {
    type Indices = BorrowedIndices<'a>;
    type Ranges = Ranges<'d>;
    type Output = FixedResult<'d>;
    type Error = SafetensorsReadError<'a>;
    fn indices(&self, selection: &'a TensorSelection) -> Self::Indices {
        match selection {
            TensorSelection::Range { start, end, .. } => BorrowedIndices::Range(*start..*end),
            TensorSelection::Indices { indices, .. } => BorrowedIndices::Slice(indices),
            _ => unreachable!(),
        }
    }
    fn product(&self, key: &'a str, values: &[usize], outer: bool) -> Result<usize, Self::Error> {
        values.iter().try_fold(1usize, |count, value| {
            count.checked_mul(*value).ok_or_else(|| {
                self.error(
                    key,
                    Cause::Geometry(if outer {
                        "outer stride overflow"
                    } else {
                        "inner stride overflow"
                    }),
                )
            })
        })
    }
    fn index_offset(&self, key: &'a str, index: usize, inner: usize) -> Result<usize, Self::Error> {
        index
            .checked_mul(inner)
            .ok_or_else(|| self.error(key, Cause::Geometry("nibble index offset overflow")))
    }
    fn initial(&mut self) -> Self::Ranges {
        match self.destination.take() {
            Some(values) => Ranges::Fill { values, used: 0 },
            None => Ranges::Count {
                count: 0,
                last: None,
            },
        }
    }
    fn last(ranges: &mut Self::Ranges) -> Option<&mut Range<usize>> {
        match ranges {
            Ranges::Count { last, .. } => last.as_mut(),
            Ranges::Fill { values, used } => used.checked_sub(1).and_then(|i| values.get_mut(i)),
        }
    }
    fn push(
        &self,
        key: &'a str,
        ranges: &mut Self::Ranges,
        value: Range<usize>,
    ) -> Result<(), Self::Error> {
        match ranges {
            Ranges::Count { count, last } => {
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| self.error(key, Cause::Geometry("range count overflow")))?;
                *last = Some(value);
            }
            Ranges::Fill { values, used } => {
                let target = values.get_mut(*used).ok_or_else(|| {
                    self.error(key, Cause::Geometry("counted range destination exhausted"))
                })?;
                *target = value;
                *used += 1;
            }
        }
        Ok(())
    }
    fn is_empty(ranges: &Self::Ranges) -> bool {
        ranges.len() == 0
    }
    fn single(mut self, range: Range<usize>, bounded: bool) -> Self::Output {
        let ranges = match self.destination.take() {
            Some(values) => {
                values[0] = range;
                Ranges::Fill { values, used: 1 }
            }
            None => Ranges::Count {
                count: 1,
                last: Some(range),
            },
        };
        FixedResult {
            ranges,
            physically_bounded: bounded,
        }
    }
    fn finish(self, ranges: Self::Ranges, physically_bounded: bool) -> Self::Output {
        FixedResult {
            ranges,
            physically_bounded,
        }
    }
    fn error(&self, key: &'a str, cause: Cause<'a>) -> Self::Error {
        SafetensorsReadError { key, cause }
    }
}

fn selected_elements<'a, P: Policy<'a>>(
    key: &'a str,
    shape: &[usize],
    policy: &P,
) -> Result<usize, P::Error> {
    // Same diagnostic and fold as the ordinary checked_elements helper.
    shape.iter().try_fold(1usize, |count, value| {
        count.checked_mul(*value).ok_or_else(|| {
            // Existing shared validation supplies the identical borrowed cause.
            policy.error(
                key,
                Cause::Selection(super::selection_validation::element_overflow(key)),
            )
        })
    })
}
fn nibble_misaligned<'a, P: Policy<'a>>(
    key: &'a str,
    indices: &P::Indices,
    inner: usize,
    storage: &P,
) -> Result<bool, P::Error> {
    for i in 0..indices.len() {
        if !storage
            .index_offset(key, indices.get(i), inner)?
            .is_multiple_of(2)
        {
            return Ok(true);
        }
    }
    Ok(false)
}
fn run<'a, P: Policy<'a>>(
    key: &'a str,
    bits: usize,
    shape: &[usize],
    payload_len: usize,
    selection: &'a TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
    mut storage: P,
) -> Result<P::Output, P::Error> {
    let bounded = matches!(policy, ReadPolicy::RequireBounded);
    if matches!(selection, TensorSelection::Full) {
        return Ok(storage.single(0..payload_len, true));
    }
    let scalar_bytes = bits.checked_div(8).filter(|_| bits.is_multiple_of(8));
    if let (
        Some(scalar_bytes),
        TensorSelection::Contiguous {
            offset_elements,
            shape,
        },
    ) = (scalar_bytes, selection)
    {
        let start = offset_elements
            .checked_mul(scalar_bytes)
            .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::ContiguousStart)))?;
        let end = selected_elements(key, shape, &storage)?
            .checked_mul(scalar_bytes)
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::ContiguousEnd)))?;
        if end > payload_len {
            return Err(storage.error(key, Cause::Invalid("contiguous byte span outside payload")));
        }
        return Ok(storage.single(start..end, true));
    }
    if let (
        Some(_),
        TensorSelection::Range {
            axis: 0,
            start,
            end,
        },
    ) = (scalar_bytes, selection)
    {
        let row_bytes = payload_len
            .checked_div(shape[0])
            .filter(|_| payload_len.is_multiple_of(shape[0]))
            .ok_or_else(|| storage.error(key, Cause::Invalid("payload is not row divisible")))?;
        let byte_start = start
            .checked_mul(row_bytes)
            .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::RowStart)))?;
        let byte_end = end
            .checked_mul(row_bytes)
            .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::RowEnd)))?;
        return Ok(storage.single(byte_start..byte_end, true));
    }
    if !bounded {
        return Ok(storage.single(0..payload_len, false));
    }
    let axis = match selection {
        TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => *axis,
        TensorSelection::Contiguous { .. } => {
            return Err(storage.error(
                key,
                Cause::Bounded("packed contiguous selection is not byte aligned"),
            ));
        }
        TensorSelection::Full => unreachable!(),
    };
    let indices = storage.indices(selection);
    let axis_len = shape[axis];
    let outer = storage.product(key, &shape[..axis], true)?;
    let inner = storage.product(key, &shape[axis + 1..], false)?;
    let output_bits = selected_elements(key, output_shape, &storage)?
        .checked_mul(bits)
        .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::SelectedBits)))?;
    if !output_bits.is_multiple_of(8) {
        return Err(storage.error(
            key,
            Cause::Bounded("selected packed payload is not byte aligned"),
        ));
    }
    let block_bytes = if bits == 4 {
        if !inner.is_multiple_of(2) || nibble_misaligned(key, &indices, inner, &storage)? {
            return Err(storage.error(
                key,
                Cause::Bounded("FP4 selection crosses a nibble boundary"),
            ));
        }
        inner / 2
    } else {
        inner
            .checked_mul(scalar_bytes.ok_or_else(|| {
                storage.error(
                    key,
                    Cause::Bounded("stored scalar width is not byte aligned"),
                )
            })?)
            .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::BlockBytes)))?
    };
    let mut ranges = storage.initial();
    for outer_index in 0..outer {
        for i in 0..indices.len() {
            let index = indices.get(i);
            let start = outer_index
                .checked_mul(axis_len)
                .and_then(|value| value.checked_add(index))
                .and_then(|value| value.checked_mul(block_bytes))
                .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::SelectionStart)))?;
            let end = start
                .checked_add(block_bytes)
                .ok_or_else(|| storage.error(key, Cause::Overflow(Overflow::SelectionEnd)))?;
            if end > payload_len {
                return Err(storage.error(key, Cause::Invalid("selection exceeds payload")));
            }
            if let Some(previous) = P::last(&mut ranges) {
                if previous.end == start {
                    previous.end = end;
                    continue;
                }
            }
            storage.push(key, &mut ranges, start..end)?;
        }
    }
    if P::is_empty(&ranges) {
        return Err(storage.error(key, Cause::Invalid("selection produced no physical ranges")));
    }
    Ok(storage.finish(ranges, true))
}
pub(super) fn ordinary(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    payload_len: usize,
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<super::SafetensorsReadPlan, StoreError> {
    run(
        key,
        dtype.bitsize(),
        shape,
        payload_len,
        selection,
        output_shape,
        policy,
        Ordinary,
    )
}

/// Reuses the actual bounded read geometry for an already inferred encoded
/// recipe. The admitted batch, not this geometry helper, owns file authority.
pub(crate) fn encoded_selection_plan<'s, 'd>(
    key: &'s str,
    bits: usize,
    shape: &'s [usize],
    payload_len: usize,
    selection: &'s TensorSelection,
    output_shape: &'d [usize],
) -> Result<SelectionReadDestinationPlan<'s, 'd>, SafetensorsReadError<'s>> {
    SelectionReadDestinationPlan::new(
        key,
        bits,
        shape,
        payload_len,
        selection,
        output_shape,
        ReadPolicy::RequireBounded,
    )
}

#[cfg(test)]
mod tests;
