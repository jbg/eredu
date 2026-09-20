//! Metadata-only projection of an admitted encoded batch into final output bytes.
use super::*;

/// Both coordinates are byte offsets. Source addresses the original batch's
/// concatenated full tensors; destination addresses the recipe's final output.
/// Construction and use remain within the checkpoint crate.
#[derive(Clone, Debug)]
pub(crate) struct EncodedRange {
    pub(crate) source: Range<usize>,
    pub(crate) destination: Range<usize>,
}

impl EncodedReadBatch {
    /// Preserve original source identities while replacing only read geometry.
    pub(crate) fn project_ranges(
        self,
        ranges: &[EncodedRange],
        output_bytes: usize,
    ) -> Result<Self, StoreError> {
        let read = PreparedEncodedRead {
            batch: self,
            _custody: (),
        };
        let plan = EncodedReadProjectionPlan::from_ranges(read, ranges, output_bytes)
            .map_err(|error| StoreError::from(error.cause))?;
        let projected = plan
            .construct(())
            .map_err(|error| StoreError::from(error.cause))?;
        Ok(projected.batch)
    }
}

/// Projection over exact retained source records. Counting may reorder existing
/// span metadata in place, but allocates nothing and reads no payload. The plan
/// owns the original read and its custody, including on later admission refusal.
pub struct EncodedReadProjectionPlan<'a, C> {
    read: PreparedEncodedRead<C>,
    ranges: &'a [EncodedRange],
    output_bytes: usize,
    backing_bytes: usize,
}
impl<'a, C> EncodedReadProjectionPlan<'a, C> {
    pub(super) fn new(
        read: PreparedEncodedRead<C>,
        mapping: &'a crate::recipe::EncodedRecipeMapping,
    ) -> Result<Self, EncodedProjectionBuildError<C>> {
        Self::from_ranges(read, mapping.encoded_ranges(), mapping.byte_len())
    }
    fn from_ranges(
        mut read: PreparedEncodedRead<C>,
        ranges: &'a [EncodedRange],
        output_bytes: usize,
    ) -> Result<Self, EncodedProjectionBuildError<C>> {
        let inspect = (|| {
            let mut end = 0;
            for row in ranges {
                if row.source.start > row.source.end
                    || row.source.end > read.batch.byte_len
                    || row.destination.start != end
                    || row.destination.start > row.destination.end
                    || row.destination.end > output_bytes
                    || row.source.len() != row.destination.len()
                {
                    return Err(EncodedProjectionError::Invalid);
                }
                end = row.destination.end;
            }
            if end != output_bytes {
                return Err(EncodedProjectionError::Invalid);
            }
            let mut covered = 0usize;
            let mut backing = 0usize;
            for spans in read
                .batch
                .shards
                .iter_mut()
                .map(|(_, source)| &mut source.spans)
                .chain(read.batch.memory.iter_mut().map(|source| &mut source.spans))
            {
                spans.sort_unstable_by_key(|span| span.destination.start);
                let plan = SpanProjectionPlan::new(spans, ranges)?;
                covered = covered
                    .checked_add(plan.covered)
                    .ok_or(EncodedProjectionError::Invalid)?;
                backing = backing
                    .checked_add(plan.layout.size())
                    .ok_or(EncodedProjectionError::Layout)?;
            }
            if covered != output_bytes {
                return Err(EncodedProjectionError::Invalid);
            }
            Ok(backing)
        })();
        match inspect {
            Ok(backing_bytes) => Ok(Self {
                read,
                ranges,
                output_bytes,
                backing_bytes,
            }),
            Err(cause) => Err(EncodedProjectionBuildError {
                cause,
                partial: read,
            }),
        }
    }
    /// New projected span backing and fixed constructor/result controls.
    /// Original read storage, mapping, allocator overhead, stack and subsequent
    /// read scratch remain separate. No source handle or tensor metadata is cloned.
    pub fn required_bytes<D>(&self) -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<SpanProjectionPlan<'static>>(),
            std::mem::size_of::<PreparedEncodedRead<(C, D)>>(),
            std::mem::size_of::<EncodedProjectionBuildError<(C, D)>>(),
            std::mem::size_of::<
                Result<PreparedEncodedRead<(C, D)>, EncodedProjectionBuildError<(C, D)>>,
            >(),
            std::mem::size_of::<Vec<ReadSpan>>(),
            std::mem::size_of::<ReadSpan>(),
            std::mem::size_of::<Result<(), EncodedProjectionError>>(),
        ]
        .into_iter()
        .try_fold(self.backing_bytes, usize::checked_add)
    }
    /// Construct projected spans under the supplied custody. Every replaced or
    /// untouched source record remains in the same owner. Reserve or geometry
    /// failure returns that exact partial read before either custody can retire.
    pub fn construct<D>(
        self,
        custody: D,
    ) -> Result<PreparedEncodedRead<(C, D)>, EncodedProjectionBuildError<(C, D)>> {
        let mut read = self.read.with_custody(custody);
        let result = (|| {
            let mut covered = 0usize;
            for spans in read
                .batch
                .shards
                .iter_mut()
                .map(|(_, source)| &mut source.spans)
                .chain(read.batch.memory.iter_mut().map(|source| &mut source.spans))
            {
                let plan = SpanProjectionPlan::new(spans, self.ranges)?;
                covered = covered
                    .checked_add(plan.covered)
                    .ok_or(EncodedProjectionError::Invalid)?;
                *spans = plan.build()?;
            }
            if covered != self.output_bytes {
                return Err(EncodedProjectionError::Invalid);
            }
            read.batch
                .shards
                .retain(|(_, shard)| !shard.spans.is_empty());
            read.batch.memory.retain(|source| !source.spans.is_empty());
            read.batch.byte_len = self.output_bytes;
            Ok(())
        })();
        match result {
            Ok(()) => Ok(read),
            Err(cause) => Err(EncodedProjectionBuildError {
                cause,
                partial: read,
            }),
        }
    }
}

/// Projection refusal retaining every original or replaced source record and
/// its existing custody. An incomplete projection is not exposed as a read.
pub struct EncodedProjectionBuildError<C> {
    cause: EncodedProjectionError,
    partial: PreparedEncodedRead<C>,
}
impl<C> EncodedProjectionBuildError<C> {
    /// Fixed geometry or allocator cause, without formatting or owner erasure.
    pub fn cause(&self) -> &EncodedProjectionError {
        &self.cause
    }
    /// Original tensor occurrences retained with the failed construction prefix.
    pub fn retained_tensors(&self) -> &[TensorMetadata] {
        self.partial.tensors()
    }
}
impl<C> std::fmt::Debug for EncodedProjectionBuildError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedProjectionBuildError")
            .field("cause", &self.cause)
            .field("partial", &self.partial)
            .finish()
    }
}
impl<C> std::fmt::Display for EncodedProjectionBuildError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<C> std::error::Error for EncodedProjectionBuildError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Original spans are ordered by their disjoint logical destination coordinates.
/// Their source coordinates may repeat or belong to different physical tensors.
struct SpanProjectionPlan<'a> {
    spans: &'a [ReadSpan],
    ranges: &'a [EncodedRange],
    count: usize,
    covered: usize,
    layout: std::alloc::Layout,
}
/// Fixed encoded projection geometry or allocation refusal.
#[derive(Clone, Debug, thiserror::Error)]
pub enum EncodedProjectionError {
    /// Source or destination coordinates do not describe the same bytes.
    #[error("encoded recipe range projection is inconsistent")]
    Invalid,
    /// Span backing or aggregate construction storage is not representable.
    #[error("encoded projection span layout overflow")]
    Layout,
    /// A supplied destination has a different count from the immutable plan.
    #[error("encoded projection destination has {actual} spans; expected {expected}")]
    Destination {
        /// Count established by the shared projection traversal.
        expected: usize,
        /// Number of supplied destination slots.
        actual: usize,
    },
    /// Exact projected-span backing could not be reserved.
    #[error("encoded projection span reserve failed: {0}")]
    Reserve(#[source] std::collections::TryReserveError),
}
impl<'a> SpanProjectionPlan<'a> {
    fn new(
        spans: &'a [ReadSpan],
        ranges: &'a [EncodedRange],
    ) -> Result<Self, EncodedProjectionError> {
        let mut previous = 0;
        for span in spans {
            if span.source.start > span.source.end
                || span.destination.start > span.destination.end
                || span.destination.start < previous
                || u64::try_from(span.destination.len()).ok()
                    != Some(span.source.end - span.source.start)
            {
                return Err(EncodedProjectionError::Invalid);
            }
            previous = span.destination.end;
        }
        let mut count = 0usize;
        let covered = project(spans, ranges, |_| {
            count = count.checked_add(1).ok_or(EncodedProjectionError::Layout)?;
            Ok(())
        })?;
        let layout = std::alloc::Layout::array::<ReadSpan>(count)
            .map_err(|_| EncodedProjectionError::Layout)?;
        Ok(Self {
            spans,
            ranges,
            count,
            covered,
            layout,
        })
    }
    fn build(&self) -> Result<Vec<ReadSpan>, EncodedProjectionError> {
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(self.count)
            .map_err(EncodedProjectionError::Reserve)?;
        spans.resize(
            self.count,
            ReadSpan {
                source: 0..0,
                destination: 0..0,
            },
        );
        self.fill_into(&mut spans)?;
        Ok(spans)
    }
    fn fill_into(&self, destination: &mut [ReadSpan]) -> Result<(), EncodedProjectionError> {
        if destination.len() != self.layout.size() / std::mem::size_of::<ReadSpan>() {
            return Err(EncodedProjectionError::Destination {
                expected: self.count,
                actual: destination.len(),
            });
        }
        let mut next = 0;
        project(self.spans, self.ranges, |span| {
            let slot = destination
                .get_mut(next)
                .ok_or(EncodedProjectionError::Invalid)?;
            *slot = span;
            next += 1;
            Ok(())
        })?;
        destination.sort_unstable_by_key(|span| span.source.start);
        Ok(())
    }
}

/// One projection traversal drives both counting and filling. It borrows the
/// original source coordinates and emits no payload or source-owner changes.
fn project(
    spans: &[ReadSpan],
    ranges: &[EncodedRange],
    mut emit: impl FnMut(ReadSpan) -> Result<(), EncodedProjectionError>,
) -> Result<usize, EncodedProjectionError> {
    let mut covered = 0usize;
    for row in ranges {
        let first = spans.partition_point(|span| span.destination.end <= row.source.start);
        for span in &spans[first..] {
            if span.destination.start >= row.source.end {
                break;
            }
            let start = span.destination.start.max(row.source.start);
            let end = span.destination.end.min(row.source.end);
            if start >= end {
                continue;
            }
            let length = end - start;
            let file_start = span
                .source
                .start
                .checked_add(
                    u64::try_from(start - span.destination.start)
                        .map_err(|_| EncodedProjectionError::Invalid)?,
                )
                .ok_or(EncodedProjectionError::Invalid)?;
            let file_end = file_start
                .checked_add(u64::try_from(length).map_err(|_| EncodedProjectionError::Invalid)?)
                .filter(|end| *end <= span.source.end)
                .ok_or(EncodedProjectionError::Invalid)?;
            let destination_start = row
                .destination
                .start
                .checked_add(start - row.source.start)
                .ok_or(EncodedProjectionError::Invalid)?;
            let destination_end = destination_start
                .checked_add(length)
                .ok_or(EncodedProjectionError::Invalid)?;
            covered = covered
                .checked_add(length)
                .ok_or(EncodedProjectionError::Invalid)?;
            emit(ReadSpan {
                source: file_start..file_end,
                destination: destination_start..destination_end,
            })?;
        }
    }

    Ok(covered)
}

#[cfg(test)]
mod tests;
